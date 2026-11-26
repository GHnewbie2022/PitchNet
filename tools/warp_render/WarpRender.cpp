// Offline check for the warp layer: runs PitchNet's own analysis and
// synthesis on a file, applies warp anchors and a pitch shift, and writes the
// result to a WAV file. Never plays audio.
//
//   WarpRender <in.wav> <out.wav> [--anchors s:d,s:d,...] [--shift semitones]
//              [--shift-range from:to] [--engine vocoder|psola] [--playback]
//              [--detector fcpe|swift|rmvpe|ensemble] [--dump-notes]
//              [--snapshot prefix]
//
// Anchors are source:timeline positions in seconds. --shift moves the pitch
// of every note whose recorded start lies in --shift-range (seconds, default
// the whole file). Run from the repository root so Resources/models is found.
// --snapshot draws the piano roll in the Warp and Timing tools to
// <prefix>_*.png, with a drag of the first anchor pushed past its limit.
#include "../../Source/Audio/SourceAudio.h"
#include "../../Source/Audio/AudioEngine.h"
#include "../../Source/Audio/KeyFinder/KeyFinder.h"
#include "../../Source/Audio/KeyFinder/ScaleTone.h"
#include "../../Source/Utils/FryDetector.h"
#include "../../Source/Utils/FryNoteCuts.h"
#include "../../Source/Audio/RecordingChange.h"
#include "../../Source/Audio/PitchEnsemble.h"
#include "../../Source/Audio/EditorController.h"
#include "../../Source/Audio/Synthesis/WarpedSource.h"
#include "../../Source/Audio/WarpLayer.h"
#include "../../Source/Utils/Constants.h"
#include "../../Source/Utils/MelSpectrogram.h"
#include "../../Source/Utils/OnnxRuntime.h"
#include "../../Source/Utils/PitchCurveProcessor.h"
#include "../../Source/Models/ProjectSerializer.h"
#include "../../Source/Models/DrawnCurve.h"
#include "../../Source/Audio/IO/MidiReference.h"
#include "../../Source/Audio/BreathSibilanceAnalysis.h"
#include "../../Source/Utils/ScaleUtils.h"
#include "../../Source/Undo/NoteActions.h"
#include "../../Source/Undo/PitchUndoManager.h"
#include "../../Source/Undo/SynthChoiceAction.h"
#include "../../Source/Undo/KeyChangeAction.h"
#include "../../Source/Undo/LockAction.h"
#include "../../Source/UI/PianoRollComponent.h"
#include "../../Source/Audio/NoteTiming.h"
#include "../../Source/UI/ParameterPanel.h"
#include "../../Source/UI/ToolbarComponent.h"
#include "../../Source/UI/MixerPanel.h"
#include "../../Source/UI/Wizard/WelcomeOverlay.h"
#include "../../Source/UI/Wizard/WizardOverlay.h"
#include "../../Source/UI/Main/ExportHelper.h"
#include "../../Source/UI/PianoRoll/NoteSplitter.h"
#include "../../Source/UI/PianoRoll/PianoRollInteractionContext.h"

#include "../../Source/JuceHeader.h"

#include <algorithm>
#include <limits>
#include <map>
#include <array>
#include <cmath>
#include <functional>
#include <atomic>
#include <vector>
#include <string>
#include <cstdio>
#include <memory>
#include <fcntl.h>
#include <unistd.h>

namespace
{
bool waitFor(const std::atomic<bool> &flag, int timeoutMs)
{
  const auto start = juce::Time::getMillisecondCounter();
  while (!flag.load())
  {
    juce::MessageManager::getInstance()->runDispatchLoopUntil(20);
    if (juce::Time::getMillisecondCounter() - start > static_cast<juce::uint32>(timeoutMs))
      return false;
  }
  return true;
}

bool writePng(juce::Component &component, const juce::File &file)
{
  const auto image = component.createComponentSnapshot(
      component.getLocalBounds(), true, 2.0f);
  file.deleteFile();
  juce::FileOutputStream stream(file);
  juce::PNGImageFormat png;
  return stream.openedOk() && png.writeImageToStream(image, stream);
}

juce::MouseEvent mouseAt(juce::Component &component, juce::Point<float> at,
                         juce::Point<float> downAt, juce::ModifierKeys mods)
{
  auto source = juce::Desktop::getInstance().getMainMouseSource();
  const auto now = juce::Time::getCurrentTime();
  return juce::MouseEvent(source, at, mods, juce::MouseInputSource::defaultPressure,
                          0.0f, 0.0f, 0.0f, 0.0f, &component, &component, now,
                          downAt, now, 1, false);
}

// "Restore note separation": merge and split notes, then restore, undo,
// redo and save/load. Prints PASS/FAIL lines.
void separationCheck(Project &project)
{
  PianoRollComponent roll;
  roll.setSize(1400, 640);
  roll.setProject(&project);
  auto &splitter = *roll.getInteractionContext().noteSplitter;
  auto cuts = [&]()
  {
    juce::String s;
    for (const auto &n : project.getNotes())
      if (!n.isRest() && n.getSrcStartFrame() < 130)
        s << "[" << n.getSrcStartFrame() << "," << n.getSrcEndFrame() << ") ";
    return s.trim();
  };
  auto find = [&](int src) -> Note *
  {
    for (auto &n : project.getNotes())
      if (!n.isRest() && n.getSrcStartFrame() == src)
        return &n;
    return nullptr;
  };
  auto check = [](const char *what, bool ok)
  { std::printf("%s %s\n", ok ? "PASS" : "FAIL", what); };

  const auto original = cuts();
  // The curve the piano roll draws: basePitch + deltaPitch, frame by frame.
  auto drawn = [&]()
  {
    const auto &ad = project.getAudioData();
    std::vector<float> curve;
    for (size_t i = 0; i < ad.basePitch.size() && i < ad.deltaPitch.size(); ++i)
      curve.push_back(ad.basePitch[i] + ad.deltaPitch[i]);
    return curve;
  };
  const auto drawnAtAnalysis = drawn();
  auto compareDrawn = [&](const char *label)
  {
    // Voiced frames inside notes: where the curve is drawn. (Gaps differ
    // after any rebuild, edit or not; see the control line.)
    const auto now = drawn();
    const auto &ad = project.getAudioData();
    float worst = 0.0f; int at = -1;
    for (const auto &n : project.getNotes())
    {
      if (n.isRest()) continue;
      for (int f = n.getStartFrame(); f < n.getEndFrame(); ++f)
      {
        const size_t i = static_cast<size_t>(f);
        if (i >= now.size() || i >= drawnAtAnalysis.size() ||
            i >= ad.voicedMask.size() || !ad.voicedMask[i]) continue;
        if (std::abs(now[i] - drawnAtAnalysis[i]) > worst)
        { worst = std::abs(now[i] - drawnAtAnalysis[i]); at = f; }
      }
    }
    std::printf("%s: drawn curve max diff %.4f semitones at frame %d\n", label, worst, at);
    return worst < 0.01f;
  };
  {
    // Is the note's stored curve what is drawn?
    const auto &ad = project.getAudioData();
    float worst = 0.0f;
    for (const auto &n : project.getNotes())
    {
      if (n.isRest() || n.getSrcStartFrame() >= 130) continue;
      const auto &od = n.getOriginalDeltaPitch();
      for (size_t k = 0; k < od.size(); ++k)
      {
        const size_t i = static_cast<size_t>(n.getStartFrame()) + k;
        if (i < ad.deltaPitch.size())
          worst = std::max(worst, std::abs(od[k] - ad.deltaPitch[i]));
      }
    }
    std::printf("analysis: note curves vs drawn delta max diff %.4f\n", worst);
  }
  {
    // Control: rebuild the curves with no edit at all.
    Project untouched = project;
    PitchCurveProcessor::rebuildBaseFromNotes(untouched);
    const auto &ad = untouched.getAudioData();
    float worst = 0.0f; int at = -1, voicedBad = 0, frames = 0;
    for (size_t i = 0; i < drawnAtAnalysis.size(); ++i)
    {
      const float d = std::abs(ad.basePitch[i] + ad.deltaPitch[i] - drawnAtAnalysis[i]);
      if (d > worst) { worst = d; at = static_cast<int>(i); }
      if (d > 0.01f) ++frames;
      if (d > 0.01f && i < ad.voicedMask.size() && ad.voicedMask[i]) ++voicedBad;
    }
    int inNotes = 0;
    for (size_t i = 0; i < drawnAtAnalysis.size(); ++i)
    {
      if (std::abs(ad.basePitch[i] + ad.deltaPitch[i] - drawnAtAnalysis[i]) <= 0.01f) continue;
      for (const auto &n : untouched.getNotes())
        if (!n.isRest() && static_cast<int>(i) >= n.getStartFrame() && static_cast<int>(i) < n.getEndFrame())
        { ++inNotes; if (inNotes <= 12) std::printf("  frame %zu (%.2f s) in note [%d,%d) voiced %d: %.2f -> %.2f\n", i, i*512/44100.0, n.getStartFrame(), n.getEndFrame(), (i < ad.voicedMask.size() && ad.voicedMask[i]) ? 1 : 0, drawnAtAnalysis[i], ad.basePitch[i] + ad.deltaPitch[i]); break; }
    }
    std::printf("control (rebuild, no edit): drawn max diff %.4f at frame %d, %d frames differ, %d voiced, %d inside notes\n",
                worst, at, frames, voicedBad, inNotes);
  }
  std::vector<Note> analysed;
  for (const auto &n : project.getNotes())
    if (!n.isRest() && n.getSrcStartFrame() < 130)
      analysed.push_back(n);
  // How far the restored notes are from the analysis: centre, curves, macro
  // source, and the absolute pitch (centre + deviation) frame by frame.
  auto compare = [&](const char *label)
  {
    float centre = 0.0f, curve = 0.0f, absolute = 0.0f, macro = 0.0f;
    for (const auto &orig : analysed)
    {
      const Note *now = find(orig.getSrcStartFrame());
      if (!now) { centre = 99.0f; continue; }
      centre = std::max(centre, std::abs(now->getMidiNote() - orig.getMidiNote()));
      macro = std::max(macro, std::abs(now->getLastNonMacroMidiNote() -
                                       orig.getLastNonMacroMidiNote()));
      const auto &a = now->getOriginalDeltaPitch();
      const auto &b = orig.getOriginalDeltaPitch();
      for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
      {
        curve = std::max(curve, std::abs(a[i] - b[i]));
        absolute = std::max(absolute, std::abs((now->getMidiNote() + a[i]) -
                                               (orig.getMidiNote() + b[i])));
      }
      if (a.size() != b.size()) curve = 99.0f;
      const bool tools = now->getTiltLeft() == orig.getTiltLeft() &&
                         now->getTiltRight() == orig.getTiltRight() &&
                         now->getVibrato() == orig.getVibrato() &&
                         now->getPitchOffset() == orig.getPitchOffset() &&
                         now->getBakedDeltaPitch() == orig.getBakedDeltaPitch() &&
                         std::abs(now->getOriginalMidiNote() - orig.getOriginalMidiNote()) < 0.001f;
      if (!tools) centre = std::max(centre, 50.0f);
    }
    std::printf("%s: max diff centre %.4f, curve %.4f, absolute %.4f, macro source %.4f\n",
                label, centre, curve, absolute, macro);
    return centre < 0.001f && curve < 0.001f;
  };
  std::printf("analysis: %s\n", original.toRawUTF8());
  Note *first = nullptr;
  for (auto &n : project.getNotes())
    if (!n.isRest()) { first = &n; break; }
  check("analysis spans recorded",
        first && first->getAnalysisSpans().size() == 1);
  check("nothing to restore after analysis", !splitter.canRestoreSeparation(*first));

  // Merge the first two notes and the result with the third.
  const int a = first->getSrcStartFrame();
  const int b = first->getSrcEndFrame();
  // Mess with the pitch first: move the second note up 2 semitones, tilt
  // the first, draw on the third.
  find(b)->setMidiNote(find(b)->getMidiNote() + 2.0f);
  find(a)->setTiltLeft(1.5f);
  {
    Note *third = find(find(b)->getSrcEndFrame());
    auto drawn = third->getOriginalDeltaPitch();
    for (auto &v : drawn) v += 0.7f;
    third->setBakedDeltaPitch(drawn);
  }
  PitchCurveProcessor::rebuildBaseFromNotes(project);
  check("merge 1+2", splitter.mergeNotes(find(a), find(b)));
  const int c = find(a)->getSrcEndFrame();
  check("merge (1+2)+3", splitter.mergeNotes(find(a), find(c)));
  const int mergedEnd = find(a)->getSrcEndFrame();
  std::printf("merged:   %s\n", cuts().toRawUTF8());
  check("merged note can be restored", splitter.canRestoreSeparation(*find(a)));

  // Restore through the undoable action, as the reset menu does.
  auto group = splitter.separationGroupFor(*find(a));
  std::vector<Note> after;
  splitter.restoreSeparation(*group, after);
  NoteSeparationAction action(&project, group->notes, after);
  std::printf("restored: %s\n", cuts().toRawUTF8());
  check("restore gives the analysis cuts", cuts() == original);
  check("restore gives the analysis pitch", compare("after merge+restore"));
  PitchCurveProcessor::rebuildBaseFromNotes(project);
  {
    const auto &ad = project.getAudioData();
    int frames = 0, voicedBad = 0, inNotes = 0;
    const auto now = drawn();
    for (size_t i = 0; i < now.size(); ++i)
    {
      const bool bad = std::abs(now[i] - drawnAtAnalysis[i]) > 0.01f;
      if (!bad) continue;
      ++frames;
      if (i < ad.voicedMask.size() && ad.voicedMask[i]) ++voicedBad;
      for (const auto &n : project.getNotes())
        if (!n.isRest() && static_cast<int>(i) >= n.getStartFrame() && static_cast<int>(i) < n.getEndFrame()) { ++inNotes; break; }
    }
    std::printf("after restore: %d frames differ, %d voiced, %d inside notes\n", frames, voicedBad, inNotes);
  }
  check("drawn curve after restore equals analysis", compareDrawn("after merge+restore"));
  action.undo();
  check("undo brings the merge back",
        find(a) && find(a)->getSrcEndFrame() == mergedEnd && !find(b) && !find(c));
  std::printf("undone:   %s\n", cuts().toRawUTF8());
  action.redo();
  check("redo restores again", cuts() == original);

  // Split a note in two by hand, then restore: the cut goes away.
  Note *second = find(b);
  const int mid = (second->getStartFrame() + second->getEndFrame()) / 2;
  check("manual split", splitter.splitNoteAtFrame(second, mid));
  // Move the new piece's pitch (the right half only: the note after it is
  // not part of this split).
  for (auto &n : project.getNotes())
    if (!n.isRest() && n.getStartFrame() == mid)
      n.setMidiNote(n.getMidiNote() - 3.0f);
  std::printf("split:    %s\n", cuts().toRawUTF8());
  auto splitGroup = splitter.separationGroupFor(*find(b));
  check("split note can be restored", splitGroup &&
        splitGroup->currentCuts != splitGroup->targetCuts);
  splitter.restoreSeparation(*splitGroup, after);
  check("restore joins the manual split", cuts() == original);
  check("restore after split gives the analysis pitch", compare("after split+restore"));

  // Split, move one half, merge back by hand: the cuts are the analysis
  // cuts again, and the merge itself gives the analysed pitch back (Cubase's
  // glue drops the halves' pitch edits; undo brings them back).
  {
    Note *n2 = find(b);
    const int m2 = (n2->getStartFrame() + n2->getEndFrame()) / 2;
    splitter.splitNoteAtFrame(n2, m2);
    Note *right = nullptr;
    for (auto &n : project.getNotes())
      if (!n.isRest() && n.getStartFrame() == m2) right = &n;
    right->setMidiNote(right->getMidiNote() + 4.0f);
    const int rightSrc = right->getSrcStartFrame();
    splitter.mergeNotes(find(b), find(rightSrc));
    const Note *merged = find(b);
    const auto *span = merged->exactAnalysisSpan();
    check("re-merged note is a detected note again", span != nullptr);
    float curveOff = span ? 0.0f : 99.0f;
    if (span)
      for (size_t i = 0; i < std::min(span->originalDelta.size(),
                                      merged->getOriginalDeltaPitch().size()); ++i)
        curveOff = std::max(curveOff, std::abs(span->originalDelta[i] -
                                               merged->getOriginalDeltaPitch()[i]));
    check("merging back gives the analysed pitch (Cubase glue)",
          span && std::abs(merged->getMidiNote() - span->midi) < 0.001f &&
              merged->isPitchUntouched() && curveOff < 0.002f);
    Note copy = *merged;
    copy.restoreAnalysisPitch();
    const Note *orig = nullptr;
    for (const auto &n : analysed) if (n.getSrcStartFrame() == b) orig = &n;
    check("restoring its pitch gives the analysed pitch",
          orig && copy.getMidiNote() == orig->getMidiNote() &&
          copy.getOriginalDeltaPitch() == orig->getOriginalDeltaPitch());
  }

  // Split, move one half (its own partial curve rebuild, as a drag does),
  // then undo the move and the split: inside the notes the drawn curve must
  // be the one before the split.
  {
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    const auto before = drawn();
    PitchUndoManager history;
    splitter.setUndoManager(&history);
    Note *target = find(b);
    const int cut = (target->getStartFrame() + target->getEndFrame()) / 2;
    splitter.splitNoteAtFrame(target, cut);
    Note *piece = nullptr;
    for (auto &n : project.getNotes())
      if (!n.isRest() && n.getStartFrame() == cut) piece = &n;
    const float oldMidi = piece->getMidiNote();
    piece->setMidiNote(oldMidi + 2.0f);
    PitchCurveProcessor::rebuildBaseFromNotesForDrag(project, {piece});
    piece->setMidiNote(oldMidi);                       // "undo" of the move
    PitchCurveProcessor::rebuildBaseFromNotesForDrag(project, {piece});
    history.undo();                                     // undo the split
    splitter.setUndoManager(nullptr);
    const auto after = drawn();
    const auto &ad = project.getAudioData();
    float worst = 0.0f;
    for (const auto &n : project.getNotes())
    {
      if (n.isRest()) continue;
      for (int i = n.getStartFrame(); i < n.getEndFrame() && i < static_cast<int>(after.size()); ++i)
        if (i < static_cast<int>(ad.voicedMask.size()) && ad.voicedMask[static_cast<size_t>(i)])
          worst = std::max(worst, std::abs(after[static_cast<size_t>(i)] - before[static_cast<size_t>(i)]));
    }
    std::printf("split + move + undo all: drawn curve in voiced note frames max diff %.4f\n", worst);
    check("undo back to the start redraws the curve", worst < 0.01f);
  }

  // Save and load keep the spans.
  splitter.mergeNotes(find(a), find(b));
  const auto json = ProjectSerializer::toJson(project);
  Project loaded;
  ProjectSerializer::fromJson(loaded, json);
  const Note *reloaded = nullptr;
  for (const auto &n : loaded.getNotes())
    if (!n.isRest() && n.getSrcStartFrame() == a) { reloaded = &n; break; }
  check("spans survive save and load",
        reloaded && reloaded->getAnalysisSpans() == find(a)->getAnalysisSpans() &&
        reloaded->getAnalysisSpans().size() == 2);
  {
    bool curves = reloaded != nullptr;
    if (reloaded)
      for (size_t i = 0; i < reloaded->getAnalysisSpans().size(); ++i)
      {
        const auto &x = reloaded->getAnalysisSpans()[i];
        const auto &y = find(a)->getAnalysisSpans()[i];
        curves = curves && !x.originalDelta.empty() && x.midi == y.midi &&
                 x.originalDelta.size() == y.originalDelta.size();
        for (size_t k = 0; curves && k < x.originalDelta.size(); ++k)
          curves = std::abs(x.originalDelta[k] - y.originalDelta[k]) < 1.0e-5f;
      }
    check("analysed pitch and curves survive save and load", curves);
  }
}


// Paint cost while scrolling, at a range of zooms: each step scrolls a bit
// and renders the piano roll the way the screen does (2x, native image).
void paintBench(Project &project)
{
  PianoRollComponent roll;
  roll.setSize(870, 900);
  roll.setProject(&project);
  std::vector<float> pitches;
  for (const auto &note : project.getNotes())
    if (!note.isRest())
      pitches.push_back(note.getMidiNote());
  std::nth_element(pitches.begin(), pitches.begin() + pitches.size() / 2,
                   pitches.end());
  const float median = pitches.empty() ? 64.0f : pitches[pitches.size() / 2];
  struct Config { float pps, semi, blob; };
  const Config configs[] = {
      {100, 64, 1}, {500, 120, 1}, {500, 120, 3}, {2000, 120, 1},
      {2000, 240, 1}, {2000, 240, 3}, {500, 240, 3}, {2000, 64, 3}};
  const int only = juce::SystemStats::getEnvironmentVariable("BENCH_ONLY", "-1")
                       .getIntValue();
  int index = -1;
  for (const auto &c : configs)
  {
    if (++index, only >= 0 && index != only)
      continue;
    roll.setPixelsPerSecond(c.pps);
    roll.setPixelsPerSemitone(c.semi);
    roll.setBlobHeightScale(c.blob);
    roll.centerOnPitchRange(median - 2.0f, median + 2.0f);
    const double startX = 4.5 * c.pps;
    std::vector<double> ms;
    const int steps = juce::SystemStats::getEnvironmentVariable(
                          "BENCH_STEPS", "24").getIntValue();
    for (int step = 0; step < steps; ++step)
    {
      roll.setScrollX(startX + step * juce::SystemStats::getEnvironmentVariable(
                                         "BENCH_STRIDE", "12").getDoubleValue());
      const double t0 = juce::Time::getMillisecondCounterHiRes();
      auto image = roll.createComponentSnapshot(roll.getLocalBounds(), true, 2.0f);
      ms.push_back(juce::Time::getMillisecondCounterHiRes() - t0);
      if (ms.back() > 40.0)
        std::printf("  slow step %d: %.1f ms\n", step, ms.back());
      juce::ignoreUnused(image);
    }
    std::sort(ms.begin(), ms.end());
    std::printf("pps %6.0f  semitone %5.0f  blob %.0f   median %6.1f ms  worst %6.1f ms\n",
                c.pps, c.semi, c.blob, ms[ms.size() / 2], ms.back());
  }
}

void snapshots(Project &project, const juce::String &prefix)
{
  PianoRollComponent roll;
  roll.setSize(1400, 640);
  roll.setProject(&project);
  roll.setPixelsPerSecond(300.0f);
  roll.setScrollX(0.0);
  // Centre on the median note: one stray low note would pull a min..max
  // range off the singing.
  std::vector<float> pitches;
  for (const auto &note : project.getNotes())
    if (!note.isRest())
      pitches.push_back(note.getMidiNote());
  if (!pitches.empty())
  {
    std::nth_element(pitches.begin(), pitches.begin() + pitches.size() / 2,
                     pitches.end());
    const float median = pitches[pitches.size() / 2];
    roll.centerOnPitchRange(median - 6.0f, median + 6.0f);
  }
  const auto dir = juce::File::getCurrentWorkingDirectory();
  auto shot = [&](const juce::String &name)
  {
    const auto file = dir.getChildFile(prefix + "_" + name + ".png");
    std::printf("snapshot %s %s\n", file.getFullPathName().toRawUTF8(),
                writePng(roll, file) ? "ok" : "FAILED");
  };

  auto view = [&](const char *label)
  {
    std::printf("%s: %.1f px/s, %.1f px/semitone, view %.2f-%.2f s\n", label,
                roll.getPixelsPerSecond(), roll.getPixelsPerSemitone(),
                roll.getScrollX() / roll.getPixelsPerSecond(),
                (roll.getScrollX() + roll.getVisibleContentWidth()) /
                    roll.getPixelsPerSecond());
  };
  view("start");
  roll.zoomHorizontallyBy(1.5f);
  view("G");
  roll.zoomHorizontallyBy(1.0f / 1.5f);
  view("H");
  roll.zoomVerticallyBy(1.25f);
  view("Shift+G");
  roll.zoomVerticallyBy(1.0f / 1.25f);
  view("Shift+H");
  roll.zoomToFullLength();
  view("Shift+F");
  shot("zoom_full");
  int picked = 0;
  for (auto &note : project.getNotes())
    if (!note.isRest() && picked < 3)
    {
      note.setSelected(true);
      ++picked;
    }
  roll.zoomToNotes();
  view("Shift+S (first 3 notes)");
  shot("zoom_selected");
  for (auto &note : project.getNotes())
    note.setSelected(false);
  roll.setPixelsPerSecond(300.0f);
  roll.setPlayheadTime(2.0);
  roll.zoomToNotes();
  view("Shift+S (nothing selected, playhead 2.00 s)");
  roll.setPixelsPerSecond(300.0f);
  roll.setScrollX(0.0);
  roll.setEditMode(EditMode::Select);
  shot("select");
  roll.setEditMode(EditMode::Warp);
  shot("warp");

  const auto &anchors = project.getAudioData().warpMap.anchors();
  if (anchors.empty())
  {
    // No anchors yet: drag the end of the first sung note (spec section 4).
    const Note *first = nullptr;
    for (const auto &note : project.getNotes())
      if (!note.isRest())
      {
        first = &note;
        break;
      }
    if (first == nullptr)
      return;
    auto &context = roll.getInteractionContext();
    const float edgeX = static_cast<float>(first->getEndFrame()) * HOP_SIZE /
                            SAMPLE_RATE * roll.getPixelsPerSecond() -
                        static_cast<float>(roll.getScrollX()) +
                        PianoRollComponent::pianoKeysWidth;
    const float rowY = context.midiToY(first->getAdjustedMidiNote()) +
                       roll.getPixelsPerSemitone() * 0.5f -
                       static_cast<float>(roll.getScrollY()) +
                       PianoRollComponent::headerHeight;
    const juce::Point<float> edge(edgeX, rowY), offRow(edgeX, rowY + 300.0f);
    const juce::ModifierKeys left(juce::ModifierKeys::leftButtonModifier);
    roll.mouseMove(mouseAt(roll, edge, edge, {}));
    shot("boundary_hover");
    std::printf("note edge at %.3f s, row %.1f px\n",
                first->getEndFrame() * HOP_SIZE / double(SAMPLE_RATE), rowY);
    roll.mouseDown(mouseAt(roll, edge, edge, left));
    roll.mouseDrag(mouseAt(roll, {edgeX + 90.0f, rowY}, edge, left));
    shot("boundary_drag");
    roll.mouseUp(mouseAt(roll, {edgeX + 90.0f, rowY}, edge, {}));
    std::printf("after boundary drag:");
    for (const auto &anchor : project.getAudioData().warpMap.anchors())
      std::printf(" %.3f->%.3f", anchor.source * HOP_SIZE / SAMPLE_RATE,
                  anchor.destination * HOP_SIZE / SAMPLE_RATE);
    std::printf("\n");
    // A click 200 px above the notes row places a single anchor instead.
    const auto count = project.getAudioData().warpMap.anchors().size();
    const float newEdgeX = edgeX + 90.0f;
    const juce::Point<float> below(newEdgeX + 40.0f, rowY - 200.0f);
    roll.mouseDown(mouseAt(roll, below, below, left));
    roll.mouseUp(mouseAt(roll, below, below, {}));
    std::printf("click off the row added %d anchor(s)\n",
                static_cast<int>(project.getAudioData().warpMap.anchors().size() - count));
    juce::ignoreUnused(offRow);
    return;
  }
  // Screen x of the first anchor: world x minus scroll plus the piano keys.
  const float worldX = static_cast<float>(
      anchors.front().destination * HOP_SIZE / SAMPLE_RATE * 300.0);
  const float x = worldX - static_cast<float>(roll.getScrollX()) +
                  PianoRollComponent::pianoKeysWidth;
  const juce::Point<float> down(x, 300.0f);
  const juce::ModifierKeys left(juce::ModifierKeys::leftButtonModifier);
  roll.mouseMove(mouseAt(roll, down, down, {}));
  shot("hover");
  roll.mouseDown(mouseAt(roll, down, down, left));
  roll.mouseDrag(mouseAt(roll, {x + 120.0f, 300.0f}, down, left));
  shot("drag");
  roll.mouseDrag(mouseAt(roll, {40.0f, 300.0f}, down, left));
  shot("limit");
  roll.mouseUp(mouseAt(roll, {40.0f, 300.0f}, down, {}));
  shot("released");

  // Shift + Alt + click on the canvas sets the playhead (MainComponent does
  // setPlayheadTime from onSeek).
  roll.onSeek = [&](double time)
  {
    roll.setPlayheadTime(time);
    std::printf("playhead set to %.3f s\n", time);
  };
  {
    const float clickX = static_cast<float>(1.25 * roll.getPixelsPerSecond() -
                                            roll.getScrollX()) +
                         PianoRollComponent::pianoKeysWidth;
    const juce::Point<float> at(clickX, 400.0f);
    const juce::ModifierKeys shiftAlt(juce::ModifierKeys::leftButtonModifier |
                                      juce::ModifierKeys::shiftModifier |
                                      juce::ModifierKeys::altModifier);
    const auto anchorsBefore = project.getAudioData().warpMap.anchors().size();
    roll.mouseDown(mouseAt(roll, at, at, shiftAlt));
    roll.mouseUp(mouseAt(roll, at, at, {}));
    std::printf("anchors unchanged by Shift+Alt+click: %s\n",
                project.getAudioData().warpMap.anchors().size() == anchorsBefore
                    ? "yes" : "NO");
    shot("playhead");
  }

  auto dumpAnchors = [&](const char *label)
  {
    std::printf("%s:", label);
    for (const auto &anchor : project.getAudioData().warpMap.anchors())
      std::printf(" %.3f->%.3f", anchor.source * HOP_SIZE / SAMPLE_RATE,
                  anchor.destination * HOP_SIZE / SAMPLE_RATE);
    std::printf("  pending %d\n", project.getAudioData().warpLayoutPending ? 1 : 0);
  };
  dumpAnchors("after drag");

  // Type a value into the ratio strip: the segment between the 2nd and 3rd
  // anchors (world x of 2.5 s), "150%".
  project.getAudioData().warpLayoutPending = false;
  const float stripX = static_cast<float>(2.5 * 300.0 - roll.getScrollX()) +
                       PianoRollComponent::pianoKeysWidth;
  const juce::Point<float> strip(stripX, PianoRollComponent::headerHeight + 8.0f);
  roll.mouseDoubleClick(mouseAt(roll, strip, strip, left));
  for (auto *child : roll.getChildren())
    if (auto *editor = dynamic_cast<juce::TextEditor *>(child))
    {
      std::printf("editor shows \"%s\"\n", editor->getText().toRawUTF8());
      shot("input");
      editor->setText("150%", false);
      editor->onReturnKey();
      break;
    }
  dumpAnchors("after typing 150%");
  shot("typed");

  // Alt + click on the last anchor deletes it.
  project.getAudioData().warpLayoutPending = false;
  const auto &now = project.getAudioData().warpMap.anchors();
  const float lastX = static_cast<float>(now.back().destination * HOP_SIZE /
                                         SAMPLE_RATE * 300.0 - roll.getScrollX()) +
                      PianoRollComponent::pianoKeysWidth;
  const juce::Point<float> last(lastX, 300.0f);
  const juce::ModifierKeys altLeft(juce::ModifierKeys::leftButtonModifier |
                                   juce::ModifierKeys::altModifier);
  roll.mouseDown(mouseAt(roll, last, last, altLeft));
  roll.mouseUp(mouseAt(roll, last, last, juce::ModifierKeys::altModifier));
  dumpAnchors("after Alt+click");
  shot("deleted");
}

bool writeWav(const juce::File &file, const juce::AudioBuffer<float> &buffer,
              double rate, int bits = 24)
{
  file.deleteFile();
  juce::WavAudioFormat format;
  std::unique_ptr<juce::OutputStream> stream(file.createOutputStream());
  if (!stream)
    return false;
  std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(
      stream.get(), rate, static_cast<unsigned int>(buffer.getNumChannels()), bits,
      {}, 0));
  if (!writer)
    return false;
  stream.release();
  return writer->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples());
}

// Split a note, edit the halves, merge them back: the merged note must sing
// the analysis again (as before the split), undo must bring the edited
// halves back and redo the merge.
void mergeCheck(Project &project)
{
  const auto noteAt = [&project](int start) -> Note *
  {
    for (auto &note : project.getNotes())
      if (!note.isRest() && note.getStartFrame() == start)
        return &note;
    return nullptr;
  };
  auto &audio = project.getAudioData();
  const auto sung = [&](int start, int end)
  {
    std::vector<float> pitch;
    for (int f = start; f < end; ++f)
      pitch.push_back(audio.basePitch[static_cast<size_t>(f)] +
                      audio.deltaPitch[static_cast<size_t>(f)]);
    return pitch;
  };
  const auto maxDiff = [](const std::vector<float> &a, const std::vector<float> &b)
  {
    float worst = a.size() == b.size() ? 0.0f : 1.0e9f;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
      worst = std::max(worst, std::abs(a[i] - b[i]));
    return worst;
  };

  // The longest notes, one per scenario.
  std::vector<int> starts;
  for (const auto &note : project.getNotes())
    if (!note.isRest() && note.getDurationFrames() >= 16)
      starts.push_back(note.getStartFrame());
  std::sort(starts.begin(), starts.end(), [&](int a, int b)
            { return noteAt(a)->getDurationFrames() >
                     noteAt(b)->getDurationFrames(); });
  struct Scenario { const char *name; std::function<void(Note &, Note &)> edit; };
  const Scenario scenarios[] = {
      {"no edit", [](Note &, Note &) {}},
      {"left drift -2.4", [](Note &l, Note &) { l.setPitchDrift(-2.4f); }},
      {"both drift 0.45", [](Note &l, Note &r) { l.setPitchDrift(0.45f); r.setPitchDrift(0.45f); }},
      {"right +1 semitone", [](Note &, Note &r) { r.setMidiNote(r.getMidiNote() + 1.0f); }},
      {"left vibrato 0 + tilt", [](Note &l, Note &) { l.setVibrato(0.0f); l.setTiltRight(0.8f); }},
  };
  int failures = 0;
  // Splitting and merging an untouched note anywhere: nothing around it may
  // move - not at the boundary, not in its neighbours' first frames.
  {
    int worstSplit = 0, worstMerge = 0, worstUndo = 0;
    float maxSplit = 0.0f, maxMerge = 0.0f, maxUndo = 0.0f;
    int tried = 0;
    for (size_t k = 0; k < starts.size() && tried < 30; ++k)
    {
      Note *note = noteAt(starts[k]);
      if (!note || note->hasBakedDeltaPitch() || note->getStartFrame() < 20)
        continue;
      const int start = note->getStartFrame(), end = note->getEndFrame();
      if (end - start < 12)
        continue;
      ++tried;
      const int split = start + (end - start) / 2;
      // Only frames some note sings: a gap's line is a transition, not pitch.
      const auto inNote = [&](int f)
      {
        return std::any_of(project.getNotes().begin(), project.getNotes().end(),
                           [f](const Note &n)
                           { return !n.isRest() && n.getStartFrame() <= f && n.getEndFrame() > f; });
      };
      const auto sungInNotes = [&](int from, int to)
      {
        auto pitch = sung(from, to);
        for (int f = from; f < to; ++f)
          if (!inNote(f))
            pitch[static_cast<size_t>(f - from)] = 0.0f;
        return pitch;
      };
      const auto around = sungInNotes(start - 12, end + 12);
      PitchUndoManager undo;
      NoteSplitter splitter;
      splitter.setProject(&project);
      splitter.setUndoManager(&undo);
      splitter.splitNoteAtFrame(note, split);
      const float s1 = maxDiff(around, sungInNotes(start - 12, end + 12));
      splitter.mergeNotes(noteAt(start), noteAt(split));
      const float m1 = maxDiff(around, sungInNotes(start - 12, end + 12));
      undo.undo();
      undo.undo();
      const float u1 = maxDiff(around, sungInNotes(start - 12, end + 12));
      if (s1 > maxSplit) { maxSplit = s1; worstSplit = start; }
      if (m1 > maxMerge) { maxMerge = m1; worstMerge = start; }
      if (u1 > maxUndo) { maxUndo = u1; worstUndo = start; }
    }
    std::printf("untouched split/merge over %d notes: split %.4f (at %d), merge %.4f (at %d), undo %.4f (at %d)\n",
                tried, maxSplit, worstSplit, maxMerge, worstMerge, maxUndo, worstUndo);
    if (maxSplit > 0.002f || maxMerge > 0.02f || maxUndo > 0.002f)
      ++failures;
  }
  size_t next = 0;
  for (const auto &scenario : scenarios)
  {
    if (next >= starts.size())
      break;
    const int start = starts[next++];
    Note *note = noteAt(start);
    const int end = note->getEndFrame();
    const int split = start + (end - start) / 2;
    PitchUndoManager undo;
    NoteSplitter splitter;
    splitter.setProject(&project);
    splitter.setUndoManager(&undo);
    const auto unsplit = sung(start, end);
    splitter.splitNoteAtFrame(note, split);
    Note *left = noteAt(start);
    Note *right = noteAt(split);
    scenario.edit(*left, *right);
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    const auto before = sung(start, end);
    left = noteAt(start);
    right = noteAt(split);
    const bool merged = splitter.mergeNotes(left, right);
    const auto after = sung(start, end);
    // Merging goes back to the note as analysed (Cubase's glue), whatever
    // was done to the halves.
    const auto &expected = unsplit;
    const float mergeDiff = maxDiff(expected, after);
    undo.undo();
    const float undoDiff = maxDiff(before, sung(start, end));
    const bool halvesBack = noteAt(split) != nullptr;
    undo.redo();
    const float redoDiff = maxDiff(expected, sung(start, end));
    const bool baked = noteAt(start)->hasBakedDeltaPitch();
    const float tolerance = 0.002f;
    const bool pass = merged && mergeDiff < tolerance && undoDiff < 0.002f &&
                      redoDiff < tolerance && halvesBack && !baked;
    failures += pass ? 0 : 1;
    std::printf("%-22s frames [%d,%d) merge %.4f undo %.4f redo %.4f baked %d  %s\n",
                scenario.name, start, end, mergeDiff, undoDiff, redoDiff,
                baked ? 1 : 0, pass ? "ok" : "FAIL");
  }

  // The same boundary split, edited and merged again and again: every merge
  // lands on the analysis, so nothing accumulates.
  if (next < starts.size())
  {
    const int start = starts[next++];
    const int end = noteAt(start)->getEndFrame();
    const int split = start + (end - start) / 2;
    const auto unsplit = sung(start, end);
    NoteSplitter splitter;
    splitter.setProject(&project);
    for (int round = 1; round <= 5; ++round)
    {
      splitter.splitNoteAtFrame(noteAt(start), split);
      noteAt(start)->setPitchDrift(-2.4f);
      noteAt(split)->setMidiNote(noteAt(split)->getMidiNote() + 1.0f);
      PitchCurveProcessor::rebuildBaseFromNotes(project);
      const auto edited = sung(start, split);
      const auto neighbour = sung(split, end);
      splitter.splitNoteAtFrame(noteAt(start), start + (split - start) / 2);
      PitchCurveProcessor::rebuildBaseFromNotes(project);
      // The split note itself keeps exactly what it sang; the base curve's
      // 80 ms smoothing may still bend the neighbour's first frames, as any
      // change of a note's centre does.
      const float splitDiff = maxDiff(edited, sung(start, split));
      const auto after = sung(split, end);
      int bent = 0;
      for (size_t i = 0; i < after.size(); ++i)
        if (std::abs(after[i] - neighbour[i]) > 0.002f)
          bent = static_cast<int>(i) + 1;
      std::printf("  neighbour frames bent: %d\n", bent);
      splitter.mergeNotes(noteAt(start), noteAt(start + (split - start) / 2));
      splitter.mergeNotes(noteAt(start), noteAt(split));
      const float mergeDiff = maxDiff(unsplit, sung(start, end));
      std::printf("round %d: split keeps edits %.4f  merge back to analysis %.4f\n",
                  round, splitDiff, mergeDiff);
      if (splitDiff > 0.002f || mergeDiff > 0.002f)
        ++failures;
    }
  }
  // Split a note and merge its right half with the next detected note, again
  // and again: every untouched note must keep singing the recording.
  {
    const auto recorded = [&](int f)
    { return freqToMidi(audio.denseF0[static_cast<size_t>(f)]); };
    Note *left = nullptr, *next = nullptr;
    for (auto &n : project.getNotes())
    {
      if (n.isRest() || !n.isPitchUntouched() || n.getDurationFrames() < 16 ||
          n.getStartFrame() < 40)
        continue;
      for (auto &m : project.getNotes())
        if (!m.isRest() && m.getStartFrame() == n.getEndFrame() &&
            m.getSrcStartFrame() == n.getSrcEndFrame() && m.isPitchUntouched() &&
            std::abs(m.getMidiNote() - n.getMidiNote()) > 2.0f)
          next = &m;
      if (next)
      {
        left = &n;
        break;
      }
      next = nullptr;
    }
    if (left && next)
    {
      const int s0 = left->getStartFrame(), boundary = next->getStartFrame(),
                e0 = next->getEndFrame(), cut = (s0 + boundary) / 2;
      NoteSplitter splitter;
      splitter.setProject(&project);
      float worst = 0.0f;
      for (int round = 0; round < 3; ++round)
      {
        splitter.splitNoteAtFrame(noteAt(s0), cut);
        splitter.mergeNotes(noteAt(cut), noteAt(boundary));
        for (int f = s0 - 12; f < e0 + 12; ++f)
        {
          const auto i = static_cast<size_t>(f);
          bool untouchedNote = false;
          for (const auto &n : project.getNotes())
            if (!n.isRest() && n.getStartFrame() <= f && n.getEndFrame() > f)
              untouchedNote = n.isPitchUntouched();
          if (untouchedNote && audio.denseF0[i] > 0.0f)
            worst = std::max(worst, std::abs(audio.basePitch[i] + audio.deltaPitch[i] - recorded(f)));
        }
        // Back to how it was cut, for the next round.
        splitter.splitNoteAtFrame(noteAt(cut), boundary);
        splitter.mergeNotes(noteAt(s0), noteAt(cut));
      }
      std::printf("cross merge rounds: untouched notes off the recording by %.4f\n", worst);
      if (worst > 0.002f)
        ++failures;
    }
    else
      std::printf("cross merge rounds: no adjacent pair found\n");
  }
  std::printf("merge check: %d failure(s)\n", failures);
}


// The Pitch Shaper tool without a confirmation: each click is applied and is one
// undo step; Esc ends the curve and keeps it; undo walks back one step at a
// time to the pitch before the curve.
void drawCheck(Project &project)
{
  auto &audio = project.getAudioData();
  PianoRollComponent roll;
  PitchUndoManager undo;
  roll.setSize(1400, 900);
  roll.setProject(&project);
  roll.setUndoManager(&undo);
  roll.setPixelsPerSecond(400.0f);
  roll.setEditMode(EditMode::PitchShaper);

  const Note *longest = nullptr;
  for (const auto &note : project.getNotes())
    if (!note.isRest() && (!longest || note.getDurationFrames() > longest->getDurationFrames()))
      longest = &note;
  const int start = longest->getStartFrame(), end = longest->getEndFrame();
  const float target = std::round(longest->getAdjustedMidiNote()) + 2.0f;
  roll.setScrollX(framesToSeconds(start) * 400.0 - 100.0);
  roll.centerOnPitchRange(target - 4.0f, target + 4.0f);

  const auto pitchAt = [&](int f)
  { return audio.basePitch[static_cast<size_t>(f)] + audio.deltaPitch[static_cast<size_t>(f)]; };
  const auto screen = [&](int frame, float midi)
  {
    const float x = PianoRollComponent::pianoKeysWidth +
                    static_cast<float>(framesToSeconds(frame) * roll.getPixelsPerSecond() -
                                       roll.getScrollX());
    const float y = PianoRollComponent::headerHeight +
                    (MAX_MIDI_NOTE - midi) * roll.getPixelsPerSemitone() +
                    roll.getPixelsPerSemitone() * 0.5f - static_cast<float>(roll.getScrollY());
    return juce::Point<float>(x, y);
  };
  const auto press = [&](juce::Point<float> at, juce::Point<float> to, bool alt)
  {
    auto source = juce::Desktop::getInstance().getMainMouseSource();
    const auto now = juce::Time::getCurrentTime();
    const auto mods = juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier |
                                         (alt ? juce::ModifierKeys::altModifier : 0));
    juce::MouseEvent down(source, at, mods, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, &roll, &roll,
                          now, at, now, 1, false);
    roll.mouseDown(down);
    if (to != at)
    {
      juce::MouseEvent drag(source, to, mods, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, &roll, &roll,
                            now, at, now, 1, true);
      roll.mouseDrag(drag);
    }
    juce::MouseEvent up(source, to, juce::ModifierKeys(), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        &roll, &roll, now, at, now, 1, to != at);
    roll.mouseUp(up);
  };
  const auto click = [&](juce::Point<float> at) { press(at, at, false); };
  const auto curves = [&]() { return project.getDrawnCurves(); };

  const int a = start + (end - start) / 4, b = start + 3 * (end - start) / 4;
  const int mid = (a + b) / 2;
  const float original = pitchAt(mid);
  int failures = 0;
  const auto expect = [&](bool ok, const char *what)
  {
    std::printf("  %-52s %s\n", what, ok ? "ok" : "FAIL");
    failures += ok ? 0 : 1;
  };

  click(screen(a, target));
  expect(undo.canUndo(), "first click is applied and undoable at once");
  click(screen(b, target));
  expect(std::abs(pitchAt(mid) - target) < 0.05f, "two anchors draw the curve");
  roll.keyPressed(juce::KeyPress(juce::KeyPress::escapeKey));
  expect(std::abs(pitchAt(mid) - target) < 0.05f, "Esc ends the curve and keeps it");
  expect(curves().size() == 1 && curves()[0].anchorFrames.size() == 2,
         "the anchors stay on the line after Esc");
  roll.cancelPitchShaperPreview(); // as MainComponent::undo does
  undo.undo();
  expect(std::abs(pitchAt(mid) - target) > 0.05f && curves()[0].anchorFrames.size() == 1,
         "one undo takes back the second anchor only");
  roll.cancelPitchShaperPreview(); // as MainComponent::undo does
  undo.undo();
  expect(std::abs(pitchAt(mid) - original) < 0.002f && curves().empty(),
         "second undo restores the pitch and drops the curve");
  roll.cancelPitchShaperPreview();
  undo.redo();
  roll.cancelPitchShaperPreview();
  undo.redo();
  expect(std::abs(pitchAt(mid) - target) < 0.05f && curves().size() == 1,
         "redo brings the curve and its anchors back");

  // Later: drag a kept anchor to reshape the curve. Anchors are where the
  // clicks landed after rounding to frames.
  const int anchorA = curves()[0].anchorFrames.front();
  const int anchorB = curves()[0].anchorFrames.back();
  press(screen(anchorB, pitchAt(anchorB)), screen(anchorB, target + 1.0f), false);
  expect(std::abs(pitchAt(anchorB) - (target + 1.0f)) < 0.05f && curves().size() == 1 &&
             curves()[0].anchorFrames.size() == 2,
         "a kept anchor can be dragged again later");
  roll.cancelPitchShaperPreview(); // as MainComponent::undo does
  undo.undo();
  expect(std::abs(pitchAt(anchorB) - target) < 0.05f, "undo takes the drag back");

  // Saved and opened again: the curve is still there.
  {
    Project reopened;
    ProjectSerializer::fromJson(reopened, ProjectSerializer::toJson(project));
    const auto &kept = reopened.getDrawnCurves();
    expect(kept.size() == 1 && kept[0].anchorFrames == curves()[0].anchorFrames &&
               kept[0].underDiff.size() == curves()[0].underDiff.size(),
           "the curve is saved with the project");
  }

  // Dragging the note moves its anchors with it (they sit on the pitch).
  {
    Note *note = nullptr;
    for (auto &n : project.getNotes())
      if (n.getStartFrame() == start)
        note = &n;
    const float before = pitchAt(a);
    note->setMidiNote(note->getMidiNote() + 1.0f);
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    expect(std::abs(pitchAt(a) - (before + 1.0f)) < 0.01f,
           "dragging the note moves the drawn shape and anchors");
    note->setMidiNote(note->getMidiNote() - 1.0f);
    PitchCurveProcessor::rebuildBaseFromNotes(project);
  }

  // Alt + click deletes anchors; the last one gives the pitch back.
  press(screen(anchorB, pitchAt(anchorB)), screen(anchorB, pitchAt(anchorB)), true);
  expect(curves().size() == 1 && curves()[0].anchorFrames.size() == 1,
         "Alt + click deletes an anchor");
  press(screen(anchorA, pitchAt(anchorA)), screen(anchorA, pitchAt(anchorA)), true);
  roll.keyPressed(juce::KeyPress(juce::KeyPress::escapeKey));
  expect(curves().empty() && std::abs(pitchAt(mid) - original) < 0.002f,
         "deleting the last anchor gives the pitch back");
  {
    bool pristine = true;
    for (const auto &n : project.getNotes())
      if (n.getStartFrame() == start)
        pristine = !n.hasBakedDeltaPitch();
    expect(pristine, "and the note plays the recording again");
  }

  // The nearest frame no note covers, brought on screen with the note.
  const auto covered = [&project](int f)
  {
    return std::any_of(project.getNotes().begin(), project.getNotes().end(),
                       [f](const Note &n)
                       { return !n.isRest() && n.getStartFrame() <= f && n.getEndFrame() > f; });
  };
  int gap = -1;
  for (int d = 1; d < 2000 && gap < 0; ++d)
  {
    if (end - 1 + d < static_cast<int>(audio.f0.size()) && !covered(end - 1 + d))
      gap = end - 1 + d;
    else if (start - d >= 0 && !covered(start - d))
      gap = start - d;
  }
  {
    int lo = gap, hi = gap;
    while (lo > 0 && !covered(lo - 1)) --lo;
    while (hi + 1 < static_cast<int>(audio.f0.size()) && !covered(hi + 1)) ++hi;
    gap = (lo + hi) / 2;
  }
  const int left = std::min(gap, start), right = std::max(gap, end);
  roll.setPixelsPerSecond(static_cast<float>(1000.0 / framesToSeconds(right - left + 10)));
  roll.setScrollX(framesToSeconds(left - 5) * roll.getPixelsPerSecond());
  click(screen(a, target - 1.0f));
  const float atMidOneAnchor = pitchAt(mid);
  click(screen(gap, target));
  click(screen(b, target + 1.0f));
  // Had the curve gone on, a..b would be one curve and the middle would move.
  expect(gap >= 0 && std::abs(pitchAt(mid) - atMidOneAnchor) < 0.002f && curves().size() == 2,
         "an empty click ends the curve; the next starts anew");
  // Cmd selects (Cubase); Alt + drag in empty space zooms (Melodyne).
  {
    roll.setEditMode(EditMode::Select);
    roll.setPixelsPerSecond(300.0f);
    roll.setPixelsPerSemitone(30.0f);
    std::vector<Note *> picks;
    for (auto &n : project.getNotes())
      if (!n.isRest() && n.getStartFrame() > 300 && n.getDurationFrames() > 15 &&
          (picks.empty() || n.getStartFrame() > picks.back()->getEndFrame() + 5))
        picks.push_back(&n);
    Note *a = picks[0], *b = picks[1];
    roll.setScrollX(framesToSeconds(a->getStartFrame()) * 300.0 - 50.0);
    roll.centerOnPitchRange(std::min(a->getMidiNote(), b->getMidiNote()) - 4.0f,
                            std::max(a->getMidiNote(), b->getMidiNote()) + 4.0f);
    const auto at = [&](const Note *n)
    {
      const double t = framesToSeconds((n->getStartFrame() + n->getEndFrame()) / 2);
      const float pps = roll.getPixelsPerSecond(), semi = roll.getPixelsPerSemitone();
      return juce::Point<float>(
          PianoRollComponent::pianoKeysWidth + static_cast<float>(t * pps - roll.getScrollX()),
          PianoRollComponent::headerHeight + (MAX_MIDI_NOTE - n->getAdjustedMidiNote()) * semi +
              semi * 0.5f - static_cast<float>(roll.getScrollY()));
    };
    auto source = juce::Desktop::getInstance().getMainMouseSource();
    const auto now = juce::Time::getCurrentTime();
    const auto gesture = [&](juce::Point<float> from, juce::Point<float> to, int mods)
    {
      const juce::ModifierKeys down(juce::ModifierKeys::leftButtonModifier | mods);
      roll.mouseMove(juce::MouseEvent(source, from, juce::ModifierKeys(mods), 0, 0, 0, 0, 0,
                                      &roll, &roll, now, from, now, 0, false));
      roll.mouseDown(juce::MouseEvent(source, from, down, 0, 0, 0, 0, 0, &roll, &roll, now,
                                      from, now, 1, false));
      if (to != from)
        roll.mouseDrag(juce::MouseEvent(source, to, down, 0, 0, 0, 0, 0, &roll, &roll, now,
                                        from, now, 1, true));
      roll.mouseUp(juce::MouseEvent(source, to, juce::ModifierKeys(mods), 0, 0, 0, 0, 0, &roll,
                                    &roll, now, from, now, 1, to != from));
    };
    // A plain click on a note drags it and leaves the selection alone (as
    // before); Cmd + click is what selects.
    project.deselectAllNotes();
    gesture(at(a), at(a), juce::ModifierKeys::commandModifier);
    const bool onlyA = a->isSelected() && !b->isSelected();
    gesture(at(b), at(b), juce::ModifierKeys::commandModifier);
    const bool both = a->isSelected() && b->isSelected();
    gesture(at(a), at(a), juce::ModifierKeys::commandModifier);
    const bool onlyB = !a->isSelected() && b->isSelected();
    std::printf("  select: onlyA %d both %d onlyB %d\n", onlyA, both, onlyB);
    expect(onlyA && both && onlyB, "Cmd + click adds a note to the selection and takes it out");

    // Cmd + box in empty space adds to the selection.
    const auto above = juce::Point<float>(at(a).x - 20.0f, 4.0f + PianoRollComponent::headerHeight);
    const auto below = juce::Point<float>(at(a).x + 20.0f, at(a).y + 40.0f);
    gesture(above, below, juce::ModifierKeys::commandModifier);
    expect(a->isSelected() && b->isSelected(), "Cmd + box adds to the selection");

    // Cmd no longer zooms; Alt + drag in empty space does, and leaves the
    // selection alone.
    const float pps = roll.getPixelsPerSecond();
    const auto empty = juce::Point<float>(at(a).x, PianoRollComponent::headerHeight + 6.0f);
    gesture(empty, empty + juce::Point<float>(80.0f, -40.0f), juce::ModifierKeys::commandModifier);
    const bool cmdZoomed = std::abs(roll.getPixelsPerSecond() - pps) > 0.5f;
    gesture(empty, empty + juce::Point<float>(80.0f, -40.0f), juce::ModifierKeys::altModifier);
    const bool altZoomed = std::abs(roll.getPixelsPerSecond() - pps) > 0.5f;
    std::printf("  zoom: Cmd %d, Alt %d (%.0f -> %.0f px/s)\n", cmdZoomed, altZoomed, pps,
                roll.getPixelsPerSecond());
    expect(!cmdZoomed && altZoomed && a->isSelected() && b->isSelected(),
           "Alt + drag in empty space zooms; Cmd does not");
    // Alt on a note is not a zoom (the note keeps its own Alt).
    const float pps2 = roll.getPixelsPerSecond();
    roll.setPixelsPerSecond(300.0f);
    roll.setPixelsPerSemitone(30.0f);
    roll.setScrollX(framesToSeconds(a->getStartFrame()) * 300.0 - 50.0);
    roll.centerOnPitchRange(std::min(a->getMidiNote(), b->getMidiNote()) - 4.0f,
                            std::max(a->getMidiNote(), b->getMidiNote()) + 4.0f);
    gesture(at(b), at(b) + juce::Point<float>(60.0f, 0.0f), juce::ModifierKeys::altModifier);
    expect(std::abs(roll.getPixelsPerSecond() - 300.0f) < 0.5f, "Alt on a note does not zoom");
    juce::ignoreUnused(pps2);
    roll.setEditMode(EditMode::PitchShaper);
  }

  // Restore all edits includes the note separation: a split, tilted note
  // goes back to one analysed note; one undo brings the split and tilt back.
  {
    roll.setEditMode(EditMode::Select);
    Note *note = nullptr;
    for (auto &n : project.getNotes())
      if (!n.isRest() && n.getStartFrame() > 40 && !n.hasBakedDeltaPitch() &&
          n.getAnalysisSpans().size() == 1 && n.exactAnalysisSpan() &&
          (!note || n.getDurationFrames() > note->getDurationFrames()))
        note = &n;
    const int s0 = note->getStartFrame(), e0 = note->getEndFrame(), cut = (s0 + e0) / 2;
    const auto pristine = [&]()
    {
      std::vector<float> p;
      for (int f = s0; f < e0; ++f)
        p.push_back(pitchAt(f));
      return p;
    }();
    NoteSplitter splitter;
    splitter.setProject(&project);
    splitter.setUndoManager(&undo);
    splitter.splitNoteAtFrame(note, cut);
    Note *left = nullptr;
    for (auto &n : project.getNotes())
      if (n.getStartFrame() == s0)
        left = &n;
    left->setTiltLeft(1.0f);
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    const auto countIn = [&]()
    {
      int count = 0;
      for (const auto &n : project.getNotes())
        if (!n.isRest() && n.getStartFrame() < e0 && n.getEndFrame() > s0)
          ++count;
      return count;
    };
    roll.resetNoteEdits(*left, PianoRollComponent::NoteRestoreMode::All);
    float worst = 0.0f;
    for (int f = s0; f < e0; ++f)
      worst = std::max(worst, std::abs(pitchAt(f) - pristine[static_cast<size_t>(f - s0)]));
    expect(countIn() == 1 && worst < 0.002f,
           "All Edits also restores the note separation and pitch");
    undo.undo();
    bool tiltBack = false;
    for (const auto &n : project.getNotes())
      if (n.getStartFrame() == s0 && std::abs(n.getTiltLeft() - 1.0f) < 1.0e-4f)
        tiltBack = true;
    expect(countIn() == 2 && tiltBack, "one undo brings the split and tilt back");
    roll.setEditMode(EditMode::PitchShaper);
  }

  // Moving the tilt pivot of a tilted note leaves its curve as it is.
  {
    roll.setEditMode(EditMode::Select);
    roll.setPixelsPerSecond(500.0f);
    roll.setPixelsPerSemitone(40.0f);
    Note *note = nullptr;
    for (auto &n : project.getNotes())
      if (!n.isRest() && n.getStartFrame() != start && !n.hasBakedDeltaPitch() &&
          (!note || n.getDurationFrames() > note->getDurationFrames()))
        note = &n;
    note->setTiltLeft(1.0f);
    note->setTiltRight(-0.5f);
    note->setTiltPivot(0.3f);
    note->setMidiNote(note->getMidiNote() + note->getTiltMean());
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    std::vector<float> before;
    for (int f = note->getStartFrame(); f < note->getEndFrame(); ++f)
      before.push_back(pitchAt(f));
    roll.setScrollX(framesToSeconds(note->getStartFrame()) * 500.0 - 200.0);
    roll.centerOnPitchRange(note->getMidiNote() - 3.0f, note->getMidiNote() + 3.0f);
    auto source = juce::Desktop::getInstance().getMainMouseSource();
    const auto now = juce::Time::getCurrentTime();
    const auto event = [&](juce::Point<float> at, juce::ModifierKeys mods, juce::Point<float> down)
    { return juce::MouseEvent(source, at, mods, 0, 0, 0, 0, 0, &roll, &roll, now, down, now, 1, at != down); };
    const auto pivotSeconds = framesToSeconds(note->getStartFrame()) +
                              0.3 * framesToSeconds(note->getDurationFrames());
    const float x = PianoRollComponent::pianoKeysWidth +
                    static_cast<float>(pivotSeconds * 500.0 - roll.getScrollX());
    const float noteY = PianoRollComponent::headerHeight +
                        (MAX_MIDI_NOTE - note->getMidiNote()) * 40.0f -
                        static_cast<float>(roll.getScrollY());
    bool moved = false;
    for (float dy = 0.0f; dy < 140.0f && !moved; dy += 2.0f)
    {
      const juce::Point<float> at(x, noteY - dy);
      roll.mouseMove(event({x, noteY + 20.0f}, {}, {}));
      roll.mouseMove(event(at, {}, {}));
      roll.mouseDown(event(at, juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier), at));
      roll.mouseDrag(event(at + juce::Point<float>(80.0f, 0.0f),
                           juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier), at));
      roll.mouseUp(event(at + juce::Point<float>(80.0f, 0.0f), {}, at));
      moved = std::abs(note->getTiltPivot() - 0.3f) > 0.01f;
    }
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    float worst = 0.0f;
    for (int f = note->getStartFrame(); f < note->getEndFrame(); ++f)
      worst = std::max(worst, std::abs(pitchAt(f) - before[static_cast<size_t>(f - note->getStartFrame())]));
    std::printf("  pivot 0.30 -> %.2f, pitch moved by at most %.4f\n", note->getTiltPivot(), worst);
    expect(moved && worst < 0.002f, "moving the tilt pivot keeps the tilted curve");
    roll.setEditMode(EditMode::PitchShaper);
  }
  std::printf("draw check: %d failure(s)\n", failures);
}

} // namespace

// Ensemble analysis: how late RMVPE and SwiftF0 report against FCPE (lag
// search, 1 ms steps, on a 2 ms grid), then how often each one is outvoted.
void ensembleCheck(EditorController &controller, const AudioData &audioData)
{
  const auto &source = controller.getLastEnsembleSource();
  const char *names[3] = {"RMVPE", "FCPE", "SwiftF0"};
  for (int d = 0; d < 3; ++d)
    if (source[static_cast<size_t>(d)].empty())
    {
      std::printf("ensemble: %s produced nothing\n", names[d]);
      return;
    }

  auto semis = [](float hz) { return 12.0f * std::log2(hz / 440.0f) + 69.0f; };
  const double fine = 0.002;
  const double seconds = source[1].size() * pitchensemble::kFrameTime[1];
  const int fineFrames = static_cast<int>(seconds / fine);
  const auto reference =
      pitchensemble::resample(source[1], pitchensemble::kFrameTime[1],
                              pitchensemble::kOffset[1], fineFrames, fine);
  for (int d : {0, 2})
  {
    double bestLag = 0.0, bestErr = 1.0e9;
    for (int ms = -40; ms <= 40; ++ms)
    {
      const auto c = pitchensemble::resample(
          source[static_cast<size_t>(d)], pitchensemble::kFrameTime[d],
          ms * 0.001, fineFrames, fine);
      double sum = 0.0;
      int n = 0;
      for (int i = 0; i < fineFrames; ++i)
      {
        if (!c.voiced[static_cast<size_t>(i)] || !reference.voiced[static_cast<size_t>(i)])
          continue;
        const double diff = std::abs(semis(c.f0[static_cast<size_t>(i)]) -
                                     semis(reference.f0[static_cast<size_t>(i)]));
        if (diff < 1.0)
        {
          sum += diff;
          ++n;
        }
      }
      const double err = n > 0 ? sum / n : 1.0e9;
      if (err < bestErr)
      {
        bestErr = err;
        bestLag = ms * 0.001;
      }
    }
    std::printf("ensemble: %s best lag %+.3f s (mean |diff| %.4f st), "
                "offset in use %+.3f s\n",
                names[d], bestLag, bestErr, pitchensemble::kOffset[d]);
  }

  // On the analysis grid, with the offsets in use.
  const int frames = audioData.getNumFrames();
  const double frameTime = static_cast<double>(HOP_SIZE) / SAMPLE_RATE;
  std::vector<pitchensemble::Curve> curves;
  for (int d = 0; d < 3; ++d)
    curves.push_back(pitchensemble::resample(
        source[static_cast<size_t>(d)], pitchensemble::kFrameTime[d],
        pitchensemble::kOffset[d], frames, frameTime));
  const auto combined = pitchensemble::combine(curves);
  int allVoiced = 0, agree = 0, outvoted[3] = {0, 0, 0}, noPair = 0;
  for (int i = 0; i < frames; ++i)
  {
    float v[3];
    bool all = true;
    for (int d = 0; d < 3; ++d)
    {
      all = all && curves[static_cast<size_t>(d)].voiced[static_cast<size_t>(i)];
      v[d] = semis(curves[static_cast<size_t>(d)].f0[static_cast<size_t>(i)]);
    }
    if (!all)
      continue;
    ++allVoiced;
    const bool ab = std::abs(v[0] - v[1]) <= 0.3f, ac = std::abs(v[0] - v[2]) <= 0.3f,
               bc = std::abs(v[1] - v[2]) <= 0.3f;
    if (ab && ac && bc)
      ++agree;
    else if (bc && !ab && !ac)
      ++outvoted[0];
    else if (ac && !ab && !bc)
      ++outvoted[1];
    else if (ab && !ac && !bc)
      ++outvoted[2];
    else if (!ab && !ac && !bc)
      ++noPair;
  }
  std::printf("ensemble: %d frames all voiced, %d agree within 0.3 st; lone "
              "outlier RMVPE %d, FCPE %d, SwiftF0 %d; no two agree %d\n",
              allVoiced, agree, outvoted[0], outvoted[1], outvoted[2], noPair);

  // The SwiftF0 dip at 12.50 s.
  const int from = static_cast<int>(12.40 / frameTime), to = static_cast<int>(12.62 / frameTime);
  std::printf("frame   RMVPE    FCPE  SwiftF0  result  (semitones)\n");
  for (int i = from; i <= to && i < frames; ++i)
  {
    std::printf("%5d", i);
    for (int d = 0; d < 3; ++d)
    {
      const auto &c = curves[static_cast<size_t>(d)];
      if (c.voiced[static_cast<size_t>(i)])
        std::printf(" %7.2f", semis(c.f0[static_cast<size_t>(i)]));
      else
        std::printf("       -");
    }
    if (combined.voiced[static_cast<size_t>(i)])
      std::printf(" %7.2f\n", semis(combined.f0[static_cast<size_t>(i)]));
    else
      std::printf("       -\n");
  }
}

// Frame ranges [from, to) where the final pitch (composed from the rebuilt
// curves) or the note gain/formant of two projects differ. Callers rebuild the
// curves first (PitchCurveProcessor::rebuildBaseFromNotes).
static std::vector<std::pair<int, int>> differingFrames(const Project &a, const Project &b,
                                                        float *maxCents)
{
  auto perFrame = [](const Project &p, int frames) {
    std::vector<std::pair<float, float>> v(static_cast<size_t>(frames), {0.0f, 0.0f});
    for (const auto &n : p.getNotes())
      if (!n.isRest())
        for (int f = std::max(0, n.getStartFrame()); f < std::min(frames, n.getEndFrame()); ++f)
          v[static_cast<size_t>(f)] = {n.getVolumeDb(), n.getFormantShift()};
    return v;
  };
  const auto fa = a.getAdjustedF0(), fb = b.getAdjustedF0();
  const int frames = static_cast<int>(std::max(fa.size(), fb.size()));
  const auto ga = perFrame(a, frames), gb = perFrame(b, frames);
  std::vector<std::pair<int, int>> ranges;
  int runStart = -1;
  for (int f = 0; f <= frames; ++f)
  {
    bool differs = false;
    if (f < frames)
    {
      const auto i = static_cast<size_t>(f);
      const float x = i < fa.size() ? fa[i] : 0.0f, y = i < fb.size() ? fb[i] : 0.0f;
      if (i >= fa.size() || i >= fb.size() || (x > 0.0f) != (y > 0.0f))
        differs = true;
      else if (x > 0.0f)
      {
        const float c = std::abs(1200.0f * std::log2(y / x));
        if (c > 0.1f) // below this is float noise from rebuilding the curves
        {
          differs = true;
          if (maxCents != nullptr)
            *maxCents = std::max(*maxCents, c);
        }
      }
      if (ga[i] != gb[i])
        differs = true;
    }
    if (differs && runStart < 0)
      runStart = f;
    if (!differs && runStart >= 0)
    {
      ranges.emplace_back(runStart, f);
      runStart = -1;
    }
  }
  return ranges;
}

int main(int argc, char **argv)
{
  juce::ScopedJuceInitialiser_GUI juce;
  if (argc < 3)
  {
    std::fprintf(stderr, "usage: WarpRender in.wav out.wav [options]\n");
    return 2;
  }
  const juce::File input = juce::File::getCurrentWorkingDirectory().getChildFile(
      juce::String::fromUTF8(argv[1]));
  const juce::File output = juce::File::getCurrentWorkingDirectory().getChildFile(
      juce::String::fromUTF8(argv[2]));
  // PITCHNET_FRY_CUT=0: notes are not cut where the fry a word starts with
  // begins (FryNoteCuts.h) - a project opens as it did before that was done,
  // to compare with. Read before any project is.
  if (const char *fryCut = std::getenv("PITCHNET_FRY_CUT"))
    frycut::setEnabled(juce::String(fryCut) != "0");
  // WarpRender <in.pitchnet> <out.pitchnet> --app-op split|merge|delete|engine
  // <noteStart> [splitFrame]: the App's own note edit, saved without
  // synthesis, as the reference an agent's direct JSON edit is checked against.
  if (argc >= 6 && juce::String::fromUTF8(argv[3]) == "--app-op")
  {
    Project project;
    if (!ProjectSerializer::loadFromFile(project, input))
    {
      std::fprintf(stderr, "could not load\n");
      return 1;
    }
    // A project that links its recording has no audio until the recording
    // is attached, as the App attaches it on open: saved without, the file
    // came out with neither the audio nor the link (and nothing could go on
    // from it).
    sourceaudio::attachRecording(project, input);
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    const juce::String op = juce::String::fromUTF8(argv[4]);
    if (op == "key")
    {
      // ... --app-op key <root 0-11> [<frame>:<root>[,<frame>:<root>...]]:
      // the key as the wizard sets it when the user confirms it
      // (KeyState::confirmed): the major scale on the root, then the changes
      // of key at those frames of the recording. Saved as the App saves.
      std::vector<KeyChange> changes;
      if (argc >= 7)
        for (const auto &item : juce::StringArray::fromTokens(juce::String::fromUTF8(argv[6]), ",", ""))
          changes.push_back({item.upToFirstOccurrenceOf(":", false, false).getIntValue(),
                             item.fromFirstOccurrenceOf(":", false, false).getIntValue()});
      const auto before = KeyState::of(project);
      const auto after =
          KeyState::confirmed(project, juce::String::fromUTF8(argv[5]).getIntValue(), std::move(changes));
      KeyChangeAction(project, before, after, "Confirm Key").redo();
      if (!ProjectSerializer::saveToFile(project, output))
        return 1;
      std::printf("key ok: root %d, %d changes\n", project.getScaleRootNote(),
                  (int)project.getKeyChanges().size());
      return 0;
    }
    const int start = juce::String::fromUTF8(argv[5]).getIntValue();
    auto noteAt = [&](int frame) -> Note * {
      for (auto &n : project.getNotes())
        if (!n.isRest() && n.getStartFrame() == frame)
          return &n;
      return nullptr;
    };
    Note *note = noteAt(start);
    if (note == nullptr)
    {
      std::fprintf(stderr, "no note starts at %d\n", start);
      return 1;
    }
    NoteSplitter splitter;
    splitter.setProject(&project);
    bool ok = false;
    if (op == "split" && argc >= 7)
      ok = splitter.splitNoteAtFrame(note, juce::String::fromUTF8(argv[6]).getIntValue());
    else if (op == "merge")
      ok = splitter.mergeNotes(note, noteAt(note->getEndFrame()));
    else if (op == "engine" && argc >= 7)
    {
      // ... --app-op engine <noteStart> auto|highs|fry|r3: the choice of the
      // synthesis menu on that note, made by the App's own action
      // (SynthChoiceAction): the pitch curve moves where the note becomes
      // whole as recorded or stops being so. (PITCHNET_RESYNTH_NOTE_ENGINE
      // sets the field alone and leaves the curve.)
      const juce::String way = juce::String::fromUTF8(argv[6]);
      const auto engine = way == "r3"      ? Note::SynthEngine::RubberBand
                          : way == "highs" ? Note::SynthEngine::VocoderRecordedHighs
                          : way == "fry"   ? Note::SynthEngine::VocoderRecordedHighsFry
                                           : Note::SynthEngine::Auto;
      const std::vector<Note *> one{note};
      SynthChoiceAction(project, one, SynthChoiceAction::current(one),
                        {SynthChoice::ofMenu(*note, engine)}, "Set Synthesis Engine")
          .redo();
      ok = true;
    }
    else if (op == "lock")
    {
      // ... --app-op lock <noteStart>: a press of the lock on that note
      // (LockAction): locked, or unlocked when it was.
      const std::vector<Note *> one{note};
      LockAction(one, LockAction::wanted(one), "Lock Notes").redo();
      ok = true;
    }
    else if (op == "delete")
    {
      std::vector<Note> removed{*note};
      splitter.removeNotes(removed, drawncurve::without(project.getDrawnCurves(),
                                                        note->getStartFrame(),
                                                        note->getEndFrame()));
      ok = true;
    }
    if (!ok)
    {
      std::fprintf(stderr, "%s failed\n", op.toRawUTF8());
      return 1;
    }
    if (!ProjectSerializer::saveToFile(project, output))
      return 1;
    std::printf("%s ok, %d notes\n", op.toRawUTF8(), (int)project.getNotes().size());
    return 0;
  }
  // WarpRender <in.pitchnet> <out.pitchnet> --snap-midi <guide.mid> [track]:
  // the App's MIDI-reference double-click on every note selected at once
  // (SelectHandler), saved without synthesis. Prints one line per note the
  // snap left alone or that needs a look.
  if (argc >= 5 && juce::String::fromUTF8(argv[3]) == "--snap-midi")
  {
    Project project;
    if (!ProjectSerializer::loadFromFile(project, input))
    {
      std::fprintf(stderr, "could not load\n");
      return 1;
    }
    const juce::File midiFile = juce::File::getCurrentWorkingDirectory().getChildFile(
        juce::String::fromUTF8(argv[4]));
    juce::FileInputStream midiIn(midiFile);
    auto tracks = midiIn.openedOk() ? MidiReference::readTracks(midiIn)
                                    : std::vector<MidiReference::Track>{};
    if (tracks.empty())
    {
      std::fprintf(stderr, "could not read guide\n");
      return 1;
    }
    size_t trackIndex = 0;
    if (argc >= 6)
    {
      const int number = juce::String::fromUTF8(argv[5]).getIntValue();
      while (trackIndex < tracks.size() && tracks[trackIndex].number != number)
        ++trackIndex;
      if (trackIndex == tracks.size())
      {
        std::fprintf(stderr, "no track %d\n", number);
        return 1;
      }
    }
    MidiReference::State state;
    state.guide = std::make_shared<MidiReference::Guide>(
        MidiReference::Guide{midiFile.getFileName(), std::move(tracks)});
    state.selectedTrack = trackIndex;
    state.visible = true;
    project.setMidiReference(state);
    project.setDoubleClickSnapMode(DoubleClickSnapMode::MidiReference);
    PitchCurveProcessor::rebuildBaseFromNotes(project);

    const auto &guide = state.guide->tracks[trackIndex].notes;
    const auto span = MidiReference::getGuideSpan(project.getAudioData());
    const float referenceOffset =
        ScaleUtils::getReferenceOffsetSemitones(project.getPitchReferenceHz());
    auto clock = [](double s) {
      return juce::String::formatted("%d:%06.3f", (int)(s / 60), std::fmod(s, 60.0));
    };
    int snapped = 0, unchanged = 0, flagged = 0;
    for (auto &note : project.getNotes())
    {
      if (note.isRest())
        continue;
      const double start = framesToSeconds(note.getStartFrame());
      const double end = framesToSeconds(note.getEndFrame());
      const float adjusted = note.getMidiNote() + note.getPitchOffset();
      // getDoubleClickSnapTarget, MidiReference case.
      const auto target = MidiReference::findSnapPitch(guide, span, start, end);
      const float snappedMidi =
          target ? static_cast<float>(*target) + referenceOffset : note.getAdjustedMidiNote();

      // Report only: the overlaps behind the choice.
      double total = 0.0, best = 0.0, second = 0.0;
      for (const auto &g : guide)
      {
        const double o = std::min(end - span.first, g.endSeconds) -
                         std::max(start - span.first, g.startSeconds);
        if (o <= 0.0)
          continue;
        total += o;
        if (o > best) { second = best; best = o; }
        else if (o > second) second = o;
      }
      juce::String reason;
      if (!target)
        reason = total > 0.0 ? "tie" : "no-guide";
      else if (std::abs(snappedMidi - adjusted) >= 6.0f)
        reason = "jump>=6";
      else if (second > 0.3 * (end - start))
        reason = "spans-two";
      if (reason.isNotEmpty())
      {
        ++flagged;
        std::printf("%s\t%s\t%d\t%.2f\t%s\t%s\n", reason.toRawUTF8(),
                    clock(start).toRawUTF8(), note.getStartFrame(), adjusted,
                    target ? juce::String(snappedMidi, 2).toRawUTF8() : "-",
                    juce::String(end - start, 3).toRawUTF8());
      }

      // SelectHandler: multi-note double-click.
      if (std::abs(snappedMidi - adjusted) <= 0.001f)
      {
        ++unchanged;
        continue;
      }
      note.setMidiNote(snappedMidi);
      note.setPitchOffset(0.0f);
      note.markDirty();
      ++snapped;
    }
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    if (!ProjectSerializer::saveToFile(project, output))
      return 1;
    std::printf("snapped %d, unchanged %d, flagged %d\n", snapped, unchanged, flagged);
    return 0;
  }
  // WarpRender <in.pitchnet> <out.pitchnet> --guide-fix <guide.mid> [track]:
  // the guide rule (spec Q10), saved without synthesis. Per note:
  //   sung pitch - the note's centre as the App analysed it (midiNote)
  //   target     - the guide pitch nearest it among those within 0.25 s of
  //                the held part (a late or early entry finds its own note);
  //                none covering 30% of the held part, none within a
  //                semitone, or two about as near: left and reported
  //   held part  - on the white line (the composed pitch), frames within
  //                10 dB of the note's loudest, trimmed of the slide in and
  //                out (first/last frame within 40 cents of the median of the
  //                later half); its line fit is the trend
  //   flatten    - held 0.15 s or longer with a trend of 16-100 cents:
  //                pitchDrift 0.5 from 0.5 s, else a tilt about the held start
  //                (right side only, so the slide in keeps its shape)
  //   shift      - the held part moves by the note's deviation; a short glide
  //                (held up to 0.2 s, trend beyond 40 cents) whose end extreme
  //                runs straight into the next note's slide the other way is
  //                placed by that extreme instead, just past the target
  // Left alone: within 10 cents and nothing to flatten.
  // Shifts are measured on the rebuilt curve and corrected three times.
  if (argc >= 5 && juce::String::fromUTF8(argv[3]) == "--guide-fix")
  {
    // Run twice. The first run (its report thrown away) finds the notes that
    // hold two pitches where only the second needs moving (a C sung on, then
    // a slide up to a sharp D): the App's own split cuts each at the start of
    // its slide. The second run judges the split project; each cut-off
    // second half is placed as the first run planned it.
    struct SplitRequest
    {
      int start, cut;             // the note, and the frame the App splits it at
      int landing0, landing1;     // absolute frames of the landing [landing0, landing1)
      int pivot;                  // absolute frame the drift starts turning from
      float level, target;        // the landing's level, and where it goes
      bool planned = true;        // the second half is placed as planned here
    };
    std::vector<SplitRequest> splitRequests;
    auto guideFix = [&](bool probe) -> int
    {
    constexpr float band = 0.4f;           // semitones, trimming band
    constexpr float tolerance = 0.10f;     // semitones, left alone within
    constexpr int windowFrames = 22;       // ~0.25 s around the held part
    constexpr double minCover = 0.3;       // share of the held part with a guide
    constexpr float maxDeviation = 1.0f;   // semitones to the nearest candidate
    constexpr float tieMargin = 0.1f;      // semitones between two candidates
    constexpr float flattenMin = 0.16f, flattenMax = 0.40f; // trend, semitones
    constexpr double driftHeldSeconds = 0.5;
    constexpr float driftFactor = 0.5f;
    constexpr float overshoot = 0.10f;     // semitones past the target
    constexpr float quietBelowPeakDb = 10.0f; // frames this far under the loudest are not judged
    constexpr double extremeHeldSeconds = 0.2; // only short glides are placed by their extreme
    constexpr float extremeReach = 1.0f;   // semitones from the note's centre
    constexpr float slideStep = 0.07f;     // semitones per frame in a slide
    constexpr int plateauBefore = 5;       // frames held before a slide
    constexpr float plateauSpread = 0.5f;  // semitones those frames stay within
    constexpr float maxExtraShift = 0.5f;  // semitones a shift may exceed the deviation by
    constexpr int settleFrames = 5;
    constexpr float tailSlideStep = 0.10f;  // semitones per frame, every frame
    constexpr int tailSlideFrames = 4;
    constexpr float tailSlideSpan = 1.0f;   // semitones in all
    constexpr float tailSlideReach = 0.6f;  // semitones around what precedes it
    constexpr float tailSlideDrag = 0.2f;   // semitones the slide must have moved the centre
    constexpr double otherNoteMaxCover = 0.5;
    constexpr double tieOverlapRatio = 2.5;  // a tie broken by this many times the overlap
    constexpr double tieOverlapShare = 0.6;  // ... covering this share of the held part
    constexpr float consonantTolerance = 0.15f; // semitones left alone on a note edged by a consonant
    constexpr float tailReach = 0.6f;        // semitones a guideless tail may be off its note
    constexpr float tailMinOff = 0.30f;      // ... and at least this, to be worth it
    constexpr double tailMaxSeconds = 0.25;  // a tail is short
    constexpr float tailMaxQuieterDb = 5.0f; // and nearly as loud as the note before
    constexpr float tailMinFall = 1.5f;      // semitones it falls from that note
    constexpr float followMin = 0.2f;        // semitones between two notes' shifts for a tail to follow
    constexpr int followFrames = 5;          // frames of the slide at a note's end
    constexpr float followRise = 0.3f;       // semitones it must move toward the next note
    constexpr float followEndTolerance = 0.35f; // semitones from the off-key note it slides to
    constexpr float followMinOffKey = 0.4f;  // semitones the next note must have moved
    constexpr float offKeyDeviation = 1.5f;  // semitones of reach for off-key singing
    constexpr double approachSeconds = 0.15; // a scoop note is no longer than this
    constexpr double passingSeconds = 0.2;   // a chromatic passing note is no longer than this
    constexpr float passingTolerance = 0.35f; // semitones its neighbours may be off a semitone away
    constexpr float passingOnNote = 0.25f;   // semitones from the off-key note it is sung on
    constexpr double ornamentSeconds = 0.12; // a turn's note is no longer than this
    constexpr float slideAwayMin = 0.6f;
    constexpr double solidHeldSeconds = 0.15; // a held part this long ...
    constexpr float solidSpread = 0.6f;      // ... and this steady (10-90%) gives the sung pitch
    constexpr double minNoteSeconds = 0.15;  // a note this long, sung late or early against the guide
    constexpr float snapMinOff = 0.2f;       // semitones off an in-key note, to snap a free note
    constexpr float snapMaxOff = 0.5f;
    constexpr float snapMaxSpread = 0.5f;
    constexpr double snapHeldSeconds = 0.1;
    constexpr int tailDriftFrames = 4;       // loud frames after the landing, drifting away
    constexpr float tailDriftMin = 0.4f;     // semitones the note's end is off its landing
    constexpr float tailDriftDrag = 0.25f;   // share of the drift the analysed centre followed
    constexpr int tailDriftLoud = 2;         // ... of them heard
    constexpr float tailDriftWobble = 0.05f; // semitones a drift may step back on its way
    constexpr float tailDriftReturn = 0.15f; // semitones its end may come back from its furthest
    constexpr float tailDriftBefore = 0.2f;  // semitones the note may go the drift's way before landing
    constexpr float tailDriftLanding = 0.5f; // semitones the landing may be off the target
    constexpr int splitPartFrames = 9;       // frames (0.1 s) each half of a two-pitch note holds
    constexpr int splitStepFrames = 8;       // frames the step between them may take
    constexpr int tailBodyFrames = 5;        // frames a tail holds before it lands, to be heard
    constexpr float tailBodyLeft = 0.5f;     // semitones the hold has left the note before
    constexpr float tailBodyReach = 1.3f;    // semitones the hold may be off the target
    constexpr float melodyFlatTolerance = 0.2f; // semitones: in tune enough to keep, slant flattened
    constexpr float splitGap = 1.5f;         // semitones between a note's first part and its landing
    constexpr float splitBand = 0.6f;        // semitones the first part wobbles around its pitch
    constexpr double splitPartSeconds = 0.1; // the first part this long, to be a note of its own
    constexpr float splitFromPrevious = 0.6f; // semitones it may be off the note before's pitch
    constexpr float tailDriftAway = 0.3f;    // semitones the drift ends further from the target
    constexpr float scaleStepReach = 0.7f;   // semitones from the step between two notes a third apart
    constexpr float scaleStepEnds = 0.3f;    // semitones the notes either side are off their own
    constexpr float turnStep = 0.5f;         // semitones to the turn's middle note
    constexpr float offKeyNearestOn = 0.35f; // semitones from the off-key note it was sung on
    constexpr float offKeyNearestMargin = 0.12f; // semitones between the in-key notes either side     // semitones an extreme must pass the held part to be a slide away
    constexpr float ornamentStep = 0.5f;     // semitones above or below both neighbours
    constexpr float approachRise = 0.2f;     // semitones its last frames move toward the target
    constexpr float approachMinOffset = 0.5f; // semitones off the target, for a real scoop
    constexpr float offKeyTolerance = 0.3f;  // semitones from the off-key note it was sung on
    constexpr float offKeyMaxSpread = 0.5f;  // semitones the held part may span
    constexpr float reviewDisagree = 0.30f; // analysed centre vs white line, semitones
    constexpr float reviewShift = 0.60f;    // shifts larger than this get a second look        // in a row inside the band to start/end the held part
    constexpr float otherNoteTolerance = 0.25f; // semitones: sung this close to another note
    constexpr float maxFlattenTrend = 1.0f; // semitones: steeper is a slide, not a sag
    constexpr double flattenHeldSeconds = 0.15;
    Project project;
    if (!ProjectSerializer::loadFromFile(project, input))
    {
      std::fprintf(stderr, "could not load\n");
      return 1;
    }
    // A project that links its recording (spec Q11) holds no audio until the
    // recording is attached, as the App does on open: without it there is no
    // level to tell the loud frames by and no composed pitch to judge (every
    // note came out "no-voiced"). One that embeds its audio is left as it is.
    if (project.getAudioData().originalWaveform.getNumSamples() == 0 &&
        !sourceaudio::attachRecording(project, input))
    {
      std::fprintf(stderr, "could not attach the recording\n");
      return 1;
    }
    const juce::File midiFile = juce::File::getCurrentWorkingDirectory().getChildFile(
        juce::String::fromUTF8(argv[4]));
    juce::FileInputStream midiIn(midiFile);
    auto tracks = midiIn.openedOk() ? MidiReference::readTracks(midiIn)
                                    : std::vector<MidiReference::Track>{};
    if (tracks.empty())
    {
      std::fprintf(stderr, "could not read guide\n");
      return 1;
    }
    size_t trackIndex = 0;
    if (argc >= 6)
    {
      const int number = juce::String::fromUTF8(argv[5]).getIntValue();
      while (trackIndex < tracks.size() && tracks[trackIndex].number != number)
        ++trackIndex;
      if (trackIndex == tracks.size())
      {
        std::fprintf(stderr, "no track %d\n", number);
        return 1;
      }
    }
    // GUIDE_FIX_OFFSET (seconds): the guide moved onto the recording's own
    // timing (a take sung to its own backing track, the guide from another
    // version). The reference saved with the project is moved too, so the App
    // shows it where the rule read it.
    if (const char *offset = std::getenv("GUIDE_FIX_OFFSET"))
      for (auto &track : tracks)
        for (auto &g : track.notes)
        {
          g.startSeconds += std::atof(offset);
          g.endSeconds += std::atof(offset);
        }
    MidiReference::State state;
    state.guide = std::make_shared<MidiReference::Guide>(
        MidiReference::Guide{midiFile.getFileName(), std::move(tracks)});
    state.selectedTrack = trackIndex;
    state.visible = true;
    project.setMidiReference(state);
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    if (!probe && !splitRequests.empty())
    {
      // The App's own split (NoteSplitter), which keeps the sung pitch.
      NoteSplitter splitter;
      splitter.setProject(&project);
      for (const auto &request : splitRequests)
        for (auto &n : project.getNotes())
          if (!n.isRest() && n.getStartFrame() == request.start)
          {
            if (!splitter.splitNoteAtFrame(&n, request.cut))
              std::fprintf(stderr, "split at %d failed\n", request.cut);
            break;
          }
      PitchCurveProcessor::rebuildBaseFromNotes(project);
    }

    const auto &audio = project.getAudioData();
    const auto span = MidiReference::getGuideSpan(audio);
    const float referenceOffset =
        ScaleUtils::getReferenceOffsetSemitones(project.getPitchReferenceHz());
    // Guide notes on the frame grid of the audio.
    struct GuideNote { double start, end; float pitch; };
    std::vector<GuideNote> guide;
    for (const auto &g : state.guide->tracks[trackIndex].notes)
      guide.push_back({(span.first + g.startSeconds) * SAMPLE_RATE / HOP_SIZE,
                       (span.first + g.endSeconds) * SAMPLE_RATE / HOP_SIZE,
                       static_cast<float>(g.midiPitch) + referenceOffset});
    // The key, from the guide itself: pitch classes holding at least
    // inScaleShare of its sounding time (a C major melody: C D E F G A B).
    std::array<bool, 12> inScale{};
    {
      std::array<double, 12> held{};
      double total = 0.0;
      for (const auto &g : guide)
      {
        const int pc = ((static_cast<int>(std::lround(g.pitch - referenceOffset)) % 12) + 12) % 12;
        held[static_cast<size_t>(pc)] += g.end - g.start;
        total += g.end - g.start;
      }
      // The major scale (or its relative minor: the same seven notes) that
      // holds the most of the guide's time.
      static constexpr int major[7] = {0, 2, 4, 5, 7, 9, 11};
      int bestRoot = 0;
      double bestHeld = -1.0;
      for (int root = 0; root < 12; ++root)
      {
        double inside = 0.0;
        for (int step : major)
          inside += held[static_cast<size_t>((root + step) % 12)];
        if (inside > bestHeld)
        {
          bestHeld = inside;
          bestRoot = root;
        }
      }
      for (int step : major)
        inScale[static_cast<size_t>((bestRoot + step) % 12)] = total > 0.0;
    }
    auto outOfKey = [&](float pitch) {
      const int pc = ((static_cast<int>(std::lround(pitch - referenceOffset)) % 12) + 12) % 12;
      return !inScale[static_cast<size_t>(pc)];
    };
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    auto white = [&](int frame) {
      const auto &a = project.getAudioData();
      if (frame < 0 || frame >= (int)a.f0.size() || frame >= (int)a.voicedMask.size() ||
          !a.voicedMask[(size_t)frame] || a.f0[(size_t)frame] <= 0.0f)
        return nan;
      return 69.0f + 12.0f * std::log2(a.f0[(size_t)frame] / 440.0f);
    };
    auto clock = [](double s) {
      return juce::String::formatted("%d:%06.3f", (int)(s / 60), std::fmod(s, 60.0));
    };
    // Level of the recording at a frame (dB RMS).
    auto levelDb = [&](int frame) {
      const auto &source = audio.originalWaveform;
      const int first = frame * HOP_SIZE;
      const int last = std::min(source.getNumSamples(), first + HOP_SIZE);
      if (source.getNumChannels() == 0 || first < 0 || last <= first)
        return -200.0f;
      const float *x = source.getReadPointer(0);
      double energy = 0.0;
      for (int i = first; i < last; ++i)
        energy += static_cast<double>(x[i]) * x[i];
      return static_cast<float>(10.0 * std::log10(energy / (last - first) + 1.0e-20));
    };

    struct Plan
    {
      Note *note;
      std::vector<int> measured; // absolute frames whose mean is the centre
      float target, want;        // want: where the measured point should land
      int extreme;               // absolute frame of a glide's extreme, or -1
      juce::String how;
      Note before = {};          // the note as it was, to put back
      float deviation = 0.0f;    // sung - target
      juce::String review;       // reasons for a second look
    };
    std::vector<Plan> plans;
    struct InTune
    {
      Note *note;
      float target, sung, centre;
      std::vector<int> measured;
    };
    std::vector<InTune> inTune;
    // The white line as recorded, before any of this pass's changes.
    std::vector<float> recorded(static_cast<size_t>(audio.f0.size()));
    for (size_t f = 0; f < recorded.size(); ++f)
      recorded[f] = white(static_cast<int>(f));
    std::map<std::string, int> counts;
    // GUIDE_FIX_TARGETS=<file>: every note the guide gave a target, one a
    // line - what was sung, the target, and what decided it - for measuring
    // how far a target found without the guide would differ.
    std::FILE *targetsFile = nullptr;
    if (const char *path = std::getenv("GUIDE_FIX_TARGETS"); path != nullptr && !probe)
      targetsFile = std::fopen(path, "w");
    const auto noteTarget = [&](const char *kind, const Note &n, float sungPitch, float targetPitch,
                                int heldFrames, const juce::String &why) {
      if (targetsFile != nullptr)
        std::fprintf(targetsFile, "%s\t%d\t%d\t%.3f\t%.3f\t%d\t%s\n", kind, n.getStartFrame(),
                     n.getEndFrame(), sungPitch, targetPitch, heldFrames, why.trim().toRawUTF8());
    };
    // A one-sided tilt turning from pivot (absolute frame) to the note's end,
    // by the amount that brings what follows closest to level (least squares).
    auto tiltTail = [&](Note &target, int start, int frames, int pivot, float level) {
      const float p = static_cast<float>(pivot - start) / static_cast<float>(frames - 1);
      double sw = 0.0, sww = 0.0;
      for (int f = pivot + 1; f < start + frames; ++f)
        if (!std::isnan(white(f)))
        {
          const double w = (static_cast<double>(f - start) / (frames - 1) - p) / (1.0 - p);
          sw += w * (white(f) - level);
          sww += w * w;
        }
      target.setTiltLeft(0.0f);
      target.setTiltPivot(p);
      target.setTiltRight(sww > 0.0 ? static_cast<float>(-sw / sww) : 0.0f);
    };
    for (auto &note : project.getNotes())
    {
      if (note.isRest())
        continue;
      const int s = note.getStartFrame(), e = note.getEndFrame(), n = e - s;
      // The second half of a note the App split for the first run: placed
      // as the first run planned it.
      if (!probe)
        if (const auto hint = std::find_if(splitRequests.begin(), splitRequests.end(),
                                           [&](const SplitRequest &r) { return r.planned && r.cut == s; });
            hint != splitRequests.end())
        {
          Plan plan{&note, {}, hint->target, hint->target, -1, "split-tail-tilt"};
          plan.before = note;
          plan.deviation = hint->level - hint->target;
          for (int f = hint->landing0; f < hint->landing1; ++f)
            if (!std::isnan(white(f)))
              plan.measured.push_back(f);
          tiltTail(note, s, n, hint->pivot, hint->level);
          note.markDirty();
          plans.push_back(plan);
          ++counts["fix-split-tail-tilt"];
          std::printf("split\t%s\t%d\t%d\n", clock(framesToSeconds(hint->start)).toRawUTF8(), hint->start, hint->cut);
          continue;
        }
      std::vector<int> voiced;
      for (int i = 0; i < n; ++i)
        if (!std::isnan(white(s + i)))
          voiced.push_back(i);
      // A note that ends held on another pitch (a D, then a step down to a C
      // held to the end): two notes the analysis joined. The first run asks
      // the App to split it where the line leaves the first one; each half is
      // then judged as a note.
      if (probe && voiced.size() >= 2 * static_cast<size_t>(splitPartFrames) &&
          voiced.back() >= n - 2 && voiced.back() - voiced.front() + 1 == static_cast<int>(voiced.size()))
      {
        auto medianOf = [&](int from, int to) { // [from, to) note-relative
          std::vector<float> v;
          for (int i = from; i < to; ++i)
            v.push_back(white(s + i));
          std::sort(v.begin(), v.end());
          return v[v.size() / 2];
        };
        const int last = voiced.back();
        const float endPitch = medianOf(last + 1 - splitPartFrames, last + 1);
        int t0 = last;
        while (t0 - 1 >= voiced.front() && std::abs(white(s + t0 - 1) - endPitch) <= splitBand)
          --t0;
        // The step into it: one way, a few frames.
        int r0 = t0 - 1;
        const float d = white(s + t0) >= white(s + t0 - 1) ? 1.0f : -1.0f;
        while (r0 - 1 >= voiced.front() && d * (white(s + r0) - white(s + r0 - 1)) > 0.0f)
          --r0;
        if (last + 1 - t0 >= splitPartFrames && t0 - r0 <= splitStepFrames &&
            r0 - voiced.front() >= splitPartFrames)
        {
          const float mainPitch = medianOf(r0 - splitPartFrames, r0);
          int m0 = r0 - 1;
          while (m0 - 1 >= voiced.front() && std::abs(white(s + m0 - 1) - mainPitch) <= splitBand)
            --m0;
          int cut = r0;
          while (cut < t0 && std::abs(white(s + cut) - mainPitch) <= splitBand)
            ++cut;
          // Both heard (loud), each long enough to be a note.
          float peak = -1000.0f;
          for (int i : voiced)
            peak = std::max(peak, levelDb(s + i));
          auto loudIn = [&](int from, int to) {
            int count = 0;
            for (int i = from; i < to; ++i)
              if (levelDb(s + i) >= peak - quietBelowPeakDb)
                ++count;
            return count;
          };
          // The analysis gave the note the first pitch: the held end is the
          // one it joined on (where it gave the end's pitch, the first part
          // is the note's slide in, kept with it).
          const float analysed = note.getMidiNote() + note.getPitchOffset();
          if (r0 - m0 >= splitPartFrames && std::abs(mainPitch - endPitch) >= splitGap &&
              std::abs(analysed - mainPitch) < std::abs(analysed - endPitch) &&
              2 * loudIn(m0, r0) >= r0 - m0 && 2 * loudIn(t0, last + 1) >= last + 1 - t0)
          {
            splitRequests.push_back({s, s + cut, 0, 0, 0, 0.0f, 0.0f, false});
            continue;
          }
        }
      }
      auto report = [&](const char *reason, float centre, float trend, float target) {
        ++counts[reason];
        std::printf("%s\t%s\t%d\t%.2f\t%+.0f\t%s\n", reason, clock(framesToSeconds(s)).toRawUTF8(),
                    s, centre, trend * 100.0f,
                    target > 0.0f ? juce::String(target, 0).toRawUTF8() : "-");
      };
      if (voiced.empty())
      {
        report("no-voiced", 0.0f, 0.0f, 0.0f);
        continue;
      }
      // Loud frames only: a note dying away (a dip in its tail) is not what
      // is heard as its pitch.
      std::vector<int> heard;
      {
        float peak = -1000.0f;
        for (int i : voiced)
          peak = std::max(peak, levelDb(s + i));
        for (int i : voiced)
          if (levelDb(s + i) >= peak - quietBelowPeakDb)
            heard.push_back(i);
        if (heard.empty())
          heard = voiced;
      }
      // Held part: trim the slide in and out against the median of the
      // later half (less its last eighth): a voice lands on its note after
      // the slide in, so the first half can be scoop and hesitation.
      // Held part: trim the slide in and out against the median of the
      // later half (less its last eighth): a voice lands on its note after
      // the slide in, so the first half can be scoop and hesitation.
      std::vector<float> middle;
      for (size_t i = (heard.size() >= 4 ? heard.size() / 2 : 0);
           i < (heard.size() >= 4 ? heard.size() - heard.size() / 8 : heard.size()); ++i)
        middle.push_back(white(s + heard[i]));
      std::sort(middle.begin(), middle.end());
      const float ref = middle.size() % 2 ? middle[middle.size() / 2]
                                          : 0.5f * (middle[middle.size() / 2 - 1] +
                                                    middle[middle.size() / 2]);
      // From the first frame that starts `settleFrames` heard frames in a row
      // inside the band to the last that ends such a run: a brief excursion
      // at either end (a dip on the way in) is left out.
      int h0 = -1, h1 = -1;
      {
        const size_t run = std::min<size_t>(settleFrames, heard.size());
        auto inside = [&](size_t k) { return std::abs(white(s + heard[k]) - ref) <= band; };
        auto settled = [&](size_t first) {
          for (size_t k = first; k < first + run; ++k)
            if (!inside(k))
              return false;
          return true;
        };
        for (size_t k = 0; k + run <= heard.size(); ++k)
          if (settled(k))
          {
            if (h0 < 0)
              h0 = heard[k];
            h1 = heard[k + run - 1] + 1;
          }
        if (h0 < 0)
          for (size_t k = 0; k < heard.size(); ++k)
            if (inside(k))
            {
              if (h0 < 0)
                h0 = heard[k];
              h1 = heard[k] + 1;
            }
      }
      if (h0 < 0)
      {
        h0 = heard.front();
        h1 = heard.back() + 1;
      }
      // The frames measured: heard frames inside the held part.
      std::vector<int> measured;
      for (int i : heard)
        if (i >= h0 && i < h1)
          measured.push_back(i);
      // Centre and trend (least-squares line).
      double sum = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
      for (int i : measured)
      {
        const float v = white(s + i);
        sum += v;
        const double x = i - h0;
        sx += x; sy += v; sxx += x * x; sxy += x * v;
      }
      const int count = static_cast<int>(measured.size());
      float centre = static_cast<float>(sum / count);
      const double denominator = count * sxx - sx * sx;
      const float slope = count >= 3 && denominator > 0.0
                              ? static_cast<float>((count * sxy - sx * sy) / denominator)
                              : 0.0f;
      float trend = slope * static_cast<float>(h1 - h0 - 1);
      // A short glide note is placed by the extreme that runs into the next
      // note's slide the other way.
      int extreme = -1;
      if (std::abs(trend) > flattenMax)
      {
        const bool falling = trend < 0.0f;
        if (framesToSeconds(h1 - h0) <= extremeHeldSeconds && voiced.back() == n - 1 &&
            !std::isnan(white(e)) && !std::isnan(white(e + 2)))
        {
          size_t at = 0;
          for (size_t i = 1; i < voiced.size(); ++i)
          {
            const float v = white(s + voiced[i]), best = white(s + voiced[at]);
            if (falling ? v < best : v > best)
              at = i;
          }
          const float after = white(e + 2) - white(e);
          // Only a landing near the note itself: a dive of semitones into the
          // next note is a transition, not where this note is sung.
          // A note that holds and then slides a long way into the next is a
          // held note with a slide, not a glide: its extreme is not a landing.
          bool heldThenSlides = false;
          {
            int run = 0;
            for (size_t i = at; i > 0 && falling == (white(s + voiced[i]) < white(s + voiced[i - 1])) &&
                                std::abs(white(s + voiced[i]) - white(s + voiced[i - 1])) >= slideStep;
                 --i)
              ++run;
            const size_t slideStart = at >= static_cast<size_t>(run) ? at - static_cast<size_t>(run) : 0;
            if (run >= 4 && slideStart >= static_cast<size_t>(plateauBefore))
            {
              float lo = 1000.0f, hi = -1000.0f;
              for (size_t i = slideStart - static_cast<size_t>(plateauBefore); i < slideStart; ++i)
              {
                lo = std::min(lo, white(s + voiced[i]));
                hi = std::max(hi, white(s + voiced[i]));
              }
              heldThenSlides = hi - lo <= plateauSpread;
            }
          }
          (void)heldThenSlides;
          if ((falling ? after > 0.0f : after < 0.0f) && at + 4 >= voiced.size() &&
              std::abs(white(s + voiced[at]) - (note.getMidiNote() + note.getPitchOffset())) <=
                  extremeReach)
            extreme = s + voiced[at];
        }
      }
      // The pitch heard: the note's centre as the App analysed it (its blob).
      // It matched the user's ear where every white-line estimate tried
      // missed one case or another (a vibrato cut to its trough, a scoop's
      // hesitation, a dying tail); the white line still decides the flatten,
      // the glide and how far the curve really moved.
      float sung = note.getMidiNote() + note.getPitchOffset();
      const float heldMean = centre;
      // A slide into the next note at the end (a steady run up or down of a
      // semitone or more) drags the analysed centre with it: judge the note
      // by the median of what comes before the slide instead.
      bool tailSlide = false;
      {
        int k = static_cast<int>(heard.size()) - 1;
        if (k >= 4 && heard[static_cast<size_t>(k)] >= n - 2)
        {
          const float last = white(s + heard[static_cast<size_t>(k)]);
          const float step0 = last - white(s + heard[static_cast<size_t>(k - 1)]);
          const float sign = step0 >= 0.0f ? 1.0f : -1.0f;
          int first = k;
          while (first > 0 && heard[static_cast<size_t>(first - 1)] == heard[static_cast<size_t>(first)] - 1 &&
                 sign * (white(s + heard[static_cast<size_t>(first)]) -
                         white(s + heard[static_cast<size_t>(first - 1)])) >= tailSlideStep)
            --first;
          const float rise = sign * (last - white(s + heard[static_cast<size_t>(first)]));
          if (k - first >= tailSlideFrames && rise >= tailSlideSpan && first >= 3)
          {
            // What comes before the slide still has its own slide in: the
            // frames near the median of its later half (wide enough for a
            // vibrato), averaged. Used only where the slide really dragged
            // the analysed centre away from it.
            std::vector<float> later;
            for (int i = first / 2; i < first; ++i)
              later.push_back(white(s + heard[static_cast<size_t>(i)]));
            std::sort(later.begin(), later.end());
            const float preRef = later[later.size() / 2];
            std::vector<int> kept;
            double keptSum = 0.0;
            for (int i = 0; i < first; ++i)
              if (std::abs(white(s + heard[static_cast<size_t>(i)]) - preRef) <= tailSlideReach)
              {
                kept.push_back(heard[static_cast<size_t>(i)]);
                keptSum += white(s + heard[static_cast<size_t>(i)]);
              }
            const float before = kept.empty() ? sung : static_cast<float>(keptSum / kept.size());
            if (!kept.empty() && std::abs(before - sung) > tailSlideDrag)
            {
              sung = before;
              centre = before;
              measured = kept;
              h0 = measured.front();
              h1 = measured.back() + 1;
              trend = 0.0f;
              tailSlide = true;
            }
          }
        }
      }
      // A long, steady held part says what was sung better than the analysed
      // centre, which a long scoop drags down or a flick at the end drags up,
      // and which reads a small deviation smaller than the line shows (the
      // ear followed the line: a G held 24 cents sharp was heard sharp).
      if (!tailSlide && framesToSeconds(h1 - h0) >= solidHeldSeconds && measured.size() >= 3)
      {
        std::vector<float> values;
        for (int i : measured)
          values.push_back(white(s + i));
        std::sort(values.begin(), values.end());
        const float spread = values[static_cast<size_t>(std::lround(0.9 * (values.size() - 1)))] -
                             values[static_cast<size_t>(std::lround(0.1 * (values.size() - 1)))];
        if (spread <= solidSpread)
          sung = centre;
      }
      // A short landing that then drifts away, loud, to the note's end (a D
      // that keeps rising toward D sharp): the analysed centre follows the
      // drift, and moving the whole note by it drags down the slide in from
      // the note before as well. The landing is the note; the drift is
      // tilted flat from where the landing ends.
      bool tailDrift = false;
      float driftBy = 0.0f, driftEnd = 0.0f;
      if (extreme < 0 && !tailSlide && framesToSeconds(h1 - h0) < flattenHeldSeconds && voiced.back() >= n - 2)
      {
        std::vector<int> tail;
        for (int i : voiced)
          if (i >= h1)
            tail.push_back(i);
        int loud = 0;
        for (int i : heard)
          if (i >= h1)
            ++loud;
        if (tail.size() >= static_cast<size_t>(tailDriftFrames) && loud >= tailDriftLoud)
        {
          driftEnd = 0.0f;
          for (size_t k = tail.size() - 3; k < tail.size(); ++k)
            driftEnd += white(s + tail[k]) / 3.0f;
          driftBy = driftEnd - centre;
          const float d = driftBy >= 0.0f ? 1.0f : -1.0f;
          // One way, to the end: up to its furthest point without turning
          // back, and staying there (a vibrato's swing comes back).
          size_t far = 0;
          for (size_t k = 1; k < tail.size(); ++k)
            if (d * (white(s + tail[k]) - white(s + tail[far])) > 0.0f)
              far = k;
          bool oneWay = d * (white(s + tail.front()) - centre) >= 0.0f;
          for (size_t k = 0; k < far; ++k)
            if (d * (white(s + tail[k + 1]) - white(s + tail[k])) < -tailDriftWobble)
              oneWay = false;
          // ... and never that side of the landing before it (a vibrato
          // swings both ways; a slide in comes from the other side).
          for (int i : voiced)
            if (i < h0 && d * (white(s + i) - centre) > tailDriftBefore)
              oneWay = false;
          const float analysed = note.getMidiNote() + note.getPitchOffset();
          tailDrift = oneWay && std::abs(driftBy) >= tailDriftMin &&
                      d * (white(s + tail[far]) - white(s + tail.back())) <= tailDriftReturn &&
                      d * (analysed - centre) >= tailDriftDrag * std::abs(driftBy);
        }
      }
      // Guide: coverage, then the nearest candidate.
      const int a = s + h0, b = s + h1;
      double covered = 0.0;
      for (const auto &g : guide)
        covered += std::max(0.0, std::min<double>(b, g.end) - std::max<double>(a, g.start));
      if (covered / std::max(1, b - a) < minCover)
      {
        // A tail without a guide note (the fall after a long note): placed by
        // where it lands, on the in-key semitone nearest that.
        const bool joined = !std::isnan(white(s)) && !std::isnan(white(s - 1)) &&
                            std::any_of(project.getNotes().begin(), project.getNotes().end(),
                                        [&](const Note &other) {
                                          return !other.isRest() && &other != &note &&
                                                 std::abs(other.getEndFrame() - s) <= 1;
                                        });
        // The previous note: its level and pitch, for a fall from it.
        const Note *before = nullptr;
        for (const auto &other : project.getNotes())
          if (!other.isRest() && &other != &note && std::abs(other.getEndFrame() - s) <= 1)
            before = &other;
        auto levelOf = [&](int f0, int f1) {
          double energy = 0.0;
          int count = 0;
          for (int f = f0; f < f1; ++f)
          {
            energy += std::pow(10.0, levelDb(f) / 10.0);
            ++count;
          }
          return count > 0 ? static_cast<float>(10.0 * std::log10(energy / count + 1e-20)) : -200.0f;
        };
        bool fallsLoud = false;
        if (before != nullptr && framesToSeconds(n) <= tailMaxSeconds)
        {
          std::vector<float> prev;
          for (int f = before->getStartFrame(); f < before->getEndFrame(); ++f)
            if (!std::isnan(white(f)))
              prev.push_back(white(f));
          if (!prev.empty())
          {
            std::sort(prev.begin(), prev.end());
            const float prevPitch = prev[prev.size() / 2];
            fallsLoud = levelOf(s, e) >= levelOf(before->getStartFrame(), before->getEndFrame()) - tailMaxQuieterDb &&
                        std::abs(prevPitch - white(s + voiced.back())) >= tailMinFall;
          }
        }
        if (joined && fallsLoud && voiced.size() >= 4)
        {
          // Where it lands: the last quarter of its voiced frames (at least
          // 4), quiet as a dying tail is.
          const size_t keep = std::max<size_t>(4, voiced.size() / 4);
          std::vector<int> landing(voiced.end() - static_cast<long>(std::min(keep, voiced.size())), voiced.end());
          std::vector<float> values;
          double landingSum = 0.0;
          for (int i : landing)
          {
            values.push_back(white(s + i));
            landingSum += white(s + i);
          }
          std::sort(values.begin(), values.end());
          const float lands = values[values.size() / 2];
          float tailTarget = 0.0f, tailDistance = 1000.0f;
          for (int k = -1; k <= 1; ++k)
          {
            const float candidate = std::round(lands - referenceOffset) + referenceOffset + k;
            if (!outOfKey(candidate) && std::abs(lands - candidate) < tailDistance)
            {
              tailTarget = candidate;
              tailDistance = std::abs(lands - candidate);
            }
          }
          if (tailDistance >= tailMinOff && tailDistance <= tailReach)
          {
            const float landingMean = static_cast<float>(landingSum / landing.size());
            Plan plan{&note, {}, tailTarget, landingMean - (lands - tailTarget), -1, "tail"};
            plan.before = note;
            plan.deviation = lands - tailTarget;
            // A tail that holds a note before it lands (a D sharp after an E,
            // then the fall to D): the held note is what is heard, the
            // landing only its end. When the hold has left the note before
            // for the landing's side, the hold goes on the target.
            float prevPitch = 0.0f;
            {
              std::vector<float> prev;
              for (int f = before->getStartFrame(); f < before->getEndFrame(); ++f)
                if (!std::isnan(white(f)))
                  prev.push_back(white(f));
              std::sort(prev.begin(), prev.end());
              prevPitch = prev.empty() ? 0.0f : prev[prev.size() / 2];
            }
            const float towardTarget = tailTarget >= prevPitch ? 1.0f : -1.0f;
            // What is heard of it: from where the fall in from the note
            // before first turns, to the end (the hold's wobble and its own
            // fall alike; the fall in is the note before's).
            size_t arrived = 0;
            while (arrived + 1 < voiced.size() &&
                   towardTarget * (white(s + voiced[arrived + 1]) - white(s + voiced[arrived])) > 0.0f)
              ++arrived;
            double heardSum = 0.0;
            for (size_t k = arrived; k < voiced.size(); ++k)
              heardSum += white(s + voiced[k]);
            const float heardMean = static_cast<float>(heardSum / (voiced.size() - arrived));
            if (h1 - h0 >= tailBodyFrames && measured.size() >= static_cast<size_t>(tailBodyFrames) &&
                towardTarget * (centre - prevPitch) >= tailBodyLeft &&
                std::abs(centre - tailTarget) <= tailBodyReach &&
                std::abs(centre - tailTarget) > std::abs(lands - tailTarget) &&
                voiced.size() - arrived >= static_cast<size_t>(tailBodyFrames))
            {
              plan.want = tailTarget;
              plan.deviation = heardMean - tailTarget;
              plan.review += "tail-body ";
              for (size_t k = arrived; k < voiced.size(); ++k)
                plan.measured.push_back(s + voiced[k]);
              note.markDirty();
              plans.push_back(plan);
              ++counts["fix-tail"];
              continue;
            }
            for (int i : landing)
              plan.measured.push_back(s + i);
            note.markDirty();
            plans.push_back(plan);
            ++counts["fix-tail"];
            continue;
          }
        }
        bool nearbyGuide = false;
        for (const auto &g : guide)
          if (framesToSeconds(voiced.back() + 1 - voiced.front()) >= minNoteSeconds)
          if (g.end > a - windowFrames && g.start < b + windowFrames)
            nearbyGuide = true;
        if (!nearbyGuide)
        {
          report("no-guide", sung, trend, 0.0f);
          continue;
        }
        // A guide note ending just before it or starting just after: a late
        // or early entry, judged against the guide notes around it.
      }
      // Guide notes sounding with the held part come first; the ones within
      // windowFrames around it (a late or early entry) only when none of
      // those is within a semitone of what was sung.
      // Sung off the key (an A sharp in C major): wrong whatever it is near,
      // so the guide note sounding with it is in reach a little further.
      // Only a note sung squarely on an off-key semitone, whose guide note is
      // the semitone next to it, gets the extra reach.
      const float offKeySemitone = std::round(sung - referenceOffset) + referenceOffset;
      float heldSpread = 0.0f;
      if (measured.size() >= 2)
      {
        std::vector<float> values;
        for (int i : measured)
          values.push_back(white(s + i));
        std::sort(values.begin(), values.end());
        const auto at = [&](double q) {
          return values[static_cast<size_t>(std::lround(q * (values.size() - 1)))];
        };
        heldSpread = at(0.9) - at(0.1); // a scoop's first frames are not the hold
      }
      // ... and held there, not passing through it on the way somewhere.
      const bool offKey = outOfKey(sung) && std::abs(sung - offKeySemitone) <= offKeyTolerance &&
                          heldSpread <= offKeyMaxSpread;
      const float reach = offKey ? offKeyDeviation : maxDeviation;
      std::vector<float> candidates;
      for (const auto &g : guide)
        if (g.end > a && g.start < b && std::abs(sung - g.pitch) <= reach &&
            (std::abs(sung - g.pitch) <= maxDeviation || std::abs(offKeySemitone - g.pitch) == 1.0f) &&
            std::find(candidates.begin(), candidates.end(), g.pitch) == candidates.end())
          candidates.push_back(g.pitch);
      const bool fromNeighbours = candidates.empty();
      if (fromNeighbours)
        for (const auto &g : guide)
          if (g.end > a - windowFrames && g.start < b + windowFrames &&
              std::find(candidates.begin(), candidates.end(), g.pitch) == candidates.end())
            candidates.push_back(g.pitch);
      std::sort(candidates.begin(), candidates.end(), [&](float x, float y) {
        return std::abs(sung - x) < std::abs(sung - y);
      });
      bool snapped = false;
      if (candidates.empty() || std::abs(sung - candidates[0]) > reach)
      {
        // The singer's own melody (or a guide note gone wrong): held near an
        // in-key note, it is that note sung off, whatever the guide says.
        const float nearest = std::round(centre - referenceOffset) + referenceOffset;
        if (!outOfKey(nearest) && std::abs(centre - nearest) >= snapMinOff &&
            std::abs(centre - nearest) <= snapMaxOff && heldSpread <= snapMaxSpread &&
            framesToSeconds(h1 - h0) >= snapHeldSeconds)
        {
          candidates.assign(1, nearest);
          sung = centre;
          snapped = true;
        }
      }
      // A step between two notes a third apart (E, D, C): the one in-key
      // note between them, when the guide has that step close by (sung late
      // or early). Without it in the guide it is a glide from one to the
      // other (C, E), not a note.
      bool scaleStep = false;
      if (candidates.empty() || std::abs(sung - candidates[0]) > reach)
      {
        const Note *prev = nullptr, *next = nullptr;
        for (const auto &other : project.getNotes())
        {
          if (other.isRest() || &other == &note)
            continue;
          if (std::abs(other.getEndFrame() - s) <= 3)
            prev = &other;
          if (std::abs(other.getStartFrame() - e) <= 3)
            next = &other;
        }
        if (prev != nullptr && next != nullptr)
        {
          const float p0 = std::round(prev->getMidiNote() - referenceOffset) + referenceOffset;
          const float p1 = std::round(next->getMidiNote() - referenceOffset) + referenceOffset;
          std::vector<float> between;
          for (float k = std::min(p0, p1) + 1.0f; k < std::max(p0, p1); k += 1.0f)
            if (!outOfKey(k))
              between.push_back(k);
          if (!outOfKey(p0) && !outOfKey(p1) && std::abs(p0 - p1) >= 3.0f && std::abs(p0 - p1) <= 4.0f &&
              between.size() == 1 && std::abs(sung - between[0]) <= scaleStepReach &&
              std::any_of(guide.begin(), guide.end(), [&](const auto &g) {
                return g.pitch == between[0] && g.end > s - 2 * windowFrames && g.start < e + 2 * windowFrames;
              }) &&
              std::abs(prev->getMidiNote() - p0) <= scaleStepEnds && std::abs(next->getMidiNote() - p1) <= scaleStepEnds)
          {
            candidates.assign(1, between[0]);
            scaleStep = true;
          }
        }
      }
      if (candidates.empty() || std::abs(sung - candidates[0]) > reach)
      {
        // The singer's own melody, sung in tune on an in-key note: its pitch
        // stays, but a slant (a vibrato sliding down) is flattened as on any
        // note, about its own centre.
        const float inKey = std::round(centre - referenceOffset) + referenceOffset;
        if (!outOfKey(inKey) && std::abs(centre - inKey) <= melodyFlatTolerance && extreme < 0 &&
            !tailSlide && std::abs(trend) >= flattenMin && std::abs(trend) <= maxFlattenTrend &&
            framesToSeconds(h1 - h0) >= flattenHeldSeconds)
        {
          Plan plan{&note, {}, inKey, centre, -1, "tilt"};
          plan.before = note;
          plan.deviation = 0.0f;
          plan.review += "melody-flatten ";
          for (int i : measured)
            plan.measured.push_back(s + i);
          if (framesToSeconds(h1 - h0) >= driftHeldSeconds)
          {
            note.setPitchDrift(driftFactor);
            plan.how = "drift";
          }
          else
          {
            const float pivot = static_cast<float>(h0) / static_cast<float>(n - 1);
            note.setTiltLeft(0.0f);
            note.setTiltRight(-slope * static_cast<float>(n - 1) * (1.0f - pivot));
            note.setTiltPivot(pivot);
          }
          note.markDirty();
          plans.push_back(plan);
          ++counts[("fix-" + plan.how).toStdString()];
          continue;
        }
        report("melody-diff", sung, trend, 0.0f);
        std::printf("review\tmelody-diff\t%s\t%d\t%.2f\t%.2f\n", clock(framesToSeconds(s)).toRawUTF8(), s, sung, heldMean);
        continue;
      }
      // Two about as near: the one sounding longer with the held part.
      auto overlapWith = [&](float pitch) {
        double total = 0.0;
        for (const auto &g : guide)
          if (g.pitch == pitch)
            total += std::max(0.0, std::min<double>(b, g.end) - std::max<double>(a, g.start));
        return total;
      };
      // A tie at the start of a turn (E, F sharp, E): the note after the next
      // returns to it, so it is that note.
      if (candidates.size() > 1 &&
          std::abs(std::abs(sung - candidates[0]) - std::abs(sung - candidates[1])) < tieMargin)
      {
        const Note *next = nullptr, *afterNext = nullptr;
        for (const auto &other : project.getNotes())
          if (!other.isRest() && std::abs(other.getStartFrame() - e) <= 3)
            next = &other;
        if (next != nullptr)
          for (const auto &other : project.getNotes())
            if (!other.isRest() && std::abs(other.getStartFrame() - next->getEndFrame()) <= 3)
              afterNext = &other;
        if (next != nullptr && afterNext != nullptr &&
            std::abs(next->getMidiNote() - sung) >= turnStep)
        {
          const float back = std::round(afterNext->getMidiNote() - referenceOffset) + referenceOffset;
          for (int k = 0; k < 2; ++k)
            if (candidates[static_cast<size_t>(k)] == back)
            {
              candidates.assign(1, back);
              break;
            }
        }
      }
      if (candidates.size() > 1 &&
          std::abs(std::abs(sung - candidates[0]) - std::abs(sung - candidates[1])) < tieMargin)
      {
        const double o0 = overlapWith(candidates[0]), o1 = overlapWith(candidates[1]);
        const double winner = std::max(o0, o1), loser = std::min(o0, o1);
        if (winner >= tieOverlapRatio * loser && winner >= tieOverlapShare * std::max(1, b - a))
        {
          if (o1 > o0)
            std::swap(candidates[0], candidates[1]);
          candidates.resize(1);
        }
      }
      if (candidates.size() > 1 &&
          std::abs(std::abs(sung - candidates[0]) - std::abs(sung - candidates[1])) < tieMargin)
      {
        report("ambiguous", sung, trend, 0.0f);
        std::printf("review\tambiguous\t%s\t%d\t%.2f\t%.2f\n", clock(framesToSeconds(s)).toRawUTF8(), s, sung, heldMean);
        continue;
      }
      float target = candidates[0];
      // The guide itself off the key (a rounded A sharp in C major): the
      // in-key semitone nearest what was sung instead.
      bool guideOffKey = false;
      if (outOfKey(target))
      {
        float best = target, bestDistance = 1000.0f;
        for (int k = -2; k <= 2; ++k)
        {
          const float candidate = std::round(sung - referenceOffset) + referenceOffset + k;
          if (!outOfKey(candidate) && std::abs(sung - candidate) < bestDistance)
          {
            best = candidate;
            bestDistance = std::abs(sung - candidate);
          }
        }
        if (bestDistance <= maxDeviation)
        {
          target = best;
          guideOffKey = true;
        }
      }
      // Sung squarely on an off-key note: the nearer in-key note, whatever
      // the guide says (a guide note gone wrong); about as near either way,
      // the guide decides.
      bool offKeyNearest = false;
      if (outOfKey(offKeySemitone) && std::abs(sung - offKeySemitone) <= offKeyNearestOn &&
          framesToSeconds(h1 - h0) >= solidHeldSeconds)
      {
        float below = offKeySemitone - 1.0f, above = offKeySemitone + 1.0f;
        while (outOfKey(below)) below -= 1.0f;
        while (outOfKey(above)) above += 1.0f;
        const float dBelow = sung - below, dAbove = above - sung;
        if (std::abs(dBelow - dAbove) >= offKeyNearestMargin)
        {
          const float nearer = dBelow < dAbove ? below : above;
          if (nearer != target)
          {
            target = nearer;
            offKeyNearest = true;
          }
        }
      }
      // Sung cleanly on another semitone where the guide note is not even
      // sounding for most of it (an anticipation, a pickup): a different
      // melody, not intonation.
      double targetCover = 0.0;
      {
        const double v0 = s + voiced.front(), v1 = s + voiced.back() + 1;
        for (const auto &g : guide)
          if (g.pitch == target)
            targetCover += std::max(0.0, std::min(v1, g.end) - std::max(v0, g.start));
        targetCover /= std::max(1.0, v1 - v0);
      }
      // An extreme further from the target than the held part is not where
      // the voice lands but a slide away from it (into an off-key next note).
      if (extreme >= 0 && std::abs(white(extreme) - target) > std::abs(centre - target) &&
          std::abs(white(extreme) - centre) >= slideAwayMin)
        extreme = -1;
      // The notes either side, sung pitch, when they touch this one.
      const Note *prevNote = nullptr, *nextNote = nullptr;
      for (const auto &other : project.getNotes())
      {
        if (other.isRest() || &other == &note)
          continue;
        if (std::abs(other.getEndFrame() - s) <= 3)
          prevNote = &other;
        if (std::abs(other.getStartFrame() - e) <= 3)
          nextNote = &other;
      }
      if (prevNote != nullptr && nextNote != nullptr)
      {
        auto inKeyNear = [&](float pitch) {
          float best = pitch, distance = 1000.0f;
          for (int k = -1; k <= 1; ++k)
          {
            const float candidate = std::round(pitch - referenceOffset) + referenceOffset + k;
            if (!outOfKey(candidate) && std::abs(pitch - candidate) < distance)
            {
              best = candidate;
              distance = std::abs(pitch - candidate);
            }
          }
          return best;
        };
        const float before = inKeyNear(prevNote->getMidiNote()), after = inKeyNear(nextNote->getMidiNote());
        const float semitone = std::round(sung - referenceOffset) + referenceOffset;
        // An off-key semitone between two notes a semitone either side, in
        // one direction (C, C sharp, D): a chromatic passing note.
        if (outOfKey(semitone) && std::abs(sung - semitone) <= passingOnNote &&
            framesToSeconds(n) <= passingSeconds &&
            std::abs(std::abs(before - semitone) - 1.0f) <= passingTolerance &&
            std::abs(std::abs(after - semitone) - 1.0f) <= passingTolerance &&
            (before - semitone) * (after - semitone) < 0.0f)
        {
          report("passing", sung, trend, target);
          continue;
        }
        // A short turn above or below both neighbours: an ornament.
        // (Not when the guide sounding with it is a semitone away: that is
        // the guide's note sung off, not a turn.)
        bool guideExplains = false;
        for (const auto &g : guide)
          if (g.end > s && g.start < e && std::abs(g.pitch - semitone) <= 1.0f)
            guideExplains = true;
        if (outOfKey(semitone) && !guideExplains && framesToSeconds(n) <= ornamentSeconds &&
            ((sung - before >= ornamentStep && sung - after >= ornamentStep) ||
             (before - sung >= ornamentStep && after - sung >= ornamentStep)))
        {
          report("ornament", sung, trend, target);
          continue;
        }
      }
      // A short note sung squarely on an off-key semitone that steps by a
      // semitone, legato, into an in-key next note (G, D sharp, E; D, D sharp,
      // D): a chromatic appoggiatura or neighbour note starting a turn - an
      // ornament, kept as sung.
      if (nextNote != nullptr)
      {
        const float semitone = std::round(sung - referenceOffset) + referenceOffset;
        const float nextPitch = nextNote->getMidiNote() + nextNote->getPitchOffset();
        const float next = std::round(nextPitch - referenceOffset) + referenceOffset;
        if (outOfKey(semitone) && std::abs(sung - semitone) <= passingOnNote &&
            framesToSeconds(voiced.back() + 1 - voiced.front()) <= passingSeconds &&
            !outOfKey(next) && std::abs(nextPitch - next) <= passingTolerance &&
            std::abs(std::abs(next - semitone) - 1.0f) < 0.01f &&
            !std::isnan(white(e - 1)) && !std::isnan(white(nextNote->getStartFrame())))
        {
          // ... where the guide already has the note it leads to: an extra
          // note before it (a D sharp where the guide holds E or D). Where the
          // guide has a note of its own there (a D between C and E), the
          // off-key note is that melody note sung wrong.
          float guideHere = 0.0f;
          double most = 0.0;
          for (const auto &g : guide)
          {
            const double overlap = std::max(0.0, std::min<double>(s + voiced.back() + 1, g.end) -
                                                     std::max<double>(s + voiced.front(), g.start));
            if (overlap > most)
            {
              most = overlap;
              guideHere = g.pitch;
            }
          }
          if (most > 0.0 && guideHere == next)
          {
            report("ornament", sung, trend, target);
            continue;
          }
        }
      }
      // A short note sliding at its end into the next note, whose guide pitch
      // is this one's target too: the next note's scoop, kept as sung.
      {
        const Note *next = nullptr;
        for (const auto &other : project.getNotes())
          if (!other.isRest() && other.getStartFrame() >= e - 1 && other.getStartFrame() <= e + 2)
            next = &other;
        // How long it is sung (a consonant before it is not part of the scoop).
        if (next != nullptr && framesToSeconds(voiced.back() + 1 - voiced.front()) <= approachSeconds &&
            voiced.size() >= 3 &&
            voiced.back() >= n - 2)
        {
          float nextGuide = 0.0f;
          const double probe = next->getStartFrame() + 3;
          for (const auto &g : guide)
            if (g.start <= probe && probe < g.end)
              nextGuide = g.pitch;
          const float end0 = white(s + voiced[voiced.size() - 3]);
          const float end1 = white(s + voiced.back());
          const float toward = (target > sung ? 1.0f : -1.0f) * (end1 - end0);
          if (nextGuide == target && std::abs(sung - target) >= approachMinOffset && toward >= approachRise)
          {
            report("approach", sung, trend, target);
            continue;
          }
        }
      }
      if (const float semitone = std::round(sung - referenceOffset) + referenceOffset;
          !offKey && std::abs(semitone - target) > 0.5f && std::abs(sung - semitone) <= otherNoteTolerance &&
          targetCover < otherNoteMaxCover)
      {
        report("other-note", sung, trend, target);
        continue;
      }
      // ... landing near the target and drifting away from it.
      if (tailDrift && std::abs(centre - target) <= tailDriftLanding &&
          std::abs(driftEnd - target) >= std::abs(centre - target) + tailDriftAway)
      {
        // The tilt turns from where the drift starts (back from its end,
        // while the line keeps moving its way, not before the landing).
        const float d = driftBy >= 0.0f ? 1.0f : -1.0f;
        int pivot = voiced.back();
        for (int i : voiced)
          if (i >= h1 && d * (white(s + i) - white(s + pivot)) > 0.0f)
            pivot = i;
        while (pivot - 1 >= h0 && !std::isnan(white(s + pivot - 1)) &&
               d * (white(s + pivot) - white(s + pivot - 1)) > 0.0f)
          --pivot;
        // The landing, before the turn, is the note's pitch.
        std::vector<int> landing;
        double landingSum = 0.0;
        for (int i : measured)
          if (i <= pivot)
          {
            landing.push_back(s + i);
            landingSum += white(s + i);
          }
        const float level = landing.empty() ? centre : static_cast<float>(landingSum / landing.size());
        sung = level;
        const bool needsShift = std::abs(level - target) > tolerance;
        // Sung on from the note before at its pitch (a C held on into the
        // syllable, then a slide up to the landing): moving the whole note
        // would move that C too. The first run asks the App to split the
        // note where the slide leaves it; only the second half moves.
        if (needsShift && probe && !landing.empty())
        {
          std::vector<float> away;
          for (int i : voiced)
            if (i < h0 && std::abs(white(s + i) - level) >= splitGap)
              away.push_back(white(s + i));
          int cut = -1;
          if (!away.empty())
          {
            std::sort(away.begin(), away.end());
            const float before = away[away.size() / 2];
            // The first part: from the first voiced frame, while it stays
            // around that pitch.
            int last = -1;
            for (int i : voiced)
            {
              if (i >= h0 || std::abs(white(s + i) - before) > splitBand)
                break;
              last = i;
            }
            int loudBefore = 0;
            for (int i : heard)
              if (i <= last)
                ++loudBefore;
            // ... long enough to be heard as a note, loud, the note before's
            // pitch, and then one slide to the landing.
            const Note *prev = nullptr;
            for (const auto &other : project.getNotes())
              if (!other.isRest() && &other != &note && std::abs(other.getEndFrame() - s) <= 3)
                prev = &other;
            bool slide = last >= 0;
            for (int i = last; slide && i + 1 < landing.front() - s; ++i)
              if (std::isnan(white(s + i + 1)) || d * (white(s + i + 1) - white(s + i)) <= 0.0f)
                slide = false;
            if (slide && framesToSeconds(last + 1 - voiced.front()) >= splitPartSeconds &&
                2 * loudBefore >= last + 1 - voiced.front() && prev != nullptr &&
                std::abs(prev->getMidiNote() + prev->getPitchOffset() - before) <= splitFromPrevious)
              cut = last + 1;
          }
          if (cut > 0)
          {
            splitRequests.push_back({s, s + cut, landing.front(), landing.back() + 1, s + pivot, level, target});
            continue;
          }
        }
        Plan plan{&note, landing, target, needsShift ? target : level, -1, "tail-tilt"};
        plan.before = note;
        plan.deviation = sung - target;
        tiltTail(note, s, n, s + pivot, level);
        note.markDirty();
        plans.push_back(plan);
        ++counts["fix-tail-tilt"];
        continue;
      }
      const bool flatten = extreme < 0 && std::abs(trend) >= flattenMin &&
                           std::abs(trend) <= maxFlattenTrend &&
                           framesToSeconds(h1 - h0) >= flattenHeldSeconds;
      // A small correction on a note that starts or ends in consonant noise
      // is not worth it: the render rebuilds the consonant's edge with the
      // note's pitch (a buzzing "f"), heard more than a few cents.
      const bool consonantEdge =
          breathsibilance::isNoise(breathsibilance::labelAt(audio.breathSibilance, s)) ||
          breathsibilance::isNoise(breathsibilance::labelAt(audio.breathSibilance, s + 1)) ||
          breathsibilance::isNoise(breathsibilance::labelAt(audio.breathSibilance, e - 1)) ||
          breathsibilance::isNoise(breathsibilance::labelAt(audio.breathSibilance, e - 2));
      if (extreme < 0 && consonantEdge && std::abs(sung - target) <= consonantTolerance && !flatten)
      {
        ++counts["consonant-small"];
        noteTarget("consonant-small", note, sung, target, h1 - h0,
                   juce::String(fromNeighbours ? "neighbour-target " : "") +
                       (snapped ? "snapped-to-key " : ""));
        continue;
      }
      if (extreme < 0 && std::abs(sung - target) <= tolerance && !flatten)
      {
        ++counts["in-tune"];
        std::vector<int> frames;
        for (int i : measured)
          frames.push_back(s + i);
        inTune.push_back({&note, target, sung, centre, frames});
        noteTarget("in-tune", note, sung, target, h1 - h0,
                   juce::String(fromNeighbours ? "neighbour-target " : "") +
                       (snapped ? "snapped-to-key " : ""));
        continue;
      }
      // The held part moves by the note's deviation (the tilt's own lift of
      // it is corrected away by the measuring passes).
      Plan plan{&note, {}, target, centre - (sung - target), -1, "shift"};
      plan.before = note;
      plan.deviation = sung - target;
      // For the agent's own second look: where the rule's evidence is thin.
      if (!tailSlide && std::abs(sung - heldMean) > reviewDisagree)
        plan.review += "centre-vs-line ";
      if (fromNeighbours)
        plan.review += "neighbour-target ";
      if (tailSlide)
        plan.review += "tail-slide ";
      if (guideOffKey)
        plan.review += "off-key-guide ";
      if (offKeyNearest)
        plan.review += "off-key-nearest ";
      if (snapped)
        plan.review += "snapped-to-key ";
      if (scaleStep)
        plan.review += "scale-step ";
      for (int i : measured)
        plan.measured.push_back(s + i);
      if (extreme >= 0)
      {
        plan.extreme = extreme;
        plan.want = target + (trend < 0.0f ? -overshoot : overshoot);
        plan.how = "extreme";
      }
      else if (flatten)
      {
        if (framesToSeconds(h1 - h0) >= driftHeldSeconds)
        {
          note.setPitchDrift(driftFactor);
          plan.how = "drift";
        }
        else
        {
          const float pivot = static_cast<float>(h0) / static_cast<float>(n - 1);
          note.setTiltLeft(0.0f);
          note.setTiltRight(-slope * static_cast<float>(n - 1) * (1.0f - pivot));
          note.setTiltPivot(pivot);
          plan.how = "tilt";
        }
      }
      note.markDirty();
      plans.push_back(plan);
      ++counts[("fix-" + plan.how).toStdString()];
      noteTarget(("fix-" + plan.how).toRawUTF8(), note, sung, target, h1 - h0, plan.review);
    }
    if (targetsFile != nullptr)
      std::fclose(targetsFile);
    if (probe)
      return 0;
    // One vowel cut into two notes of the same pitch: when one is corrected,
    // the other goes with it, so the two render as one and no splice sits
    // inside the vowel.
    auto continuous = [&](const Note &left, const Note &right) {
      return std::abs(right.getStartFrame() - left.getEndFrame()) <= 1 &&
             !std::isnan(white(left.getEndFrame() - 1)) && !std::isnan(white(right.getStartFrame()));
    };
    for (const auto &held : inTune)
      for (const auto &plan : std::vector<Plan>(plans))
        if (plan.target == held.target &&
            (continuous(*plan.note, *held.note) || continuous(*held.note, *plan.note)) &&
            !held.measured.empty())
        {
          Plan joined{held.note, held.measured, held.target, held.centre - (held.sung - held.target),
                      -1, "join"};
          joined.before = *held.note;
          joined.deviation = held.sung - held.target;
          held.note->markDirty();
          plans.push_back(joined);
          --counts["in-tune"];
          ++counts["fix-join"];
          break;
        }
    // A note cut where the fry it starts with begins (FryNoteCuts.h) left the
    // frames before the fry as a note of their own, with no voiced frame to
    // judge ("no-voiced" above). It goes with the note it led into, so that
    // the two move as the one note they were: left where it was, the base
    // took a step between them, and the note before glided up to 17 cents
    // off over its last two frames (煙幕 at 1:19.3).
    struct Lead
    {
      Note *note;
      Note *into;
      Note before;   // the lead as it was
      float intoWas; // the centre of the note it leads into, before
    };
    std::vector<Lead> leads;
    for (const auto &[first, last] : fry::onsets(project.getAudioData().rawF0))
    {
      juce::ignoreUnused(last);
      Note *lead = nullptr;
      Note *into = nullptr;
      for (auto &n : project.getNotes())
      {
        if (n.isRest())
          continue;
        if (n.getSrcEndFrame() == first)
          lead = &n;
        if (n.getSrcStartFrame() == first)
          into = &n;
      }
      const auto plan = std::find_if(plans.begin(), plans.end(),
                                     [&](const Plan &p) { return p.note == into; });
      const auto own = std::find_if(plans.begin(), plans.end(),
                                    [&](const Plan &p) { return p.note == lead; });
      // Only such a piece: a note that sings and happens to end where the
      // fry begins is judged on its own.
      bool silent = lead != nullptr;
      for (int f = lead != nullptr ? lead->getStartFrame() : 0;
           silent && f < lead->getEndFrame(); ++f)
        if (f >= 0 && static_cast<size_t>(f) < recorded.size() &&
            !std::isnan(recorded[static_cast<size_t>(f)]))
          silent = false;
      if (silent && into != nullptr && plan != plans.end() && own == plans.end())
        leads.push_back({lead, into, *lead, plan->before.getMidiNote()});
    }
    // A note put back where it was ("over-shift" below) takes its lead back
    // with it, untouched.
    const auto leadsFollow = [&] {
      for (const auto &lead : leads)
      {
        const float shift = lead.into->getMidiNote() - lead.intoWas;
        *lead.note = lead.before;
        if (shift != 0.0f)
        {
          lead.note->setMidiNote(lead.before.getMidiNote() + shift);
          lead.note->markDirty();
        }
      }
    };
    // Shift: measure on the rebuilt curve, correct, three times.
    for (int pass = 0; pass < 3; ++pass)
    {
      leadsFollow();
      PitchCurveProcessor::rebuildBaseFromNotes(project);
      for (const auto &plan : plans)
      {
        float now = 0.0f;
        if (plan.extreme >= 0)
          now = white(plan.extreme);
        else
        {
          double sum = 0.0;
          int count = 0;
          for (int f : plan.measured)
            if (const float v = white(f); !std::isnan(v))
            {
              sum += v;
              ++count;
            }
          now = count > 0 ? static_cast<float>(sum / count) : plan.want;
        }
        if (std::getenv("GUIDE_FIX_DEBUG") != nullptr &&
            plan.note->getStartFrame() == std::atoi(std::getenv("GUIDE_FIX_DEBUG")))
          std::fprintf(stderr, "debug pass %d: how %s tilt L%.2f R%.2f P%.2f drift %.2f | midi %.3f want %.3f now %.3f measured %zu frames [%d..%d]\n",
                       pass, plan.how.toRawUTF8(), plan.note->getTiltLeft(), plan.note->getTiltRight(),
                       plan.note->getTiltPivot(), plan.note->getPitchDrift(),
                       plan.note->getMidiNote(), plan.want, now, plan.measured.size(),
                       plan.measured.empty() ? -1 : plan.measured.front(),
                       plan.measured.empty() ? -1 : plan.measured.back());
        if (!std::isnan(now))
          plan.note->setMidiNote(plan.note->getMidiNote() + plan.want - now);
      }
    }
    // Never move a note further than it is off (the guide video's rule):
    // anything that did is put back and reported.
    {
      bool reverted = false;
      for (auto &plan : plans)
        if (std::abs(plan.note->getMidiNote() - plan.before.getMidiNote()) >
            std::abs(plan.deviation) + maxExtraShift)
        {
          std::printf("over-shift\t%s\t%d\t%.2f\t%+.0f\t%.0f\n",
                      clock(framesToSeconds(plan.note->getStartFrame())).toRawUTF8(),
                      plan.note->getStartFrame(), plan.before.getMidiNote(),
                      (plan.note->getMidiNote() - plan.before.getMidiNote()) * 100.0f,
                      plan.target);
          *plan.note = plan.before;
          --counts[("fix-" + plan.how).toStdString()];
          plan.how = "";
          ++counts["over-shift"];
          reverted = true;
        }
      plans.erase(std::remove_if(plans.begin(), plans.end(),
                                 [](const Plan &p) { return p.how.isEmpty(); }),
                  plans.end());
      (void)reverted;
    }
    // A note whose end slides into the next note lands where that note now
    // is: its tail turns (a tilt from where the slide starts) by the
    // difference between the two notes' shifts.
    {
      std::map<int, float> shiftOf;
      for (const auto &plan : plans)
        shiftOf[plan.note->getStartFrame()] = plan.note->getMidiNote() - plan.before.getMidiNote();
      auto &notes = project.getNotes();
      for (auto &left : notes)
      {
        if (left.isRest())
          continue;
        const Note *right = nullptr;
        for (const auto &other : notes)
          if (!other.isRest() && &other != &left && continuous(left, other))
            right = &other;
        if (right == nullptr)
          continue;
        const float leftShift = shiftOf.count(left.getStartFrame()) ? shiftOf[left.getStartFrame()] : 0.0f;
        const float rightShift = shiftOf.count(right->getStartFrame()) ? shiftOf[right->getStartFrame()] : 0.0f;
        const float follow = rightShift - leftShift;
        const int e = left.getEndFrame(), n = e - left.getStartFrame();
        if (std::abs(follow) < followMin || n < 8 || std::abs(left.getTiltLeft()) > 0.001f ||
            std::abs(left.getTiltRight()) > 0.001f)
          continue;
        // The slide: the last frames moving steadily toward the next note.
        const float goal = recorded[static_cast<size_t>(right->getStartFrame())];
        const int from = e - followFrames;
        const float dir = goal - recorded[static_cast<size_t>(from)] >= 0.0f ? 1.0f : -1.0f;
        bool slides = true;
        for (int f = from; f < e - 1; ++f)
          if (std::isnan(recorded[static_cast<size_t>(f)]) || std::isnan(recorded[static_cast<size_t>(f + 1)]) ||
              dir * (recorded[static_cast<size_t>(f + 1)] - recorded[static_cast<size_t>(f)]) < 0.0f)
            slides = false;
        if (!slides || dir * (recorded[static_cast<size_t>(e - 1)] - recorded[static_cast<size_t>(from)]) < followRise)
          continue;
        // Only where the slide lands off the key (a D sharp the next note was
        // corrected away from): elsewhere the slide is the singer's own.
        const float end = recorded[static_cast<size_t>(e - 1)];
        const float endNote = std::round(end - referenceOffset) + referenceOffset;
        if (!outOfKey(endNote) || std::abs(end - endNote) > followEndTolerance || std::abs(follow) < followMinOffKey)
          continue;
        auto existing = std::find_if(plans.begin(), plans.end(), [&](const Plan &p) { return p.note == &left; });
        if (existing == plans.end())
        {
          Plan follower{&left, {}, 0.0f, 0.0f, -1, "follow"};
          follower.before = left;
          plans.push_back(follower);
          existing = plans.end() - 1;
          ++counts["fix-follow"];
        }
        else
          existing->how += "+follow";
        left.setTiltLeft(0.0f);
        left.setTiltPivot(static_cast<float>(from - left.getStartFrame()) / static_cast<float>(n - 1));
        left.setTiltRight(follow);
        left.markDirty();
      }
    }
    for (auto &plan : plans)
    {
      if (std::abs(plan.note->getMidiNote() - plan.before.getMidiNote()) > reviewShift)
        plan.review += "big-shift ";
      if (plan.review.isNotEmpty())
        std::printf("review\t%s\t%s\t%d\t%.2f\t%+.0f\t%.0f\n", plan.review.trim().toRawUTF8(),
                    clock(framesToSeconds(plan.note->getStartFrame())).toRawUTF8(),
                    plan.note->getStartFrame(), plan.before.getMidiNote(),
                    (plan.note->getMidiNote() - plan.before.getMidiNote()) * 100.0f, plan.target);
    }
    leadsFollow();
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    for (const auto &plan : plans)
      std::printf("fix-%s\t%s\t%d\t%.2f\t%.2f\t%s\n", plan.how.toRawUTF8(),
                  clock(framesToSeconds(plan.note->getStartFrame())).toRawUTF8(),
                  plan.note->getStartFrame(), plan.note->getMidiNote(), plan.target,
                  plan.how.endsWith("tilt") ? juce::String::formatted("R%+.2f P%.2f",
                      plan.note->getTiltRight(), plan.note->getTiltPivot()).toRawUTF8()
                  : plan.how == "drift" ? "drift 0.5" : "");
    if (!ProjectSerializer::saveToFile(project, output))
      return 1;
    for (const auto &[k, v] : counts)
      std::printf("# %s %d\n", k.c_str(), v);
    return 0;
    };
    {
      std::fflush(stdout);
      const int saved = dup(1), sink = open("/dev/null", O_WRONLY);
      dup2(sink, 1);
      const int probed = guideFix(true);
      std::fflush(stdout);
      dup2(saved, 1);
      close(sink);
      close(saved);
      if (probed != 0)
        return probed;
    }
    return guideFix(false);
  }
  // WarpRender <in.pitchnet> <out.f32> --vocoder-probe <start> <end>
  // [<from> <to> <semitones>]: the vocoder's raw output for frames
  // [start,end) of the project's composed F0 and analysis mel, optionally with
  // the F0 of frames [from,to) moved by some semitones. Float32, mono, written
  // as is. For finding what an edit's waveform depends on.
  if (argc >= 6 && juce::String::fromUTF8(argv[3]) == "--vocoder-probe")
  {
    Project loaded;
    if (!ProjectSerializer::loadFromFile(loaded, input))
      return 1;
    sourceaudio::attachRecording(loaded, input);
    juce::String ortError;
    if (!OnnxRuntime::initialise(&ortError))
      return 1;
    EditorController controller(false);
    controller.setProject(std::make_unique<Project>(loaded));
    auto &p = *controller.getProject();
    auto &data = p.getAudioData();
    if (data.melSpectrogram.empty())
    {
      MelSpectrogram mel(data.sampleRate, N_FFT, HOP_SIZE, NUM_MELS, FMIN, FMAX);
      data.melSpectrogram = mel.compute(data.originalWaveform.getReadPointer(0),
                                        data.originalWaveform.getNumSamples());
    }
    PitchCurveProcessor::rebuildBaseFromNotes(p);
    const int start = juce::String::fromUTF8(argv[4]).getIntValue();
    const int end = juce::String::fromUTF8(argv[5]).getIntValue();
    auto f0 = p.getAdjustedF0ForRange(start, end);
    if (argc >= 9)
    {
      const int from = juce::String::fromUTF8(argv[6]).getIntValue();
      const int to = juce::String::fromUTF8(argv[7]).getIntValue();
      const float ratio = std::pow(2.0f, juce::String::fromUTF8(argv[8]).getFloatValue() / 12.0f);
      for (int f = std::max(from, start); f < std::min(to, end); ++f)
        if (f0[(size_t)(f - start)] > 0.0f)
          f0[(size_t)(f - start)] *= ratio;
    }
    std::vector<std::vector<float>> melRange(data.melSpectrogram.begin() + start,
                                             data.melSpectrogram.begin() + end);
    auto *vocoder = controller.getVocoder();
    // Loaded as resynthesizeIncrementalAsync loads it. PITCHNET_VOCODER_MODEL
    // loads another file instead: an .onnx with no Core ML bundle beside it
    // runs on ONNX Runtime (CPU, float32), to compare with Core ML.
    if (vocoder != nullptr && !vocoder->isLoaded())
    {
      const char *modelOverride = std::getenv("PITCHNET_VOCODER_MODEL");
      const juce::File model = modelOverride != nullptr ? juce::File(modelOverride)
                                                        : PlatformPaths::getVocoderModelFile();
      vocoder->loadModel(model);
      std::printf("vocoder model %s (Core ML bundle beside it: %s), device %s\n",
                  model.getFullPathName().toRawUTF8(),
                  model.getSiblingFile("pc_nsf_hifigan.mlmodelc").isDirectory() ? "yes" : "no",
                  vocoder->getExecutionDevice().toRawUTF8());
    }
    if (vocoder == nullptr || !vocoder->isLoaded())
    {
      std::fprintf(stderr, "vocoder not loaded\n");
      return 1;
    }
    const auto wave = vocoder->infer(melRange, f0);
    juce::FileOutputStream out(output);
    if (!out.openedOk())
      return 1;
    out.setPosition(0);
    out.truncate();
    out.write(wave.data(), wave.size() * sizeof(float));
    std::printf("probe [%d,%d) -> %zu samples\n", start, end, wave.size());
    return 0;
  }
  // WarpRender <in.pitchnet> <ignored> --labels <start> <end>: the breath /
  // sibilance label and voiced flag of each frame, as a load computes them.
  if (argc >= 6 && juce::String::fromUTF8(argv[3]) == "--labels")
  {
    Project project;
    if (!ProjectSerializer::loadFromFile(project, input))
      return 1;
    // A project that links its recording has no audio to label until the
    // recording is attached, as the App does on open.
    sourceaudio::attachRecording(project, input);
    breathsibilance::ensure(project.getAudioData());
    const auto &a = project.getAudioData();
    const int start = juce::String::fromUTF8(argv[4]).getIntValue();
    const int end = juce::String::fromUTF8(argv[5]).getIntValue();
    for (int f = start; f < end; ++f)
      std::printf("%d\tvoiced %d\tlabel %d\traw %.1f\tdense %.1f\n", f,
                  f < (int)a.voicedMask.size() && a.voicedMask[(size_t)f] ? 1 : 0,
                  f < (int)a.breathSibilance.size() ? (int)a.breathSibilance[(size_t)f] : -1,
                  f < (int)a.rawF0.size() ? a.rawF0[(size_t)f] : -1.0f,
                  f < (int)a.denseF0.size() ? a.denseF0[(size_t)f] : -1.0f);
    return 0;
  }
  // WarpRender <in.pitchnet> <out.pitchnet> --rebuild: the pitch curves
  // rebuilt from the notes as the App does after an edit, saved without
  // synthesis. The saved pitchData f0 is then the white line the App would
  // draw for these note values.
  if (argc >= 4 && juce::String::fromUTF8(argv[3]) == "--rebuild")
  {
    Project project;
    if (!ProjectSerializer::loadFromFile(project, input))
      return 1;
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    if (!ProjectSerializer::saveToFile(project, output))
      return 1;
    std::printf("rebuilt\n");
    return 0;
  }
  // WarpRender <before.pitchnet> <after.pitchnet> --diff: frames whose final
  // pitch or note gain/formant differ between two projects, independent of
  // how the notes were cut.
  if (argc >= 4 && juce::String::fromUTF8(argv[3]) == "--diff")
  {
    Project a, b;
    if (!ProjectSerializer::loadFromFile(a, input) ||
        !ProjectSerializer::loadFromFile(b, output))
    {
      std::fprintf(stderr, "could not load\n");
      return 1;
    }
    PitchCurveProcessor::rebuildBaseFromNotes(a);
    PitchCurveProcessor::rebuildBaseFromNotes(b);
    float maxCents = 0.0f;
    const auto differs = differingFrames(a, b, &maxCents);
    int total = 0;
    const auto fa = a.getAdjustedF0(), fb = b.getAdjustedF0();
    for (const auto &[from, to] : differs)
    {
      float worst = 0.0f;
      for (int f = from; f < to && f < (int)std::min(fa.size(), fb.size()); ++f)
        if (fa[(size_t)f] > 0.0f && fb[(size_t)f] > 0.0f)
          worst = std::max(worst, std::abs(1200.0f * std::log2(fb[(size_t)f] / fa[(size_t)f])));
      std::printf("differs frames [%d,%d) max %.3f cents\n", from, to, worst);
      total += to - from;
    }
    std::printf("notes %d -> %d, total %d frames, max %.2f cents\n",
                (int)a.getNotes().size(), (int)b.getNotes().size(), total, maxCents);
    return 0;
  }
  // WarpRender <in.pitchnet> <out.pitchnet> --apply --before <original.pitchnet>:
  // an agent edited the note data directly. Every note on frames whose final
  // pitch, gain or formant differ from the original is synthesized now (the
  // same incremental path as an edit in the App), and the project is saved
  // with the new composite so Cubase plays the edit.
  // WarpRender <in.pitchnet> <out.pitchnet> --reopen [recording]: open as
  // the App does (spec Q11). A project that links its recording reads it
  // (the saved path, or the file given): refused if its shape differs,
  // analysed again where its content changed, and the edited notes there
  // rendered. An embedded project finds its recording and is saved linked.
  // WarpRender <in.pitchnet> <out.png> --shot from:to [i,j,...]: the piano
  // roll over [from, to] seconds as the App draws it, notes i, j... set to
  // render with Rubber Band R3 (spec Q12, their outline).
  if (argc >= 5 && juce::String::fromUTF8(argv[3]) == "--shot")
  {
    Project project;
    if (!ProjectSerializer::loadFromFile(project, input))
      return 1;
    {
      // --recording <wav>: the project's recording, when it is no longer
      // where the project says (the view cannot scroll past a recording
      // that was not found).
      juce::File recording;
      for (int k = 5; k + 1 < argc; ++k)
        if (juce::String::fromUTF8(argv[k]) == "--recording")
          recording = juce::File::getCurrentWorkingDirectory().getChildFile(juce::String::fromUTF8(argv[k + 1]));
      if (!sourceaudio::attachRecording(project, input, recording))
        std::printf("recording not found: the view stays at the start\n");
    }
    if (argc >= 6 && !juce::String::fromUTF8(argv[5]).startsWith("--"))
    {
      juce::StringArray indices;
      indices.addTokens(juce::String::fromUTF8(argv[5]), ",", "");
      for (const auto &index : indices)
        project.getNotes()[static_cast<size_t>(index.getIntValue())].setSynthEngine(
            Note::SynthEngine::RubberBand);
    }
    // ... --hover <note>: with the pointer on that note (index in the
    // project), so its handles are drawn - the green squares of what it
    // keeps as recorded among them. --keep <note>:<head>:<tail>: that note
    // marked by hand first (frames from its start, up to its end, 0 for
    // none). --told <note>: told from the menu to keep its fry.
    int hoverIndex = -1;
    for (int k = 5; k + 1 < argc; ++k)
    {
      const juce::String arg = juce::String::fromUTF8(argv[k]);
      const juce::String value = juce::String::fromUTF8(argv[k + 1]);
      if (arg == "--hover")
        hoverIndex = value.getIntValue();
      else if (arg == "--keys")
      {
        // --keys <root>,<frame>:<root>,...: the key as the wizard confirms
        // it (the major scale on the root, then the changes of key at those
        // frames of the recording).
        const auto items = juce::StringArray::fromTokens(value, ",", "");
        std::vector<KeyChange> changes;
        for (int item = 1; item < items.size(); ++item)
          changes.push_back({items[item].upToFirstOccurrenceOf(":", false, false).getIntValue(),
                             items[item].fromFirstOccurrenceOf(":", false, false).getIntValue()});
        KeyState::confirmed(project, items[0].getIntValue(), std::move(changes)).applyTo(project);
      }
      else if (arg == "--lock")
      {
        // --lock <noteStart>,...: those notes locked.
        for (const auto &start : juce::StringArray::fromTokens(value, ",", ""))
          for (auto &note : project.getNotes())
            if (!note.isRest() && note.getStartFrame() == start.getIntValue())
              note.setLocked(true);
      }
      else if (arg == "--scale")
      {
        // --scale on: the rows shaded by the scale, as with the toolbar's
        // switch (off in a new project) - what a change of key is seen by.
        if (value == "on")
          project.setScaleMode(project.getPreferredScaleMode());
      }
      else if (arg == "--told")
        project.getNotes()[static_cast<size_t>(value.getIntValue())].setSynthEngine(
            Note::SynthEngine::VocoderRecordedHighsFry);
      else if (arg == "--keep")
      {
        juce::StringArray parts;
        parts.addTokens(value, ":", "");
        if (parts.size() >= 3)
        {
          auto &note = project.getNotes()[static_cast<size_t>(parts[0].getIntValue())];
          const int head = parts[1].getIntValue(), tail = parts[2].getIntValue();
          note.setKeptRange(head > 0 ? note.getSrcStartFrame() + head : Note::kKeepsNone,
                            tail > 0 ? note.getSrcEndFrame() - tail : Note::kKeepsNone);
        }
      }
    }
    const juce::String range = juce::String::fromUTF8(argv[4]);
    const double from = range.upToFirstOccurrenceOf(":", false, false).getDoubleValue();
    const double to = range.fromFirstOccurrenceOf(":", false, false).getDoubleValue();
    PianoRollComponent roll;
    roll.setSize(1400, 700);
    roll.setProject(&project);
    roll.refreshScaleFromProject();
    {
      // What the ruler shows of the key (spec section 15).
      const auto flags = roll.keyFlags();
      int locked = 0;
      for (const auto &note : project.getNotes())
        locked += note.isLocked() ? 1 : 0;
      std::printf("key changes %d, flags %d, locked notes %d\n",
                  static_cast<int>(project.getKeyChanges().size()), static_cast<int>(flags.size()), locked);
    }
    const float pps = static_cast<float>(1200.0 / std::max(0.05, to - from));
    roll.setPixelsPerSecond(pps);
    roll.setScrollX(from * pps);
    float low = 127.0f, high = 0.0f;
    for (const auto &note : project.getNotes())
      if (!note.isRest() && note.getEndFrame() * HOP_SIZE / double(SAMPLE_RATE) > from &&
          note.getStartFrame() * HOP_SIZE / double(SAMPLE_RATE) < to)
      {
        low = std::min(low, note.getMidiNote());
        high = std::max(high, note.getMidiNote());
      }
    roll.centerOnPitchRange(low - 2.0f, high + 2.0f);
    if (hoverIndex >= 0 && hoverIndex < static_cast<int>(project.getNotes().size()))
    {
      const auto &note = project.getNotes()[static_cast<size_t>(hoverIndex)];
      auto &context = roll.getInteractionContext();
      roll.setEditMode(EditMode::Select);
      const juce::Point<float> at(
          (note.getStartFrame() + note.getEndFrame()) * 0.5f * HOP_SIZE / SAMPLE_RATE *
                  roll.getPixelsPerSecond() -
              static_cast<float>(roll.getScrollX()) + PianoRollComponent::pianoKeysWidth,
          context.midiToY(note.getAdjustedMidiNote()) + roll.getPixelsPerSemitone() * 0.5f -
              static_cast<float>(roll.getScrollY()) + PianoRollComponent::headerHeight);
      roll.mouseMove(mouseAt(roll, at, at, {}));
      const auto found = fry::find(project.getAudioData());
      const auto shown = fry::shownRange(note, project.getNotes(),
                                         project.getAudioData().rawF0, found);
      std::printf("hover note %d: src %d-%d, engine %d, by hand %d, squares at src %d and %d\n",
                  hoverIndex, note.getSrcStartFrame(), note.getSrcEndFrame(),
                  static_cast<int>(note.getSynthEngine()), note.hasKeptRange() ? 1 : 0,
                  shown.headUntil, shown.tailFrom);
      for (const auto &handle : context.pitchToolHandles->getHandles())
        if (PitchToolController::isKeepHandle(handle.type))
          std::printf("  %s square: x %.1f-%.1f, y %.1f-%.1f (world)\n",
                      handle.type == PitchToolHandles::HandleType::KeepHead ? "left" : "right",
                      handle.bounds.getX(), handle.bounds.getRight(), handle.bounds.getY(),
                      handle.bounds.getBottom());
    }
    {
      // How long the whole view takes to draw (20 times, the median): to
      // compare a song with changes of key against one without.
      std::vector<double> times;
      for (int k = 0; k < 20; ++k)
      {
        const double began = juce::Time::getMillisecondCounterHiRes();
        const auto image = roll.createComponentSnapshot(roll.getLocalBounds(), true, 2.0f);
        times.push_back(juce::Time::getMillisecondCounterHiRes() - began);
        juce::ignoreUnused(image);
      }
      std::sort(times.begin(), times.end());
      std::printf("draw %.2f ms (median of 20, 1400 x 700 at 2x)\n", times[times.size() / 2]);
    }
    std::printf("shot %s\n", writePng(roll, output) ? "ok" : "FAILED");
    return 0;
  }
  // WarpRender - <out.png> --panel-shot [device,device,...]: the right-hand
  // panel at its preferred height, with these inference devices offered (the
  // Rendering card shows only when there are two or more).
  if (argc >= 4 && juce::String::fromUTF8(argv[3]) == "--panel-shot")
  {
    ParameterPanel panel;
    panel.setSynthesisEngine(SynthesisEngineType::Vocoder);
    juce::StringArray devices;
    if (argc >= 5)
      devices.addTokens(juce::String::fromUTF8(argv[4]), ",", "");
    panel.setRenderDeviceOptions(devices, 0);
    panel.setSize(280, panel.getPreferredHeight());
    std::printf("preferred height %d\n", panel.getPreferredHeight());
    std::printf("shot %s\n", writePng(panel, output) ? "ok" : "FAILED");
    return 0;
  }
  // WarpRender <in.pitchnet> <out.png> --synth-menu-shot <note>[,<note>...]:
  // the synthesis menu as it opens on each of these notes (index in the
  // project), one under the other: what is marked, with a ring or a dot,
  // and whether "Back to Automatic" can be chosen.
  if (argc >= 5 && juce::String::fromUTF8(argv[3]) == "--synth-menu-shot")
  {
    Project project;
    if (!ProjectSerializer::loadFromFile(project, input))
      return 1;
    sourceaudio::attachRecording(project, input);
    PianoRollComponent roll;
    roll.setSize(1400, 700);
    roll.setProject(&project);
    juce::StringArray indices;
    indices.addTokens(juce::String::fromUTF8(argv[4]), ",", "");
    std::vector<juce::PopupMenu> menus;
    int width = 0, height = 8;
    constexpr int kGap = 18, kSeparator = 9;
    for (const auto &index : indices)
    {
      auto &note = project.getNotes()[static_cast<size_t>(index.getIntValue())];
      std::printf("note %d: frames %d-%d, engine %d, shown as %d\n", index.getIntValue(),
                  note.getStartFrame(), note.getEndFrame(),
                  static_cast<int>(note.getSynthEngine()),
                  static_cast<int>(fry::shownAs(note, fry::find(project.getAudioData()))));
      menus.push_back(roll.synthMenuFor(note));
      for (juce::PopupMenu::MenuItemIterator it(menus.back()); it.next();)
      {
        const auto &item = it.getItem();
        int w = 0, h = kSeparator;
        if (!item.isSeparator && item.customComponent != nullptr)
          item.customComponent->getIdealSize(w, h);
        width = std::max(width, w);
        height += h;
      }
      height += kGap;
    }
    juce::Image image(juce::Image::ARGB, 2 * (width + 16), 2 * height, true);
    {
      juce::Graphics g(image);
      g.addTransform(juce::AffineTransform::scale(2.0f));
      g.fillAll(juce::Colour(0xff2a2a2a));
      int y = 8;
      for (auto &menu : menus)
      {
        for (juce::PopupMenu::MenuItemIterator it(menu); it.next();)
        {
          const auto &item = it.getItem();
          if (item.isSeparator || item.customComponent == nullptr)
          {
            g.setColour(juce::Colours::white.withAlpha(0.15f));
            g.fillRect(8 + 8, y + kSeparator / 2, width - 16, 1);
            y += kSeparator;
            continue;
          }
          int w = 0, h = 0;
          item.customComponent->getIdealSize(w, h);
          item.customComponent->setBounds(0, 0, width, h);
          juce::Graphics::ScopedSaveState state(g);
          g.setOrigin(8, y);
          // The component reads "enabled" from the menu window's item, which
          // there is none of here: dim it by hand.
          if (!item.isEnabled)
            g.beginTransparencyLayer(0.35f);
          item.customComponent->paintEntireComponent(g, true);
          if (!item.isEnabled)
            g.endTransparencyLayer();
          y += h;
        }
        y += kGap;
      }
    }
    output.deleteFile();
    juce::FileOutputStream stream(output);
    juce::PNGImageFormat png;
    std::printf("shot %s\n",
                stream.openedOk() && png.writeImageToStream(image, stream) ? "ok" : "FAILED");
    return 0;
  }
  // WarpRender - <out.png> --toolbar-shot <width> <project file> [recording]:
  // the toolbar at this width, naming the project and, on the line under
  // it, its recording (the standalone; neither file is read).
  if (argc >= 6 && juce::String::fromUTF8(argv[3]) == "--toolbar-shot")
  {
    ToolbarComponent toolbar;
    toolbar.setSize(juce::String::fromUTF8(argv[4]).getIntValue(), 60);
    toolbar.setDocumentName(juce::File(juce::String::fromUTF8(argv[5])));
    if (argc >= 7)
      toolbar.setSourceFile(juce::File(juce::String::fromUTF8(argv[6])));
    std::printf("shot %s\n", writePng(toolbar, output) ? "ok" : "FAILED");
    return 0;
  }
  if (argc >= 4 && juce::String::fromUTF8(argv[3]) == "--reopen")
  {
    Project loaded;
    if (!ProjectSerializer::loadFromFile(loaded, input))
    {
      std::fprintf(stderr, "could not load %s\n", input.getFullPathName().toRawUTF8());
      return 1;
    }
    juce::String ortError;
    if (!OnnxRuntime::initialise(&ortError))
    {
      std::fprintf(stderr, "ONNX Runtime: %s\n", ortError.toRawUTF8());
      return 1;
    }
    EditorController controller(false);
    controller.reloadInferenceModels(false);
    {
      const auto start = juce::Time::getMillisecondCounter();
      while (!controller.isSelectedPitchDetectorLoaded() || controller.isInferenceBusy())
      {
        juce::MessageManager::getInstance()->runDispatchLoopUntil(50);
        if (juce::Time::getMillisecondCounter() - start > 120000)
        {
          std::fprintf(stderr, "pitch model did not load\n");
          return 1;
        }
      }
    }
    controller.setProject(std::make_unique<Project>(loaded));
    auto &p = *controller.getProject();
    const auto started = juce::Time::getMillisecondCounterHiRes();
    if (p.linksRecording())
    {
      const juce::File file =
          argc >= 5 ? juce::File::getCurrentWorkingDirectory().getChildFile(
                          juce::String::fromUTF8(argv[4]))
                    : p.getFilePath();
      const auto opened = recordingchange::openLinked(
          p, file,
          [&](const float *samples, int numSamples, int frames, std::vector<float> &f0,
              std::vector<bool> &voiced) {
            return controller.detectPitch(samples, numSamples, frames, f0, voiced);
          });
      if (opened.result != recordingchange::Opened::Result::opened)
      {
        std::printf("refused: %s (file %d Hz %d ch %lld samples, saved %d Hz %d ch %lld)\n",
                    opened.result == recordingchange::Opened::Result::otherShape
                        ? "another shape" : "unreadable",
                    opened.found.sampleRate, opened.found.channels,
                    static_cast<long long>(opened.found.samples),
                    p.getRecordingFingerprint().sampleRate,
                    p.getRecordingFingerprint().channels,
                    static_cast<long long>(p.getRecordingFingerprint().samples));
        controller.requestShutdown();
        return 3;
      }
      std::printf("opened %s, changed %d, detected %d, %.0f ms\n",
                  file.getFullPathName().toRawUTF8(), opened.changed ? 1 : 0,
                  opened.update.detected ? 1 : 0,
                  juce::Time::getMillisecondCounterHiRes() - started);
      for (const auto &[from, to] : opened.update.audioFrames)
        std::printf("audio changed frames [%d,%d)\n", from, to);
      for (const auto &[from, to] : opened.update.pitchFrames)
        std::printf("pitch changed frames [%d,%d)\n", from, to);
      std::printf("notes to render %d\n", opened.update.notesToRender);
      for (const auto &note : p.getNotes())
        if (!note.isRest() && note.isDirty())
          std::printf("render note %d-%d midi %.2f\n", note.getStartFrame(), note.getEndFrame(),
                      note.getMidiNote());
      if (opened.update.notesToRender > 0)
      {
        const auto renderStart = juce::Time::getMillisecondCounterHiRes();
        std::atomic<bool> pending{false}, done{false}, ok{false};
        controller.resynthesizeIncrementalAsync(
            p, nullptr, [&](bool success) { ok = success; done = true; }, pending, false);
        if (!waitFor(done, 600000) || !ok.load())
        {
          std::fprintf(stderr, "resynth failed\n");
          controller.requestShutdown();
          return 1;
        }
        std::printf("rendered in %.0f ms\n",
                    juce::Time::getMillisecondCounterHiRes() - renderStart);
      }
      // PITCHNET_RERENDER_R3=1: render the edited R3 notes again with the
      // current R3 settings, as switching them to the vocoder and back in the
      // App's synthesis menu would (SetSynthEngineAction marks them the same
      // way). =vocoder: switch them to the vocoder instead, as choosing it in
      // that menu would. One pass per run of touching R3 notes, as selecting
      // each run would.
      const char *rerender = std::getenv("PITCHNET_RERENDER_R3");
      const bool toVocoder = rerender != nullptr && juce::String(rerender) == "vocoder";
      if (rerender != nullptr && (juce::String(rerender) == "1" || toVocoder))
      {
        std::vector<std::vector<size_t>> runs;
        auto &notes = p.getNotes();
        for (size_t i = 0; i < notes.size(); ++i)
        {
          const auto &note = notes[i];
          if (note.isRest() || note.getSynthEngine() != Note::SynthEngine::RubberBand ||
              note.isNeutralForOriginalWaveform())
            continue;
          if (!runs.empty() && notes[runs.back().back()].getEndFrame() == note.getStartFrame())
            runs.back().push_back(i);
          else
            runs.push_back({i});
        }
        const double hop = static_cast<double>(HOP_SIZE) / SAMPLE_RATE;
        for (const auto &run : runs)
        {
          for (const size_t i : run)
          {
            if (toVocoder)
              notes[i].setSynthEngine(Note::SynthEngine::Vocoder);
            notes[i].markDirty();
            notes[i].markSynthDirty();
            std::printf("R3 note %zu %.3f-%.3f s\n", i, notes[i].getStartFrame() * hop,
                        notes[i].getEndFrame() * hop);
          }
          std::atomic<bool> pending{false}, done{false}, ok{false};
          controller.resynthesizeIncrementalAsync(
              p, nullptr, [&](bool success) { ok = success; done = true; }, pending, false);
          if (!waitFor(done, 600000) || !ok.load())
          {
            std::fprintf(stderr, "resynth failed\n");
            controller.requestShutdown();
            return 1;
          }
          std::printf("rendered run of %zu notes\n", run.size());
        }
        p.setModified(true);
      }
    }
    else
    {
      if (!sourceaudio::attachRecording(p, input))
      {
        std::fprintf(stderr, "recording not found\n");
        controller.requestShutdown();
        return 1;
      }
      std::printf("embedded project, recording %s attached, fingerprint %d blocks\n",
                  p.getFilePath().getFullPathName().toRawUTF8(),
                  static_cast<int>(p.getRecordingFingerprint().hashes.size()));
    }
    const bool saved = ProjectSerializer::saveToFile(p, output);
    controller.requestShutdown();
    std::printf("saved %s (%lld bytes)\n", output.getFullPathName().toRawUTF8(),
                static_cast<long long>(output.getSize()));
    return saved ? 0 : 1;
  }
  if (argc >= 4 && juce::String::fromUTF8(argv[3]) == "--apply")
  {
    if (argc < 6 || juce::String::fromUTF8(argv[4]) != "--before")
    {
      std::fprintf(stderr, "usage: WarpRender in.pitchnet out.pitchnet --apply "
                           "--before original.pitchnet\n");
      return 2;
    }
    Project loaded;
    if (!ProjectSerializer::loadFromFile(loaded, input))
    {
      std::fprintf(stderr, "could not load %s\n", input.getFullPathName().toRawUTF8());
      return 1;
    }
    sourceaudio::attachRecording(loaded, input);
    juce::String ortError;
    if (!OnnxRuntime::initialise(&ortError))
    {
      std::fprintf(stderr, "ONNX Runtime: %s\n", ortError.toRawUTF8());
      return 1;
    }
    EditorController controller(false);
    controller.setProject(std::make_unique<Project>(loaded));
    auto &p = *controller.getProject();
    auto &data = p.getAudioData();
    if (data.melSpectrogram.empty())
    {
      MelSpectrogram mel(data.sampleRate, N_FFT, HOP_SIZE, NUM_MELS, FMIN, FMAX);
      data.melSpectrogram = mel.compute(data.originalWaveform.getReadPointer(0),
                                        data.originalWaveform.getNumSamples());
    }
    // The stored pitch curves do not follow note values edited outside the
    // App; rebuild them from the notes as an edit in the App does.
    PitchCurveProcessor::rebuildBaseFromNotes(p);
    // Compare with the original frame by frame, so how the notes were cut
    // does not matter and notes rendered before and edited again are caught.
    std::vector<std::pair<int, int>> changed;
    {
      Project before;
      const juce::File beforeFile = juce::File::getCurrentWorkingDirectory().getChildFile(
          juce::String::fromUTF8(argv[5]));
      if (!ProjectSerializer::loadFromFile(before, beforeFile))
      {
        std::fprintf(stderr, "could not load %s\n",
                     beforeFile.getFullPathName().toRawUTF8());
        return 1;
      }
      // Linked, it has no curves to compare until its recording is attached.
      if (before.getAudioData().originalWaveform.getNumSamples() == 0)
        sourceaudio::attachRecording(before, beforeFile);
      PitchCurveProcessor::rebuildBaseFromNotes(before);
      changed = differingFrames(before, p, nullptr);
      for (const auto &[from, to] : changed)
        std::printf("changed frames [%d,%d)\n", from, to);
    }
    auto overlapsChange = [&](const Note &note) {
      for (const auto &[from, to] : changed)
        if (note.getStartFrame() < to && note.getEndFrame() > from)
          return true;
      return false;
    };
    int marked = 0;
    for (auto &note : p.getNotes())
      // A rendered note set back to neutral is rendered again too, so the
      // composite returns to the recording there.
      if (!note.isRest() && overlapsChange(note) &&
          (!note.isNeutralForOriginalWaveform() || note.hasRenderedEdit()))
      {
        note.markDirty();
        note.markSynthDirty();
        ++marked;
        std::printf("note @%d offset %.2f\n", note.getStartFrame(),
                    note.getPitchOffset());
      }
    if (marked > 0)
    {
      std::atomic<bool> pending{false}, done{false}, ok{false};
      controller.resynthesizeIncrementalAsync(
          p, nullptr, [&](bool success) { ok = success; done = true; }, pending,
          false);
      if (!waitFor(done, 600000) || !ok.load())
      {
        std::fprintf(stderr, "resynth failed\n");
        controller.requestShutdown();
        return 1;
      }
    }
    const bool saved = ProjectSerializer::saveToFile(p, output);
    controller.requestShutdown();
    if (!saved)
    {
      std::fprintf(stderr, "could not save %s\n", output.getFullPathName().toRawUTF8());
      return 1;
    }
    std::printf("synthesized %d notes, wrote %s\n", marked,
                output.getFullPathName().toRawUTF8());
    return 0;
  }
  // WarpRender <project.pitchnet> <ignored> --debug-report: the project's
  // debug recording (DebugJournal.h) read back: every step, every render pass
  // and its splices, and at each mark the passes that wrote there and how the
  // composite there compares with the recording.
  if (argc >= 4 && juce::String::fromUTF8(argv[3]) == "--debug-report")
  {
    Project project;
    if (!ProjectSerializer::loadFromFile(project, input))
      return 1;
    auto entries = project.getDebugJournal().copyEntries();
    // --debug-report <time> [dump folder]: also analyse that time (seconds) as
    // if marked last, and write the audio each pass wrote around every mark
    // (before / after, float32 raw, 44.1 kHz) into the folder.
    if (argc >= 5)
    {
      debugjournal::Entry extra;
      extra.kind = "mark";
      extra.name = "given on the command line";
      extra.playhead = juce::String::fromUTF8(argv[4]).getDoubleValue();
      extra.time = std::numeric_limits<juce::int64>::max();
      entries.push_back(extra);
    }
    const juce::File dumpDir = argc >= 6 ? juce::File::getCurrentWorkingDirectory().getChildFile(
                                               juce::String::fromUTF8(argv[5]))
                                         : juce::File();
    const auto &audio = project.getAudioData();
    const int rate = audio.sampleRate > 0 ? audio.sampleRate : SAMPLE_RATE;
    auto clockOf = [](double s) {
      const int m = static_cast<int>(s / 60.0);
      return juce::String::formatted("%02d:%06.3f", m, s - m * 60.0);
    };
    auto noteText = [](const debugjournal::NoteState &n) {
      return juce::String::formatted("%d-%d midi %.2f%s%s%s%s", n.start, n.end, n.midi,
                                     std::abs(n.tiltLeft) + std::abs(n.tiltRight) > 1e-4f
                                         ? juce::String::formatted(" tilt %.2f/%.2f@%.2f", n.tiltLeft, n.tiltRight, n.tiltPivot).toRawUTF8() : "",
                                     std::abs(n.drift - 1.0f) > 1e-4f ? juce::String::formatted(" drift %.2f", n.drift).toRawUTF8() : "",
                                     std::abs(n.volume) > 1e-4f ? juce::String::formatted(" vol %.1fdB", n.volume).toRawUTF8() : "",
                                     std::abs(n.formant) > 1e-4f ? juce::String::formatted(" formant %.2f", n.formant).toRawUTF8() : "");
    };
    auto whenOf = [](juce::int64 ms) {
      const juce::Time when(ms);
      return when.formatted("%Y-%m-%d %H:%M:%S") + juce::String::formatted(".%03d", when.getMilliseconds());
    };
    std::printf("debug recording: %zu entries, recording %s\n", entries.size(),
                project.getDebugJournal().isEnabled() ? "on" : "off");
    int index = 0;
    for (const auto &e : entries)
    {
      ++index;
      if (e.kind == "start")
        std::printf("\n");
      std::printf("#%d %s %-6s %s  (playhead %s)\n", index, whenOf(e.time).toRawUTF8(), e.kind.toRawUTF8(),
                  e.name.toRawUTF8(), clockOf(e.playhead).toRawUTF8());
      if (e.kind == "start")
      {
        std::printf("     %zu notes\n", e.after.size());
        continue;
      }
      for (const auto &n : e.before)
        std::printf("     - %s\n", noteText(n).toRawUTF8());
      for (const auto &n : e.after)
        std::printf("     + %s\n", noteText(n).toRawUTF8());
      if (e.detail.isNotEmpty())
        std::printf("     %s\n", e.detail.toRawUTF8());
      for (const auto &s : e.splices)
        std::printf("     splice %s %-10s lag %+.1f angle %+.2f\n",
                    clockOf(static_cast<double>(s.at) / rate).toRawUTF8(), s.kind.toRawUTF8(), s.lag, s.angle);
      if (!e.audioAfter.empty())
        std::printf("     wrote %s - %s\n", clockOf(static_cast<double>(e.audioStart) / rate).toRawUTF8(),
                    clockOf(static_cast<double>(e.audioStart + e.audioAfter.size()) / rate).toRawUTF8());
    }
    // At each mark: the passes that wrote there (latest last), their splices
    // near it, and the composite against the recording around it.
    auto levelDb = [](const float *x, int n) {
      double e = 0.0;
      for (int i = 0; i < n; ++i)
        e += static_cast<double>(x[i]) * x[i];
      return 10.0 * std::log10(e / std::max(1, n) + 1e-24);
    };
    const auto &reference = warp::reference(audio);
    for (const auto &mark : entries)
    {
      if (mark.kind != "mark")
        continue;
      const int at = static_cast<int>(mark.playhead * rate);
      std::printf("\n== mark %s (set %s)\n", clockOf(mark.playhead).toRawUTF8(),
                  whenOf(mark.time).toRawUTF8());
      int pass = 0;
      for (const auto &e : entries)
      {
        ++pass;
        if (e.kind != "render" || e.time > mark.time)
          continue;
        const int from = e.audioStart, to = e.audioStart + static_cast<int>(e.audioAfter.size());
        const int near = static_cast<int>(0.3 * rate);
        if (to < at - near || from > at + near)
          continue;
        std::printf("   pass #%d wrote %s - %s: %s\n", pass,
                    clockOf(static_cast<double>(from) / rate).toRawUTF8(),
                    clockOf(static_cast<double>(to) / rate).toRawUTF8(), e.detail.toRawUTF8());
        if (dumpDir != juce::File())
        {
          dumpDir.createDirectory();
          for (const auto &[suffix, samples] : {std::pair<const char *, const std::vector<float> *>{"before", &e.audioBefore},
                                                 {"after", &e.audioAfter}})
          {
            const auto file = dumpDir.getChildFile(juce::String::formatted("pass%04d_%d_%s.f32", pass, e.audioStart, suffix));
            file.deleteFile();
            juce::FileOutputStream out(file);
            out.write(samples->data(), samples->size() * sizeof(float));
          }
        }
        for (const auto &s : e.splices)
          if (std::abs(s.at - at) < near)
            std::printf("      splice %s (%+.0f ms from the mark) %s lag %+.1f angle %+.2f\n",
                        clockOf(static_cast<double>(s.at) / rate).toRawUTF8(),
                        1000.0 * (s.at - at) / rate, s.kind.toRawUTF8(), s.lag, s.angle);
      }
      // Composite vs recording, 256-sample windows over +-150 ms: level
      // difference and correlation (a cancelling crossfade: a dip with the
      // correlation going through zero or below).
      if (audio.waveform.getNumSamples() > 0 && reference.getNumSamples() > 0)
      {
        // The composite as it was when the mark was set: each sample from the
        // last pass before the mark that wrote it, else the saved composite.
        std::vector<float> heard(audio.waveform.getReadPointer(0),
                                 audio.waveform.getReadPointer(0) + audio.waveform.getNumSamples());
        for (const auto &e : entries)
          if (e.kind == "render" && e.time <= mark.time && !e.audioAfter.empty())
            for (size_t i = 0; i < e.audioAfter.size(); ++i)
              if (const auto k = static_cast<size_t>(e.audioStart) + i; k < heard.size())
                heard[k] = e.audioAfter[i];
        const float *c = heard.data();
        const float *o = reference.getReadPointer(0);
        const int total = std::min(static_cast<int>(heard.size()), reference.getNumSamples());
        std::printf("   composite when marked vs recording (ms: level diff dB / correlation):\n     ");
        for (int s = at - static_cast<int>(0.15 * rate); s < at + static_cast<int>(0.15 * rate); s += 512)
        {
          if (s < 0 || s + 256 > total)
            continue;
          double ab = 0, aa = 0, bb = 0;
          for (int i = s; i < s + 256; ++i)
          {
            ab += static_cast<double>(c[i]) * o[i];
            aa += static_cast<double>(c[i]) * c[i];
            bb += static_cast<double>(o[i]) * o[i];
          }
          std::printf("%+.0f:%+.1f/%+.2f  ", 1000.0 * (s - at) / rate, levelDb(c + s, 256) - levelDb(o + s, 256),
                      ab / std::sqrt(aa * bb + 1e-24));
        }
        std::printf("\n");
      }
    }
    return 0;
  }
  // WarpRender <in.pitchnet> <out.pitchnet> --drags <noteStart:midi>...: the
  // App's single-note pitch drags (SelectHandler's drag end), one after
  // another, each rendered as the App renders after it. Prints the notes each
  // render was asked to redo; saves the project with its composite.
  if (argc >= 5 && juce::String::fromUTF8(argv[3]) == "--drags")
  {
    Project loaded;
    if (!ProjectSerializer::loadFromFile(loaded, input))
      return 1;
    sourceaudio::attachRecording(loaded, input);
    juce::String ortError;
    if (!OnnxRuntime::initialise(&ortError))
      return 1;
    EditorController controller(false);
    controller.setProject(std::make_unique<Project>(loaded));
    auto &p = *controller.getProject();
    auto &data = p.getAudioData();
    if (data.melSpectrogram.empty())
    {
      MelSpectrogram mel(data.sampleRate, N_FFT, HOP_SIZE, NUM_MELS, FMIN, FMAX);
      data.melSpectrogram = mel.compute(data.originalWaveform.getReadPointer(0),
                                        data.originalWaveform.getNumSamples());
    }
    // PITCHNET_DEBUG_RECORD: record the drags in the project's debug journal,
    // as the App does with recording on (history steps, render passes) and a
    // mark at each dragged note's start, as if heard there.
    const bool record = std::getenv("PITCHNET_DEBUG_RECORD") != nullptr;
    if (record)
      p.getDebugJournal().start(p.getNotes(), 0.0);
    for (int k = 4; k < argc; ++k)
    {
      const juce::String arg = juce::String::fromUTF8(argv[k]);
      const int start = arg.upToFirstOccurrenceOf(":", false, false).getIntValue();
      const float midi = arg.fromFirstOccurrenceOf(":", false, false).getFloatValue();
      Note *note = nullptr;
      for (auto &n : p.getNotes())
        if (!n.isRest() && n.getStartFrame() == start)
          note = &n;
      if (note == nullptr)
        return 1;
      note->setMidiNote(midi);
      note->setPitchOffset(0.0f);
      note->markDirty();
      note->markSynthDirty();
      PitchCurveProcessor::rebuildBaseFromNotes(p);
      p.markNoteEditDirtyRange(note->getStartFrame(), note->getEndFrame());
      if (record)
        p.getDebugJournal().recordHistory(p.getNotes(), "apply", "Drag Note Pitch",
                                          framesToSeconds(start));
      std::atomic<bool> pending{false}, done{false}, ok{false};
      controller.resynthesizeIncrementalAsync(
          p, nullptr, [&](bool success) { ok = success; done = true; }, pending, false);
      const bool finished = waitFor(done, 600000);
      if (record)
        p.getDebugJournal().recordMark(framesToSeconds(start) + 0.05);
      std::printf("drag %d -> %.2f: render %s; rendered notes:", start, midi,
                  finished && ok.load() ? "ok" : "failed");
      for (const auto &n : p.getNotes())
        if (!n.isRest() && n.hasRenderedEdit() && n.getEndFrame() > start - 40 &&
            n.getStartFrame() < start + 60)
          std::printf(" %d", n.getStartFrame());
      std::printf("\n");
    }
    const bool saved = ProjectSerializer::saveToFile(p, output);
    controller.requestShutdown();
    return saved ? 0 : 1;
  }
  // WarpRender <in.pitchnet> <out.pitchnet> --keep-drag <noteStart> head|tail
  // <frames> [...] [--undo] [--shot <out.png>]: the App's drag of a note's
  // green square (the stretch kept as recorded by hand), through the piano
  // roll's own mouse handling - the pointer onto the note, the square taken
  // hold of, moved by `frames` frames of the recording (the left one right,
  // the right one left, for a positive count), let go - each rendered as the
  // App renders after it. --undo takes the last one back the same way.
  // Prints the note's choice before and after; saves the project.
  if (argc >= 7 && juce::String::fromUTF8(argv[3]) == "--keep-drag")
  {
    Project loaded;
    if (!ProjectSerializer::loadFromFile(loaded, input))
      return 1;
    sourceaudio::attachRecording(loaded, input);
    juce::String ortError;
    if (!OnnxRuntime::initialise(&ortError))
      return 1;
    EditorController controller(false);
    controller.setProject(std::make_unique<Project>(loaded));
    auto &p = *controller.getProject();
    auto &data = p.getAudioData();
    if (data.melSpectrogram.empty())
    {
      MelSpectrogram mel(data.sampleRate, N_FFT, HOP_SIZE, NUM_MELS, FMIN, FMAX);
      data.melSpectrogram = mel.compute(data.originalWaveform.getReadPointer(0),
                                        data.originalWaveform.getNumSamples());
    }
    PianoRollComponent roll;
    PitchUndoManager history;
    roll.setSize(1400, 700);
    roll.setProject(&p);
    roll.setUndoManager(&history);
    roll.setEditMode(EditMode::Select);
    auto &context = roll.getInteractionContext();
    const juce::ModifierKeys left(juce::ModifierKeys::leftButtonModifier);
    const auto render = [&](const char *what)
    {
      std::atomic<bool> pending{false}, done{false}, ok{false};
      controller.resynthesizeIncrementalAsync(
          p, nullptr, [&](bool success) { ok = success; done = true; }, pending, false);
      const bool finished = waitFor(done, 600000);
      std::printf("%s: render %s\n", what, finished && ok.load() ? "ok" : "failed");
    };
    const auto say = [](const char *when, const Note &note)
    {
      std::printf("  %s: engine %d, by hand %d, head until %d, tail from %d (src %d-%d)\n",
                  when, static_cast<int>(note.getSynthEngine()), note.hasKeptRange() ? 1 : 0,
                  note.getKeptHeadUntil(), note.getKeptTailFrom(), note.getSrcStartFrame(),
                  note.getSrcEndFrame());
    };
    const auto onScreen = [&](juce::Point<float> world)
    {
      return juce::Point<float>(
          world.x + PianoRollComponent::pianoKeysWidth - static_cast<float>(roll.getScrollX()),
          world.y + PianoRollComponent::headerHeight - static_cast<float>(roll.getScrollY()));
    };
    juce::File shotFile;
    int shotNote = -1;
    for (int k = 4; k < argc; ++k)
    {
      const juce::String arg = juce::String::fromUTF8(argv[k]);
      if (arg == "--undo")
      {
        history.undo();
        render("undo");
        continue;
      }
      if (arg == "--shot" && k + 1 < argc)
      {
        shotFile = juce::File::getCurrentWorkingDirectory().getChildFile(
            juce::String::fromUTF8(argv[++k]));
        continue;
      }
      if (k + 2 >= argc)
        break;
      const int start = arg.getIntValue();
      const bool head = juce::String::fromUTF8(argv[k + 1]) == "head";
      const int frames = juce::String::fromUTF8(argv[k + 2]).getIntValue();
      k += 2;
      Note *note = nullptr;
      for (auto &n : p.getNotes())
        if (!n.isRest() && n.getStartFrame() == start)
          note = &n;
      if (note == nullptr)
      {
        std::fprintf(stderr, "no note starts at %d\n", start);
        return 1;
      }
      shotNote = start;
      // The note 600 px wide on screen, 300 px from the left.
      const double seconds =
          std::max(0.02, (note->getEndFrame() - note->getStartFrame()) * double(HOP_SIZE) / SAMPLE_RATE);
      roll.setPixelsPerSecond(static_cast<float>(std::min(2000.0, 600.0 / seconds)));
      const float pps = roll.getPixelsPerSecond();
      roll.setScrollX(std::max(0.0, note->getStartFrame() * double(HOP_SIZE) / SAMPLE_RATE * pps - 300.0));
      roll.centerOnPitchRange(note->getAdjustedMidiNote() - 2.0f, note->getAdjustedMidiNote() + 2.0f);
      const auto map = notetiming::forNoteOnScreen(data, *note);
      const auto xOf = [&](int sourceFrame)
      {
        return framesToSeconds(static_cast<float>(map.destinationAt(sourceFrame))) * pps;
      };
      const juce::Point<float> body(
          (xOf(note->getSrcStartFrame()) + xOf(note->getSrcEndFrame())) * 0.5f,
          context.midiToY(note->getAdjustedMidiNote()) + roll.getPixelsPerSemitone() * 0.5f);
      roll.mouseMove(mouseAt(roll, onScreen(body), onScreen(body), {}));
      const auto shown = fry::shownRange(*note, p.getNotes(), data.rawF0, fry::find(data));
      const auto type = head ? PitchToolHandles::HandleType::KeepHead
                             : PitchToolHandles::HandleType::KeepTail;
      const PitchToolHandles::Handle *square = nullptr;
      for (const auto &handle : context.pitchToolHandles->getHandles())
        if (handle.type == type)
          square = &handle;
      if (square == nullptr)
      {
        std::fprintf(stderr, "no square on the note at %d\n", start);
        return 1;
      }
      std::printf("drag @%d %s by %d frames\n", start, head ? "head" : "tail", frames);
      say("before", *note);
      const int from = head ? shown.headUntil : shown.tailFrom;
      const int to = head ? from + frames : from - frames;
      const auto grab = onScreen(square->bounds.getCentre());
      const auto release = grab.translated(xOf(to) - xOf(from), 0.0f);
      roll.mouseMove(mouseAt(roll, grab, grab, {}));
      roll.mouseDown(mouseAt(roll, grab, grab, left));
      roll.mouseDrag(mouseAt(roll, release, grab, left));
      roll.mouseUp(mouseAt(roll, release, grab, {}));
      say("after", *note);
      std::printf("  history: %s\n", history.canUndo() ? "one step" : "nothing");
      render("drag");
    }
    if (shotFile != juce::File() && shotNote >= 0)
    {
      for (const auto &n : p.getNotes())
        if (!n.isRest() && n.getStartFrame() == shotNote)
        {
          const juce::Point<float> body(
              (n.getStartFrame() + n.getEndFrame()) * 0.5f * HOP_SIZE / SAMPLE_RATE *
                  roll.getPixelsPerSecond(),
              context.midiToY(n.getAdjustedMidiNote()) + roll.getPixelsPerSemitone() * 0.5f);
          roll.mouseMove(mouseAt(roll, onScreen({8.0f, body.y - 300.0f}), onScreen(body), {}));
          roll.mouseMove(mouseAt(roll, onScreen(body), onScreen(body), {}));
        }
      std::printf("shot %s\n", writePng(roll, shotFile) ? "ok" : "FAILED");
    }
    roll.setProject(nullptr);
    const bool saved = ProjectSerializer::saveToFile(p, output);
    controller.requestShutdown();
    return saved ? 0 : 1;
  }
  // WarpRender <in.pitchnet> <out-prefix> --reset-undo pitch|all <noteStart>...:
  // the App's Restore on the selected notes, then undo, each rendered as the
  // App renders after a history change (afterHistoryChange). Writes the
  // composite before, after the restore and after the undo as float32 mono
  // <out-prefix>_0.f32, _1.f32, _2.f32.
  if (argc >= 6 && juce::String::fromUTF8(argv[3]) == "--reset-undo")
  {
    Project loaded;
    if (!ProjectSerializer::loadFromFile(loaded, input))
      return 1;
    sourceaudio::attachRecording(loaded, input);
    juce::String ortError;
    if (!OnnxRuntime::initialise(&ortError))
    {
      std::fprintf(stderr, "ONNX Runtime: %s\n", ortError.toRawUTF8());
      return 1;
    }
    EditorController controller(false);
    controller.setProject(std::make_unique<Project>(loaded));
    auto &p = *controller.getProject();
    auto &data = p.getAudioData();
    if (data.melSpectrogram.empty())
    {
      MelSpectrogram mel(data.sampleRate, N_FFT, HOP_SIZE, NUM_MELS, FMIN, FMAX);
      data.melSpectrogram = mel.compute(data.originalWaveform.getReadPointer(0),
                                        data.originalWaveform.getNumSamples());
    }
    const bool all = juce::String::fromUTF8(argv[4]) == "all";
    PianoRollComponent roll;
    PitchUndoManager undo;
    roll.setSize(1400, 900);
    roll.setProject(&p);
    roll.setUndoManager(&undo);
    roll.setEditMode(EditMode::Select);
    p.deselectAllNotes();
    Note *first = nullptr;
    for (int k = 5; k < argc; ++k)
      for (auto &n : p.getNotes())
        if (!n.isRest() && n.getStartFrame() == juce::String::fromUTF8(argv[k]).getIntValue())
        {
          n.setSelected(true);
          if (first == nullptr)
            first = &n;
        }
    if (first == nullptr)
      return 1;
    // The selected notes and one note's worth around them, for the report.
    int nearFrom = std::numeric_limits<int>::max(), nearTo = 0;
    for (const auto &n : p.getNotes())
      if (!n.isRest() && n.isSelected())
      {
        nearFrom = std::min(nearFrom, n.getStartFrame() - 100);
        nearTo = std::max(nearTo, n.getEndFrame() + 100);
      }
    const auto write = [&](int step) {
      const juce::File out(output.getFullPathName() + "_" + juce::String(step) + ".f32");
      out.deleteFile();
      juce::FileOutputStream stream(out);
      stream.write(data.waveform.getReadPointer(0),
                   static_cast<size_t>(data.waveform.getNumSamples()) * sizeof(float));
    };
    const auto render = [&](const char *what) {
      for (const auto &n : p.getNotes())
        if (!n.isRest() && n.getEndFrame() > nearFrom && n.getStartFrame() < nearTo)
          std::printf("%s: note %d dirty %d neutral %d renderedEdit %d\n", what, n.getStartFrame(),
                      n.isDirty() ? 1 : 0, n.isNeutralForOriginalWaveform() ? 1 : 0,
                      n.hasRenderedEdit() ? 1 : 0);
      std::printf("%s: f0 dirty [%d,%d)\n", what, p.getF0DirtyRange().first, p.getF0DirtyRange().second);
      std::atomic<bool> pending{false}, done{false}, ok{false};
      controller.resynthesizeIncrementalAsync(
          p, nullptr, [&](bool success) { ok = success; done = true; }, pending, false);
      const bool finished = waitFor(done, 600000);
      std::printf("%s: render %s\n", what, finished && ok.load() ? "ok" : "failed");
    };
    write(0);
    roll.onHistoryActionApplied = [&](bool needs) {
      if (needs)
        render("restore");
    };
    roll.resetNoteEdits(*first, all ? PianoRollComponent::NoteRestoreMode::All
                                    : PianoRollComponent::NoteRestoreMode::Pitch);
    write(1);
    roll.cancelPitchShaperPreview();
    if (undo.undo())
      render("undo");
    write(2);
    for (const auto &n : p.getNotes())
      if (!n.isRest() && n.getEndFrame() > nearFrom && n.getStartFrame() < nearTo)
        std::printf("note %d-%d midi %.3f tiltR %.3f\n", n.getStartFrame(), n.getEndFrame(),
                    n.getMidiNote(), n.getTiltRight());
    controller.requestShutdown();
    return 0;
  }
  // WarpRender - <out.png> --mixer-shot none|set|on|missing|loading [width]:
  // the standalone's left panel (MixerPanel) in one of its states, in the
  // language of the settings.
  if (argc >= 5 && juce::String::fromUTF8(argv[3]) == "--mixer-shot")
  {
    Localization::loadFromSettings();
    if (const char *language = std::getenv("PITCHNET_LANG"))
      Localization::getInstance().setLanguage(language);
    const juce::String which = juce::String::fromUTF8(argv[4]);
    MixerPanel panel;
    MixerPanel::State state;
    state.hasProject = true;
    if (which != "none")
    {
      state.hasBacking = true;
      state.backingName = juce::String::fromUTF8("晚餐歌_伴奏.wav");
      state.backingAudible = which == "on";
      state.backingDb = which == "on" ? -9.5f : -6.0f;
      state.voiceDb = which == "on" ? -2.0f : 0.0f;
      state.backingMissing = which == "missing";
      state.backingLoading = which == "loading";
    }
    panel.setState(state);
    const int width = argc >= 6 ? juce::String::fromUTF8(argv[5]).getIntValue() : 141;
    panel.setSize(width, panel.getPreferredHeight());
    std::printf("mixer %s (%s): %d x %d, shot %s\n", which.toRawUTF8(),
                Localization::getInstance().getLanguage().toRawUTF8(), width, panel.getPreferredHeight(),
                writePng(panel, output) ? "ok" : "FAILED");
    return 0;
  }
  // WarpRender - <out.png> --export-shot voice|tracks [rate]: the export
  // dialog's content, without and with the row that asks which track.
  if (argc >= 5 && juce::String::fromUTF8(argv[3]) == "--export-shot")
  {
    Localization::loadFromSettings();
    if (const char *language = std::getenv("PITCHNET_LANG"))
      Localization::getInstance().setLanguage(language);
    const bool tracks = juce::String::fromUTF8(argv[4]) == "tracks";
    const int rate = argc >= 6 ? juce::String::fromUTF8(argv[5]).getIntValue() : 48000;
    auto content = ExportHelper::makeExportSettingsContent(rate, tracks);
    std::printf("export dialog %d x %d, shot %s\n", content->getWidth(), content->getHeight(),
                writePng(*content, output) ? "ok" : "FAILED");
    return 0;
  }
  // WarpRender <notes.txt | in.pitchnet | -> <out.png> --wizard-shot <page>
  // [--backing <wav>] [--size <w> <h>] [--recent <path>]... [--at <seconds>]
  // [--add <seconds>]: the welcome screen or a page of the wizard as the
  // window shows it. Pages: welcome, files, files-ready, files-project,
  // analysing, keys, next. `keys` with notes (and an accompaniment) is the
  // song's own answer, found by the calls the App makes; with "-" it is one
  // stretch, C before G. --at: the playhead; --add: a change of key added
  // there by hand.
  if (argc >= 5 && juce::String::fromUTF8(argv[3]) == "--wizard-shot")
  {
    Localization::loadFromSettings();
    if (const char *language = std::getenv("PITCHNET_LANG"))
      Localization::getInstance().setLanguage(language);
    const juce::String page = juce::String::fromUTF8(argv[4]);
    juce::File backingFile;
    juce::StringArray recent;
    int width = 1280, height = 800;
    double at = 0.0, add = -1.0;
    for (int i = 5; i < argc; ++i)
    {
      const juce::String arg = juce::String::fromUTF8(argv[i]);
      const auto next = [&] { return i + 1 < argc ? juce::String::fromUTF8(argv[++i]) : juce::String(); };
      if (arg == "--backing")
        backingFile = juce::File::getCurrentWorkingDirectory().getChildFile(next());
      else if (arg == "--recent")
        recent.add(juce::File::getCurrentWorkingDirectory().getChildFile(next()).getFullPathName());
      else if (arg == "--at")
        at = next().getDoubleValue();
      else if (arg == "--add")
        add = next().getDoubleValue();
      else if (arg == "--size")
      {
        width = next().getIntValue();
        height = next().getIntValue();
      }
    }
    if (page == "welcome")
    {
      WelcomeOverlay welcome;
      welcome.setRecentFiles(recent);
      welcome.setSize(width, height);
      std::printf("welcome (%s): %d recent, card %s, shot %s\n",
                  Localization::getInstance().getLanguage().toRawUTF8(), welcome.recentCount(),
                  welcome.cardBounds().toString().toRawUTF8(), writePng(welcome, output) ? "ok" : "FAILED");
      return 0;
    }
    wizard::Model model;
    WizardOverlay overlay(model);
    const juce::File voice = juce::File::getSpecialLocation(juce::File::tempDirectory)
                                 .getChildFile(juce::String::fromUTF8("晚餐歌_LeadVocal 1.wav"));
    const juce::File backing = juce::File::getSpecialLocation(juce::File::tempDirectory)
                                   .getChildFile(juce::String::fromUTF8("晚餐歌_伴奏.wav"));
    if (page == "files-ready")
    {
      model.setVoice(voice);
      model.setBacking(backing);
      model.aligned = true;
    }
    else if (page == "files-project")
    {
      model.existingProject = true;
      overlay.setVoiceName(voice.getFileName());
    }
    else if (page == "analysing")
      model.page = wizard::Page::analysing;
    else if (page == "next")
      model.page = wizard::Page::nextStep;
    else if (page == "keys")
    {
      keyfinder::Result found;
      if (input.getFileName() == "-")
      {
        found.hasBacking = true;
        found.stretches.push_back({0.0, 215.0, 0, 7, 3.0});
      }
      else
      {
        std::vector<keyfinder::VocalNote> notes;
        if (input.hasFileExtension("pitchnet"))
        {
          Project project;
          if (!ProjectSerializer::loadFromFile(project, input))
            return 1;
          notes = keyfinder::vocalNotes(project);
        }
        else
        {
          juce::StringArray lines;
          input.readLines(lines);
          for (const auto &line : lines)
          {
            int a = 0, b = 0, c = 0, d = 0;
            float midi = 0.0f;
            if (std::sscanf(line.toRawUTF8(), "note src [%d, %d) -> [%d, %d) midi %f", &a, &b, &c, &d,
                            &midi) != 5 ||
                line.endsWith("rest") || b - a < keyfinder::kMinNoteFrames)
              continue;
            keyfinder::VocalNote note;
            note.startSeconds = static_cast<double>(c) * HOP_SIZE / SAMPLE_RATE;
            note.weight = b - a;
            note.pitch = midi;
            notes.push_back(note);
          }
        }
        keyfinder::Chroma chroma;
        if (backingFile != juce::File{})
        {
          juce::AudioFormatManager formats;
          formats.registerBasicFormats();
          std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(backingFile));
          if (reader == nullptr)
            return 1;
          juce::AudioBuffer<float> audio(juce::jlimit(1, 2, static_cast<int>(reader->numChannels)),
                                         static_cast<int>(reader->lengthInSamples));
          reader->read(&audio, 0, audio.getNumSamples(), 0, true, true);
          chroma = keyfinder::analyse(audio, reader->sampleRate);
        }
        found = keyfinder::find(notes, chroma.empty() ? nullptr : &chroma);
      }
      model.setResult(found);
      std::printf("mode: %s (fit major %.3f, minor %.3f)\n", found.minor ? "minor" : "major",
                  found.majorFit, found.minorFit);
      if (add >= 0.0)
        std::printf("change added at %.1f s: %s\n", add, model.addChangeAt(add) ? "yes" : "no");
      model.page = wizard::Page::keyCheck;
      auto &keys = overlay.keyPage();
      keys.setVoiceToo(!found.hasBacking, found.hasBacking);
      keys.setScaleLevelDb(0.0f);
      keys.setPlayback(at, at > 0.0);
      if (at > 0.0 && !model.getStretches().empty())
      {
        const int sounding = model.stretchAt(at);
        keys.setSounding(sounding, model.getStretches()[static_cast<size_t>(sounding)].chosen);
      }
      for (const auto &stretch : model.getStretches())
        std::printf("stretch %.1f-%.1f: %s then %s\n", stretch.startSeconds, stretch.endSeconds,
                    wizardui::noteName(stretch.first), wizardui::noteName(stretch.second));
    }
    overlay.setSize(width, height);
    overlay.setVisible(true);
    overlay.refresh();
    std::printf("wizard %s (%s): card %s, shot %s\n", page.toRawUTF8(),
                Localization::getInstance().getLanguage().toRawUTF8(),
                overlay.cardBounds().toString().toRawUTF8(), writePng(overlay, output) ? "ok" : "FAILED");
    return 0;
  }
  // WarpRender <voice.wav> <out.wav> --key-audition <backing.wav> <root 0-11>
  // <startSeconds> [voice] [around <midi>]: what the key check plays, through
  // the playback engine itself (AudioEngine): the accompaniment from
  // startSeconds, and after 1 s the sine scale of the major key on `root`
  // laid over it, 5.5 s in all, at the levels the App starts with
  // (accompaniment -6 dB, scale -18 dBFS). `voice`: the voice is heard as
  // well. `around`: the pitch the scale's octave is chosen by (default 64).
  if (argc >= 7 && juce::String::fromUTF8(argv[3]) == "--key-audition")
  {
    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    const auto read = [&](const juce::File &file, juce::AudioBuffer<float> &audio, int &rate)
    {
      std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
      if (reader == nullptr)
        return false;
      audio.setSize(static_cast<int>(reader->numChannels), static_cast<int>(reader->lengthInSamples));
      reader->read(&audio, 0, audio.getNumSamples(), 0, true, true);
      rate = static_cast<int>(std::lround(reader->sampleRate));
      return true;
    };
    juce::AudioBuffer<float> voice, backing;
    int voiceRate = 0, backingRate = 0;
    const juce::File backingFile =
        juce::File::getCurrentWorkingDirectory().getChildFile(juce::String::fromUTF8(argv[4]));
    if (!read(input, voice, voiceRate) || !read(backingFile, backing, backingRate))
    {
      std::fprintf(stderr, "cannot read the voice or the accompaniment\n");
      return 1;
    }
    const int root = juce::String::fromUTF8(argv[5]).getIntValue();
    const double start = juce::String::fromUTF8(argv[6]).getDoubleValue();
    bool withVoice = false;
    double around = 64.0;
    for (int i = 7; i < argc; ++i)
    {
      const juce::String arg = juce::String::fromUTF8(argv[i]);
      if (arg == "voice")
        withVoice = true;
      else if (arg == "around" && i + 1 < argc)
        around = juce::String::fromUTF8(argv[++i]).getDoubleValue();
    }
    const double deviceRate = 48000.0;
    const int block = 512;
    const auto chroma = keyfinder::analyse(backing, backingRate);
    AudioEngine engine;
    engine.prepareToPlay(block, deviceRate);
    engine.loadWaveform(voice, voiceRate, false);
    engine.setBacking(backing, backingRate);
    engine.setBackingAudible(true);
    engine.setBackingGainDb(-6.0f);
    engine.setVoiceAudible(withVoice);
    engine.seek(start);
    engine.play();
    const int total = static_cast<int>(5.5 * deviceRate) / block;
    const int scaleAt = static_cast<int>(1.0 * deviceRate) / block;
    juce::AudioBuffer<float> out(2, total * block), one(2, block);
    for (int k = 0; k < total; ++k)
    {
      if (k == scaleAt)
        engine.playOverlay(keyfinder::scaleTone(keyfinder::scaleRootNear(root, around),
                                                chroma.tuneSemitones, deviceRate),
                           static_cast<int>(deviceRate));
      juce::AudioSourceChannelInfo info(&one, 0, block);
      engine.getNextAudioBlock(info);
      for (int ch = 0; ch < 2; ++ch)
        out.copyFrom(ch, k * block, one, ch, 0, block);
    }
    writeWav(output, out, deviceRate, 24);
    std::printf("key audition: root %d from MIDI %d, tune %+.1f cents, peak %.2f dBFS\n", root,
                keyfinder::scaleRootNear(root, around), 100.0 * chroma.tuneSemitones,
                juce::Decibels::gainToDecibels(out.getMagnitude(0, out.getNumSamples())));
    return 0;
  }
  // WarpRender <in.pitchnet | notes.txt> <out.txt> --key [--backing <wav>]
  // [--change-cost <points>] [--scale <wav>]: the key of a song as the
  // wizard finds it (KeyFinder.h), written out to compare with the Python it
  // was tried in: the accompaniment's tuning and the share of its energy on
  // each major scale, whole and by window; the singer's offset and the
  // notes' distance to each scale; each window's cost and key on the
  // cheapest path; the stretches and their two likeliest keys. The notes
  // come from a project, or from the text --dump-notes prints (2 decimals).
  // --scale writes the sine scale of the first stretch's key at 48 kHz.
  if (argc >= 4 && juce::String::fromUTF8(argv[3]) == "--key")
  {
    juce::File backingFile, scaleFile;
    double changeCost = keyfinder::kChangeCost;
    for (int i = 4; i + 1 < argc; i += 2)
    {
      const juce::String arg = juce::String::fromUTF8(argv[i]);
      const juce::String value = juce::String::fromUTF8(argv[i + 1]);
      if (arg == "--backing")
        backingFile = juce::File::getCurrentWorkingDirectory().getChildFile(value);
      else if (arg == "--change-cost")
        changeCost = value.getDoubleValue();
      else if (arg == "--scale")
        scaleFile = juce::File::getCurrentWorkingDirectory().getChildFile(value);
    }
    std::vector<keyfinder::VocalNote> notes;
    if (input.hasFileExtension("pitchnet"))
    {
      Project project;
      if (!ProjectSerializer::loadFromFile(project, input))
        return 1;
      notes = keyfinder::vocalNotes(project);
    }
    else
    {
      juce::StringArray lines;
      input.readLines(lines);
      for (const auto &line : lines)
      {
        int a = 0, b = 0, c = 0, d = 0;
        float midi = 0.0f;
        if (std::sscanf(line.toRawUTF8(), "note src [%d, %d) -> [%d, %d) midi %f", &a, &b, &c, &d,
                        &midi) != 5 ||
            line.endsWith("rest") || b - a < keyfinder::kMinNoteFrames)
          continue;
        keyfinder::VocalNote note;
        note.startSeconds = static_cast<double>(c) * HOP_SIZE / SAMPLE_RATE;
        note.weight = b - a;
        note.pitch = midi;
        notes.push_back(note);
      }
    }
    keyfinder::Chroma chroma;
    if (backingFile != juce::File{})
    {
      juce::AudioFormatManager formats;
      formats.registerBasicFormats();
      std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(backingFile));
      if (reader == nullptr)
      {
        std::fprintf(stderr, "cannot read %s\n", backingFile.getFullPathName().toRawUTF8());
        return 1;
      }
      juce::AudioBuffer<float> audio(static_cast<int>(reader->numChannels),
                                     static_cast<int>(reader->lengthInSamples));
      reader->read(&audio, 0, audio.getNumSamples(), 0, true, true);
      chroma = keyfinder::analyse(audio, reader->sampleRate);
    }
    const auto result = keyfinder::find(notes, chroma.empty() ? nullptr : &chroma, changeCost);
    static const char *names[12] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    juce::String out;
    int placed = 0;
    for (const auto &note : notes)
      placed += note.placed ? 1 : 0;
    out << "notes\t" << static_cast<int>(notes.size()) << "\tplaced\t" << placed << "\n";
    out << "tune_cents\t" << juce::String(100.0 * result.tuneSemitones, 2) << "\n";
    out << "offset_cents\t" << juce::String(100.0 * result.vocalOffsetSemitones, 2) << "\n";
    const auto row = [&](const juce::String &label, const std::array<double, 12> &values, double scale,
                         int decimals)
    {
      out << label;
      for (const double value : values)
        out << "\t" << juce::String(value * scale, decimals);
      out << "\n";
    };
    out << "roots";
    for (const char *name : names)
      out << "\t" << name;
    out << "\n";
    if (!chroma.empty())
      row("whole_backing_share", keyfinder::scaleShares(keyfinder::sum(
                                     chroma, 0, static_cast<int>(chroma.frames.size()))), 100.0, 2);
    std::array<double, 12> fit{};
    if (keyfinder::vocalFit(notes, result.vocalOffsetSemitones, fit))
      row("whole_voice_fit", fit, 100.0, 2);
    for (const auto &window : result.windows)
    {
      const juce::String at = juce::String(window.startSeconds, 2);
      if (window.hasBacking)
        row("window_backing\t" + at, window.backingShare, 100.0, 2);
      if (window.hasVoice)
        row("window_voice\t" + at + "\t" + juce::String(window.notes), window.voiceFit, 100.0, 2);
      row("window_cost\t" + at + "\t" + names[window.key], window.cost, 1.0, 2);
    }
    for (const auto &stretch : result.stretches)
      out << "stretch\t" << juce::String(stretch.startSeconds, 2) << "\t"
          << juce::String(stretch.endSeconds, 2) << "\t" << names[stretch.key] << "\t"
          << names[stretch.second] << "\t" << juce::String(stretch.lead, 2) << "\n";
    output.replaceWithText(out);
    for (const auto &stretch : result.stretches)
      std::printf("%7.2f - %7.2f  %-2s  then %-2s  lead %.2f\n", stretch.startSeconds, stretch.endSeconds,
                  names[stretch.key], names[stretch.second], stretch.lead);
    if (scaleFile != juce::File{} && !result.stretches.empty())
    {
      double weight = 0.0, mean = 0.0;
      for (const auto &note : notes)
      {
        weight += note.weight;
        mean += note.weight * note.pitch;
      }
      const int root = keyfinder::scaleRootNear(result.stretches.front().key,
                                                weight > 0.0 ? mean / weight : 60.0);
      writeWav(scaleFile,
               keyfinder::scaleTone(root, result.hasBacking ? result.tuneSemitones
                                                            : result.vocalOffsetSemitones, 48000.0),
               48000, 24);
    }
    return 0;
  }
  // WarpRender <project.pitchnet> - --notes [from:to]: the notes (index,
  // seconds, recorded and current pitch, offset, edited, synthesis engine),
  // to pick one for a listening test.
  if (argc >= 4 && juce::String::fromUTF8(argv[3]) == "--notes")
  {
    Project project;
    if (!ProjectSerializer::loadFromFile(project, input))
      return 1;
    const juce::String range = argc >= 5 ? juce::String::fromUTF8(argv[4]) : "0:1e9";
    const double from = range.upToFirstOccurrenceOf(":", false, false).getDoubleValue();
    const double to = range.fromFirstOccurrenceOf(":", false, false).getDoubleValue();
    const double hop = static_cast<double>(HOP_SIZE) / SAMPLE_RATE;
    const auto &notes = project.getNotes();
    for (size_t i = 0; i < notes.size(); ++i)
    {
      const auto &n = notes[i];
      if (n.isRest() || n.getEndFrame() * hop <= from || n.getStartFrame() * hop >= to)
        continue;
      std::printf("%zu\t%.3f\t%.3f\torig %.2f\tnow %.2f\toffset %.2f\tedited %d\tengine %d\n",
                  i, n.getStartFrame() * hop, n.getEndFrame() * hop, n.getOriginalMidiNote(),
                  n.getAdjustedMidiNote(), n.getPitchOffset(),
                  n.isNeutralForOriginalWaveform() ? 0 : 1,
                  static_cast<int>(n.getSynthEngine()));
    }
    return 0;
  }
  // WarpRender <project.pitchnet> <out.wav> --project: the saved project's
  // playback as the App builds it (embedded composite + the recording).
  if (argc >= 4 && juce::String::fromUTF8(argv[3]) == "--project")
  {
    Project project;
    if (!ProjectSerializer::loadFromFile(project, input))
    {
      std::fprintf(stderr, "could not load %s\n", input.getFullPathName().toRawUTF8());
      return 1;
    }
    auto &audioData = project.getAudioData();
    // The same lookup the App does on open.
    if (sourceaudio::attachRecording(project, input))
      std::printf("recording %s\n",
                  project.getFilePath().getFullPathName().toRawUTF8());
    else
      std::fprintf(stderr, "recording not found\n");

    // --resynth from:to (seconds): render every edited note in the range
    // again from the original, as a fresh edit would.
    std::unique_ptr<EditorController> resynthController;
    AudioData *rendered = &audioData;
    // --touch N: the user edits note N (+1 semitone) in the UI after opening;
    // only what that edit marks is rendered again (other notes keep their
    // loaded flags).
    const bool touch =
        argc >= 6 && juce::String::fromUTF8(argv[4]) == "--touch";
    const bool resynthAsked =
        argc >= 6 && (juce::String::fromUTF8(argv[4]) == "--resynth" || touch);
    // As the App does on open (MainComponent::layOutProjectStretchedTheOldWay):
    // a project stretched before format 9 is laid out again, the recording
    // stretched by kind and every edit rendered onto it. (--resynth renders
    // below, and that render lays it out first.)
    std::unique_ptr<EditorController> layoutController;
    if (audioData.warpLayoutPending && !resynthAsked)
    {
      juce::String ortError;
      if (!OnnxRuntime::initialise(&ortError))
      {
        std::fprintf(stderr, "ONNX Runtime: %s\n", ortError.toRawUTF8());
        return 1;
      }
      layoutController = std::make_unique<EditorController>(false);
      layoutController->setProject(std::make_unique<Project>(project));
      auto &p = *layoutController->getProject();
      std::atomic<bool> pending{false}, done{false}, ok{false};
      layoutController->resynthesizeIncrementalAsync(
          p, nullptr, [&](bool success) { ok = success; done = true; }, pending, false);
      if (!waitFor(done, 600000) || !ok.load())
      {
        std::fprintf(stderr, "layout failed\n");
        return 1;
      }
      std::printf("laid out again: stretched before format 9\n");
      rendered = &p.getAudioData();
    }
    if (resynthAsked)
    {
      const juce::String range = touch ? juce::String("0:0")
                                       : juce::String::fromUTF8(argv[5]);
      const double from = range.upToFirstOccurrenceOf(":", false, false).getDoubleValue();
      const double to = range.fromFirstOccurrenceOf(":", false, false).getDoubleValue();
      juce::String ortError;
      if (!OnnxRuntime::initialise(&ortError))
      {
        std::fprintf(stderr, "ONNX Runtime: %s\n", ortError.toRawUTF8());
        return 1;
      }
      resynthController = std::make_unique<EditorController>(false);
      // Listening tests: PITCHNET_RESYNTH_ENGINE=psola renders with PSOLA;
      // PITCHNET_RESYNTH_SHIFT=x sets the notes rendered in the range to
      // their analysed centre + x semitones (a tiny x renders the note again
      // without moving it), unedited notes included; PITCHNET_R3_FORMANT=0
      // lets the R3 notes' formants move with the pitch (the App keeps them).
      if (const char *engineName = std::getenv("PITCHNET_RESYNTH_ENGINE"))
        resynthController->setSynthesisEngineType(
            juce::String(engineName) == "psola" ? SynthesisEngineType::Psola
                                                : SynthesisEngineType::Vocoder);
      if (const char *formantFlag = std::getenv("PITCHNET_R3_FORMANT"))
        if (auto *synth = resynthController->getIncrementalSynth())
          synth->setRubberBandFormantPreserved(juce::String(formantFlag) != "0");
      // PITCHNET_R3_ONE_PASS=0: on a stretched part R3 shifts the stretched
      // recording (two passes) instead of reading the recording itself.
      if (const char *onePass = std::getenv("PITCHNET_R3_ONE_PASS"))
        if (auto *synth = resynthController->getIncrementalSynth())
          synth->setRubberBandOnePass(juce::String(onePass) != "0");
      // PITCHNET_VOICE_TAIL=0: what an edited vowel leaves under the sibilant
      // after it stays (the render before that was taken out).
      if (const char *voiceTail = std::getenv("PITCHNET_VOICE_TAIL"))
        if (auto *synth = resynthController->getIncrementalSynth())
          synth->setVoiceTailRemoved(juce::String(voiceTail) != "0");
      // PITCHNET_SPLICE_ALIGN=0: the render is not shifted or rotated to
      // meet the recording at its splices.
      if (const char *align = std::getenv("PITCHNET_SPLICE_ALIGN"))
        if (auto *synth = resynthController->getIncrementalSynth())
          synth->setSpliceAligned(juce::String(align) != "0");
      resynthController->setProject(std::make_unique<Project>(project));
      auto &p = *resynthController->getProject();
      auto &data = p.getAudioData();
      // The composite starts from the recording as the warp layer lays it
      // out. (It was reset to the unstretched recording, which a project
      // saved with anchors then showed through around each rendered note:
      // the phrase at 45.96 s of the speech project began 50 ms early for
      // 30 ms. A project with no anchors is the same either way.)
      if (!touch)
        data.waveform.makeCopyOf(warp::reference(data));
      // PITCHNET_RESYNTH_ANCHORS=src:dst,... (seconds): warp anchors placed
      // as the App places them, before the notes are rendered.
      if (const char *anchorText = std::getenv("PITCHNET_RESYNTH_ANCHORS"))
      {
        const double framesPerSecond = static_cast<double>(SAMPLE_RATE) / HOP_SIZE;
        warp::fitToProject(data);
        warp::WarpMap map = data.warpMap;
        for (const auto &pair : juce::StringArray::fromTokens(juce::String(anchorText), ",", ""))
        {
          if (pair.trim().isEmpty())
            continue;
          const double source =
              pair.upToFirstOccurrenceOf(":", false, false).getDoubleValue() * framesPerSecond;
          const double destination =
              pair.fromFirstOccurrenceOf(":", false, false).getDoubleValue() * framesPerSecond;
          const int index = map.addAnchorAtSource(source);
          if (index < 0)
          {
            std::fprintf(stderr, "anchor at %.3f s refused\n", source / framesPerSecond);
            continue;
          }
          const double placed = map.moveAnchor(index, destination);
          std::printf("anchor %.3f s -> %.3f s (asked %.3f s)\n", source / framesPerSecond,
                      placed / framesPerSecond, destination / framesPerSecond);
        }
        if (!map.isIdentity())
          warp::apply(p, map);
      }
      if (touch)
      {
        // --touch N[,M...]: notes by index. PITCHNET_TOUCH_ENGINE=rubberband
        // switches them to R3 (spec Q12) as the App's synthesis menu does,
        // instead of moving them a semitone.
        const char *touchEngine = std::getenv("PITCHNET_TOUCH_ENGINE");
        juce::StringArray indices;
        indices.addTokens(juce::String::fromUTF8(argv[5]), ",", "");
        for (const auto &index : indices)
        {
          auto &n = p.getNotes()[static_cast<size_t>(index.getIntValue())];
          if (touchEngine != nullptr && juce::String(touchEngine) == "rubberband")
            n.setSynthEngine(Note::SynthEngine::RubberBand);
          else
            n.setPitchOffset(n.getPitchOffset() + 1.0f);
          n.markDirty();
          n.markSynthDirty();
        }
      }
      for (const auto &n : p.getNotes())
        if (!n.isRest() && (n.isDirty() || n.getPitchOffset() != 0.0f))
          std::printf("note @%d dirty %d synthDirty %d rendered %d offset %.2f\n",
                      n.getStartFrame(), n.isDirty() ? 1 : 0,
                      n.isSynthDirty() ? 1 : 0, n.hasRenderedEdit() ? 1 : 0,
                      n.getPitchOffset());
      if (data.melSpectrogram.empty())
      {
        MelSpectrogram mel(data.sampleRate, N_FFT, HOP_SIZE, NUM_MELS, FMIN, FMAX);
        data.melSpectrogram = mel.compute(data.originalWaveform.getReadPointer(0),
                                          data.originalWaveform.getNumSamples());
      }
      std::printf("mel %d frames, f0 %d\n", (int)data.melSpectrogram.size(),
                  (int)data.f0.size());
      const double hop = static_cast<double>(HOP_SIZE) / SAMPLE_RATE;
      int marked = 0;
      // PITCHNET_RESYNTH_DEFAULT_EXCEPT=from:to (seconds): every note goes
      // back to the default synthesis, the ones with no edit included,
      // except those starting in the range.
      if (const char *except = std::getenv("PITCHNET_RESYNTH_DEFAULT_EXCEPT"))
      {
        const double keepFrom = juce::String(except).upToFirstOccurrenceOf(":", false, false).getDoubleValue();
        const double keepTo = juce::String(except).fromFirstOccurrenceOf(":", false, false).getDoubleValue();
        for (auto &note : p.getNotes())
        {
          const double start = note.getStartFrame() * hop;
          if (note.getSynthEngine() == Note::SynthEngine::Auto ||
              (start >= keepFrom && start < keepTo))
            continue;
          std::printf("to default @%d (%.3f s)\n", note.getStartFrame(), start);
          note.setSynthEngine(Note::SynthEngine::Auto);
        }
      }
      const char *shiftText = std::getenv("PITCHNET_RESYNTH_SHIFT");
      for (auto &note : p.getNotes())
        if (!note.isRest() &&
            (shiftText != nullptr || std::getenv("PITCHNET_RESYNTH_FORMANT") != nullptr ||
             !note.isNeutralForOriginalWaveform()) &&
            note.getEndFrame() * hop > from && note.getStartFrame() * hop < to)
        {
          // PITCHNET_RESYNTH_RESET=1: every pitch edit on the note is dropped
          // first (pitch tools too: PITCHNET_RESYNTH_SHIFT alone only sets
          // the centre, and a note with a tilt stays edited).
          if (std::getenv("PITCHNET_RESYNTH_RESET") != nullptr)
            PitchCurveProcessor::resetNoteToUnedited(note);
          // PITCHNET_RESYNTH_UNEDIT_RANGE=from:to (seconds): the notes starting
          // there go back to the recording (no pitch edit, default synthesis),
          // the others keeping what the project has.
          if (const char *unedit = std::getenv("PITCHNET_RESYNTH_UNEDIT_RANGE"))
          {
            const double uneditFrom = juce::String(unedit).upToFirstOccurrenceOf(":", false, false).getDoubleValue();
            const double uneditTo = juce::String(unedit).fromFirstOccurrenceOf(":", false, false).getDoubleValue();
            if (note.getStartFrame() * hop >= uneditFrom && note.getStartFrame() * hop < uneditTo)
            {
              PitchCurveProcessor::resetNoteToUnedited(note);
              note.setSynthEngine(Note::SynthEngine::Auto);
            }
          }
          // PITCHNET_RESYNTH_SHIFT_ONLY=from:to (seconds): only the notes
          // starting there are moved; the others in the range are put back
          // on the recording (an R3 note between unedited neighbours).
          if (shiftText != nullptr)
          {
            const char *only = std::getenv("PITCHNET_RESYNTH_SHIFT_ONLY");
            const double onlyFrom = only ? juce::String(only).upToFirstOccurrenceOf(":", false, false).getDoubleValue() : -1.0;
            const double onlyTo = only ? juce::String(only).fromFirstOccurrenceOf(":", false, false).getDoubleValue() : 1.0e9;
            const bool moved = note.getStartFrame() * hop >= onlyFrom &&
                               note.getStartFrame() * hop < onlyTo;
            note.setMidiNote(note.getOriginalMidiNote() +
                             (moved ? juce::String(shiftText).getFloatValue() : 0.0f));
          }
          // PITCHNET_RESYNTH_NOTE_ENGINE=rubberband: these notes use R3 (Q12);
          // =vocoder: they follow the engine (a saved R3 choice is cleared).
          // PITCHNET_RESYNTH_R3_RANGE=from:to: only the notes starting there.
          const char *r3Range = std::getenv("PITCHNET_RESYNTH_R3_RANGE");
          const double r3From = r3Range ? juce::String(r3Range).upToFirstOccurrenceOf(":", false, false).getDoubleValue() : -1.0;
          const double r3To = r3Range ? juce::String(r3Range).fromFirstOccurrenceOf(":", false, false).getDoubleValue() : 1.0e9;
          if (const char *noteEngine = std::getenv("PITCHNET_RESYNTH_NOTE_ENGINE");
              noteEngine != nullptr &&
              note.getStartFrame() * hop >= r3From && note.getStartFrame() * hop < r3To)
            note.setSynthEngine(juce::String(noteEngine) == "rubberband"
                                    ? Note::SynthEngine::RubberBand
                                    : juce::String(noteEngine) == "highs"
                                        ? Note::SynthEngine::VocoderRecordedHighs
                                    : juce::String(noteEngine) == "fry"
                                        ? Note::SynthEngine::VocoderRecordedHighsFry
                                        : juce::String(noteEngine) == "auto"
                                            ? Note::SynthEngine::Auto
                                            : Note::SynthEngine::Vocoder);
          // PITCHNET_RESYNTH_FORMANT=x: the notes' formant shift, in semitones.
          if (const char *formantText = std::getenv("PITCHNET_RESYNTH_FORMANT"))
            note.setFormantShift(juce::String(formantText).getFloatValue());
          note.markDirty();
          note.markSynthDirty();
          if (std::getenv("PITCHNET_RESYNTH_LIST") != nullptr)
            std::printf("marked @%d-%d src %d-%d midi %.3f orig %.3f engine %s\n",
                        note.getStartFrame(), note.getEndFrame(), note.getSrcStartFrame(),
                        note.getSrcEndFrame(), note.getMidiNote(), note.getOriginalMidiNote(),
                        [&] {
                          const auto engine = fry::resolved(note);
                          const bool isAuto = note.getSynthEngine() == Note::SynthEngine::Auto;
                          return engine == Note::SynthEngine::RubberBand
                                     ? (isAuto ? "r3 (auto)" : "r3")
                                     : engine == Note::SynthEngine::VocoderRecordedHighs
                                           ? (isAuto ? "highs (auto)"
                                              : note.getSynthEngine() ==
                                                        Note::SynthEngine::VocoderRecordedHighsFry
                                                    ? "highs + fry"
                                                    : "highs")
                                           : "vocoder";
                        }());
          ++marked;
        }
      if (shiftText != nullptr || std::getenv("PITCHNET_RESYNTH_UNEDIT_RANGE") != nullptr)
        PitchCurveProcessor::rebuildBaseFromNotes(p);
      // PITCHNET_RESYNTH_VOICED_NOT_NOISE=1 (listening tests): frames the
      // pitch detector called voiced lose their breath / sibilance label, so
      // the render sings them instead of keeping the recording there.
      if (std::getenv("PITCHNET_RESYNTH_VOICED_NOT_NOISE") != nullptr)
      {
        breathsibilance::ensure(data);
        int cleared = 0;
        for (size_t f = 0; f < data.breathSibilance.size() && f < data.voicedMask.size(); ++f)
          if (data.voicedMask[f] && breathsibilance::isNoise(data.breathSibilance[f]))
          {
            data.breathSibilance[f] = breathsibilance::None;
            ++cleared;
          }
        std::printf("voiced frames no longer noise: %d\n", cleared);
      }
      // PITCHNET_RESYNTH_KEEP=<noteStart>:<head>:<tail>[,...]: the stretches
      // these notes keep as recorded by hand (the two green squares): `head`
      // frames of the recording from the note's start, `tail` up to its end,
      // 0 for none. Made as the App makes it (SynthChoiceAction), so the
      // pitch curve follows where a note becomes whole.
      if (const char *keep = std::getenv("PITCHNET_RESYNTH_KEEP"))
      {
        juce::StringArray entries;
        entries.addTokens(juce::String(keep), ",", "");
        for (const auto &entry : entries)
        {
          juce::StringArray parts;
          parts.addTokens(entry, ":", "");
          if (parts.size() < 3)
            continue;
          Note *target = nullptr;
          for (auto &note : p.getNotes())
            if (!note.isRest() && note.getStartFrame() == parts[0].getIntValue())
              target = &note;
          if (target == nullptr)
          {
            std::fprintf(stderr, "keep: no note starts at %d\n", parts[0].getIntValue());
            return 1;
          }
          const int head = parts[1].getIntValue(), tail = parts[2].getIntValue();
          const std::vector<Note *> one{target};
          SynthChoiceAction(p, one, SynthChoiceAction::current(one),
                            {SynthChoice::byHand(
                                head > 0 ? target->getSrcStartFrame() + head : Note::kKeepsNone,
                                tail > 0 ? target->getSrcEndFrame() - tail : Note::kKeepsNone)},
                            "Set Kept Range")
              .redo();
          std::printf("keep @%d src %d-%d: head %d, tail %d%s\n", target->getStartFrame(),
                      target->getSrcStartFrame(), target->getSrcEndFrame(), head, tail,
                      fry::wholeByHand(*target) ? " (whole)" : "");
        }
      }
      // PITCHNET_FRY_LIST=1: the vocal fry found in the recording (frames
      // and seconds of the recording), and what of it the notes left on Auto
      // keep as recorded. PITCHNET_FRY_LIST=only stops after the list.
      if (const char *fryList = std::getenv("PITCHNET_FRY_LIST"))
      {
        const auto list = [hop](const char *label, const fry::Ranges &ranges)
        {
          for (const auto &[first, last] : ranges)
            std::printf("fry %s src %d-%d (%.3f-%.3f s)\n", label, first, last,
                        first * hop, last * hop);
        };
        const auto ends = fry::detect(data);
        const auto starts = fry::onsets(data.rawF0);
        list("end", ends);
        list("end kept", fry::kept(p.getNotes(), data.rawF0, ends));
        list("onset", starts);
        list("onset kept", fry::keptOnsets(p.getNotes(), starts));
        list("told kept", fry::keptAsTold(p.getNotes(), fry::Found{ends, starts}));
        // The notes marked by hand, and everything the synthesizer is told
        // stays the recording (what it renders by, fry::keptAsRecorded).
        for (const auto &note : p.getNotes())
          if (fry::byHand(note))
          {
            const auto head = fry::handHead(note), tail = fry::handTail(note);
            std::printf("fry by hand @%d src %d-%d: start %d-%d, end %d-%d%s\n",
                        note.getStartFrame(), note.getSrcStartFrame(),
                        note.getSrcEndFrame(), head.first, head.second, tail.first,
                        tail.second, fry::wholeByHand(note) ? " (whole)" : "");
          }
        const auto keptNow =
            fry::keptAsRecorded(p.getNotes(), data.rawF0, fry::Found{ends, starts});
        list("kept in all", keptNow.all);
        list("kept as a head", keptNow.heads);
        if (juce::String(fryList) == "only")
          return 0;
      }
      std::printf("marked %d notes dirty, hasDirty %d\n", marked,
                  p.hasDirtyNotes() ? 1 : 0);
      std::atomic<bool> pending{false}, done{false}, ok{false};
      resynthController->resynthesizeIncrementalAsync(
          p, nullptr, [&](bool success) { ok = success; done = true; }, pending,
          false);
      if (!waitFor(done, 600000) || !ok.load())
      {
        std::fprintf(stderr, "resynth failed\n");
        return 1;
      }
      std::printf("resynthesized %d notes\n", marked);
      // PITCHNET_RESYNTH_SAVE=<out.pitchnet>: the project as it stands after
      // the render, saved as the App saves it (a whole project rendered
      // again with the current synthesis, kept as a new file).
      if (const char *savePath = std::getenv("PITCHNET_RESYNTH_SAVE"))
      {
        const juce::File saveFile(juce::String::fromUTF8(savePath));
        if (saveFile.existsAsFile() || !ProjectSerializer::saveToFile(p, saveFile))
        {
          std::fprintf(stderr, "could not save %s (it must not exist yet)\n", savePath);
          return 1;
        }
        std::printf("saved %s (%lld bytes)\n", savePath,
                    static_cast<long long>(saveFile.getSize()));
      }
      rendered = &data;
    }
    auto &audioDataOut = *rendered;
    const auto heard = sourceaudio::build(audioDataOut);
    writeWav(output, heard.audio, heard.sampleRate, 32);
    // The internal 44.1 kHz composite and reference, for debugging only:
    // written when PITCHNET_PROJECT_INTERNALS is set.
    if (std::getenv("PITCHNET_PROJECT_INTERNALS") != nullptr)
    {
      writeWav(output.withFileExtension("composite.wav"), audioDataOut.waveform,
               audioDataOut.sampleRate, 32);
      writeWav(output.withFileExtension("reference.wav"),
               warp::reference(audioDataOut), audioDataOut.sampleRate, 32);
    }
    if (resynthController)
      resynthController->requestShutdown();
    if (layoutController)
      layoutController->requestShutdown();
    std::printf("wrote %s (%d ch, %d Hz)\n", output.getFullPathName().toRawUTF8(),
                heard.audio.getNumChannels(), heard.sampleRate);
    return 0;
  }
  juce::String anchorsArg, engine = "vocoder", detector = "fcpe", snapshotPrefix;
  double shift = 0.0, shiftFrom = 0.0, shiftTo = 1.0e9;
  bool playback = false; // write what playback and export use (spec Q5)
  bool saveProject = false; // --save-project: the analysed project, as the App saves a dropped file
  bool ensembleLag = false, dumpNotes = false, separationTest = false, benchPaint = false, testMerge = false, testDraw = false;
  for (int i = 3; i < argc; ++i)
  {
    const juce::String arg = juce::String::fromUTF8(argv[i]);
    const juce::String next = i + 1 < argc ? juce::String::fromUTF8(argv[i + 1]) : "";
    if (arg == "--anchors") { anchorsArg = next; ++i; }
    else if (arg == "--playback") playback = true;
    else if (arg == "--save-project") saveProject = true;
    else if (arg == "--shift") { shift = next.getDoubleValue(); ++i; }
    else if (arg == "--shift-range")
    {
      shiftFrom = next.upToFirstOccurrenceOf(":", false, false).getDoubleValue();
      shiftTo = next.fromFirstOccurrenceOf(":", false, false).getDoubleValue();
      ++i;
    }
    else if (arg == "--engine") { engine = next; ++i; }
    else if (arg == "--detector") { detector = next; ++i; }
    else if (arg == "--dump-notes") dumpNotes = true;
    else if (arg == "--separation-test") separationTest = true;
    else if (arg == "--paint-bench") benchPaint = true;
    else if (arg == "--merge-test") testMerge = true;
    else if (arg == "--draw-test") testDraw = true;
    else if (arg == "--ensemble-lag") ensembleLag = true;
    else if (arg == "--snapshot") { snapshotPrefix = next; ++i; }
  }

  // ONNX Runtime is loaded from <executable>/../Frameworks on macOS.
  juce::String ortError;
  if (!OnnxRuntime::initialise(&ortError))
  {
    std::fprintf(stderr, "ONNX Runtime: %s\n", ortError.toRawUTF8());
    return 1;
  }
  EditorController controller(false);
  controller.setPitchDetectorType(detector == "swift" ? PitchDetectorType::SwiftF0
                                  : detector == "rmvpe" ? PitchDetectorType::RMVPE
                                  : detector == "ensemble" ? PitchDetectorType::Ensemble
                                                        : PitchDetectorType::FCPE);
  controller.setSynthesisEngineType(engine == "psola" ? SynthesisEngineType::Psola
                                                      : SynthesisEngineType::Vocoder);

  controller.reloadInferenceModels(false);
  // Models load in the background; analysing before they are ready shows a
  // "model failed to load" dialog.
  {
    const auto start = juce::Time::getMillisecondCounter();
    while (!controller.isSelectedPitchDetectorLoaded() || controller.isInferenceBusy())
    {
      juce::MessageManager::getInstance()->runDispatchLoopUntil(50);
      if (juce::Time::getMillisecondCounter() - start > 120000)
      {
        std::fprintf(stderr, "pitch model did not load\n");
        return 1;
      }
    }
  }

  std::atomic<bool> loaded{false}, failed{false};
  controller.loadAudioFileAsync(
      input, [](double, const juce::String &) {},
      [&](const juce::AudioBuffer<float> &) { loaded = true; },
      [&]() { failed = true; loaded = true; });
  if (!waitFor(loaded, 600000) || failed.load() || !controller.getProject())
  {
    std::fprintf(stderr, "analysis failed\n");
    return 1;
  }
  auto &project = *controller.getProject();
  auto &audioData = project.getAudioData();
  const double framesPerSecond = static_cast<double>(SAMPLE_RATE) / HOP_SIZE;
  std::printf("analysed: %d frames, %d notes\n", audioData.getNumFrames(),
              static_cast<int>(project.getNotes().size()));
  if (ensembleLag)
  {
    ensembleCheck(controller, audioData);
    return 0;
  }
  if (saveProject)
  {
    const bool saved = ProjectSerializer::saveToFile(project, output);
    std::printf("%s %s (audio %s)\n", saved ? "saved" : "could not save",
                output.getFullPathName().toRawUTF8(),
                project.getFilePath().getFullPathName().toRawUTF8());
    controller.requestShutdown();
    return saved ? 0 : 1;
  }

  // Warp anchors.
  warp::fitToProject(audioData);
  warp::WarpMap map = audioData.warpMap;
  for (const auto &pair : juce::StringArray::fromTokens(anchorsArg, ",", ""))
  {
    if (pair.trim().isEmpty())
      continue;
    const double source =
        pair.upToFirstOccurrenceOf(":", false, false).getDoubleValue() * framesPerSecond;
    const double destination =
        pair.fromFirstOccurrenceOf(":", false, false).getDoubleValue() * framesPerSecond;
    const int index = map.addAnchorAtSource(source);
    if (index < 0)
    {
      std::fprintf(stderr, "anchor at %.3f s refused\n", source / framesPerSecond);
      continue;
    }
    const double placed = map.moveAnchor(index, destination);
    std::printf("anchor %.3f s -> %.3f s (asked %.3f s)\n", source / framesPerSecond,
                placed / framesPerSecond, destination / framesPerSecond);
  }
  if (!map.isIdentity())
    warp::apply(project, map);

  // Pitch shift.
  if (shift != 0.0)
  {
    int shifted = 0;
    for (auto &note : project.getNotes())
    {
      const double start = note.getSrcStartFrame() / framesPerSecond;
      if (note.isRest() || start < shiftFrom || start >= shiftTo)
        continue;
      note.setMidiNote(note.getMidiNote() + static_cast<float>(shift));
      note.markDirty();
      note.markSynthDirty();
      ++shifted;
    }
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    std::printf("shifted %d notes by %.2f semitones\n", shifted, shift);
  }

  if (dumpNotes)
    for (const auto &note : project.getNotes())
      std::printf("note src [%d, %d) -> [%d, %d) midi %.2f%s\n",
                  note.getSrcStartFrame(), note.getSrcEndFrame(),
                  note.getStartFrame(), note.getEndFrame(), note.getMidiNote(),
                  note.isRest() ? " rest" : "");

  if (separationTest)
  {
    separationCheck(project);
    controller.requestShutdown();
    return 0;
  }
  if (testDraw)
  {
    drawCheck(project);
    controller.requestShutdown();
    return 0;
  }
  if (testMerge)
  {
    mergeCheck(project);
    controller.requestShutdown();
    return 0;
  }
  if (benchPaint)
  {
    paintBench(project);
    controller.requestShutdown();
    return 0;
  }
  if (snapshotPrefix.isNotEmpty())
  {
    snapshots(project, snapshotPrefix);
    controller.requestShutdown();
    return 0;
  }

  std::atomic<bool> pending{false};
  std::atomic<bool> rendered{false}, ok{false};
  const bool anyEdit = audioData.warpLayoutPending || project.hasDirtyNotes();
  if (anyEdit)
  {
    controller.resynthesizeIncrementalAsync(
        project, nullptr,
        [&](bool success) { ok = success; rendered = true; }, pending, false);
    if (!waitFor(rendered, 600000))
    {
      std::fprintf(stderr, "render timed out\n");
      return 1;
    }
    std::printf("render %s\n", ok.load() ? "ok" : "FAILED");
  }

  if (playback)
  {
    const auto heard = sourceaudio::build(audioData);
    if (!writeWav(output, heard.audio, heard.sampleRate, 32))
    {
      std::fprintf(stderr, "could not write %s\n", output.getFullPathName().toRawUTF8());
      return 1;
    }
    // The 44.1 kHz composite of the same run, for comparison.
    writeWav(output.withFileExtension("composite.wav"), audioData.waveform,
             audioData.sampleRate, 32);
    std::printf("wrote playback %s (%d ch, %.2f s at %d Hz)\n",
                output.getFullPathName().toRawUTF8(), heard.audio.getNumChannels(),
                heard.audio.getNumSamples() / double(heard.sampleRate), heard.sampleRate);
    controller.requestShutdown();
    return ok.load() || !anyEdit ? 0 : 1;
  }
  if (!writeWav(output, audioData.waveform, audioData.sampleRate))
  {
    std::fprintf(stderr, "could not write %s\n", output.getFullPathName().toRawUTF8());
    return 1;
  }
  std::printf("wrote %s (%.2f s at %d Hz)\n", output.getFullPathName().toRawUTF8(),
              audioData.waveform.getNumSamples() / double(audioData.sampleRate),
              audioData.sampleRate);
  controller.requestShutdown();
  return ok.load() || !anyEdit ? 0 : 1;
}
