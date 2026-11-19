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
#include "../../Source/Audio/PitchEnsemble.h"
#include "../../Source/Audio/EditorController.h"
#include "../../Source/Audio/WarpLayer.h"
#include "../../Source/Utils/Constants.h"
#include "../../Source/Utils/MelSpectrogram.h"
#include "../../Source/Utils/OnnxRuntime.h"
#include "../../Source/Utils/PitchCurveProcessor.h"
#include "../../Source/Models/ProjectSerializer.h"
#include "../../Source/Models/DrawnCurve.h"
#include "../../Source/Audio/IO/MidiReference.h"
#include "../../Source/Utils/ScaleUtils.h"
#include "../../Source/Undo/NoteActions.h"
#include "../../Source/Undo/PitchUndoManager.h"
#include "../../Source/UI/PianoRollComponent.h"
#include "../../Source/UI/PianoRoll/NoteSplitter.h"
#include "../../Source/UI/PianoRoll/PianoRollInteractionContext.h"

#include "../../Source/JuceHeader.h"

#include <algorithm>
#include <limits>
#include <map>
#include <cmath>
#include <functional>
#include <atomic>
#include <vector>
#include <string>
#include <cstdio>
#include <memory>

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
  // WarpRender <in.pitchnet> <out.pitchnet> --app-op split|merge|delete
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
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    const juce::String op = juce::String::fromUTF8(argv[4]);
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
    constexpr float maxExtraShift = 0.5f;  // semitones a shift may exceed the deviation by
    constexpr int settleFrames = 5;        // in a row inside the band to start/end the held part
    constexpr float otherNoteTolerance = 0.25f; // semitones: sung this close to another note
    constexpr float maxFlattenTrend = 1.0f; // semitones: steeper is a slide, not a sag
    constexpr double flattenHeldSeconds = 0.15;
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
    PitchCurveProcessor::rebuildBaseFromNotes(project);

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
    };
    std::vector<Plan> plans;
    std::map<std::string, int> counts;
    for (auto &note : project.getNotes())
    {
      if (note.isRest())
        continue;
      const int s = note.getStartFrame(), e = note.getEndFrame(), n = e - s;
      std::vector<int> voiced;
      for (int i = 0; i < n; ++i)
        if (!std::isnan(white(s + i)))
          voiced.push_back(i);
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
      const float sung = note.getMidiNote() + note.getPitchOffset();
      // Guide: coverage, then the nearest candidate.
      const int a = s + h0, b = s + h1;
      double covered = 0.0;
      for (const auto &g : guide)
        covered += std::max(0.0, std::min<double>(b, g.end) - std::max<double>(a, g.start));
      if (covered / std::max(1, b - a) < minCover)
      {
        report("no-guide", sung, trend, 0.0f);
        continue;
      }
      std::vector<float> candidates;
      for (const auto &g : guide)
        if (g.end > a - windowFrames && g.start < b + windowFrames &&
            std::find(candidates.begin(), candidates.end(), g.pitch) == candidates.end())
          candidates.push_back(g.pitch);
      std::sort(candidates.begin(), candidates.end(), [&](float x, float y) {
        return std::abs(sung - x) < std::abs(sung - y);
      });
      if (candidates.empty() || std::abs(sung - candidates[0]) > maxDeviation)
      {
        report("melody-diff", sung, trend, 0.0f);
        continue;
      }
      if (candidates.size() > 1 &&
          std::abs(std::abs(sung - candidates[0]) - std::abs(sung - candidates[1])) < tieMargin)
      {
        report("ambiguous", sung, trend, 0.0f);
        continue;
      }
      const float target = candidates[0];
      // Sung cleanly on another semitone: a different melody, not intonation.
      if (const float semitone = std::round(sung - referenceOffset) + referenceOffset;
          std::abs(semitone - target) > 0.5f && std::abs(sung - semitone) <= otherNoteTolerance)
      {
        report("other-note", sung, trend, target);
        continue;
      }
      const bool flatten = extreme < 0 && std::abs(trend) >= flattenMin &&
                           std::abs(trend) <= maxFlattenTrend &&
                           framesToSeconds(h1 - h0) >= flattenHeldSeconds;
      if (extreme < 0 && std::abs(sung - target) <= tolerance && !flatten)
      {
        ++counts["in-tune"];
        continue;
      }
      // The held part moves by the note's deviation (the tilt's own lift of
      // it is corrected away by the measuring passes).
      Plan plan{&note, {}, target, centre - (sung - target), -1, "shift"};
      plan.before = note;
      plan.deviation = sung - target;
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
    }
    // Shift: measure on the rebuilt curve, correct, three times.
    for (int pass = 0; pass < 3; ++pass)
    {
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
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    for (const auto &plan : plans)
      std::printf("fix-%s\t%s\t%d\t%.2f\t%.2f\t%s\n", plan.how.toRawUTF8(),
                  clock(framesToSeconds(plan.note->getStartFrame())).toRawUTF8(),
                  plan.note->getStartFrame(), plan.note->getMidiNote(), plan.target,
                  plan.how == "tilt" ? juce::String::formatted("R%+.2f P%.2f",
                      plan.note->getTiltRight(), plan.note->getTiltPivot()).toRawUTF8()
                  : plan.how == "drift" ? "drift 0.5" : "");
    if (!ProjectSerializer::saveToFile(project, output))
      return 1;
    for (const auto &[k, v] : counts)
      std::printf("# %s %d\n", k.c_str(), v);
    return 0;
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
    // Loaded as resynthesizeIncrementalAsync loads it.
    if (vocoder != nullptr && !vocoder->isLoaded())
      vocoder->loadModel(PlatformPaths::getVocoderModelFile());
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
    const auto &a = project.getAudioData();
    const int start = juce::String::fromUTF8(argv[4]).getIntValue();
    const int end = juce::String::fromUTF8(argv[5]).getIntValue();
    for (int f = start; f < end; ++f)
      std::printf("%d\tvoiced %d\tlabel %d\n", f,
                  f < (int)a.voicedMask.size() && a.voicedMask[(size_t)f] ? 1 : 0,
                  f < (int)a.breathSibilance.size() ? (int)a.breathSibilance[(size_t)f] : -1);
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
    if (argc >= 6 && (juce::String::fromUTF8(argv[4]) == "--resynth" || touch))
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
      resynthController->setProject(std::make_unique<Project>(project));
      auto &p = *resynthController->getProject();
      auto &data = p.getAudioData();
      if (!touch)
        data.waveform.makeCopyOf(data.originalWaveform);
      if (touch)
      {
        auto &n = p.getNotes()[static_cast<size_t>(
            juce::String::fromUTF8(argv[5]).getIntValue())];
        n.setPitchOffset(n.getPitchOffset() + 1.0f);
        n.markDirty();
        n.markSynthDirty();
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
      for (auto &note : p.getNotes())
        if (!note.isRest() && !note.isNeutralForOriginalWaveform() &&
            note.getEndFrame() * hop > from && note.getStartFrame() * hop < to)
        {
          note.markDirty();
          note.markSynthDirty();
          ++marked;
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
      rendered = &data;
    }
    auto &audioDataOut = *rendered;
    const auto heard = sourceaudio::build(audioDataOut);
    writeWav(output, heard.audio, heard.sampleRate, 32);
    writeWav(output.withFileExtension("composite.wav"), audioDataOut.waveform,
             audioDataOut.sampleRate, 32);
    writeWav(output.withFileExtension("reference.wav"),
             warp::reference(audioDataOut), audioDataOut.sampleRate, 32);
    if (resynthController)
      resynthController->requestShutdown();
    std::printf("wrote %s (%d ch, %d Hz)\n", output.getFullPathName().toRawUTF8(),
                heard.audio.getNumChannels(), heard.sampleRate);
    return 0;
  }
  juce::String anchorsArg, engine = "vocoder", detector = "fcpe", snapshotPrefix;
  double shift = 0.0, shiftFrom = 0.0, shiftTo = 1.0e9;
  bool playback = false; // write what playback and export use (spec Q5)
  bool ensembleLag = false, dumpNotes = false, separationTest = false, benchPaint = false, testMerge = false, testDraw = false;
  for (int i = 3; i < argc; ++i)
  {
    const juce::String arg = juce::String::fromUTF8(argv[i]);
    const juce::String next = i + 1 < argc ? juce::String::fromUTF8(argv[i + 1]) : "";
    if (arg == "--anchors") { anchorsArg = next; ++i; }
    else if (arg == "--playback") playback = true;
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
