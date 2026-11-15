// Offline check for the warp layer: runs PitchNet's own analysis and
// synthesis on a file, applies warp anchors and a pitch shift, and writes the
// result to a WAV file. Never plays audio.
//
//   WarpRender <in.wav> <out.wav> [--anchors s:d,s:d,...] [--shift semitones]
//              [--shift-range from:to] [--engine vocoder|psola]
//              [--detector fcpe|swift|rmvpe] [--dump-notes]
//              [--snapshot prefix]
//
// Anchors are source:timeline positions in seconds. --shift moves the pitch
// of every note whose recorded start lies in --shift-range (seconds, default
// the whole file). Run from the repository root so Resources/models is found.
// --snapshot draws the piano roll in the Warp and Timing tools to
// <prefix>_*.png, with a drag of the first anchor pushed past its limit.
#include "../../Source/Audio/EditorController.h"
#include "../../Source/Audio/WarpLayer.h"
#include "../../Source/Utils/Constants.h"
#include "../../Source/Utils/OnnxRuntime.h"
#include "../../Source/Utils/PitchCurveProcessor.h"
#include "../../Source/Models/ProjectSerializer.h"
#include "../../Source/Undo/NoteActions.h"
#include "../../Source/Undo/PitchUndoManager.h"
#include "../../Source/UI/PianoRollComponent.h"
#include "../../Source/UI/PianoRoll/NoteSplitter.h"
#include "../../Source/UI/PianoRoll/PianoRollInteractionContext.h"

#include "../../Source/JuceHeader.h"

#include <algorithm>
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
              double rate)
{
  file.deleteFile();
  juce::WavAudioFormat format;
  std::unique_ptr<juce::OutputStream> stream(file.createOutputStream());
  if (!stream)
    return false;
  std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(
      stream.get(), rate, static_cast<unsigned int>(buffer.getNumChannels()), 24,
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

} // namespace

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
  juce::String anchorsArg, engine = "vocoder", detector = "fcpe", snapshotPrefix;
  double shift = 0.0, shiftFrom = 0.0, shiftTo = 1.0e9;
  bool dumpNotes = false, separationTest = false, benchPaint = false, testMerge = false;
  for (int i = 3; i < argc; ++i)
  {
    const juce::String arg = juce::String::fromUTF8(argv[i]);
    const juce::String next = i + 1 < argc ? juce::String::fromUTF8(argv[i + 1]) : "";
    if (arg == "--anchors") { anchorsArg = next; ++i; }
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
