#include "../Source/UI/PianoRollComponent.h"
#include "../Source/UI/PianoRoll/NoteSplitter.h"
#include "../Source/Audio/NoteTiming.h"
#include "../Source/Audio/WarpLayer.h"
#include "../Source/Undo/PitchUndoManager.h"
#include "../Source/Undo/DragActions.h"
#include "../Source/Utils/PitchCurveProcessor.h"
#include "TestAssert.h"
#include <iostream>

static juce::MouseEvent event(PianoRollComponent& roll, juce::Point<float> point,
    int modifiers = juce::ModifierKeys::leftButtonModifier) {
  return {juce::Desktop::getInstance().getMainMouseSource(), point,
      juce::ModifierKeys(modifiers), 1.0f, 0, 0, 0, 0,
      &roll, &roll, juce::Time::getCurrentTime(), point, juce::Time::getCurrentTime(), 1, true};
}

static void initWarpedOriginalDelta(Project& project) {
  const auto& audioData = project.getAudioData();
  const int maxFrame = static_cast<int>(audioData.basePitch.size()) - 1;
  if (maxFrame < 0)
    return;
  for (auto& n : project.getNotes()) {
    const auto map = notetiming::forNote(audioData, n);
    const int sLen = n.getSrcDurationFrames();
    std::vector<float> od(static_cast<size_t>(sLen), 0.0f);
    for (int i = 0; i < sLen; ++i) {
      const int s = n.getSrcStartFrame() + i;
      const double dst = std::clamp(
          map.destinationAt(static_cast<double>(s) + 0.5) - 0.5,
          0.0, static_cast<double>(maxFrame));
      const size_t t0 = static_cast<size_t>(std::floor(dst));
      const size_t t1 = std::min(t0 + 1, static_cast<size_t>(maxFrame));
      const float frac = static_cast<float>(dst - static_cast<double>(t0));
      const float base = audioData.basePitch[t0] * (1.0f - frac) +
                         audioData.basePitch[t1] * frac;
      od[static_cast<size_t>(i)] =
          freqToMidi(audioData.denseF0[static_cast<size_t>(s)]) - base;
    }
    n.setOriginalDeltaPitch(std::move(od));
  }
}

static void warpedSplitMergeAndDeletePreservePitchCurve(PianoRollComponent& roll) {
  Project proj;
  auto& audio = proj.getAudioData();
  audio.sampleRate = SAMPLE_RATE;
  audio.melSpectrogram.resize(100);
  audio.denseF0.assign(100, midiToFreq(55.4f));
  for (int i = 30; i < 35; ++i)
    audio.denseF0[static_cast<size_t>(i)] = midiToFreq(55.5f);
  for (int i = 35; i < 45; ++i)
    audio.denseF0[static_cast<size_t>(i)] = midiToFreq(44.0f);
  for (int i = 45; i < 62; ++i)
    audio.denseF0[static_cast<size_t>(i)] = midiToFreq(55.4f);
  audio.f0 = audio.denseF0;
  audio.voicedMask.assign(100, true);
  audio.warpMap.setLength(100.0);
  audio.warpMap.setAnchors({{30.0, 40.0}, {45.0, 60.0}, {62.0, 82.0}});

  Note target(40, 60, 55.1f);
  target.setSrcStartFrame(30);
  target.setSrcEndFrame(45);
  Note rightNeighbour(60, 82, 55.4f);
  rightNeighbour.setSrcStartFrame(45);
  rightNeighbour.setSrcEndFrame(62);
  proj.addNote(target);
  proj.addNote(rightNeighbour);
  PitchCurveProcessor::rebuildBaseFromNotes(proj);
  initWarpedOriginalDelta(proj);
  proj.getNotes()[0].setAnalysisSpans(
      {{30, 45, 55.1f, proj.getNotes()[0].getOriginalDeltaPitch()}});
  proj.getNotes()[1].setAnalysisSpans(
      {{45, 62, 55.4f, proj.getNotes()[1].getOriginalDeltaPitch()}});
  PitchCurveProcessor::rebuildBaseFromNotes(proj);
  PitchCurveProcessor::refitUntouchedCurves(proj, 0, 100);

  const auto f0Before = proj.getAudioData().f0;

  auto findNoteAt = [&](int startFrame) -> Note* {
    for (auto& n : proj.getNotes())
      if (!n.isRest() && n.getStartFrame() == startFrame)
        return &n;
    return nullptr;
  };
  auto checkRangeMatches = [&](int from, int to) {
    for (int i = from; i < to; ++i) {
      const float gotMidi = freqToMidi(proj.getAudioData().f0[static_cast<size_t>(i)]);
      const float expMidi = freqToMidi(f0Before[static_cast<size_t>(i)]);
      CHECK(std::abs(gotMidi - expMidi) < 0.02f);
    }
  };

  PitchUndoManager history;
  NoteSplitter splitter;
  splitter.setProject(&proj);
  splitter.setUndoManager(&history);

  // Split the warped note across the high/low pitch jump: zero boundary spikes.
  CHECK(splitter.splitNoteAtFrame(findNoteAt(40), 46));
  CHECK(proj.getNotes().size() == 3);
  CHECK(findNoteAt(40) != nullptr && findNoteAt(46) != nullptr && findNoteAt(60) != nullptr);
  CHECK(findNoteAt(40)->getMidiNote() > 54.0f);
  CHECK(findNoteAt(46)->getMidiNote() < 46.0f);
  checkRangeMatches(40, 82);

  history.undo();
  CHECK(proj.getNotes().size() == 2);
  checkRangeMatches(40, 82);

  history.redo();
  CHECK(proj.getNotes().size() == 3);
  checkRangeMatches(40, 82);

  // Merge the warped halves back together: zero seam or neighbour spikes.
  CHECK(splitter.mergeNotes(findNoteAt(40), findNoteAt(46)));
  CHECK(proj.getNotes().size() == 2);
  checkRangeMatches(40, 82);

  history.undo();
  CHECK(proj.getNotes().size() == 3);
  checkRangeMatches(40, 82);

  history.redo();
  CHECK(proj.getNotes().size() == 2);
  checkRangeMatches(40, 82);

  // Back to split state, then delete the low half: surviving warped neighbours keep f0.
  history.undo();
  CHECK(proj.getNotes().size() == 3);
  roll.setProject(&proj);
  roll.setUndoManager(&history);
  roll.setTrackViewMode(false);
  proj.deselectAllNotes();
  findNoteAt(46)->setSelected(true);
  CHECK(roll.keyPressed(juce::KeyPress(juce::KeyPress::backspaceKey)));
  CHECK(proj.getNotes().size() == 2);
  CHECK(findNoteAt(46) == nullptr);
  checkRangeMatches(40, 46);
  checkRangeMatches(60, 82);

  history.undo();
  CHECK(proj.getNotes().size() == 3);
  checkRangeMatches(40, 82);

  history.redo();
  CHECK(proj.getNotes().size() == 2);
  checkRangeMatches(40, 46);
  checkRangeMatches(60, 82);

  // Editing one half then merging a warped note without an exact analysis span
  // still returns to a spike-free curve.
  history.undo();
  CHECK(proj.getNotes().size() == 3);
  findNoteAt(40)->setAnalysisSpans({});
  findNoteAt(46)->setAnalysisSpans({});
  findNoteAt(46)->setMidiNote(findNoteAt(46)->getMidiNote() + 4.0f);
  PitchCurveProcessor::rebuildBaseFromNotes(proj);
  CHECK(splitter.mergeNotes(findNoteAt(40), findNoteAt(46)));
  CHECK(proj.getNotes().size() == 2);
  checkRangeMatches(40, 82);

  roll.setUndoManager(nullptr);
}

// Three warped notes with a moving contour: left neighbour, target, right
// neighbour. Each carries its exact analysis span.
static void buildWarpedTrio(Project& proj) {
  auto& audio = proj.getAudioData();
  audio.sampleRate = SAMPLE_RATE;
  audio.melSpectrogram.resize(120);
  audio.denseF0.assign(120, midiToFreq(57.0f));
  for (int i = 0; i < 120; ++i) {
    float midi = 57.0f + 0.4f * std::sin(0.45f * static_cast<float>(i));
    // Sung glides between notes (a few frames each), not instant jumps.
    const auto glide = [i](float at) {
      return 0.5f * (1.0f + std::tanh((static_cast<float>(i) - at) / 1.5f));
    };
    midi += 2.0f * glide(37.5f) - 3.5f * glide(46.5f) + 4.5f * glide(57.5f);
    audio.denseF0[static_cast<size_t>(i)] = midiToFreq(midi);
  }
  audio.f0 = audio.denseF0;
  audio.voicedMask.assign(120, true);
  audio.warpMap.setLength(120.0);
  audio.warpMap.setAnchors({{20.0, 20.0}, {38.0, 40.0}, {58.0, 70.0}, {75.0, 92.0}});

  Note left(20, 40, 57.0f);
  left.setSrcStartFrame(20);
  left.setSrcEndFrame(38);
  Note target(40, 70, 58.0f);
  target.setSrcStartFrame(38);
  target.setSrcEndFrame(58);
  Note right(70, 92, 60.0f);
  right.setSrcStartFrame(58);
  right.setSrcEndFrame(75);
  proj.addNote(left);
  proj.addNote(target);
  proj.addNote(right);
  PitchCurveProcessor::rebuildBaseFromNotes(proj);
  initWarpedOriginalDelta(proj);
  for (auto& n : proj.getNotes()) {
    // Detected notes carry their recorded f0, as a loaded project's do.
    n.setF0Values({audio.denseF0.begin() + n.getSrcStartFrame(),
                   audio.denseF0.begin() + n.getSrcEndFrame()});
  }
  PitchCurveProcessor::rebuildBaseFromNotes(proj);
  initWarpedOriginalDelta(proj);
  for (auto& n : proj.getNotes())
    n.setAnalysisSpans({{n.getSrcStartFrame(), n.getSrcEndFrame(),
                         n.getOriginalMidiNote(), n.getOriginalDeltaPitch()}});
  PitchCurveProcessor::rebuildBaseFromNotes(proj);
  // As a loaded project is (ProjectSerializer): unedited notes sing the
  // recording exactly.
  PitchCurveProcessor::refitUntouchedCurves(proj, 0, 120);
}

static Note* noteAt(Project& proj, int startFrame) {
  for (auto& n : proj.getNotes())
    if (!n.isRest() && n.getStartFrame() == startFrame)
      return &n;
  return nullptr;
}

static void shiftNote(Project& proj, int startFrame, float semitones) {
  auto* n = noteAt(proj, startFrame);
  CHECK(n != nullptr);
  if (!n)
    return;
  n->setMidiNote(n->getMidiNote() + semitones);
  PitchCurveProcessor::rebuildBaseFromNotes(proj);
}

static float maxF0ErrorSemitones(const Project& proj, const std::vector<float>& ref,
                                 int from, int to) {
  float worst = 0.0f;
  for (int i = from; i < to; ++i)
    worst = std::max(worst, std::abs(
        freqToMidi(proj.getAudioData().f0[static_cast<size_t>(i)]) -
        freqToMidi(ref[static_cast<size_t>(i)])));
  return worst;
}

// Split a warped note in 4, drag every piece, merge in pairs, then merge the
// pairs: the analysed pitch comes back, and undo/redo walk it exactly.
static void warpedQuarteredShiftedPairMergeRestoresAnalysis() {
  Project proj;
  buildWarpedTrio(proj);
  const auto f0Initial = proj.getAudioData().f0;
  PitchUndoManager history;
  NoteSplitter splitter;
  splitter.setProject(&proj);
  splitter.setUndoManager(&history);

  CHECK(splitter.splitNoteAtFrame(noteAt(proj, 40), 55));
  CHECK(splitter.splitNoteAtFrame(noteAt(proj, 40), 47));
  CHECK(splitter.splitNoteAtFrame(noteAt(proj, 55), 62));
  CHECK(proj.getNotes().size() == 6);
  CHECK(maxF0ErrorSemitones(proj, f0Initial, 20, 92) < 0.02f);
  shiftNote(proj, 40, 5.0f);
  shiftNote(proj, 47, -8.0f);
  shiftNote(proj, 55, 9.0f);
  shiftNote(proj, 62, -6.0f);
  const auto f0Shifted = proj.getAudioData().f0;

  CHECK(splitter.mergeNotes(noteAt(proj, 40), noteAt(proj, 47)));
  CHECK(splitter.mergeNotes(noteAt(proj, 55), noteAt(proj, 62)));
  const auto f0Pairs = proj.getAudioData().f0;
  CHECK(maxF0ErrorSemitones(proj, f0Initial, 20, 92) < 0.02f);
  CHECK(splitter.mergeNotes(noteAt(proj, 40), noteAt(proj, 55)));
  CHECK(proj.getNotes().size() == 3);
  const float err = maxF0ErrorSemitones(proj, f0Initial, 20, 92);
  if (err >= 0.02f)
    std::cerr << "quartered pair merge max error " << err << " st\n";
  CHECK(err < 0.02f);

  history.undo();
  CHECK(maxF0ErrorSemitones(proj, f0Pairs, 20, 92) < 0.02f);
  history.undo();
  history.undo();
  CHECK(maxF0ErrorSemitones(proj, f0Shifted, 20, 92) < 0.02f);
  history.redo();
  history.redo();
  CHECK(maxF0ErrorSemitones(proj, f0Pairs, 20, 92) < 0.02f);
  history.redo();
  CHECK(maxF0ErrorSemitones(proj, f0Initial, 20, 92) < 0.02f);
}

// Split and drag, split a dragged half again and drag, drag the neighbours,
// then glue everything across the original note boundaries into one note.
static void warpedResplitShiftedCrossNoteMergeRestoresAnalysis() {
  Project proj;
  buildWarpedTrio(proj);
  const auto f0Initial = proj.getAudioData().f0;
  PitchUndoManager history;
  NoteSplitter splitter;
  splitter.setProject(&proj);
  splitter.setUndoManager(&history);

  CHECK(splitter.splitNoteAtFrame(noteAt(proj, 40), 52));
  shiftNote(proj, 40, 6.0f);
  shiftNote(proj, 52, -7.0f);
  CHECK(splitter.splitNoteAtFrame(noteAt(proj, 52), 61));
  shiftNote(proj, 52, 4.0f);
  shiftNote(proj, 61, -5.0f);
  shiftNote(proj, 20, -3.0f);
  shiftNote(proj, 70, -6.0f);
  CHECK(proj.getNotes().size() == 5);

  CHECK(splitter.mergeNotes(noteAt(proj, 52), noteAt(proj, 61)));
  CHECK(splitter.mergeNotes(noteAt(proj, 40), noteAt(proj, 52)));
  CHECK(splitter.mergeNotes(noteAt(proj, 20), noteAt(proj, 40)));
  CHECK(splitter.mergeNotes(noteAt(proj, 20), noteAt(proj, 70)));
  CHECK(proj.getNotes().size() == 1);
  const float err = maxF0ErrorSemitones(proj, f0Initial, 20, 92);
  if (err >= 0.02f)
    std::cerr << "cross-note merge max error " << err << " st\n";
  CHECK(err < 0.02f);
}

// Split, drag a piece far up, delete it, stretch the left neighbour's right
// edge over the gap to meet the other piece, then glue: the analysed pitch
// comes back with no seam.
static void warpedDeleteShiftedPieceResizeMergeRestoresAnalysis(PianoRollComponent& roll) {
  Project proj;
  buildWarpedTrio(proj);
  const auto f0Initial = proj.getAudioData().f0;
  PitchUndoManager history;
  NoteSplitter splitter;
  splitter.setProject(&proj);
  splitter.setUndoManager(&history);
  auto report = [&](const char* step, int from, int to) {
    const float err = maxF0ErrorSemitones(proj, f0Initial, from, to);
    if (err >= 0.02f)
      std::cerr << "delete/resize/merge " << step << " max error " << err << " st\n";
    return err;
  };

  CHECK(splitter.splitNoteAtFrame(noteAt(proj, 40), 52));
  shiftNote(proj, 40, 8.0f);

  roll.setProject(&proj);
  roll.setUndoManager(&history);
  roll.setTrackViewMode(false);
  proj.deselectAllNotes();
  noteAt(proj, 40)->setSelected(true);
  CHECK(roll.keyPressed(juce::KeyPress(juce::KeyPress::backspaceKey)));
  CHECK(noteAt(proj, 40) == nullptr);
  const float afterDelete = std::max(report("after delete (left)", 20, 40),
                                     report("after delete (right)", 52, 92));

  Note* left = noteAt(proj, 20);
  Note* piece = noteAt(proj, 52);
  CHECK(left != nullptr && piece != nullptr);
  const Note leftBefore = *left;
  CHECK(splitter.commitResizeNote(left, leftBefore, leftBefore.getSrcStartFrame(),
                                  piece->getSrcStartFrame()));
  CHECK(noteAt(proj, 20)->getEndFrame() == 52);
  const float afterResize = report("after resize", 20, 92);

  CHECK(splitter.mergeNotes(noteAt(proj, 20), noteAt(proj, 52)));
  const float afterMerge = report("after merge", 20, 92);
  roll.setUndoManager(nullptr);
  CHECK(afterDelete < 0.02f);
  CHECK(afterResize < 0.02f);
  CHECK(afterMerge < 0.02f);
}

// Split, drag and glue, then Restore All Edits over the notes: they sing
// the recording again, through the warp as it is (Restore leaves the
// timing), with no spike at any boundary.
static void warpedRestoreAllEditsSingsRecording(PianoRollComponent& roll) {
  Project proj;
  buildWarpedTrio(proj);
  PitchUndoManager history;
  NoteSplitter splitter;
  splitter.setProject(&proj);
  splitter.setUndoManager(&history);

  CHECK(splitter.splitNoteAtFrame(noteAt(proj, 40), 52));
  shiftNote(proj, 40, 6.0f);
  shiftNote(proj, 52, -7.0f);
  CHECK(splitter.splitNoteAtFrame(noteAt(proj, 52), 61));
  shiftNote(proj, 61, 4.0f);
  shiftNote(proj, 70, -5.0f);
  CHECK(splitter.mergeNotes(noteAt(proj, 40), noteAt(proj, 52)));

  roll.setProject(&proj);
  roll.setUndoManager(&history);
  roll.setTrackViewMode(false);
  proj.deselectAllNotes();
  for (auto& n : proj.getNotes())
    if (!n.isRest() && n.getStartFrame() >= 40)
      n.setSelected(true);
  const auto f0Edited = proj.getAudioData().f0;
  const auto warpEdited = proj.getAudioData().warpMap;
  CHECK(!warpEdited.anchors().empty());
  roll.resetNoteEdits(*noteAt(proj, 40), PianoRollComponent::NoteRestoreMode::All);
  const auto f0Restored = proj.getAudioData().f0;
  // Timing belongs to the warp layer and its tool: Restore leaves it.
  CHECK(proj.getAudioData().warpMap == warpEdited);
  history.undo();
  CHECK(maxF0ErrorSemitones(proj, f0Edited, 20, 92) < 0.02f);
  history.redo();
  CHECK(maxF0ErrorSemitones(proj, f0Restored, 20, 92) < 0.02f);

  // The recording, through the warp as it now is.
  const auto& audio = proj.getAudioData();
  float worst = 0.0f;
  for (const auto& n : proj.getNotes()) {
    if (n.isRest() || n.getEndFrame() <= 40)
      continue;
    const auto& map = audio.warpMap;
    for (int t = n.getStartFrame(); t < n.getEndFrame(); ++t) {
      const int last = static_cast<int>(audio.denseF0.size()) - 1;
      const double src = std::clamp(map.sourceAt(t + 0.5) - 0.5, 0.0,
                                    static_cast<double>(last));
      const int a = static_cast<int>(std::floor(src));
      const int b = std::min(a + 1, last);
      const float k = static_cast<float>(src - a);
      const float expected = freqToMidi(audio.denseF0[size_t(a)]) * (1.0f - k) +
                             freqToMidi(audio.denseF0[size_t(b)]) * k;
      const float err = std::abs(freqToMidi(audio.f0[size_t(t)]) - expected);
      worst = std::max(worst, err);
    }
  }
  if (worst >= 0.02f)
    std::cerr << "restore all max error " << worst << " st\n";
  roll.setUndoManager(nullptr);
  CHECK(worst < 0.02f);
}

// Worst distance (semitones) between the sung pitch and the recording read
// through the warp, over the notes starting in [from, to).
static float maxErrorFromRecording(const Project& proj, int from, int to) {
  const auto& audio = proj.getAudioData();
  float worst = 0.0f;
  for (const auto& n : proj.getNotes()) {
    if (n.isRest() || n.getStartFrame() < from || n.getStartFrame() >= to)
      continue;
    const auto& map = audio.warpMap;
    for (int t = n.getStartFrame(); t < n.getEndFrame(); ++t) {
      const int last = static_cast<int>(audio.denseF0.size()) - 1;
      const double src = std::clamp(map.sourceAt(t + 0.5) - 0.5, 0.0,
                                    static_cast<double>(last));
      const int a = static_cast<int>(std::floor(src));
      const int b = std::min(a + 1, last);
      const float k = static_cast<float>(src - a);
      const float expected = freqToMidi(audio.denseF0[size_t(a)]) * (1.0f - k) +
                             freqToMidi(audio.denseF0[size_t(b)]) * k;
      worst = std::max(worst, std::abs(freqToMidi(audio.f0[size_t(t)]) - expected));
    }
  }
  return worst;
}

// A stretched note dragged up, its neighbour dragged down, then Restore Pitch
// on the stretched note: it sings as it would had only the neighbour been
// dragged - the recording, its end bending smoothly toward the neighbour.
static void warpedRestorePitchNextToShiftedNeighbour(PianoRollComponent& roll) {
  Project reference;
  buildWarpedTrio(reference);
  shiftNote(reference, 70, -5.0f);
  const auto f0Expected = reference.getAudioData().f0;

  Project proj;
  buildWarpedTrio(proj);
  PitchUndoManager history;
  shiftNote(proj, 40, 3.0f);
  shiftNote(proj, 70, -5.0f);

  roll.setProject(&proj);
  roll.setUndoManager(&history);
  roll.setTrackViewMode(false);
  proj.deselectAllNotes();
  roll.resetNoteEdits(*noteAt(proj, 40), PianoRollComponent::NoteRestoreMode::Pitch);
  const float err = maxF0ErrorSemitones(proj, f0Expected, 20, 92);
  if (err >= 0.02f)
    std::cerr << "restore pitch next to shifted neighbour max error " << err << " st\n";
  roll.setUndoManager(nullptr);
  // Before the refit covered stretched notes, the end bent 1.24 st off.
  CHECK(err < 0.02f);
}

// Stretched note dragged up, its neighbour dragged down; Restore Pitch on the
// stretched note, then on the neighbour: the stretched note, already back to
// the recording, follows its neighbour back without a spike at the boundary.
static void warpedRestorePitchThenNeighbourKeepsBoundary(PianoRollComponent& roll) {
  Project proj;
  buildWarpedTrio(proj);
  PitchUndoManager history;
  shiftNote(proj, 40, 4.0f);
  shiftNote(proj, 70, -5.0f);

  roll.setProject(&proj);
  roll.setUndoManager(&history);
  roll.setTrackViewMode(false);
  proj.deselectAllNotes();
  roll.resetNoteEdits(*noteAt(proj, 40), PianoRollComponent::NoteRestoreMode::Pitch);
  const auto f0OneRestored = proj.getAudioData().f0;
  roll.resetNoteEdits(*noteAt(proj, 70), PianoRollComponent::NoteRestoreMode::Pitch);
  const float err = maxErrorFromRecording(proj, 20, 92);
  if (err >= 0.02f)
    std::cerr << "restore pitch then neighbour max error " << err << " st\n";
  CHECK(err < 0.02f);

  history.undo();
  CHECK(maxF0ErrorSemitones(proj, f0OneRestored, 20, 92) < 0.02f);
  roll.setUndoManager(nullptr);
}

// Drag a note's pitch (undoable), Restore Pitch on its stretched neighbour,
// then undo both: back to the starting pitch, with no spike at the boundary.
static void dragThenRestoreNeighbourThenUndoAll(PianoRollComponent& roll) {
  Project proj;
  buildWarpedTrio(proj);
  const auto f0Initial = proj.getAudioData().f0;
  PitchUndoManager history;
  auto drag = [&](int start, float semitones) {
    Note* n = noteAt(proj, start);
    const float from = n->getMidiNote();
    auto action = std::make_unique<NotePitchDragAction>(
        n, nullptr, from, from + semitones, std::vector<F0FrameEdit>{},
        [&proj](Note*) { PitchCurveProcessor::rebuildBaseFromNotes(proj); });
    action->redo();
    history.addAction(std::move(action));
  };

  drag(70, 6.0f);
  drag(40, 3.0f);
  roll.setProject(&proj);
  roll.setUndoManager(&history);
  roll.setTrackViewMode(false);
  proj.deselectAllNotes();
  roll.resetNoteEdits(*noteAt(proj, 40), PianoRollComponent::NoteRestoreMode::Pitch);
  history.undo();
  history.undo();
  history.undo();
  const float err = maxF0ErrorSemitones(proj, f0Initial, 20, 92);
  if (err >= 0.02f)
    std::cerr << "drag, restore neighbour, undo all max error " << err << " st\n";
  roll.setUndoManager(nullptr);
  CHECK(err < 0.02f);
}

// A note edited and then stretched kept its stored contour over its own
// frames, a fraction of a frame off the recording as the track's map plays
// it: target over recording then wobbled wherever the pitch moved, and
// Rubber Band sang that as a quiver (spec Q12). Its analysed contour is now
// fitted to the recording through the new map (warp::apply), so only the
// edit is left, measured against the note's own unedited curve and against
// the recording alike.
static void stretchedEditedNoteLeavesOnlyTheEdit() {
  Project proj;
  buildWarpedTrio(proj);
  shiftNote(proj, 40, 1.0f);
  auto map = proj.getAudioData().warpMap;
  map.setAnchors({{20.0, 20.0}, {38.0, 40.0}, {58.0, 75.4}, {75.0, 97.0}});
  warp::apply(proj, map);
  const auto* note = noteAt(proj, 40);
  CHECK(note != nullptr);
  if (!note)
    return;
  CHECK(note->getDurationFrames() != note->getSrcDurationFrames());
  const auto& audio = proj.getAudioData();
  const auto unedited = PitchCurveProcessor::uneditedPitchOnMovedNotes(proj, 0, 120);
  const auto target = proj.getAdjustedF0ForRange(0, 120);
  CHECK(unedited.size() == 120 && target.size() == 120);
  // Clear of the base pitch's bend toward the neighbours at both ends.
  const int from = note->getStartFrame() + 13, to = note->getEndFrame() - 13;
  CHECK(to - from >= 6);
  float editLow = 1.0e9f, editHigh = -1.0e9f, misfitLow = 1.0e9f, misfitHigh = -1.0e9f;
  const int last = static_cast<int>(audio.denseF0.size()) - 1;
  for (int t = from; t < to; ++t) {
    CHECK(unedited[size_t(t)] > 0.0f);
    const float sung = freqToMidi(target[size_t(t)]);
    const float edit = sung - unedited[size_t(t)];
    editLow = std::min(editLow, edit);
    editHigh = std::max(editHigh, edit);
    const double src = std::clamp(audio.warpMap.sourceAt(t + 0.5) - 0.5, 0.0,
                                  static_cast<double>(last));
    const int a = static_cast<int>(std::floor(src));
    const int b = std::min(a + 1, last);
    const float k = static_cast<float>(src - a);
    const float recorded = freqToMidi(audio.denseF0[size_t(a)]) * (1.0f - k) +
                           freqToMidi(audio.denseF0[size_t(b)]) * k;
    misfitLow = std::min(misfitLow, sung - recorded);
    misfitHigh = std::max(misfitHigh, sung - recorded);
  }
  if (editHigh - editLow >= 0.001f || misfitHigh - misfitLow >= 0.001f)
    std::cerr << "edit " << editLow << ".." << editHigh << " misfit " << misfitLow
              << ".." << misfitHigh << "\n";
  // The edit alone: the semitone the note was moved by, on every frame.
  CHECK(editHigh - editLow < 0.001f);
  CHECK(std::abs(0.5f * (editHigh + editLow) - 1.0f) < 0.01f);
  // Against the recording as the warp plays it too.
  CHECK(misfitHigh - misfitLow < 0.001f);
  CHECK(std::abs(0.5f * (misfitHigh + misfitLow) - 1.0f) < 0.01f);
  // Notes on their recorded frames are left to the recording itself.
  Project still;
  buildWarpedTrio(still);
  still.getAudioData().warpMap.setAnchors({});
  for (auto& n : still.getNotes()) {
    n.setStartFrame(n.getSrcStartFrame());
    n.setEndFrame(n.getSrcEndFrame());
  }
  PitchCurveProcessor::rebuildBaseFromNotes(still);
  for (float value : PitchCurveProcessor::uneditedPitchOnMovedNotes(still, 0, 120))
    CHECK(value == 0.0f);
}

// An edited note shortened next to a note read far lower (fry at the end of
// a phrase, read an octave down while the voice holds its pitch into it).
// The first note's contour ends in a rise that cancels the base's bend toward
// the low note, frame for frame. Spread over the shorter note that rise came
// early or late against the bend, which stays a fixed number of frames long:
// moved up a semitone and shortened to 0.80x, the note's last frames sang
// under the recording and then above the semitone, and stepped down where the
// next note played the recording. Fitted through the new map, the edit runs
// from the semitone down toward the neighbour and nowhere outside it.
static void editedNoteShortenedNextToLowNoteFollowsRecording() {
  Project proj;
  auto& audio = proj.getAudioData();
  audio.sampleRate = SAMPLE_RATE;
  audio.melSpectrogram.resize(120);
  audio.denseF0.assign(120, midiToFreq(62.0f));
  for (int i = 63; i < 120; ++i)
    audio.denseF0[static_cast<size_t>(i)] = midiToFreq(48.0f);
  audio.f0 = audio.denseF0;
  audio.voicedMask.assign(120, true);
  audio.warpMap.setLength(120.0);
  Note held(20, 60, 62.0f);
  held.setSrcStartFrame(20);
  held.setSrcEndFrame(60);
  Note low(60, 100, 48.0f);
  low.setSrcStartFrame(60);
  low.setSrcEndFrame(100);
  proj.addNote(held);
  proj.addNote(low);
  PitchCurveProcessor::rebuildBaseFromNotes(proj);
  initWarpedOriginalDelta(proj);
  PitchCurveProcessor::rebuildBaseFromNotes(proj);
  PitchCurveProcessor::refitUntouchedCurves(proj, 0, 120); // as a loaded project is
  shiftNote(proj, 20, 1.0f);

  auto map = audio.warpMap;
  map.setAnchors({{20.0, 20.0}, {100.0, 84.0}});
  // What the note sings is the recording as the warp plays it, plus what the
  // edit adds to the base: the semitone, bending away toward the neighbour.
  const auto worstMisfit = [](Project& laidOut) {
    const auto& data = laidOut.getAudioData();
    const auto sung = laidOut.getAdjustedF0ForRange(0, 120);
    const auto edited = data.basePitch;
    Project plain = laidOut;
    for (auto& n : plain.getNotes())
      n.setMidiNote(n.getOriginalMidiNote());
    PitchCurveProcessor::rebuildBaseFromNotes(plain);
    const auto& unedited = plain.getAudioData().basePitch;
    const auto* note = noteAt(laidOut, 20);
    float worst = 0.0f;
    for (int t = note->getStartFrame(); t < note->getEndFrame(); ++t) {
      const double src = std::clamp(data.warpMap.sourceAt(t + 0.5) - 0.5, 0.0, 119.0);
      const int a = static_cast<int>(std::floor(src));
      const int b = std::min(a + 1, 119);
      const float k = static_cast<float>(src - a);
      const float recorded = freqToMidi(data.denseF0[size_t(a)]) * (1.0f - k) +
                             freqToMidi(data.denseF0[size_t(b)]) * k;
      worst = std::max(worst, std::abs(freqToMidi(sung[size_t(t)]) -
                                       (recorded + edited[size_t(t)] - unedited[size_t(t)])));
    }
    return worst;
  };
  // What this guards against: the layout as it was, the stored contour spread
  // over the shorter note. Two frames from its end the note sang 62.43 where
  // the recording plus the edit is 62.90.
  Project spread = proj;
  spread.getAudioData().warpMap = map;
  warp::syncNotes(spread);
  PitchCurveProcessor::rebuildBaseFromNotes(spread);
  PitchCurveProcessor::refitUntouchedCurves(spread, 0, 120);
  CHECK(noteAt(spread, 20) != nullptr);
  CHECK(worstMisfit(spread) > 0.3f);

  warp::apply(proj, map);
  const auto* note = noteAt(proj, 20);
  CHECK(note != nullptr);
  if (!note)
    return;
  CHECK(note->getEndFrame() == 52 && note->getSrcEndFrame() == 60);
  const float fitted = worstMisfit(proj);
  if (fitted >= 0.001f)
    std::cerr << "misfit after the warp " << fitted << "\n";
  CHECK(fitted < 0.001f);

  // The stretch taken back: the note is on its recorded frames again and its
  // contour, written at the shorter length, is fitted there too.
  auto back = audio.warpMap;
  back.setAnchors({});
  warp::apply(proj, back);
  note = noteAt(proj, 20);
  CHECK(note != nullptr);
  if (!note)
    return;
  CHECK(note->getEndFrame() == 60);
  CHECK(static_cast<int>(note->getOriginalDeltaPitch().size()) == 40);
  CHECK(worstMisfit(proj) < 0.001f);
}

int main() {
  juce::ScopedJuceInitialiser_GUI gui;
  stretchedEditedNoteLeavesOnlyTheEdit();
  editedNoteShortenedNextToLowNoteFollowsRecording();
  {
    PianoRollComponent scratchRoll;
    scratchRoll.setSize(800, 400);
    warpedDeleteShiftedPieceResizeMergeRestoresAnalysis(scratchRoll);
    warpedRestoreAllEditsSingsRecording(scratchRoll);
    warpedRestorePitchNextToShiftedNeighbour(scratchRoll);
    warpedRestorePitchThenNeighbourKeepsBoundary(scratchRoll);
    dragThenRestoreNeighbourThenUndoAll(scratchRoll);
  }
  warpedQuarteredShiftedPairMergeRestoresAnalysis();
  warpedResplitShiftedCrossNoteMergeRestoresAnalysis();
  Project a, b;
  for (auto* project : {&a, &b}) {
    project->getAudioData().sampleRate = SAMPLE_RATE;
    project->getAudioData().f0.assign(50, 440.0f);
    project->getAudioData().basePitch.assign(50, 88.0f);
    project->getAudioData().deltaPitch.assign(50, 0.0f);
    project->addNote(Note(3, 20, 88));
  }
  PianoRollComponent roll;
  PitchUndoManager history;
  roll.setSize(800, 400);
  roll.setProject(&a);
  roll.setUndoManager(&history);
  roll.setTrackViewMode(true);
  roll.setScrollX(0); roll.setScrollY(0);
  auto regions = std::make_shared<std::vector<MainViewRegionPreview>>();
  MainViewRegionPreview first;
  first.active = true; first.sourceProject = &a; first.startSeconds = 0; first.endSeconds = 0.5;
  regions->push_back(first);
  MainViewRegionPreview second;
  second.project = std::make_shared<Project>(b);
  second.sourceProject = &b; second.startSeconds = 1; second.endSeconds = 1.5;
  second.projectHostOffsetSeconds = 1;
  regions->push_back(second);
  roll.setRegionPreviews(regions);
  bool bAlive = true;
  int aRenders = 0, bRenders = 0;
  roll.onPitchEditFinished = [&] { ++aRenders; };
  roll.getTrackEditTargets = [&] {
    return std::vector<TrackPitchMacroTarget>{
        {&a, {{0, 50}}, [] { return true; }, {}},
        {&b, {{0, 50}}, [&] { return bAlive; }, [&](bool render) { if (render) ++bRenders; }}};
  };
  const float h = roll.getPixelsPerSemitone();
  const float header = PianoRollComponent::headerHeight;
  // A newly focused region can be published before its analysis completes.
  // Its canvas then gains notes, but selection must also receive the new
  // source binding without recreating the component/editor.
  (*regions)[0].sourceProject = nullptr;
  roll.setRegionPreviews(regions);
  CHECK(roll.beginTrackGesture(event(roll, {31, header + 1})));
  roll.updateTrackGesture(event(roll, {220, header + 3*h}));
  roll.finishTrackGesture();
  CHECK(!a.getNotes()[0].isSelected());
  a.getNotes()[0].setSelected(false);
  b.getNotes()[0].setSelected(false);
  auto analyzedRegions = std::make_shared<std::vector<MainViewRegionPreview>>(*regions);
  (*analyzedRegions)[0].sourceProject = &a;
  roll.setRegionPreviews(analyzedRegions);
  CHECK(roll.beginTrackGesture(event(roll, {31, header + 1})));
  roll.updateTrackGesture(event(roll, {220, header + 3*h}));
  roll.finishTrackGesture();
  CHECK(a.getNotes()[0].isSelected() && b.getNotes()[0].isSelected());
  CHECK(second.project->getNotes()[0].isSelected());
  CHECK(!history.canUndo());
  const float x = PianoRollComponent::pianoKeysWidth + framesToSeconds(10) * roll.getPixelsPerSecond();
  CHECK(roll.beginTrackGesture(event(roll, {x, header + 1.5f*h})));
  roll.updateTrackGesture(event(roll, {x, header + 0.5f*h}));
  roll.finishTrackGesture();
  CHECK(a.getNotes()[0].getMidiNote() == 89 && b.getNotes()[0].getMidiNote() == 89);
  CHECK(history.getUndoName() == "Drag Track Notes");
  CHECK(aRenders == 1 && bRenders == 1);
  bAlive = false;
  history.undo();
  CHECK(a.getNotes()[0].getMidiNote() == 88 && b.getNotes()[0].getMidiNote() == 89);
  CHECK(bRenders == 1);
  history.redo();
  CHECK(a.getNotes()[0].getMidiNote() == 89);
  // A note in an overlapping peer region must get hover/tools even when
  // the bound region covers that same timeline position at another pitch.
  bAlive = true;
  a.getNotes()[0].setMidiNote(88);
  b.getNotes()[0].setMidiNote(86);
  auto peers = std::make_shared<std::vector<MainViewRegionPreview>>(*analyzedRegions);
  (*peers)[0].key = "a";
  (*peers)[0].active = true;
  (*peers)[0].project = std::make_shared<Project>(a);
  (*peers)[1].key = "b";
  (*peers)[1].active = false;
  (*peers)[1].startSeconds = 0;
  (*peers)[1].endSeconds = 0.5;
  (*peers)[1].projectHostOffsetSeconds = 0;
  (*peers)[1].project = std::make_shared<Project>(b);
  roll.setRegionPreviews(peers);
  juce::String routed;
  roll.onInactiveRegionClicked = [&](const juce::String& key) {
    routed = key;
    roll.setProject(key == "a" ? &a : &b);
    roll.setTimelineDisplayOffset(0);
    (*peers)[0].active = key == "a";
    (*peers)[1].active = key == "b";
    roll.setRegionPreviews(peers);
  };
  roll.mouseMove(event(roll, {x, header + 3.5f*h}, 0));
  CHECK(routed == "b" && roll.getProject() == &b);
  // Hover returns to A without a selection/activation click.
  roll.mouseMove(event(roll, {x, header + 1.5f*h}, 0));
  CHECK(routed == "a" && roll.getProject() == &a);
  // Flags affect only routing metadata, not which overlapping note can be hit.
  (*peers)[0].active = false;
  (*peers)[1].active = true;
  roll.setRegionPreviews(peers);
  routed.clear();
  roll.mouseMove(event(roll, {x, header + 3.5f*h}, 0));
  CHECK(routed == "b");
  // Clip Mode retains its existing local hover behavior.
  roll.setTrackViewMode(false);
  roll.setProject(&a);
  routed.clear();
  roll.mouseMove(event(roll, {x, header + 3.5f*h}, 0));
  CHECK(routed.isEmpty() && roll.getProject() == &a);

  // Select-mode left-edge resize expands frame coverage while preserving the
  // note's pitch center and keeping the pitch contour spike-free.
  Project c;
  c.getAudioData().sampleRate = SAMPLE_RATE;
  c.getAudioData().melSpectrogram.resize(60);
  c.getAudioData().denseF0.assign(60, midiToFreq(86.0f));
  for (int i = 20; i < 40; ++i)
    c.getAudioData().denseF0[static_cast<size_t>(i)] = midiToFreq(88.0f);
  c.getAudioData().f0 = c.getAudioData().denseF0;
  c.getAudioData().voicedMask.assign(60, true);
  c.addNote(Note(20, 40, 88.0f));
  PitchCurveProcessor::rebuildBaseFromNotes(c);

  PitchUndoManager resizeHistory;
  roll.setUndoManager(&resizeHistory);
  roll.setEditMode(EditMode::Select);
  roll.setTrackViewMode(false);
  roll.setProject(&c);
  roll.setScrollX(0);
  roll.setScrollY(0);

  const float pps = roll.getPixelsPerSecond();
  const float leftEdgeX = PianoRollComponent::pianoKeysWidth + framesToSeconds(20) * pps;
  const float dragTargetX = PianoRollComponent::pianoKeysWidth + framesToSeconds(10) * pps;
  const float noteCenterY = header + 1.5f * h;

  roll.mouseMove(event(roll, {leftEdgeX, noteCenterY}, 0));
  roll.mouseDown(event(roll, {leftEdgeX, noteCenterY}));
  roll.mouseDrag(event(roll, {dragTargetX, noteCenterY}));
  roll.mouseUp(event(roll, {dragTargetX, noteCenterY}, 0));

  CHECK(c.getNotes().size() == 1);
  CHECK(c.getNotes()[0].getStartFrame() == 10 && c.getNotes()[0].getEndFrame() == 40);
  CHECK(c.getNotes()[0].getSrcStartFrame() == 10 && c.getNotes()[0].getSrcEndFrame() == 40);
  CHECK(c.getAudioData().warpMap.isIdentity());
  CHECK(std::abs(c.getNotes()[0].getMidiNote() - 88.0f) < 0.01f);
  CHECK(std::abs(c.getNotes()[0].getOriginalMidiNote() - 88.0f) < 0.01f);
  CHECK(c.getNotes()[0].getF0Values().size() == 30);
  CHECK(c.getNotes()[0].getOriginalDeltaPitch().size() == 30);
  for (int i = 10; i < 20; ++i)
    CHECK(std::abs(freqToMidi(c.getAudioData().f0[static_cast<size_t>(i)]) - 86.0f) < 0.05f);
  for (int i = 20; i < 40; ++i)
    CHECK(std::abs(freqToMidi(c.getAudioData().f0[static_cast<size_t>(i)]) - 88.0f) < 0.05f);
  CHECK(resizeHistory.canUndo());
  CHECK(resizeHistory.getUndoName() == "Resize Note");

  resizeHistory.undo();
  CHECK(c.getNotes()[0].getStartFrame() == 20 && c.getNotes()[0].getEndFrame() == 40);
  CHECK(c.getNotes()[0].getSrcStartFrame() == 20 && c.getNotes()[0].getSrcEndFrame() == 40);
  CHECK(std::abs(c.getNotes()[0].getMidiNote() - 88.0f) < 0.01f);

  resizeHistory.redo();
  CHECK(c.getNotes()[0].getStartFrame() == 10 && c.getNotes()[0].getEndFrame() == 40);
  CHECK(std::abs(c.getNotes()[0].getMidiNote() - 88.0f) < 0.01f);

  // Backspace deletes selected notes with Undo/Redo support.
  c.getNotes()[0].setSelected(true);
  CHECK(roll.keyPressed(juce::KeyPress(juce::KeyPress::backspaceKey)));
  CHECK(c.getNotes().empty());
  CHECK(resizeHistory.getUndoName() == "Delete Note");

  resizeHistory.undo();
  CHECK(c.getNotes().size() == 1);
  CHECK(c.getNotes()[0].getStartFrame() == 10 && c.getNotes()[0].getEndFrame() == 40);

  resizeHistory.redo();
  CHECK(c.getNotes().empty());

  // Resizing a warped note adjacent to another pitch preserves its pitch center
  // and does not introduce a boundary spike on either note.
  Project d;
  d.getAudioData().sampleRate = SAMPLE_RATE;
  d.getAudioData().melSpectrogram.resize(80);
  d.getAudioData().denseF0.assign(80, midiToFreq(88.0f));
  for (int i = 0; i < 20; ++i)
    d.getAudioData().denseF0[static_cast<size_t>(i)] = midiToFreq(82.0f);
  for (int i = 32; i < 80; ++i)
    d.getAudioData().denseF0[static_cast<size_t>(i)] = midiToFreq(76.0f);
  d.getAudioData().f0 = d.getAudioData().denseF0;
  d.getAudioData().voicedMask.assign(80, true);
  d.getAudioData().warpMap.setLength(80.0);
  d.getAudioData().warpMap.setAnchors({{50.0, 60.0}});

  Note leftNote(0, 24, 82.0f);
  leftNote.setSrcStartFrame(0);
  leftNote.setSrcEndFrame(20);
  Note rightNote(24, 48, 88.0f);
  rightNote.setSrcStartFrame(20);
  rightNote.setSrcEndFrame(40);
  d.addNote(leftNote);
  d.addNote(rightNote);
  PitchCurveProcessor::rebuildBaseFromNotes(d);
  initWarpedOriginalDelta(d);
  PitchCurveProcessor::rebuildBaseFromNotes(d);
  // As a loaded project is: unedited notes sing the recording through the
  // warp. Frames 23 and 24 play recorded frames 19.08 and 19.92, either side
  // of the 82 -> 88 step, so each reads a little toward the other note.
  PitchCurveProcessor::refitUntouchedCurves(d, 0, 80);

  const float f0LeftBoundaryBefore = d.getAudioData().f0[23];
  const float f0RightBoundaryBefore = d.getAudioData().f0[24];
  CHECK(std::abs(freqToMidi(f0LeftBoundaryBefore) - 82.0f) < 0.6f);
  CHECK(std::abs(freqToMidi(f0RightBoundaryBefore) - 88.0f) < 0.6f);

  PitchUndoManager warpResizeHistory;
  roll.setUndoManager(&warpResizeHistory);
  roll.setProject(&d);
  roll.setScrollX(0);
  roll.setScrollY(0);

  const float rightEdgeX = PianoRollComponent::pianoKeysWidth + framesToSeconds(48) * pps;
  const float dragRightTargetX = PianoRollComponent::pianoKeysWidth + framesToSeconds(36) * pps;

  roll.mouseMove(event(roll, {rightEdgeX, noteCenterY}, 0));
  roll.mouseDown(event(roll, {rightEdgeX, noteCenterY}));
  roll.mouseDrag(event(roll, {dragRightTargetX, noteCenterY}));
  roll.mouseUp(event(roll, {dragRightTargetX, noteCenterY}, 0));

  CHECK(d.getNotes().size() == 2);
  CHECK(std::abs(d.getNotes()[1].getMidiNote() - 88.0f) < 0.01f);
  CHECK(std::abs(d.getNotes()[1].getOriginalMidiNote() - 88.0f) < 0.01f);
  CHECK(d.getNotes()[1].getStartFrame() == 24 && d.getNotes()[1].getEndFrame() == 36);
  CHECK(d.getNotes()[1].getSrcStartFrame() == 20 && d.getNotes()[1].getSrcEndFrame() == 30);
  CHECK(!d.getAudioData().warpMap.isIdentity());
  CHECK(std::abs(freqToMidi(d.getAudioData().f0[23]) - freqToMidi(f0LeftBoundaryBefore)) < 0.05f);
  CHECK(std::abs(freqToMidi(d.getAudioData().f0[24]) - freqToMidi(f0RightBoundaryBefore)) < 0.05f);
  CHECK(std::abs(freqToMidi(d.getAudioData().f0[28]) - 88.0f) < 0.05f);

  // Wiggling a note boundary across multiple mouseDrag steps must not
  // cumulatively drift neighbouring notes' white pitch curve, and dragging the
  // boundary back to re-touch the neighbour must leave f0 unchanged.
  Project e;
  e.getAudioData().sampleRate = SAMPLE_RATE;
  e.getAudioData().melSpectrogram.resize(80);
  e.getAudioData().denseF0.assign(80, midiToFreq(58.0f));
  for (int i = 0; i < 20; ++i)
    e.getAudioData().denseF0[static_cast<size_t>(i)] =
        (i >= 15) ? midiToFreq(45.0f) : midiToFreq(55.1f);
  for (int i = 20; i < 40; ++i)
    e.getAudioData().denseF0[static_cast<size_t>(i)] =
        (i < 22) ? midiToFreq(45.0f) : midiToFreq(55.4f);
  e.getAudioData().f0 = e.getAudioData().denseF0;
  e.getAudioData().voicedMask.assign(80, true);
  e.getAudioData().warpMap.setLength(80.0);
  e.getAudioData().warpMap.setAnchors({{60.0, 72.0}});

  Note eLeft(0, 24, 55.1f);
  eLeft.setSrcStartFrame(0);
  eLeft.setSrcEndFrame(20);
  Note eRight(24, 48, 55.4f);
  eRight.setSrcStartFrame(20);
  eRight.setSrcEndFrame(40);
  Note eThird(48, 72, 58.0f);
  eThird.setSrcStartFrame(40);
  eThird.setSrcEndFrame(60);
  e.addNote(eLeft);
  e.addNote(eRight);
  e.addNote(eThird);
  PitchCurveProcessor::rebuildBaseFromNotes(e);
  initWarpedOriginalDelta(e);
  PitchCurveProcessor::rebuildBaseFromNotes(e);
  PitchCurveProcessor::refitUntouchedCurves(e, 0, 80); // as a loaded project is
  const auto initialEF0 = e.getAudioData().f0;

  PitchUndoManager wiggleHistory;
  roll.setUndoManager(&wiggleHistory);
  roll.setProject(&e);
  roll.setScrollX(0);
  roll.setScrollY((MAX_MIDI_NOTE - 58.0f) * h);

  const float eRightLeftEdgeX = PianoRollComponent::pianoKeysWidth + framesToSeconds(24) * pps;
  const float eWiggleAx = PianoRollComponent::pianoKeysWidth + framesToSeconds(30) * pps;
  const float eWiggleBx = PianoRollComponent::pianoKeysWidth + framesToSeconds(33) * pps;
  const float eRightCenterY =
      header + (MAX_MIDI_NOTE - 55.4f) * h - static_cast<float>(roll.getScrollY()) + 0.5f * h;

  roll.mouseMove(event(roll, {eRightLeftEdgeX, eRightCenterY}, 0));
  roll.mouseDown(event(roll, {eRightLeftEdgeX, eRightCenterY}));
  for (int step = 0; step < 10; ++step)
    roll.mouseDrag(event(roll, {(step % 2 == 0) ? eWiggleBx : eWiggleAx, eRightCenterY}));
  roll.mouseUp(event(roll, {eWiggleAx, eRightCenterY}, 0));

  CHECK(e.getNotes()[1].getStartFrame() == 30);
  for (int i = 0; i < 24; ++i)
  {
    const float gotMidi = freqToMidi(e.getAudioData().f0[static_cast<size_t>(i)]);
    const float expMidi = freqToMidi(initialEF0[static_cast<size_t>(i)]);
    CHECK(std::abs(gotMidi - expMidi) < 0.02f);
  }

  roll.mouseMove(event(roll, {eWiggleAx, eRightCenterY}, 0));
  roll.mouseDown(event(roll, {eWiggleAx, eRightCenterY}));
  roll.mouseDrag(event(roll, {eRightLeftEdgeX, eRightCenterY}));
  roll.mouseUp(event(roll, {eRightLeftEdgeX, eRightCenterY}, 0));

  CHECK(e.getNotes()[1].getStartFrame() == 24);
  for (int i = 0; i < 48; ++i)
  {
    const float got = e.getAudioData().f0[static_cast<size_t>(i)];
    const float exp = initialEF0[static_cast<size_t>(i)];
    CHECK(std::abs(got - exp) < 0.02f);
  }

  warpedSplitMergeAndDeletePreservePitchCurve(roll);

  std::cout << "Track gesture tests passed\n";
}
