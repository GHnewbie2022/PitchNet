#include "../Source/UI/PianoRollComponent.h"
#include "../Source/Audio/NoteTiming.h"
#include "../Source/Undo/PitchUndoManager.h"
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

int main() {
  juce::ScopedJuceInitialiser_GUI gui;
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

  const float f0LeftBoundaryBefore = d.getAudioData().f0[23];
  const float f0RightBoundaryBefore = d.getAudioData().f0[24];
  CHECK(std::abs(freqToMidi(f0LeftBoundaryBefore) - 82.0f) < 0.25f);
  CHECK(std::abs(freqToMidi(f0RightBoundaryBefore) - 88.0f) < 0.25f);

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

  std::cout << "Track gesture tests passed\n";
}
