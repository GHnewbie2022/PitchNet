// The changes of key on the piano roll's time ruler and the lock on a note
// (spec section 15, plan items 35-40) - through the mouse calls the App
// gets, on the real PianoRollComponent, and the calls its menus make.
#include "../Source/UI/PianoRollComponent.h"
#include "../Source/UI/PianoRoll/PianoRollInteractionContext.h"
#include "../Source/Undo/KeyFlagEdits.h"
#include "../Source/Undo/PitchUndoManager.h"
#include "../Source/Utils/PitchCurveProcessor.h"
#include "TestAssert.h"

#include <cmath>
#include <cstdio>

namespace
{
constexpr int kFrames = 240;

// 240 frames of one steady voice and three notes: [20, 50), [50, 90), [150, 190).
void fill(Project &project)
{
  auto &audio = project.getAudioData();
  audio.sampleRate = SAMPLE_RATE;
  audio.melSpectrogram.resize(kFrames);
  audio.denseF0.assign(kFrames, midiToFreq(60.0f));
  audio.f0 = audio.denseF0;
  audio.rawF0 = audio.denseF0;
  audio.voicedMask.assign(kFrames, true);
  audio.warpMap.setLength(static_cast<double>(kFrames));
  project.addNote(Note(20, 50, 60.0f));
  project.addNote(Note(50, 90, 62.0f));
  project.addNote(Note(150, 190, 64.0f));
  PitchCurveProcessor::rebuildBaseFromNotes(project);
  PitchCurveProcessor::refitUntouchedCurves(project, 0, kFrames);
}

// C major, then the changes given.
void setKey(Project &project, std::vector<KeyChange> changes)
{
  KeyState::confirmed(project, 0, std::move(changes)).applyTo(project);
}

juce::MouseEvent mouse(PianoRollComponent &roll, juce::Point<float> point,
                       int modifiers = juce::ModifierKeys::leftButtonModifier)
{
  return {juce::Desktop::getInstance().getMainMouseSource(), point,
          juce::ModifierKeys(modifiers), 1.0f, 0, 0, 0, 0, &roll, &roll,
          juce::Time::getCurrentTime(), point, juce::Time::getCurrentTime(), 1, true};
}

constexpr float kPixelsPerFrame = 4.0f;

// undo() and redo() say whether the audio is to be rendered again (never,
// for these), not whether there was a step: that is asked first.
void stepBack(PitchUndoManager &history)
{
  CHECK(history.canUndo());
  history.undo();
}
void stepForward(PitchUndoManager &history)
{
  CHECK(history.canRedo());
  history.redo();
}

void show(PianoRollComponent &roll, Project &project, PitchUndoManager &history)
{
  roll.setProject(&project);
  roll.setUndoManager(&history);
  roll.setTrackViewMode(false);
  roll.setEditMode(EditMode::Select);
  roll.setPixelsPerSecond(kPixelsPerFrame * SAMPLE_RATE / HOP_SIZE);
  roll.setScrollX(0.0);
  roll.centerOnPitchRange(58.0f, 66.0f);
  roll.refreshScaleFromProject();
}

float flagX(int sourceFrame)
{
  return PianoRollComponent::pianoKeysWidth + sourceFrame * kPixelsPerFrame;
}

// The edits a flag makes, by themselves.
void theEditsOfAFlag()
{
  Project project;
  fill(project);
  setKey(project, {{80, 7}, {160, 2}});

  // Moved: as far as its neighbours and the ends let it.
  CHECK(keyflags::moved(project, 0, 100).changes[0].sourceFrame == 100);
  CHECK(keyflags::moved(project, 0, 500).changes[0].sourceFrame == 160 - keyflags::kMinGapFrames);
  CHECK(keyflags::moved(project, 0, -50).changes[0].sourceFrame == keyflags::kMinGapFrames);
  CHECK(keyflags::moved(project, 1, 0).changes[1].sourceFrame == 80 + keyflags::kMinGapFrames);
  CHECK(keyflags::moved(project, 1, 9999).changes[1].sourceFrame == kFrames - keyflags::kMinGapFrames);
  CHECK(keyflags::moved(project, 2, 100) == KeyState::of(project)); // there is no third

  // Another key: stretch 0 is the project's own (its mode stays), the rest are the changes'.
  const auto first = keyflags::rekeyed(project, 0, 5);
  CHECK(first.projectRoot == 5 && first.projectMode == ScaleMode::Major);
  CHECK(first.changes == project.getKeyChanges());
  const auto second = keyflags::rekeyed(project, 2, 14);
  CHECK(second.changes[1].root == 2 && second.changes[1].sourceFrame == 160);
  CHECK(keyflags::rekeyed(project, 1, 9).changes[0].root == 9);
  CHECK(keyflags::rekeyed(project, 3, 9) == KeyState::of(project));

  // Taken away; added (not on top of another, nor at the ends).
  CHECK(keyflags::removed(project, 0).changes.size() == 1);
  CHECK(keyflags::removed(project, 0).changes[0].sourceFrame == 160);
  CHECK(keyflags::removed(project, 5) == KeyState::of(project));
  CHECK(keyflags::canAdd(project, 120) && !keyflags::canAdd(project, 84));
  CHECK(!keyflags::canAdd(project, 3) && !keyflags::canAdd(project, kFrames - 2));
  const auto more = keyflags::added(project, 120, 4);
  CHECK(more.changes.size() == 3 && more.changes[1].sourceFrame == 120 && more.changes[1].root == 4);
  CHECK(keyflags::added(project, 84, 4) == KeyState::of(project));

  // The stretches, on the timeline.
  const auto stretches = keyflags::stretches(project);
  CHECK(stretches.size() == 3 && stretches[0].root == 0 && stretches[1].root == 7);
  CHECK(std::abs(stretches[2].startSeconds - 160.0 * HOP_SIZE / SAMPLE_RATE) < 1.0e-9);
}

// 35, 37. A flag is dragged along the ruler: the change follows, one step
// when it is let go, never past its neighbours; the first stretch's does
// not move.
void aFlagIsDragged(PianoRollComponent &roll)
{
  Project project;
  fill(project);
  setKey(project, {{80, 7}, {160, 2}});
  PitchUndoManager history;
  show(roll, project, history);
  int edited = 0;
  roll.onKeyChangesEdited = [&] { ++edited; };

  const auto flags = roll.keyFlags();
  CHECK(flags.size() == 3);
  CHECK(std::abs(flags[1].x - flagX(80)) < 0.51f && std::abs(flags[2].x - flagX(160)) < 0.51f);
  CHECK(flags[1].root == 7 && flags[2].root == 2 && flags[0].root == 0);
  const auto grab = flags[1].bounds.getCentre();
  CHECK(roll.keyFlagAt(grab) == 1 && roll.keyFlagAt(flags[2].bounds.getCentre()) == 2);
  CHECK(roll.keyFlagAt({flagX(120), 6.0f}) == -1); // the bare ruler

  // 20 frames to the right.
  roll.mouseMove(mouse(roll, grab, 0));
  roll.mouseDown(mouse(roll, grab));
  roll.mouseDrag(mouse(roll, grab.translated(10 * kPixelsPerFrame, 0.0f)));
  CHECK(project.getKeyChanges()[0].sourceFrame == 90); // shown as it goes
  CHECK(!history.canUndo());                            // not a step yet
  roll.mouseDrag(mouse(roll, grab.translated(20 * kPixelsPerFrame, 0.0f)));
  roll.mouseUp(mouse(roll, grab.translated(20 * kPixelsPerFrame, 0.0f), 0));
  CHECK(project.getKeyChanges()[0].sourceFrame == 100 && project.getKeyChanges()[0].root == 7);
  CHECK(history.canUndo() && edited >= 2);
  stepBack(history);
  CHECK(project.getKeyChanges()[0].sourceFrame == 80 && !history.canUndo());
  stepForward(history);
  CHECK(project.getKeyChanges()[0].sourceFrame == 100);
  stepBack(history);

  // Dragged past the next one: it stops short of it.
  const auto again = roll.keyFlags()[1].bounds.getCentre();
  roll.mouseDown(mouse(roll, again));
  roll.mouseDrag(mouse(roll, again.translated(150 * kPixelsPerFrame, 0.0f)));
  roll.mouseUp(mouse(roll, again.translated(150 * kPixelsPerFrame, 0.0f), 0));
  CHECK(project.getKeyChanges().size() == 2);
  CHECK(project.getKeyChanges()[0].sourceFrame == 160 - keyflags::kMinGapFrames);
  CHECK(project.getKeyChanges()[1].sourceFrame == 160);
  stepBack(history);
  CHECK(!history.canUndo());

  // Pressed and let go where it was: nothing to undo.
  roll.mouseDown(mouse(roll, again));
  roll.mouseUp(mouse(roll, again, 0));
  CHECK(!history.canUndo() && project.getKeyChanges()[0].sourceFrame == 80);

  // The first stretch's flag is the project's key: a press on it is the
  // ruler's (the playhead), and nothing moves.
  const auto firstFlag = roll.keyFlags()[0].bounds.getCentre();
  const auto before = KeyState::of(project);
  roll.mouseDown(mouse(roll, firstFlag));
  roll.mouseDrag(mouse(roll, firstFlag.translated(40.0f, 0.0f)));
  roll.mouseUp(mouse(roll, firstFlag.translated(40.0f, 0.0f), 0));
  CHECK(KeyState::of(project) == before && !history.canUndo());
  CHECK(!roll.removeKeyChange(-1));

  roll.onKeyChangesEdited = nullptr;
  roll.setProject(nullptr);
}

// A change of key is kept on the recording: where the recording is
// stretched, its flag stands where that frame plays, and a flag dragged to
// a place on the timeline puts the change on the frame that plays there.
void aStretchedRecording(PianoRollComponent &roll)
{
  Project project;
  fill(project);
  // Frame 120 of the recording plays at frame 150 of the timeline: the
  // first half slower, the second faster.
  project.getAudioData().warpMap.setAnchors({{120.0, 150.0}});
  CHECK(!project.getAudioData().warpMap.isIdentity());
  setKey(project, {{120, 7}});
  PitchUndoManager history;
  show(roll, project, history);

  const auto flags = roll.keyFlags();
  CHECK(flags.size() == 2 && std::abs(flags[1].x - flagX(150)) < 0.51f);
  const auto stretches = keyflags::stretches(project);
  CHECK(std::abs(stretches[1].startSeconds - 150.0 * HOP_SIZE / SAMPLE_RATE) < 1.0e-9);

  // Dragged to where frame 180 of the timeline is: frame 160 of the
  // recording plays there (30 frames on, at 120 / 90 of the speed).
  const auto grab = flags[1].bounds.getCentre();
  roll.mouseDown(mouse(roll, grab));
  roll.mouseDrag(mouse(roll, grab.translated(30 * kPixelsPerFrame, 0.0f)));
  roll.mouseUp(mouse(roll, grab.translated(30 * kPixelsPerFrame, 0.0f), 0));
  CHECK(std::abs(project.getKeyChanges()[0].sourceFrame - 160) <= 1);
  CHECK(std::abs(roll.keyFlags()[1].x - flagX(180)) <= kPixelsPerFrame);
  // A note is in the key of where it is on the recording.
  CHECK(project.keyRootFor(project.getNotes()[2]) == 7); // src [150, 190): its middle is past 160
  CHECK(project.keyRootFor(project.getNotes()[1]) == 0);
  stepBack(history);
  CHECK(project.getKeyChanges()[0].sourceFrame == 120);
  roll.setProject(nullptr);
}

// 36. What the menus do: a stretch in another key, a change taken away, one
// added on the bare ruler - each one step.
void theMenusOfTheRuler(PianoRollComponent &roll)
{
  Project project;
  fill(project);
  setKey(project, {{80, 7}});
  PitchUndoManager history;
  show(roll, project, history);

  CHECK(roll.setStretchKey(1, 9));
  CHECK(project.getKeyChanges()[0].root == 9 && roll.keyFlags()[1].root == 9);
  CHECK(!roll.setStretchKey(1, 9)); // the key it has: nothing
  CHECK(roll.setStretchKey(0, 5));
  CHECK(project.getScaleRootNote() == 5 && project.getPreferredScaleMode() == ScaleMode::Major);
  CHECK(roll.keyFlags()[0].root == 5);

  // Added at 2 s on the timeline (frame 172 of the recording).
  const double seconds = 172.0 * HOP_SIZE / SAMPLE_RATE + 0.001;
  CHECK(roll.addKeyChangeAt(seconds, 2));
  CHECK(project.getKeyChanges().size() == 2 && project.getKeyChanges()[1].sourceFrame == 172);
  CHECK(project.getKeyChanges()[1].root == 2 && roll.keyFlags().size() == 3);
  CHECK(!roll.addKeyChangeAt(seconds, 4)); // on top of it

  CHECK(roll.removeKeyChange(0));
  CHECK(project.getKeyChanges().size() == 1 && project.getKeyChanges()[0].sourceFrame == 172);

  // Four steps, back to where it began.
  for (int i = 0; i < 4; ++i)
    stepBack(history);
  CHECK(!history.canUndo());
  CHECK(project.getScaleRootNote() == 0 && project.getKeyChanges().size() == 1);
  CHECK(project.getKeyChanges()[0].sourceFrame == 80 && project.getKeyChanges()[0].root == 7);

  // The last change taken away: a song in one key shows no flag.
  CHECK(roll.removeKeyChange(0));
  CHECK(roll.keyFlags().empty() && roll.keyFlagAt({flagX(80) + 4.0f, 6.0f}) == -1);
  roll.setProject(nullptr);
}

// 40. The grid's rows are shaded by the key of each stretch.
void theGridFollowsTheStretches(PianoRollComponent &roll)
{
  Project project;
  fill(project);
  setKey(project, {{120, 1}}); // C major, then C# major
  // The rows are shaded by the scale only while the scale is shown (the
  // toolbar's switch; off in a new project).
  project.setScaleMode(ScaleMode::Major);
  PitchUndoManager history;
  show(roll, project, history);
  roll.centerOnPitchRange(70.0f, 76.0f); // above the notes: bare rows
  auto &context = roll.getInteractionContext();
  const auto image = roll.createComponentSnapshot(roll.getLocalBounds(), true, 1.0f);
  const auto rowColour = [&](float midi, int sourceFrame)
  {
    const float y = context.midiToY(midi) + roll.getPixelsPerSemitone() * 0.5f +
                    PianoRollComponent::headerHeight - static_cast<float>(roll.getScrollY());
    return image.getPixelAt(static_cast<int>(flagX(sourceFrame)), static_cast<int>(y));
  };
  // 73 is C#, 74 D, 77 F. D is in C major and not in C# major; C# the other
  // way; F is in both (it is the E# of C# major).
  const auto dLeft = rowColour(74.0f, 100), dRight = rowColour(74.0f, 140);
  const auto dbLeft = rowColour(73.0f, 100), dbRight = rowColour(73.0f, 140);
  CHECK(dLeft != dRight && dbLeft != dbRight);
  CHECK(dLeft == dbRight && dRight == dbLeft);
  CHECK(rowColour(77.0f, 100) == rowColour(77.0f, 140));

  // In one key again: one shade across.
  CHECK(roll.removeKeyChange(0));
  const auto plain = roll.createComponentSnapshot(roll.getLocalBounds(), true, 1.0f);
  const float y = context.midiToY(73.0f) + roll.getPixelsPerSemitone() * 0.5f +
                  PianoRollComponent::headerHeight - static_cast<float>(roll.getScrollY());
  CHECK(plain.getPixelAt(static_cast<int>(flagX(100)), static_cast<int>(y)) ==
        plain.getPixelAt(static_cast<int>(flagX(140)), static_cast<int>(y)));
  roll.setProject(nullptr);
}

// 38, 39. The lock under a note and L: the selection at once, one step; a
// locked note is still moved by hand.
void notesAreLocked(PianoRollComponent &roll)
{
  Project project;
  fill(project);
  PitchUndoManager history;
  show(roll, project, history);
  roll.centerOnPitchRange(55.0f, 59.0f); // the notes high in the view: room under them for their buttons
  auto &notes = project.getNotes();
  auto &context = roll.getInteractionContext();
  const auto hover = [&](const Note &note)
  {
    const auto onScreen = [&](juce::Point<float> world)
    {
      return juce::Point<float>(world.x + PianoRollComponent::pianoKeysWidth - static_cast<float>(roll.getScrollX()),
                                world.y + PianoRollComponent::headerHeight - static_cast<float>(roll.getScrollY()));
    };
    roll.mouseMove(mouse(roll, onScreen({8.0f, context.midiToY(57.0f) + 10.0f}), 0));
    const float x = (note.getStartFrame() + note.getEndFrame()) * 0.5f * kPixelsPerFrame;
    const float y = context.midiToY(note.getAdjustedMidiNote()) + roll.getPixelsPerSemitone() * 0.5f;
    roll.mouseMove(mouse(roll, onScreen({x, y}), 0));
  };

  // The note under the pointer, not selected: that note alone.
  hover(notes[0]);
  auto &lock = roll.lockButtonForTests();
  // The size of the three beside it (their images are drawn at half size).
  CHECK(lock.isVisible() && lock.getWidth() == 21 && lock.getHeight() == 21);
  lock.onClick();
  CHECK(notes[0].isLocked() && !notes[1].isLocked() && !notes[2].isLocked());
  CHECK(history.canUndo());
  lock.onClick();
  CHECK(!notes[0].isLocked());

  // Part of a selection: all of it, and unlocked only when all of it is locked.
  notes[0].setSelected(true);
  notes[1].setSelected(true);
  notes[1].setLocked(true);
  hover(notes[0]);
  lock.onClick();
  CHECK(notes[0].isLocked() && notes[1].isLocked() && !notes[2].isLocked());
  lock.onClick();
  CHECK(!notes[0].isLocked() && !notes[1].isLocked());
  stepBack(history); // one step for the two
  CHECK(notes[0].isLocked() && notes[1].isLocked());

  // L does the same to the selection.
  CHECK(roll.keyPressed(juce::KeyPress('l')));
  CHECK(!notes[0].isLocked() && !notes[1].isLocked());
  CHECK(roll.keyPressed(juce::KeyPress('l')));
  CHECK(notes[0].isLocked() && notes[1].isLocked() && !notes[2].isLocked());

  // Locked, and still the user's to move: the lock is for the wizard.
  const float was = notes[0].getMidiNote();
  CHECK(roll.keyPressed(juce::KeyPress(juce::KeyPress::upKey)));
  CHECK(std::abs(notes[0].getMidiNote() - (was + 1.0f)) < 1.0e-4f && notes[0].isLocked());

  // Nothing selected: L does nothing.
  for (auto &note : notes)
    note.setSelected(false);
  CHECK(!roll.keyPressed(juce::KeyPress('l')));
  roll.setProject(nullptr);
}
} // namespace

int main()
{
  juce::ScopedJuceInitialiser_GUI gui;
  theEditsOfAFlag();
  {
    PianoRollComponent roll;
    roll.setSize(1100, 400);
    aFlagIsDragged(roll);
    aStretchedRecording(roll);
    theMenusOfTheRuler(roll);
    theGridFollowsTheStretches(roll);
    notesAreLocked(roll);
  }
  std::printf("KeyFlagGestureTests: all passed\n");
  return 0;
}
