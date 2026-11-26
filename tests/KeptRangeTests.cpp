// The stretches a note keeps as recorded by hand (spec Q12, the two green
// squares on its lower corners): the note's own rule, the file, a split, a
// merge and a moved edge, the history step shared with the synthesis menu,
// and the drag in the piano roll - through the calls the App makes.
#include "../Source/UI/PianoRollComponent.h"
#include "../Source/UI/PianoRoll/NoteSplitter.h"
#include "../Source/UI/PianoRoll/PianoRollInteractionContext.h"
#include "../Source/Models/ProjectSerializer.h"
#include "../Source/Undo/PitchUndoManager.h"
#include "../Source/Undo/SynthChoiceAction.h"
#include "../Source/Utils/FryDetector.h"
#include "../Source/Utils/PitchCurveProcessor.h"
#include "TestAssert.h"
#include <cmath>
#include <iostream>

namespace
{
constexpr int kNone = Note::kKeepsNone;
constexpr auto kKeepsFry = Note::SynthEngine::VocoderRecordedHighsFry;

// 120 frames of one steady voice (so the detector finds no fry), and two
// notes on it: [20, 50) and [50, 90).
void fill(Project &project)
{
  auto &audio = project.getAudioData();
  audio.sampleRate = SAMPLE_RATE;
  audio.melSpectrogram.resize(120);
  audio.denseF0.assign(120, midiToFreq(60.0f));
  audio.f0 = audio.denseF0;
  audio.rawF0 = audio.denseF0;
  audio.voicedMask.assign(120, true);
  audio.warpMap.setLength(120.0);
  project.addNote(Note(20, 50, 60.0f));
  project.addNote(Note(50, 90, 60.0f));
  PitchCurveProcessor::rebuildBaseFromNotes(project);
  PitchCurveProcessor::refitUntouchedCurves(project, 0, 120);
}

Note *noteAt(Project &project, int startFrame)
{
  for (auto &note : project.getNotes())
    if (!note.isRest() && note.getStartFrame() == startFrame)
      return &note;
  return nullptr;
}

// 8. A range kept by hand and a way of rendering other than "keep the fry"
//    never meet: marking sets that way, any other way drops the marks.
void theMarksBelongToOneWay()
{
  Note note(20, 50, 60.0f);
  CHECK(!note.hasKeptRange());
  note.setSynthEngine(Note::SynthEngine::RubberBand);
  note.setKeptRange(26, kNone);
  CHECK(note.hasKeptRange());
  CHECK(note.getSynthEngine() == kKeepsFry);
  CHECK(note.getKeptHeadUntil() == 26 && note.getKeptTailFrom() == kNone);
  note.setSynthEngine(kKeepsFry); // the same way again: the marks stay
  CHECK(note.hasKeptRange() && note.getKeptHeadUntil() == 26);
  for (const auto other : {Note::SynthEngine::Auto, Note::SynthEngine::RubberBand,
                           Note::SynthEngine::VocoderRecordedHighs})
  {
    note.setKeptRange(26, 44);
    note.setSynthEngine(other);
    CHECK(!note.hasKeptRange());
    CHECK(note.getKeptHeadUntil() == kNone && note.getKeptTailFrom() == kNone);
    CHECK(note.getSynthEngine() == other);
  }
}

// 9, 10. Saved and read back: by the call the App saves with, and through
//    the plug-in's state. The file is format 10; a format 9 file, which has
//    no such marks, opens with none.
void theMarksAreSaved()
{
  Project project;
  fill(project);
  project.getNotes()[0].setKeptRange(26, kNone);
  project.getNotes()[1].setKeptRange(kNone, kNone); // marked, keeping nothing

  const auto saved = ProjectSerializer::toSavedJson(project);
  CHECK(static_cast<int>(saved.getProperty("formatVersion", 0)) == ProjectSerializer::FORMAT_VERSION);
  const auto text = juce::JSON::toString(saved);
  Project reopened;
  CHECK(ProjectSerializer::fromJson(reopened, juce::JSON::parse(text)));
  CHECK(reopened.getNotes().size() == 2);
  CHECK(reopened.getNotes()[0].hasKeptRange());
  CHECK(reopened.getNotes()[0].getSynthEngine() == kKeepsFry);
  CHECK(reopened.getNotes()[0].getKeptHeadUntil() == 26);
  CHECK(reopened.getNotes()[0].getKeptTailFrom() == kNone);
  CHECK(reopened.getNotes()[1].hasKeptRange());
  CHECK(reopened.getNotes()[1].getKeptHeadUntil() == kNone);
  CHECK(reopened.getNotes()[1].getKeptTailFrom() == kNone);

  juce::MemoryBlock state;
  CHECK(ProjectSerializer::toBinaryArchive(project, state));
  Project fromState;
  CHECK(ProjectSerializer::fromBinaryArchive(fromState, state.getData(), state.getSize()));
  CHECK(fromState.getNotes()[0].hasKeptRange());
  CHECK(fromState.getNotes()[0].getKeptHeadUntil() == 26);
  CHECK(fromState.getNotes()[1].hasKeptRange());

  // A note not marked writes nothing, and a file from before has nothing.
  Project plain;
  fill(plain);
  plain.getNotes()[0].setSynthEngine(kKeepsFry);
  auto nine = ProjectSerializer::toSavedJson(plain);
  CHECK(!nine["notes"][0].hasProperty("keptHeadUntil"));
  CHECK(!nine["notes"][0].hasProperty("keptTailFrom"));
  nine.getDynamicObject()->setProperty("formatVersion", 9);
  Project fromNine;
  CHECK(ProjectSerializer::fromJson(fromNine, juce::JSON::parse(juce::JSON::toString(nine))));
  CHECK(fromNine.getNotes()[0].getSynthEngine() == kKeepsFry);
  CHECK(!fromNine.getNotes()[0].hasKeptRange());
  CHECK(!fromNine.getNotes()[1].hasKeptRange());
  // A file newer than this build is refused, as before.
  nine.getDynamicObject()->setProperty("formatVersion",
                                       ProjectSerializer::FORMAT_VERSION + 1);
  Project fromLater;
  CHECK(!ProjectSerializer::fromJson(fromLater, nine));
}

// 11. A split: both halves are rendered the way the note was, and each
//     keeps what of the marked stretches lies in it. Undo gives the note back.
void aSplitSharesThem()
{
  Project project;
  fill(project);
  PitchUndoManager history;
  NoteSplitter splitter;
  splitter.setProject(&project);
  splitter.setUndoManager(&history);

  noteAt(project, 50)->setKeptRange(56, 84); // [50, 56) and [84, 90)
  CHECK(splitter.splitNoteAtFrame(noteAt(project, 50), 70));
  CHECK(project.getNotes().size() == 3);
  const Note *left = noteAt(project, 50);
  const Note *right = noteAt(project, 70);
  CHECK(left != nullptr && right != nullptr);
  CHECK(left->getSynthEngine() == kKeepsFry && right->getSynthEngine() == kKeepsFry);
  CHECK(left->hasKeptRange() && right->hasKeptRange());
  CHECK((fry::handHead(*left) == std::pair<int, int>{50, 56}));
  CHECK(fry::handTail(*left).first == fry::handTail(*left).second); // none of the end's
  CHECK(fry::handHead(*right).first == fry::handHead(*right).second);
  CHECK((fry::handTail(*right) == std::pair<int, int>{84, 90}));

  history.undo();
  CHECK(project.getNotes().size() == 2);
  CHECK(noteAt(project, 50)->getKeptHeadUntil() == 56);
  CHECK(noteAt(project, 50)->getKeptTailFrom() == 84);

  // Cut inside a marked stretch: the left half is kept whole, the right one
  // keeps the rest of it.
  noteAt(project, 50)->setKeptRange(76, kNone);
  CHECK(splitter.splitNoteAtFrame(noteAt(project, 50), 70));
  CHECK(fry::wholeByHand(*noteAt(project, 50)));
  CHECK((fry::handHead(*noteAt(project, 70)) == std::pair<int, int>{70, 76}));
  history.undo();

  // A note on another way keeps it on both halves too (the right half of a
  // Rubber Band note used to go back to Auto).
  noteAt(project, 50)->setSynthEngine(Note::SynthEngine::RubberBand);
  CHECK(splitter.splitNoteAtFrame(noteAt(project, 50), 70));
  CHECK(noteAt(project, 50)->getSynthEngine() == Note::SynthEngine::RubberBand);
  CHECK(noteAt(project, 70)->getSynthEngine() == Note::SynthEngine::RubberBand);
  CHECK(!noteAt(project, 70)->hasKeptRange());
}

// 12. A merge: what the halves show kept from the start on and up to the
//     end stays; a stretch left in the middle goes. Undo gives both back.
void aMergeKeepsTheEnds()
{
  Project project;
  fill(project);
  PitchUndoManager history;
  NoteSplitter splitter;
  splitter.setProject(&project);
  splitter.setUndoManager(&history);

  // [20, 26) of the first, and [44, 50) of it which ends up in the middle;
  // [84, 90) of the second.
  noteAt(project, 20)->setKeptRange(26, 44);
  noteAt(project, 50)->setKeptRange(kNone, 84);
  CHECK(splitter.mergeNotes(noteAt(project, 20), noteAt(project, 50)));
  CHECK(project.getNotes().size() == 1);
  const Note *merged = noteAt(project, 20);
  CHECK(merged != nullptr && merged->hasKeptRange());
  CHECK(merged->getSynthEngine() == kKeepsFry);
  CHECK((fry::handHead(*merged) == std::pair<int, int>{20, 26}));
  CHECK((fry::handTail(*merged) == std::pair<int, int>{84, 90}));

  history.undo();
  CHECK(project.getNotes().size() == 2);
  CHECK(noteAt(project, 20)->getKeptTailFrom() == 44);
  CHECK(noteAt(project, 50)->getKeptTailFrom() == 84);

  // The first kept whole: its stretch runs on into the second's start.
  noteAt(project, 20)->setKeptRange(50, kNone);
  noteAt(project, 50)->setKeptRange(58, kNone);
  CHECK(splitter.mergeNotes(noteAt(project, 20), noteAt(project, 50)));
  CHECK((fry::handHead(*noteAt(project, 20)) == std::pair<int, int>{20, 58}));
  CHECK(fry::handTail(*noteAt(project, 20)).first == 90);
  history.undo();

  // Neither marked: as it was, the way of the first half and no marks.
  noteAt(project, 20)->setSynthEngine(Note::SynthEngine::Auto);
  noteAt(project, 50)->setSynthEngine(Note::SynthEngine::Auto);
  CHECK(splitter.mergeNotes(noteAt(project, 20), noteAt(project, 50)));
  CHECK(!noteAt(project, 20)->hasKeptRange());
  CHECK(noteAt(project, 20)->getSynthEngine() == Note::SynthEngine::Auto);
  history.undo();

  // The first on a way that keeps no fry: the merged note is on it, unmarked.
  noteAt(project, 20)->setSynthEngine(Note::SynthEngine::RubberBand);
  noteAt(project, 50)->setKeptRange(58, kNone);
  CHECK(splitter.mergeNotes(noteAt(project, 20), noteAt(project, 50)));
  CHECK(!noteAt(project, 20)->hasKeptRange());
  CHECK(noteAt(project, 20)->getSynthEngine() == Note::SynthEngine::RubberBand);
}

// 13. The marks are places in the recording: a note whose edge is moved
//     keeps the same sound, not the same count of frames.
void aMovedEdgeKeepsTheSameSound()
{
  Note note(20, 50, 60.0f);
  note.setKeptRange(26, 44);
  note.setSrcStartFrame(23); // the left edge pulled in
  CHECK((fry::handHead(note) == std::pair<int, int>{23, 26}));
  note.setSrcStartFrame(30); // past the mark: nothing left of it
  CHECK(fry::handHead(note).first == fry::handHead(note).second);
  note.setSrcEndFrame(46);
  CHECK((fry::handTail(note) == std::pair<int, int>{44, 46}));
  // A side that keeps nothing keeps nothing however the edge moves.
  Note plain(20, 50, 60.0f);
  plain.setKeptRange(kNone, kNone);
  plain.setSrcStartFrame(10);
  plain.setSrcEndFrame(60);
  CHECK(fry::handHead(plain).first == fry::handHead(plain).second);
  CHECK(fry::handTail(plain).first == fry::handTail(plain).second);
}

// 18. The synthesis menu and the squares share one history step. Any way but
//     "keep the fry" drops the marks, and one undo brings them back; that
//     way chosen again on a marked note changes nothing.
void theMenuAndTheMarksShareAStep(PianoRollComponent &roll)
{
  Project project;
  fill(project);
  PitchUndoManager history;
  roll.setProject(&project);
  roll.setUndoManager(&history);
  project.deselectAllNotes();
  Note &note = *noteAt(project, 20);

  for (const auto other : {Note::SynthEngine::VocoderRecordedHighs,
                           Note::SynthEngine::RubberBand, Note::SynthEngine::Auto})
  {
    note.setKeptRange(26, 44);
    roll.chooseSynthEngine(note, other);
    CHECK(note.getSynthEngine() == other);
    CHECK(!note.hasKeptRange());
    history.undo();
    CHECK(note.getSynthEngine() == kKeepsFry);
    CHECK(note.hasKeptRange());
    CHECK(note.getKeptHeadUntil() == 26 && note.getKeptTailFrom() == 44);
    history.redo();
    CHECK(note.getSynthEngine() == other && !note.hasKeptRange());
    history.undo();
  }

  // The same way again: nothing to do, and no history step for it.
  note.setKeptRange(26, 44);
  const bool couldUndo = history.canUndo();
  history.clear();
  roll.chooseSynthEngine(note, kKeepsFry);
  CHECK(note.hasKeptRange() && note.getKeptHeadUntil() == 26);
  CHECK(!history.canUndo());
  juce::ignoreUnused(couldUndo);
  roll.setProject(nullptr);
}

// 19. A note kept whole as recorded sings its recorded pitch: the choice
//     that makes it so moves the curve there at once (the menu's own action
//     left the moved curve until some other pitch edit rebuilt it), and its
//     undo moves it back.
void aWholeNoteSingsWhatWasRecordedAtOnce(PianoRollComponent &roll)
{
  Project project;
  fill(project);
  PitchUndoManager history;
  roll.setProject(&project);
  roll.setUndoManager(&history);
  project.deselectAllNotes();
  Note &note = *noteAt(project, 20);
  note.setMidiNote(62.0f); // moved up 2 semitones
  PitchCurveProcessor::rebuildBaseFromNotes(project);
  const auto &base = project.getAudioData().basePitch;
  CHECK(std::abs(base[35] - 62.0f) < 0.05f);

  // Told to keep fry the detector finds none of: all of it is the recording.
  roll.chooseSynthEngine(note, kKeepsFry);
  CHECK(fry::wholeAsRecorded(note, fry::find(project.getAudioData())));
  CHECK(std::abs(base[35] - 60.0f) < 0.05f);
  history.undo();
  CHECK(std::abs(base[35] - 62.0f) < 0.05f);

  // The same by hand: the two stretches meeting.
  const std::vector<Note *> one{&note};
  auto meet = std::make_unique<SynthChoiceAction>(
      project, one, SynthChoiceAction::current(one),
      std::vector<SynthChoice>{SynthChoice::byHand(30, 30)}, "Set Kept Range");
  meet->redo();
  CHECK(fry::wholeByHand(note));
  CHECK(std::abs(base[35] - 60.0f) < 0.05f);
  meet->undo();
  CHECK(!note.hasKeptRange());
  CHECK(std::abs(base[35] - 62.0f) < 0.05f);

  // Part of it only: the note is rendered at the pitch it was moved to.
  auto part = std::make_unique<SynthChoiceAction>(
      project, one, SynthChoiceAction::current(one),
      std::vector<SynthChoice>{SynthChoice::byHand(26, kNone)}, "Set Kept Range");
  part->redo();
  CHECK(!fry::wholeAsRecorded(note, fry::find(project.getAudioData())));
  CHECK(std::abs(base[35] - 62.0f) < 0.05f);
  CHECK(note.isDirty());
  roll.setProject(nullptr);
}
// ---- the drag in the piano roll, through the mouse calls the App gets --------

juce::MouseEvent mouse(PianoRollComponent &roll, juce::Point<float> point,
                       int modifiers = juce::ModifierKeys::leftButtonModifier)
{
  return {juce::Desktop::getInstance().getMainMouseSource(), point,
          juce::ModifierKeys(modifiers), 1.0f, 0, 0, 0, 0, &roll, &roll,
          juce::Time::getCurrentTime(), point, juce::Time::getCurrentTime(), 1, true};
}

struct Roll
{
  PianoRollComponent &roll;
  Project &project;

  // A point of the roll's world (where the handles are laid out), as the
  // component gets it from the mouse.
  juce::Point<float> onScreen(juce::Point<float> world) const
  {
    return {world.x + PianoRollComponent::pianoKeysWidth - static_cast<float>(roll.getScrollX()),
            world.y + PianoRollComponent::headerHeight - static_cast<float>(roll.getScrollY())};
  }
  float pixelsPerFrame() const
  {
    return roll.getPixelsPerSecond() * HOP_SIZE / static_cast<float>(SAMPLE_RATE);
  }
  // The pointer onto the note's body, from somewhere empty: its handles
  // are laid out anew.
  void hover(const Note &note)
  {
    auto &context = roll.getInteractionContext();
    roll.mouseMove(mouse(roll, onScreen({8.0f, context.midiToY(63.0f) + 10.0f}), 0));
    const float x = (note.getStartFrame() + note.getEndFrame()) * 0.5f * pixelsPerFrame();
    const float y = context.midiToY(note.getAdjustedMidiNote()) +
                    roll.getPixelsPerSemitone() * 0.5f;
    roll.mouseMove(mouse(roll, onScreen({x, y}), 0));
  }
  const PitchToolHandles::Handle *square(PitchToolHandles::HandleType type)
  {
    for (const auto &handle : roll.getInteractionContext().pitchToolHandles->getHandles())
      if (handle.type == type)
        return &handle;
    return nullptr;
  }
  // The note's square taken hold of, moved by `frames` (and back by `back`
  // before it is let go), let go.
  void drag(const Note &note, PitchToolHandles::HandleType type, float frames,
            float back = 0.0f)
  {
    hover(note);
    const auto *handle = square(type);
    CHECK(handle != nullptr);
    const auto from = onScreen(handle->bounds.getCentre());
    roll.mouseMove(mouse(roll, from, 0));
    roll.mouseDown(mouse(roll, from));
    if (frames != 0.0f)
      roll.mouseDrag(mouse(roll, from.translated(frames * pixelsPerFrame(), 0.0f)));
    const float end = frames - back;
    if (back != 0.0f)
      roll.mouseDrag(mouse(roll, from.translated(end * pixelsPerFrame(), 0.0f)));
    roll.mouseUp(mouse(roll, from.translated(end * pixelsPerFrame(), 0.0f), 0));
  }
};

// 14 to 17. The squares appear on the note under the pointer, where it keeps
// what it keeps; a drag marks the note by hand as one history step; a click
// changes nothing; neither square passes the other, and the two meeting is
// all of the note.
void theSquaresAreDragged(PianoRollComponent &roll)
{
  using Type = PitchToolHandles::HandleType;
  Project project;
  fill(project);
  PitchUndoManager history;
  roll.setProject(&project);
  roll.setUndoManager(&history);
  roll.setTrackViewMode(false);
  roll.setEditMode(EditMode::Select);
  roll.setPixelsPerSecond(8.0f * SAMPLE_RATE / HOP_SIZE); // 8 px a frame
  roll.setScrollX(0.0);
  roll.centerOnPitchRange(58.0f, 62.0f);
  project.deselectAllNotes();
  Roll at{roll, project};
  CHECK(std::abs(at.pixelsPerFrame() - 8.0f) < 0.001f);
  Note &note = *noteAt(project, 20);

  // 14. Nothing kept: on the note's own two lower corners.
  at.hover(note);
  CHECK(at.square(Type::KeepHead) != nullptr && at.square(Type::KeepTail) != nullptr);
  CHECK(std::abs(at.square(Type::KeepHead)->bounds.getCentreX() - 20 * 8.0f) < 0.6f);
  CHECK(std::abs(at.square(Type::KeepTail)->bounds.getCentreX() - 50 * 8.0f) < 0.6f);
  CHECK(at.square(Type::KeepHead)->bounds.getY() > at.square(Type::TiltLeft)->bounds.getBottom());

  // 16. A click, and a drag that ends where it began: the note stays left
  //     to the detector, and the history has nothing new.
  at.drag(note, Type::KeepHead, 0.0f);
  CHECK(!note.hasKeptRange() && note.getSynthEngine() == Note::SynthEngine::Auto);
  at.drag(note, Type::KeepHead, 5.0f, 5.0f);
  CHECK(!note.hasKeptRange() && note.getSynthEngine() == Note::SynthEngine::Auto);
  CHECK(!history.canUndo());

  // 15. The left square 6 frames to the right: [20, 26) kept, by hand, the
  //     user's own choice of "keep the fry". One undo: left to the detector
  //     again.
  at.drag(note, Type::KeepHead, 6.0f);
  CHECK(note.hasKeptRange());
  CHECK(note.getSynthEngine() == kKeepsFry);
  CHECK(note.getKeptHeadUntil() == 26 && note.getKeptTailFrom() == kNone);
  CHECK(history.canUndo());
  at.hover(note);
  CHECK(std::abs(at.square(Type::KeepHead)->bounds.getCentreX() - 26 * 8.0f) < 0.6f);
  history.undo();
  CHECK(!note.hasKeptRange() && note.getSynthEngine() == Note::SynthEngine::Auto);
  CHECK(!history.canUndo());
  history.redo();
  CHECK(note.hasKeptRange() && note.getKeptHeadUntil() == 26);

  // The right square 4 frames to the left: [46, 50) as well; the left one
  // stays where it was put.
  at.drag(note, Type::KeepTail, -4.0f);
  CHECK(note.getKeptHeadUntil() == 26 && note.getKeptTailFrom() == 46);
  // Back to its corner is nothing kept there.
  at.drag(note, Type::KeepTail, 9.0f);
  CHECK(note.getKeptHeadUntil() == 26 && note.getKeptTailFrom() == kNone);
  at.drag(note, Type::KeepTail, -4.0f);
  CHECK(note.getKeptTailFrom() == 46);

  // 17. The left one as far right as it goes: it stops at the right one,
  //     and the two touching is all of the note.
  at.drag(note, Type::KeepHead, 40.0f);
  CHECK(note.getKeptHeadUntil() == 46 && note.getKeptTailFrom() == 46);
  CHECK(fry::wholeByHand(note));
  at.hover(note);
  CHECK(std::abs(at.square(Type::KeepHead)->bounds.getRight() -
                 at.square(Type::KeepTail)->bounds.getX()) < 0.01f);
  CHECK(std::abs(at.square(Type::KeepTail)->bounds.getX() - 46 * 8.0f) < 0.6f);
  // Each is still taken hold of on its own side: the left one back by 10.
  at.drag(note, Type::KeepHead, -10.0f);
  CHECK(note.getKeptHeadUntil() == 36 && note.getKeptTailFrom() == 46);
  CHECK(!fry::wholeByHand(note));
  // The right one cannot pass the left one either.
  at.drag(note, Type::KeepTail, -40.0f);
  CHECK(note.getKeptHeadUntil() == 36 && note.getKeptTailFrom() == 36);

  // A note on Rubber Band has the squares too, and a drag puts it on "keep
  // the fry" (undo: Rubber Band again).
  Note &other = *noteAt(project, 50);
  other.setSynthEngine(Note::SynthEngine::RubberBand);
  at.drag(other, Type::KeepTail, -5.0f);
  CHECK(other.getSynthEngine() == kKeepsFry);
  CHECK(other.getKeptHeadUntil() == kNone && other.getKeptTailFrom() == 85);
  history.undo();
  CHECK(other.getSynthEngine() == Note::SynthEngine::RubberBand && !other.hasKeptRange());

  // Told from the menu to keep fry the detector finds none of: all of it,
  // shown as the left square at the far right, which is dragged back from there.
  roll.chooseSynthEngine(other, kKeepsFry);
  CHECK(!other.hasKeptRange());
  at.hover(other);
  CHECK(std::abs(at.square(Type::KeepHead)->bounds.getRight() - 90 * 8.0f) < 0.6f);
  at.drag(other, Type::KeepHead, -30.0f);
  CHECK(other.hasKeptRange());
  CHECK(other.getKeptHeadUntil() == 60 && other.getKeptTailFrom() == kNone);
  // One undo: told from the menu again, with nothing marked by hand (all of
  // it); a drag that comes back to where it began leaves it so too.
  history.undo();
  CHECK(other.getSynthEngine() == kKeepsFry && !other.hasKeptRange());
  CHECK(fry::wholeAsRecorded(other, fry::find(project.getAudioData())));
  history.redo();
  CHECK(other.hasKeptRange() && other.getKeptHeadUntil() == 60);
  history.undo();
  at.drag(other, Type::KeepHead, -12.0f, -12.0f);
  CHECK(other.getSynthEngine() == kKeepsFry && !other.hasKeptRange());
  roll.setProject(nullptr);
}
} // namespace

int main()
{
  juce::ScopedJuceInitialiser_GUI gui;
  theMarksBelongToOneWay();
  theMarksAreSaved();
  aSplitSharesThem();
  aMergeKeepsTheEnds();
  aMovedEdgeKeepsTheSameSound();
  {
    PianoRollComponent roll;
    roll.setSize(800, 400);
    theMenuAndTheMarksShareAStep(roll);
    aWholeNoteSingsWhatWasRecordedAtOnce(roll);
    theSquaresAreDragged(roll);
  }
  std::printf("KeptRangeTests passed\n");
  return 0;
}
