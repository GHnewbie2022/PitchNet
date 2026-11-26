// The wizard's data on a project (spec section 15): changes of key along
// the recording, the accompaniment it links, notes locked - on the model, in
// the file and the plug-in's state, through a split and a merge, in the
// history, and in what a note snaps to. With the calls the App makes.
#include "../Source/UI/PianoRoll/NoteSplitter.h"
#include "../Source/UI/PianoRoll/PitchEditor.h"
#include "../Source/Models/ProjectSerializer.h"
#include "../Source/Undo/KeyChangeAction.h"
#include "../Source/Undo/LockAction.h"
#include "../Source/Undo/PitchUndoManager.h"
#include "../Source/Utils/PitchCurveProcessor.h"
#include "../Source/Utils/ScaleUtils.h"
#include "TestAssert.h"

#include <cmath>
#include <cstdio>

namespace
{
// 120 frames of one steady voice and two notes on it: [20, 50) and [70, 100).
void fill(Project &project)
{
  auto &audio = project.getAudioData();
  audio.sampleRate = SAMPLE_RATE;
  audio.melSpectrogram.resize(120);
  audio.denseF0.assign(120, midiToFreq(65.0f));
  audio.f0 = audio.denseF0;
  audio.rawF0 = audio.denseF0;
  audio.voicedMask.assign(120, true);
  audio.warpMap.setLength(120.0);
  project.addNote(Note(20, 50, 65.0f));
  project.addNote(Note(70, 100, 65.0f));
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

BackingTrack someBacking()
{
  BackingTrack backing;
  backing.file = juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("伴奏 檔.wav");
  backing.sampleRate = 48000;
  backing.channels = 2;
  backing.samples = 11946000;
  backing.aligned = true;
  backing.audible = true;
  backing.gainDb = -9.5f;
  return backing;
}

// 10. The key at a frame of the recording.
void theKeyAtAFrame()
{
  Project project;
  project.setProjectKey(8, ScaleMode::Major); // G#
  CHECK(project.getKeyChanges().empty());
  CHECK(project.keyRootAt(0) == 8 && project.keyRootAt(100000) == 8);

  project.setModified(false);
  project.setKeyChanges({{14650, 9}});
  CHECK(project.isModified());
  CHECK(project.keyRootAt(0) == 8);
  CHECK(project.keyRootAt(14649) == 8);
  CHECK(project.keyRootAt(14650) == 9); // the frame of the change is the later key's
  CHECK(project.keyRootAt(99999) == 9);

  // Kept in order, one a frame (the last given), roots 0..11, none at 0.
  project.setKeyChanges({{900, 14}, {300, 5}, {0, 3}, {-4, 1}, {300, 7}});
  const auto &changes = project.getKeyChanges();
  CHECK(changes.size() == 2);
  CHECK((changes[0] == KeyChange{300, 7}) && (changes[1] == KeyChange{900, 2}));
  CHECK(project.keyRootAt(299) == 8 && project.keyRootAt(300) == 7 && project.keyRootAt(900) == 2);

  // The project's own key still applies up to the first change.
  project.setProjectKey(0, ScaleMode::Major);
  CHECK(project.keyRootAt(299) == 0 && project.keyRootAt(300) == 7);

  // A note is of the key where the middle of it was recorded.
  Note note(280, 330, 60.0f); // middle at 305
  CHECK(project.keyRootFor(note) == 7);
  Note early(280, 310, 60.0f); // middle at 295
  CHECK(project.keyRootFor(early) == 0);

  // The same list again is no change.
  project.setModified(false);
  project.setKeyChanges({{300, 7}, {900, 2}});
  CHECK(!project.isModified());
}

// 11, 13. The file: written only where there is something, read back whole;
// format 11; a file of format 10 has none of it.
void theFile()
{
  Project plain;
  fill(plain);
  const auto plainJson = ProjectSerializer::toSavedJson(plain);
  CHECK(static_cast<int>(plainJson.getProperty("formatVersion", 0)) == 11);
  CHECK(!plainJson.hasProperty("keyChanges") && !plainJson.hasProperty("backing") &&
        !plainJson.hasProperty("vocalGainDb"));
  CHECK(!juce::JSON::toString(plainJson).contains("\"locked\""));

  Project project;
  fill(project);
  project.setProjectKey(8, ScaleMode::Major);
  project.setKeyChanges({{60, 9}, {90, 10}});
  project.setBacking(someBacking());
  project.setVocalGainDb(-3.0f);
  noteAt(project, 70)->setLocked(true);

  const auto text = juce::JSON::toString(ProjectSerializer::toSavedJson(project));
  Project reopened;
  CHECK(ProjectSerializer::fromJson(reopened, juce::JSON::parse(text)));
  CHECK(reopened.getKeyChanges() == project.getKeyChanges());
  CHECK(reopened.getBacking() == project.getBacking());
  CHECK(reopened.getBacking().file.getFileName() == juce::String::fromUTF8("伴奏 檔.wav"));
  CHECK(reopened.getVocalGainDb() == -3.0f);
  CHECK(!noteAt(reopened, 20)->isLocked() && noteAt(reopened, 70)->isLocked());
  CHECK(reopened.keyRootAt(59) == 8 && reopened.keyRootAt(60) == 9 && reopened.keyRootAt(95) == 10);

  // Format 10: a file without the properties, as a build before wrote it.
  auto ten = ProjectSerializer::toSavedJson(plain);
  ten.getDynamicObject()->setProperty("formatVersion", 10);
  Project fromTen;
  CHECK(ProjectSerializer::fromJson(fromTen, juce::JSON::parse(juce::JSON::toString(ten))));
  CHECK(fromTen.getKeyChanges().empty() && !fromTen.getBacking().isSet() &&
        fromTen.getVocalGainDb() == 0.0f);
  for (const auto &note : fromTen.getNotes())
    CHECK(!note.isLocked());

  // A later format is refused, as before.
  ten.getDynamicObject()->setProperty("formatVersion", ProjectSerializer::FORMAT_VERSION + 1);
  Project fromLater;
  CHECK(!ProjectSerializer::fromJson(fromLater, ten));
}

// 12. The plug-in's state carries it too (it keeps the accompaniment with
// the project and does nothing with it).
void thePlugInsState()
{
  Project project;
  fill(project);
  project.getAudioData().waveform.setSize(1, 120 * HOP_SIZE);
  project.getAudioData().waveform.clear();
  project.getAudioData().originalWaveform.makeCopyOf(project.getAudioData().waveform);
  project.setProjectKey(8, ScaleMode::Major);
  project.setKeyChanges({{60, 9}});
  project.setBacking(someBacking());
  project.setVocalGainDb(2.5f);
  noteAt(project, 20)->setLocked(true);

  for (const auto mode : {ProjectSerializer::BinaryArchiveMode::selfContained,
                          ProjectSerializer::BinaryArchiveMode::hostBackedARA})
  {
    juce::MemoryBlock archive;
    CHECK(ProjectSerializer::toBinaryArchive(project, archive, mode));
    Project restored;
    CHECK(ProjectSerializer::fromBinaryArchive(restored, archive.getData(), archive.getSize()));
    CHECK(restored.getKeyChanges() == project.getKeyChanges());
    CHECK(restored.getBacking() == project.getBacking());
    CHECK(restored.getVocalGainDb() == 2.5f);
    CHECK(noteAt(restored, 20)->isLocked() && !noteAt(restored, 70)->isLocked());
  }
}

// 14. A split leaves both pieces locked; a merge with a locked note is locked.
void aSplitAndAMerge()
{
  Project project;
  fill(project);
  PitchUndoManager history;
  NoteSplitter splitter;
  splitter.setProject(&project);
  splitter.setUndoManager(&history);

  noteAt(project, 70)->setLocked(true);
  CHECK(splitter.splitNoteAtFrame(noteAt(project, 70), 85));
  CHECK(noteAt(project, 70) != nullptr && noteAt(project, 85) != nullptr);
  CHECK(noteAt(project, 70)->isLocked() && noteAt(project, 85)->isLocked());
  history.undo();
  CHECK(noteAt(project, 85) == nullptr && noteAt(project, 70)->isLocked());
  CHECK(!noteAt(project, 20)->isLocked());

  // An unlocked note split: neither piece is locked.
  CHECK(splitter.splitNoteAtFrame(noteAt(project, 20), 35));
  CHECK(!noteAt(project, 20)->isLocked() && !noteAt(project, 35)->isLocked());

  // Merged with a locked one, either way round.
  noteAt(project, 35)->setLocked(true);
  CHECK(splitter.mergeNotes(noteAt(project, 20), noteAt(project, 35)));
  CHECK(noteAt(project, 35) == nullptr && noteAt(project, 20)->isLocked());
  history.undo();
  CHECK(!noteAt(project, 20)->isLocked() && noteAt(project, 35)->isLocked());
  noteAt(project, 35)->setLocked(false);
  noteAt(project, 20)->setLocked(true);
  CHECK(splitter.mergeNotes(noteAt(project, 20), noteAt(project, 35)));
  CHECK(noteAt(project, 20)->isLocked());
  history.undo();
  noteAt(project, 20)->setLocked(false);
  CHECK(splitter.mergeNotes(noteAt(project, 20), noteAt(project, 35)));
  CHECK(!noteAt(project, 20)->isLocked());
}

// 15. A note snaps to the key it was recorded in, wherever the timeline has
// taken it: C major up to frame 60 of the recording, G major from there.
void aNoteSnapsToItsOwnKey()
{
  Project project;
  fill(project);
  project.setProjectKey(0, ScaleMode::Major);
  project.setKeyChanges({{60, 7}});
  Note *inC = noteAt(project, 20);
  Note *inG = noteAt(project, 70);
  CHECK(project.keyRootFor(*inC) == 0 && project.keyRootFor(*inG) == 7);

  PitchEditor editor;
  editor.setProject(&project);
  editor.setDragSnapMode(DragSnapMode::Scale);
  // From F (65), dragged up 0.6 of a semitone: F# is not of C major, so the
  // note in C stays on F; it is of G major, where F is not.
  CHECK(editor.getSnappedDragOffset(0.6f, 65.0f, inC) == 0.0f);
  CHECK(editor.getSnappedDragOffset(0.6f, 65.0f, inG) == 1.0f);
  // Down 0.6: E is of both.
  CHECK(editor.getSnappedDragOffset(-0.6f, 65.0f, inC) == -1.0f);
  CHECK(editor.getSnappedDragOffset(-0.6f, 65.0f, inG) == -1.0f);
  // With no note given, the project's key (as before there were changes).
  CHECK(editor.getSnappedDragOffset(0.6f, 65.0f) == 0.0f);

  // The timeline moved the second note in front of the first (a stretch):
  // its recorded frames, and so its key, are what they were.
  inG->setStartFrame(2);
  inG->setEndFrame(18);
  CHECK(inG->getSrcStartFrame() == 70 && inG->getSrcEndFrame() == 100);
  CHECK(project.keyRootFor(*inG) == 7);
  CHECK(editor.getSnappedDragOffset(0.6f, 65.0f, inG) == 1.0f);

  // The scale a pitch snaps to (the double click, the Pitch Center macro).
  CHECK(ScaleUtils::snapMidiToScale(65.6f, ScaleMode::Major, project.keyRootFor(*inC), 440) == 65.0f);
  CHECK(ScaleUtils::snapMidiToScale(65.6f, ScaleMode::Major, project.keyRootFor(*inG), 440) == 66.0f);
}

// 16. The key in the history: one step each for a change added, moved, set
// to another key and removed; and the key the user confirmed, always major.
void theKeyInTheHistory()
{
  Project project;
  fill(project);
  // As the analysis may leave it: D minor (the notes rounded and counted).
  project.setProjectKey(2, ScaleMode::Minor);
  PitchUndoManager history;
  const auto step = [&](KeyState after, const char *name)
  {
    auto before = KeyState::of(project);
    CHECK(before != after);
    after.applyTo(project);
    history.addAction(std::make_unique<KeyChangeAction>(project, std::move(before), after, name));
  };
  const auto withChanges = [&](std::vector<KeyChange> changes)
  {
    auto state = KeyState::of(project);
    state.changes = std::move(changes);
    return state;
  };

  // Confirmed by ear: C, with a change to D at frame 60.
  step(KeyState::confirmed(project, 0, {{60, 2}}), "confirm");
  CHECK(project.getScaleRootNote() == 0 && project.getPreferredScaleMode() == ScaleMode::Major);
  CHECK(project.getUseProjectKey());
  CHECK(project.keyRootAt(10) == 0 && project.keyRootAt(60) == 2);

  step(withChanges({{60, 2}, {90, 4}}), "add");
  CHECK(project.getKeyChanges().size() == 2 && project.keyRootAt(95) == 4);
  step(withChanges({{60, 2}, {80, 4}}), "move");
  CHECK(project.keyRootAt(85) == 4 && project.keyRootAt(79) == 2);
  step(withChanges({{60, 2}, {80, 9}}), "another key");
  CHECK(project.keyRootAt(85) == 9);
  step(withChanges({{80, 9}}), "remove");
  CHECK(project.keyRootAt(60) == 0 && project.keyRootAt(80) == 9);

  history.undo(); // the removal
  CHECK(project.getKeyChanges().size() == 2 && project.keyRootAt(60) == 2);
  history.undo(); // the other key
  CHECK(project.keyRootAt(85) == 4);
  history.undo(); // the move
  CHECK(project.keyRootAt(85) == 2 && project.keyRootAt(95) == 4);
  history.undo(); // the addition
  CHECK(project.getKeyChanges().size() == 1);
  history.undo(); // the confirmation: D minor again, no change
  CHECK(project.getKeyChanges().empty());
  CHECK(project.getScaleRootNote() == 2 && project.getPreferredScaleMode() == ScaleMode::Minor);
  history.redo();
  CHECK(project.getScaleRootNote() == 0 && project.getPreferredScaleMode() == ScaleMode::Major);
  CHECK(project.getKeyChanges().size() == 1 && project.keyRootAt(60) == 2);

  // A region reading its own key is given it on that level.
  Project region;
  fill(region);
  region.setProjectKey(5, ScaleMode::Major);
  region.setScaleRootNote(9);
  region.setPreferredScaleMode(ScaleMode::Minor);
  CHECK(!region.getUseProjectKey());
  KeyState::confirmed(region, 4, {}).applyTo(region);
  CHECK(!region.getUseProjectKey());
  CHECK(region.getScaleRootNote() == 4 && region.getPreferredScaleMode() == ScaleMode::Major);
  CHECK(region.getProjectKeyRoot() == 5);

  KeyChangeAction action(project, KeyState::of(project), KeyState::of(project), "none");
  CHECK(!action.requiresAudioResynthesis());
}

// 17. Notes locked and unlocked as one step.
void theLockInTheHistory()
{
  Project project;
  fill(project);
  PitchUndoManager history;
  Note *first = noteAt(project, 20);
  Note *second = noteAt(project, 70);
  second->setLocked(true);
  const auto press = [&](std::vector<Note *> notes)
  {
    const bool lock = LockAction::wanted(notes);
    auto action = std::make_unique<LockAction>(notes, lock, "lock");
    action->redo();
    history.addAction(std::move(action));
    return lock;
  };

  // One of two locked: a press locks both.
  CHECK(press({first, second}));
  CHECK(first->isLocked() && second->isLocked());
  // Both locked: a press unlocks both.
  CHECK(!press({first, second}));
  CHECK(!first->isLocked() && !second->isLocked());
  history.undo();
  CHECK(first->isLocked() && second->isLocked());
  history.undo(); // each as it was
  CHECK(!first->isLocked() && second->isLocked());
  history.redo();
  CHECK(first->isLocked() && second->isLocked());

  // The step finds its notes after the list has moved them.
  history.undo();
  project.getNotes().insert(project.getNotes().begin(), Note(2, 12, 60.0f));
  history.redo();
  CHECK(noteAt(project, 20)->isLocked() && noteAt(project, 70)->isLocked());
  CHECK(!noteAt(project, 2)->isLocked());

  LockAction action({}, true, "none");
  CHECK(!action.requiresAudioResynthesis());
}
} // namespace

int main()
{
  juce::ScopedJuceInitialiser_GUI juce;
  theKeyAtAFrame();
  theFile();
  thePlugInsState();
  aSplitAndAMerge();
  aNoteSnapsToItsOwnKey();
  theKeyInTheHistory();
  theLockInTheHistory();
  std::printf("KeyChangeTests: all passed\n");
  return 0;
}
