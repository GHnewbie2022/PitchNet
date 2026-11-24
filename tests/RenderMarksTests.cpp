// The plug-in against the standalone (spec, section 11): what the plug-in
// does on its own paths is done with the calls tested here.
//   - a render made on a copy clears the marks it answered on the project
//     shown, as the standalone's render clears them (RenderMarks.h);
//   - the host's audio is made mono as a file's is (recordingchange::monoOf).
#include "TestAssert.h"

#include "../Source/Audio/RecordingChange.h"
#include "../Source/Models/ProjectSerializer.h"
#include "../Source/Models/RenderMarks.h"
#include "../Source/Utils/Constants.h"

#include <cstdio>
#include <vector>

namespace
{
Note *noteAt(Project &project, int start)
{
  for (auto &note : project.getNotes())
    if (note.getStartFrame() == start)
      return &note;
  return nullptr;
}

// Three notes; the first is edited and marked, as an edit in the App marks
// it (the note, and the range of curve around it).
void lay(Project &project)
{
  // The marked range of curve is held inside the project's frames.
  project.getAudioData().f0.assign(200, 220.0f);
  for (const int start : {10, 40, 70})
  {
    Note note(start, start + 20, 60.0f);
    note.setOriginalDeltaPitch(std::vector<float>(20, 0.0f));
    project.addNote(note);
  }
  auto *edited = noteAt(project, 10);
  edited->setMidiNote(61.0f);
  edited->markDirty();
  project.markNoteEditDirtyRange(10, 30);
}

// The copy as the render leaves it: every mark cleared (Project::clearAllDirty).
Project renderedCopyOf(const Project &shown)
{
  Project copy(shown);
  copy.clearAllDirty();
  return copy;
}

// A render lands with nothing changed since it was asked for: the marks it
// answered are gone, and the next render has nothing left over to take.
void theMarksARenderAnsweredAreCleared()
{
  Project shown;
  lay(shown);
  CHECK(shown.hasDirtyNotes() && shown.hasF0DirtyRange());
  const auto marks = rendermarks::of(shown);
  CHECK(marks.notes.size() == 1 && marks.hasCurveRange);
  const Project rendered = renderedCopyOf(shown);
  rendermarks::clearAnswered(shown, rendered, marks);
  CHECK(!shown.hasDirtyNotes());
  CHECK(!shown.hasF0DirtyRange());
}

// The note is moved again while its render runs: what landed is not what it
// is now, and its mark stays for the render the new edit asks for.
void aNoteEditedAgainKeepsItsMark()
{
  Project shown;
  lay(shown);
  const auto marks = rendermarks::of(shown);
  const Project rendered = renderedCopyOf(shown);
  noteAt(shown, 10)->setMidiNote(62.0f);
  rendermarks::clearAnswered(shown, rendered, marks);
  CHECK(noteAt(shown, 10)->isDirty());
}

// Another note is edited while the render runs: it was not in what was asked
// for, so it stays marked, and so does the range of curve that grew with it.
void anEditMadeMeanwhileKeepsItsMarks()
{
  Project shown;
  lay(shown);
  const auto marks = rendermarks::of(shown);
  const Project rendered = renderedCopyOf(shown);
  auto *later = noteAt(shown, 70);
  later->setMidiNote(59.0f);
  later->markDirty();
  shown.markNoteEditDirtyRange(70, 90);
  rendermarks::clearAnswered(shown, rendered, marks);
  CHECK(!noteAt(shown, 10)->isDirty());
  CHECK(noteAt(shown, 70)->isDirty());
  CHECK(shown.hasF0DirtyRange());
}

// Notes compare by what a project file holds of them, not by their marks.
void notesCompareAsTheyAreSaved()
{
  Project project;
  lay(project);
  Note same = *noteAt(project, 10);
  same.clearDirty();
  CHECK(ProjectSerializer::writtenAlike(same, *noteAt(project, 10)));
  Note moved = same;
  moved.setTiltRight(0.5f);
  CHECK(!ProjectSerializer::writtenAlike(moved, same));
  Note kept = same;
  kept.setKeptRange(14, -1);
  CHECK(!ProjectSerializer::writtenAlike(kept, same));
}

// One channel is itself; two are averaged, sample for sample, and two that
// are the same give that channel exactly (a mono file a host hands over as
// stereo is analysed as the file is).
void theHostsAudioIsMadeMonoAsAFileIs()
{
  juce::AudioBuffer<float> one(1, 4);
  const float values[4] = {0.25f, -0.5f, 0.125f, 1.0f};
  for (int i = 0; i < 4; ++i)
    one.setSample(0, i, values[i]);
  const auto mono = recordingchange::monoOf(one);
  CHECK(mono.getNumChannels() == 1 && mono.getNumSamples() == 4);
  for (int i = 0; i < 4; ++i)
    CHECK(mono.getSample(0, i) == values[i]);

  juce::AudioBuffer<float> two(2, 4);
  for (int i = 0; i < 4; ++i)
  {
    two.setSample(0, i, values[i]);
    two.setSample(1, i, -values[3 - i]);
  }
  const auto mid = recordingchange::monoOf(two);
  CHECK(mid.getNumChannels() == 1 && mid.getNumSamples() == 4);
  for (int i = 0; i < 4; ++i)
    CHECK(mid.getSample(0, i) == (values[i] - values[3 - i]) * 0.5f);
  // ...which the left channel alone is not.
  CHECK(mid.getSample(0, 0) != values[0]);

  for (int i = 0; i < 4; ++i)
    two.setSample(1, i, values[i]);
  const auto alike = recordingchange::monoOf(two);
  for (int i = 0; i < 4; ++i)
    CHECK(alike.getSample(0, i) == values[i]);

  // The analysis copy is that mono, converted.
  const auto copy = recordingchange::analysisCopy(two, SAMPLE_RATE);
  CHECK(copy.getNumSamples() == 4);
  for (int i = 0; i < 4; ++i)
    CHECK(copy.getSample(0, i) == values[i]);
}
} // namespace

int main()
{
  theMarksARenderAnsweredAreCleared();
  aNoteEditedAgainKeepsItsMark();
  anEditMadeMeanwhileKeepsItsMarks();
  notesCompareAsTheyAreSaved();
  theHostsAudioIsMadeMonoAsAFileIs();
  std::printf("RenderMarksTests passed\n");
  return 0;
}
