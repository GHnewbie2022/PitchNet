// Spec Q12: a note is cut where the vocal fry a word starts with begins, so
// that the fry it keeps as recorded starts the note (FryNoteCuts.h) - as a
// project is read, as its recording is read, and by the call itself.
#include "TestAssert.h"

#include "../Source/Audio/NoteTiming.h"
#include "../Source/Audio/RecordingChange.h"
#include "../Source/Models/ProjectSerializer.h"
#include "../Source/UI/PianoRoll/NoteSplitter.h"
#include "../Source/Utils/Constants.h"
#include "../Source/Utils/FryDetector.h"
#include "../Source/Utils/FryNoteCuts.h"
#include "../Source/Utils/MelSpectrogram.h"
#include "../Source/Utils/PitchCurveProcessor.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
constexpr double kPi = 3.14159265358979323846;

// A voice at 250 Hz on [20, 40), ten unvoiced frames, a low run of two
// frames at 110 Hz and one frame part-way at 150 Hz, then the voice at
// 220 Hz up to frame 100, and nothing after. What the detector keeps for
// the word starting in fry: the 6 unvoiced frames leading in, the run and
// the frame part-way - [44, 53).
std::vector<float> curveOf(int frames)
{
  std::vector<float> curve(static_cast<size_t>(frames), 0.0f);
  const auto put = [&](int from, int to, float hz) {
    for (int i = from; i < to && i < frames; ++i)
      curve[static_cast<size_t>(i)] = hz;
  };
  put(20, 40, 250.0f);
  put(50, 52, 110.0f);
  put(52, 53, 150.0f);
  put(53, 100, 220.0f);
  return curve;
}

// The pitch curves of a project of `frames` frames over that voice, and two
// notes: [20, 40) and [second, 100). The second starts in the unvoiced
// frames before the fry, as the analysis placed the two in 煙幕.
void lay(Project &project, int frames, int second = 42)
{
  auto &audio = project.getAudioData();
  audio.sampleRate = SAMPLE_RATE;
  if (audio.getNumFrames() != frames)
    audio.melSpectrogram.assign(static_cast<size_t>(frames), std::vector<float>(NUM_MELS, 0.0f));
  audio.rawF0 = curveOf(frames);
  audio.cleanedF0 = audio.rawF0;
  // The curve the notes are fitted to has no holes: an unvoiced frame takes
  // the next voiced one.
  audio.denseF0 = audio.rawF0;
  float next = 220.0f;
  for (int i = frames - 1; i >= 0; --i)
  {
    if (audio.rawF0[static_cast<size_t>(i)] > 0.0f)
      next = audio.rawF0[static_cast<size_t>(i)];
    else
      audio.denseF0[static_cast<size_t>(i)] = next;
  }
  audio.f0 = audio.denseF0;
  audio.voicedMask.assign(static_cast<size_t>(frames), false);
  for (int i = 0; i < frames; ++i)
    audio.voicedMask[static_cast<size_t>(i)] = audio.rawF0[static_cast<size_t>(i)] > 0.0f;
  audio.vadMask = audio.voicedMask;
  audio.warpMap.setLength(static_cast<double>(frames));
  // As an analysis leaves them: each note with a curve of its own length,
  // which the refit then lays on the recorded pitch.
  Note first(20, 40, freqToMidi(250.0f));
  first.setOriginalDeltaPitch(std::vector<float>(20, 0.0f));
  Note word(second, 100, freqToMidi(220.0f));
  word.setOriginalDeltaPitch(std::vector<float>(static_cast<size_t>(100 - second), 0.0f));
  project.addNote(first);
  project.addNote(word);
  PitchCurveProcessor::rebuildBaseFromNotes(project);
  PitchCurveProcessor::refitUntouchedCurves(project, 0, frames);
}

const Note *noteFrom(const Project &project, int sourceStart)
{
  for (const auto &note : project.getNotes())
    if (!note.isRest() && note.getSrcStartFrame() == sourceStart)
      return &note;
  return nullptr;
}

Note *noteFrom(Project &project, int sourceStart)
{
  return const_cast<Note *>(noteFrom(static_cast<const Project &>(project), sourceStart));
}

int notesIn(const Project &project)
{
  int count = 0;
  for (const auto &note : project.getNotes())
    count += note.isRest() ? 0 : 1;
  return count;
}

// What plays at each frame: the base with the notes' curves on it.
std::vector<float> sung(const Project &project, int from, int to)
{
  const auto &audio = project.getAudioData();
  std::vector<float> pitch;
  for (int i = from; i < to; ++i)
    pitch.push_back(audio.basePitch[static_cast<size_t>(i)] +
                    audio.deltaPitch[static_cast<size_t>(i)]);
  return pitch;
}

// The fixture is what it says: one onset, [44, 53), in the middle of the
// second note - whose squares therefore stand in its corners.
void theFryIsInTheMiddleOfTheNote()
{
  Project project;
  lay(project, 140);
  const auto &audio = project.getAudioData();
  const auto onsets = fry::onsets(audio.rawF0);
  CHECK(onsets.size() == 1);
  CHECK(onsets[0].first == 44 && onsets[0].second == 53);
  const auto wanted = frycut::wanted(project);
  CHECK(wanted.size() == 1);
  CHECK(wanted[0].noteStart == 42 && wanted[0].at == 44);
  const auto shown = fry::shownRange(*noteFrom(project, 42), project.getNotes(), audio.rawF0,
                                     fry::find(audio));
  CHECK(shown.headUntil == 42 && shown.tailFrom == 100);
}

// Cut there: the frames before the fry are a note of their own, and the
// note that holds the fry starts with it - its left square at the end of
// what is kept. A second call finds nothing to do.
void theNoteIsCutWhereTheFryStarts()
{
  Project project;
  lay(project, 140);
  const auto base = project.getAudioData().basePitch;
  const auto delta = project.getAudioData().deltaPitch;
  const auto line = project.getAudioData().f0;
  const Note other = *noteFrom(project, 20);
  CHECK(frycut::apply(project) == 1);
  CHECK(notesIn(project) == 3);
  // The pitch curves are the very values they were, and so is the note that
  // was not cut.
  CHECK(project.getAudioData().basePitch == base);
  CHECK(project.getAudioData().deltaPitch == delta);
  CHECK(project.getAudioData().f0 == line);
  CHECK(noteFrom(project, 20)->getUid() == other.getUid());
  CHECK(noteFrom(project, 20)->getOriginalDeltaPitch() == other.getOriginalDeltaPitch());
  CHECK(noteFrom(project, 20)->getDeltaPitch() == other.getDeltaPitch());
  const auto *lead = noteFrom(project, 42);
  const auto *word = noteFrom(project, 44);
  CHECK(lead != nullptr && word != nullptr);
  CHECK(lead->getSrcEndFrame() == 44 && lead->getEndFrame() == 44);
  CHECK(word->getStartFrame() == 44 && word->getSrcEndFrame() == 100);
  CHECK(noteFrom(project, 20)->getSrcEndFrame() == 40);
  // Both pieces stay on the note's pitch. Centred on its own voiced frames
  // (as a split by hand is), the piece that starts in fry would sit 0.6
  // semitones under the voice here, and 2 under it in 煙幕 at 230.3 s.
  const float voice = freqToMidi(220.0f);
  CHECK(std::abs(lead->getMidiNote() - voice) < 1.0e-3f);
  CHECK(std::abs(word->getMidiNote() - voice) < 1.0e-3f);
  CHECK(std::abs(word->getOriginalMidiNote() - voice) < 1.0e-3f);

  const auto &audio = project.getAudioData();
  const auto found = fry::find(audio);
  const auto shown = fry::shownRange(*word, project.getNotes(), audio.rawF0, found);
  CHECK(shown.headUntil == 53 && shown.tailFrom == 100);
  // The detector keeps the same frames as before the cut.
  const auto kept = fry::keptAsRecorded(project.getNotes(), audio.rawF0, found, true).all;
  CHECK(kept.size() == 1 && kept[0].first == 44 && kept[0].second == 53);

  CHECK(frycut::wanted(project).empty());
  CHECK(frycut::apply(project) == 0);
  CHECK(notesIn(project) == 3);
}

// A file can hold a pitch curve between its notes that the notes do not lay
// out (a format 6 file of 煙幕: a deviation on 5088 such frames, up to 7
// semitones, where a rebuild leaves none). The cut is held against what the
// notes lay out, not against those values - against them it looked like a
// change of pitch, nothing was put back, and the rest of the song came out a
// digit off in the fourth place - and the project opens with the curves it
// was read with, value for value.
void aCurveBetweenTheNotesStaysAsRead()
{
  Project project;
  lay(project, 140);
  auto &audio = project.getAudioData();
  for (int i = 0; i < 140; ++i)
    if (i < 20 || (i >= 40 && i < 42) || i >= 100)
    {
      audio.deltaPitch[static_cast<size_t>(i)] = 3.0f;
      audio.f0[static_cast<size_t>(i)] = 300.0f;
    }
  const auto base = audio.basePitch;
  const auto delta = audio.deltaPitch;
  const auto line = audio.f0;
  const auto baseLine = audio.baseF0;
  const Note other = *noteFrom(project, 20);
  CHECK(frycut::apply(project) == 1);
  CHECK(notesIn(project) == 3);
  CHECK(audio.basePitch == base);
  CHECK(audio.deltaPitch == delta);
  CHECK(audio.f0 == line);
  CHECK(audio.baseF0 == baseLine);
  CHECK(noteFrom(project, 20)->getOriginalDeltaPitch() == other.getOriginalDeltaPitch());
  CHECK(noteFrom(project, 20)->getDeltaPitch() == other.getDeltaPitch());
  CHECK(noteFrom(project, 44) != nullptr && noteFrom(project, 42)->getSrcEndFrame() == 44);
}

// What a project is given once its frames are known is one call wherever it
// happens (recordingchange::settle): the standalone's hydrate() ends with
// it, and the plug-in makes it when it gives a restored project its host's
// audio. It cuts the notes - unless told to leave the note list alone, for
// a project a piano roll is showing; a later call then cuts them.
void settlingAProjectCutsItsNotesWhenAsked()
{
  Project project;
  lay(project, 140);
  CHECK(recordingchange::settle(project, false) == 0);
  CHECK(notesIn(project) == 2);
  CHECK(frycut::wanted(project).size() == 1);
  // It tells how many it cut: the plug-in forgets the region's undo history
  // then, whose steps hold the note whole.
  CHECK(recordingchange::settle(project, true) == 1);
  CHECK(notesIn(project) == 3);
  CHECK(noteFrom(project, 44) != nullptr);
  CHECK(recordingchange::settle(project, true) == 0);
  CHECK(notesIn(project) == 3);
}

// An edited note keeps its edit on both pieces, and what it sang: moved a
// semitone up before the cut, the line over the voice is where it was.
void theEditStaysOnBothPieces()
{
  Project project;
  lay(project, 140);
  auto *note = noteFrom(project, 42);
  note->setMidiNote(note->getMidiNote() + 1.0f);
  note->markDirty();
  note->setRenderedEdit(true);
  note->setSynthEngine(Note::SynthEngine::RubberBand);
  PitchCurveProcessor::rebuildBaseFromNotes(project);
  const auto before = sung(project, 53, 100);

  CHECK(frycut::apply(project) == 1);
  const auto after = sung(project, 53, 100);
  float furthest = 0.0f;
  for (size_t i = 0; i < before.size(); ++i)
    furthest = std::max(furthest, std::abs(after[i] - before[i]));
  std::printf("  the sung line over the voice moved by %.4f semitones at most\n", furthest);
  CHECK(furthest < 0.05f);
  for (const int start : {42, 44})
  {
    const auto *piece = noteFrom(project, start);
    CHECK(piece != nullptr);
    CHECK(std::abs(piece->getAdjustedMidiNote() - piece->getOriginalMidiNote() - 1.0f) < 1.0e-3f);
    CHECK(piece->getSynthEngine() == Note::SynthEngine::RubberBand);
    CHECK(piece->hasRenderedEdit());
  }
}

// Nothing to cut where the kept frames already start the note, where they
// run to its end, or where they lie over the edge of two notes.
void aNoteThatStartsWithItsFryIsLeft()
{
  Project starts;
  lay(starts, 140, 44);
  CHECK(frycut::wanted(starts).empty());
  CHECK(frycut::apply(starts) == 0 && notesIn(starts) == 2);

  Project inTheLead; // the note starts inside the kept frames
  lay(inTheLead, 140, 47);
  CHECK(frycut::apply(inTheLead) == 0 && notesIn(inTheLead) == 2);

  Project toTheEnd; // [42, 53): the kept frames are the note's end
  lay(toTheEnd, 140);
  auto *note = noteFrom(toTheEnd, 42);
  note->setEndFrame(53);
  note->setSrcEndFrame(53);
  PitchCurveProcessor::rebuildBaseFromNotes(toTheEnd);
  CHECK(frycut::apply(toTheEnd) == 0 && notesIn(toTheEnd) == 2);
}

// A note as the analysis made it gives two analysed notes: there is no
// separation to restore on either. One the user had merged from two keeps
// the account of them it had.
void thePiecesAreAnalysedNotes()
{
  Project project;
  lay(project, 140);
  auto *note = noteFrom(project, 42);
  note->setAnalysisSpans({{42, 100, note->getMidiNote(), note->getOriginalDeltaPitch()}});
  CHECK(frycut::apply(project) == 1);
  NoteSplitter splitter;
  splitter.setProject(&project);
  for (const int start : {42, 44})
  {
    const auto *piece = noteFrom(project, start);
    CHECK(piece->getAnalysisSpans().size() == 1);
    CHECK(piece->getAnalysisSpans()[0].first == piece->getSrcStartFrame());
    CHECK(piece->getAnalysisSpans()[0].second == piece->getSrcEndFrame());
    CHECK(piece->exactAnalysisSpan() != nullptr);
    CHECK(!splitter.canRestoreSeparation(*piece));
  }

  Project merged;
  lay(merged, 140);
  auto *two = noteFrom(merged, 42);
  two->setAnalysisSpans({{42, 70, 57.0f, {}}, {70, 100, 57.0f, {}}});
  CHECK(frycut::apply(merged) == 1);
  NoteSplitter other;
  other.setProject(&merged);
  CHECK(noteFrom(merged, 44)->getAnalysisSpans().size() == 2);
  CHECK(other.canRestoreSeparation(*noteFrom(merged, 44)));
}

// A stretched note is cut at the same sound: at the recorded frame the fry
// starts at, wherever the stretch put that on the timeline. Stretched to
// 1.24 times its length, and squeezed to 0.79 - where the timeline frame of
// the cut reads back as the recorded frame after it, and the pieces' edges
// are set to the frame itself.
void aStretchedNoteIsCutAtTheSameSound()
{
  for (const double end : {114.0, 88.0})
  {
    Project project;
    auto &audio = project.getAudioData();
    const int timeline = 140 + static_cast<int>(end) - 100;
    audio.sampleRate = SAMPLE_RATE;
    audio.melSpectrogram.assign(static_cast<size_t>(timeline), std::vector<float>(NUM_MELS, 0.0f));
    // The detector's curves are over the recording: as lay() fills them.
    const int recorded = 160;
    audio.rawF0 = curveOf(recorded);
    audio.cleanedF0 = audio.rawF0;
    audio.denseF0 = audio.rawF0;
    float next = 220.0f;
    for (int i = recorded - 1; i >= 0; --i)
    {
      if (audio.rawF0[static_cast<size_t>(i)] > 0.0f)
        next = audio.rawF0[static_cast<size_t>(i)];
      else
        audio.denseF0[static_cast<size_t>(i)] = next;
    }
    audio.f0 = audio.denseF0;
    audio.voicedMask.assign(static_cast<size_t>(recorded), false);
    for (int i = 0; i < recorded; ++i)
      audio.voicedMask[static_cast<size_t>(i)] = audio.rawF0[static_cast<size_t>(i)] > 0.0f;
    audio.warpMap.setLength(static_cast<double>(timeline));
    audio.warpMap.setAnchors({{42.0, 42.0}, {100.0, end}});

    project.addNote(Note(20, 40, freqToMidi(250.0f)));
    Note stretched(42, static_cast<int>(end), freqToMidi(220.0f));
    stretched.setSrcStartFrame(42);
    stretched.setSrcEndFrame(100);
    project.addNote(stretched);
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    // Each note's analysed curve over its recorded frames, against the base
    // where the stretch plays them.
    const int last = static_cast<int>(audio.basePitch.size()) - 1;
    for (auto &note : project.getNotes())
    {
      const auto map = notetiming::forNote(audio, note);
      std::vector<float> curve;
      for (int s = note.getSrcStartFrame(); s < note.getSrcEndFrame(); ++s)
      {
        const double at = std::clamp(map.destinationAt(s + 0.5) - 0.5, 0.0, static_cast<double>(last));
        const auto low = static_cast<size_t>(std::floor(at));
        const auto high = std::min(low + 1, static_cast<size_t>(last));
        const float share = static_cast<float>(at - static_cast<double>(low));
        curve.push_back(freqToMidi(audio.denseF0[static_cast<size_t>(s)]) -
                        (audio.basePitch[low] * (1.0f - share) + audio.basePitch[high] * share));
      }
      note.setOriginalDeltaPitch(std::move(curve));
    }
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    PitchCurveProcessor::refitUntouchedCurves(project, 0, timeline);

    const auto wanted = frycut::wanted(project);
    CHECK(wanted.size() == 1 && wanted[0].noteStart == 42 && wanted[0].at == 44);
    const int at = static_cast<int>(std::lround(audio.warpMap.destinationAt(44.0)));
    CHECK(frycut::apply(project) == 1);
    const auto *lead = noteFrom(project, 42);
    const auto *word = noteFrom(project, 44);
    CHECK(lead != nullptr && word != nullptr);
    CHECK(lead->getSrcEndFrame() == 44 && lead->getEndFrame() == at);
    CHECK(word->getStartFrame() == at && word->getSrcEndFrame() == 100);
    CHECK(word->getEndFrame() == static_cast<int>(end));
    const auto shown = fry::shownRange(*word, project.getNotes(), audio.rawF0, fry::find(audio));
    CHECK(shown.headUntil == 53 && shown.tailFrom == 100);
    CHECK(frycut::wanted(project).empty());
    std::printf("  stretched to %.2fx: cut at recorded frame 44, timeline frame %d\n",
                (end - 42.0) / 58.0, at);
  }
}

// Before a project's frames are known its curves cannot be rebuilt (a linked
// project between the file and its recording): nothing is touched.
void theCutWaitsForTheFrames()
{
  Project project;
  lay(project, 140);
  auto &audio = project.getAudioData();
  const auto base = audio.basePitch;
  audio.melSpectrogram.clear();
  CHECK(audio.getNumFrames() == 0);
  CHECK(frycut::apply(project) == 0);
  CHECK(notesIn(project) == 2);
  CHECK(audio.basePitch == base);
}

// The plug-in's state: read back with its audio, the project is cut.
void aStateReadBackIsCut()
{
  // The frames a mel of this many samples has, so that the curves fit it.
  const int samples = 140 * HOP_SIZE;
  juce::AudioBuffer<float> voice(1, samples);
  for (int i = 0; i < samples; ++i)
    voice.setSample(0, i, static_cast<float>(0.2 * std::sin(2 * kPi * 220.0 * i / SAMPLE_RATE)));
  MelSpectrogram mel(SAMPLE_RATE, N_FFT, HOP_SIZE, NUM_MELS, FMIN, FMAX);
  const int frames = static_cast<int>(mel.compute(voice.getReadPointer(0), samples).size());
  CHECK(frames >= 110);

  Project project;
  auto &audio = project.getAudioData();
  audio.originalWaveform.makeCopyOf(voice);
  audio.waveform.makeCopyOf(voice);
  audio.melSpectrogram = mel.compute(voice.getReadPointer(0), samples);
  lay(project, frames);
  CHECK(notesIn(project) == 2);

  juce::MemoryBlock state;
  CHECK(ProjectSerializer::toBinaryArchive(project, state));
  Project restored;
  CHECK(ProjectSerializer::fromBinaryArchive(restored, state.getData(), state.getSize()));
  CHECK(restored.getAudioData().getNumFrames() == frames);
  CHECK(notesIn(restored) == 3);
  CHECK(noteFrom(restored, 44) != nullptr && noteFrom(restored, 42)->getSrcEndFrame() == 44);
  CHECK(!restored.isModified());

  // ...and as JSON with its analysis (the plug-in's editor and processor
  // hand the project to each other so).
  Project handed;
  CHECK(ProjectSerializer::fromJson(
      handed, juce::JSON::parse(juce::JSON::toString(ProjectSerializer::toJson(project, true, true)))));
  if (handed.getAudioData().getNumFrames() > 0)
    CHECK(notesIn(handed) == 3);
}

// A project that links its recording: read from its file it has no frames
// yet and is left as saved; the recording read, it is cut.
void aLinkedProjectIsCutWhenItsRecordingIsRead()
{
  constexpr int kSourceRate = 48000;
  const int sourceSamples = 2 * kSourceRate;
  juce::AudioBuffer<float> recording(1, sourceSamples);
  for (int i = 0; i < sourceSamples; ++i)
    recording.setSample(0, i, static_cast<float>(0.3 * std::sin(2 * kPi * 220.0 * i / kSourceRate)));

  Project project;
  auto &audio = project.getAudioData();
  audio.sourceAudio.makeCopyOf(recording);
  audio.sourceSampleRate = kSourceRate;
  project.setLinksRecording(true);
  CHECK(recordingchange::hydrate(project));
  const int frames = audio.getNumFrames();
  CHECK(frames >= 110);
  lay(project, frames);
  project.setRecordingFingerprint(recordingchange::fingerprint(recording, kSourceRate));
  CHECK(notesIn(project) == 2);

  const auto saved = juce::JSON::toString(ProjectSerializer::toSavedJson(project));
  Project reopened;
  CHECK(ProjectSerializer::fromJson(reopened, juce::JSON::parse(saved)));
  CHECK(reopened.linksRecording());
  CHECK(reopened.getAudioData().getNumFrames() == 0);
  CHECK(notesIn(reopened) == 2);
  const auto baseAsSaved = reopened.getAudioData().basePitch;
  CHECK(!baseAsSaved.empty());

  reopened.getAudioData().sourceAudio.makeCopyOf(recording);
  reopened.getAudioData().sourceSampleRate = kSourceRate;
  CHECK(recordingchange::hydrate(reopened));
  CHECK(notesIn(reopened) == 3);
  CHECK(noteFrom(reopened, 44) != nullptr && noteFrom(reopened, 42)->getSrcEndFrame() == 44);
  CHECK(reopened.getAudioData().basePitch.size() == baseAsSaved.size());

  // Saved cut, it opens cut and is not cut again.
  const auto savedCut = juce::JSON::toString(ProjectSerializer::toSavedJson(reopened));
  Project again;
  CHECK(ProjectSerializer::fromJson(again, juce::JSON::parse(savedCut)));
  CHECK(notesIn(again) == 3);
  again.getAudioData().sourceAudio.makeCopyOf(recording);
  again.getAudioData().sourceSampleRate = kSourceRate;
  CHECK(recordingchange::hydrate(again));
  CHECK(notesIn(again) == 3);
}
} // namespace

int main()
{
  theFryIsInTheMiddleOfTheNote();
  theNoteIsCutWhereTheFryStarts();
  aCurveBetweenTheNotesStaysAsRead();
  settlingAProjectCutsItsNotesWhenAsked();
  theEditStaysOnBothPieces();
  aNoteThatStartsWithItsFryIsLeft();
  thePiecesAreAnalysedNotes();
  aStretchedNoteIsCutAtTheSameSound();
  theCutWaitsForTheFrames();
  aStateReadBackIsCut();
  aLinkedProjectIsCutWhenItsRecordingIsRead();
  std::printf("FryNoteCutsTests passed\n");
  return 0;
}
