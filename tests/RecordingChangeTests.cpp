// Spec Q11: a project links its recording; when the recording is edited
// outside (RX), the changed blocks are found, analysed again, and the edited
// notes there are queued for rendering (RecordingChange.h).
#include "TestAssert.h"

#include "../Source/Audio/RecordingChange.h"
#include "../Source/Audio/SourceAudio.h"
#include "../Source/Audio/WarpLayer.h"
#include "../Source/Models/ProjectSerializer.h"
#include "../Source/Utils/Constants.h"
#include "../Source/Utils/PitchCurveProcessor.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr int kSourceRate = 48000;

juce::AudioBuffer<float> recordingOf(double seconds)
{
  const int samples = static_cast<int>(seconds * kSourceRate);
  juce::AudioBuffer<float> source(2, samples);
  for (int i = 0; i < samples; ++i)
  {
    const double t = static_cast<double>(i) / kSourceRate;
    source.setSample(0, i, static_cast<float>(0.4 * std::sin(2 * kPi * 220 * t)));
    source.setSample(1, i, static_cast<float>(0.3 * std::sin(2 * kPi * 220 * t + 0.5)));
  }
  return source;
}

// A linked project over `source`, hydrated as on open: two notes at A3,
// the second dragged up two semitones and rendered (its composite halved).
void makeProject(Project &project, const juce::AudioBuffer<float> &source)
{
  auto &audio = project.getAudioData();
  audio.sourceAudio.makeCopyOf(source);
  audio.sourceSampleRate = kSourceRate;
  project.setLinksRecording(true);
  CHECK(recordingchange::hydrate(project));
  const int frames = audio.getNumFrames();
  CHECK(frames > 200);
  audio.rawF0.assign(static_cast<size_t>(frames), 220.0f);
  audio.cleanedF0 = audio.rawF0;
  audio.denseF0 = audio.rawF0;
  audio.f0 = audio.rawF0;
  audio.voicedMask.assign(static_cast<size_t>(frames), true);
  audio.vadMask.assign(static_cast<size_t>(frames), true);

  Note untouched(100, 140, 57.0f);
  untouched.setSrcStartFrame(100);
  untouched.setSrcEndFrame(140);
  untouched.setOriginalMidiNote(57.0f);
  untouched.setOriginalDeltaPitch(std::vector<float>(40, 0.0f));
  Note edited(140, 180, 59.0f);
  edited.setSrcStartFrame(140);
  edited.setSrcEndFrame(180);
  edited.setOriginalMidiNote(57.0f);
  edited.setOriginalDeltaPitch(std::vector<float>(40, 0.0f));
  edited.setRenderedEdit(true);
  project.addNote(untouched);
  project.addNote(edited);
  PitchCurveProcessor::rebuildBaseFromNotes(project);
  for (int i = 140 * HOP_SIZE; i < 180 * HOP_SIZE; ++i)
    audio.waveform.setSample(0, i, audio.waveform.getSample(0, i) * 0.5f);
  project.setRecordingFingerprint(recordingchange::fingerprint(source, kSourceRate));
}

void changedBlocksAreFound()
{
  auto source = recordingOf(2.0);
  const auto before = recordingchange::fingerprint(source, kSourceRate);
  CHECK(before.isValid());
  CHECK(before.hashes.size() == static_cast<size_t>((96000 + 1023) / 1024));
  CHECK(recordingchange::changedSamples(before, before).empty());
  source.setSample(1, 50000, source.getSample(1, 50000) + 1.0e-4f);
  const auto after = recordingchange::fingerprint(source, kSourceRate);
  const auto changed = recordingchange::changedSamples(before, after);
  CHECK(changed.size() == 1);
  CHECK(changed[0].first == 48 * 1024 && changed[0].second == 49 * 1024);
  std::printf("  one sample changed: block [%lld, %lld)\n",
              static_cast<long long>(changed[0].first),
              static_cast<long long>(changed[0].second));
}

// The plug-in reads the host's recording block by block; the App reads the
// file whole (sourceaudio::load). Both must give the same fingerprint, or a
// state saved in one would read as edited in the other.
void readerAndBufferAgree()
{
  const auto source = recordingOf(1.5);
  const auto file = juce::File::createTempFile(".wav");
  {
    juce::WavAudioFormat wav;
    std::unique_ptr<juce::OutputStream> stream = std::make_unique<juce::FileOutputStream>(file);
    auto writer = wav.createWriterFor(stream, juce::AudioFormatWriterOptions{}
                                                  .withSampleRate(kSourceRate)
                                                  .withNumChannels(2)
                                                  .withBitsPerSample(24));
    CHECK(writer != nullptr);
    CHECK(writer->writeFromAudioSampleBuffer(source, 0, source.getNumSamples()));
  }
  AudioData loaded;
  CHECK(sourceaudio::load(loaded, file));
  juce::AudioFormatManager formats;
  formats.registerBasicFormats();
  std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
  CHECK(reader != nullptr);
  const auto streamed = recordingchange::fingerprint(*reader);
  const auto whole = recordingchange::fingerprint(loaded.sourceAudio, loaded.sourceSampleRate);
  CHECK(streamed.isValid() && streamed == whole);
  const auto shape = recordingchange::shapeOf(file);
  CHECK(shape.sameShape(whole));
  reader.reset();
  file.deleteFile();
  std::printf("  streamed and whole-file fingerprints agree (%d blocks)\n",
              static_cast<int>(whole.hashes.size()));
}

void linkedProjectRoundTrips()
{
  const auto source = recordingOf(3.0);
  Project project;
  makeProject(project, source);
  const auto json = ProjectSerializer::toJson(project, true, true);
  CHECK(!json.hasProperty("waveform") && !json.hasProperty("originalWaveform"));
  CHECK(json.hasProperty("editPieces") && json.hasProperty("recording"));

  Project reopened;
  CHECK(ProjectSerializer::fromJson(reopened, juce::JSON::parse(juce::JSON::toString(json))));
  CHECK(reopened.linksRecording());
  CHECK(reopened.getRecordingFingerprint() == project.getRecordingFingerprint());
  CHECK(reopened.getLinkedEditPieces().size() == 1);
  reopened.getAudioData().sourceAudio.makeCopyOf(source);
  reopened.getAudioData().sourceSampleRate = kSourceRate;
  CHECK(recordingchange::hydrate(reopened));
  const auto &a = project.getAudioData().waveform;
  const auto &b = reopened.getAudioData().waveform;
  CHECK(a.getNumSamples() == b.getNumSamples());
  int differing = 0;
  for (int i = 0; i < a.getNumSamples(); ++i)
    differing += a.getSample(0, i) != b.getSample(0, i) ? 1 : 0;
  std::printf("  reopened composite: %d of %d samples differ\n", differing, a.getNumSamples());
  CHECK(differing == 0);
}

void changedPitchMovesTheEditedLine()
{
  auto source = recordingOf(3.0);
  Project project;
  makeProject(project, source);
  const auto saved = project.getRecordingFingerprint();
  auto &audio = project.getAudioData();
  const auto lineBefore = audio.f0;

  // Edit the recording inside the edited note (frames 155-157), re-read it.
  const int from = static_cast<int>(155.0 * HOP_SIZE * kSourceRate / SAMPLE_RATE);
  for (int c = 0; c < 2; ++c)
    for (int i = from; i < from + 100; ++i)
      source.setSample(c, i, source.getSample(c, i) * 0.8f);
  audio.sourceAudio.makeCopyOf(source);
  CHECK(recordingchange::hydrate(project));

  // A detector that hears a semitone higher than before.
  const float higher = 220.0f * std::pow(2.0f, 1.0f / 12.0f);
  int calls = 0;
  const auto update = recordingchange::apply(
      project, saved,
      [&](const float *, int, int frames, std::vector<float> &f0, std::vector<bool> &voiced) {
        ++calls;
        f0.assign(static_cast<size_t>(frames), higher);
        voiced.assign(static_cast<size_t>(frames), true);
        return true;
      });
  CHECK(calls == 1);
  CHECK(update.audioFrames.size() == 1);
  CHECK(update.audioFrames[0].first > 140 && update.audioFrames[0].second < 180);
  CHECK(!update.pitchFrames.empty());
  std::printf("  audio changed frames [%d, %d), pitch changed frames [%d, %d)\n",
              update.audioFrames[0].first, update.audioFrames[0].second,
              update.pitchFrames.front().first, update.pitchFrames.back().second);

  // The edited note's line moved by the semitone where the pitch changed,
  // and stayed elsewhere; it is still two semitones over the recording.
  const Note *edited = nullptr;
  const Note *untouched = nullptr;
  for (const auto &note : project.getNotes())
    (note.getStartFrame() == 140 ? edited : untouched) = &note;
  CHECK(edited != nullptr && untouched != nullptr);
  for (int t = 145; t < 175; ++t)
  {
    const bool changed = t >= update.pitchFrames.front().first &&
                         t < update.pitchFrames.back().second;
    const float moved = freqToMidi(audio.f0[static_cast<size_t>(t)]) -
                        freqToMidi(lineBefore[static_cast<size_t>(t)]);
    CHECK(std::abs(moved - (changed ? 1.0f : 0.0f)) < 0.02f);
    if (changed)
      CHECK(std::abs(freqToMidi(audio.f0[static_cast<size_t>(t)]) -
                     freqToMidi(audio.denseF0[static_cast<size_t>(t)]) - 2.0f) < 0.02f);
  }

  // The old render there is gone and the note is queued; the other note
  // (outside the change) is not.
  CHECK(update.notesToRender == 1);
  CHECK(edited->isDirty() && edited->isSynthDirty());
  CHECK(!untouched->isDirty());
  const auto &reference = warp::reference(audio);
  for (int i = 140 * HOP_SIZE; i < 180 * HOP_SIZE; ++i)
    CHECK(audio.waveform.getSample(0, i) == reference.getSample(0, i));
  CHECK(project.getRecordingFingerprint() ==
        recordingchange::fingerprint(source, kSourceRate));
}
// Two edited notes sung with vibrato, both committed (baked): one kept the
// vibrato, one flattened it. The recording changes over both and the
// detector now hears a steady pitch a semitone up.
void bakedContoursFollowAsFarAsTheyFollowedTheRecording()
{
  auto source = recordingOf(3.0);
  Project project;
  auto &audio = project.getAudioData();
  audio.sourceAudio.makeCopyOf(source);
  audio.sourceSampleRate = kSourceRate;
  project.setLinksRecording(true);
  CHECK(recordingchange::hydrate(project));
  const int frames = audio.getNumFrames();
  // Vibrato of 0.4 semitones, a period of 20 frames, around A3.
  std::vector<float> vibrato(static_cast<size_t>(frames));
  for (int t = 0; t < frames; ++t)
    vibrato[static_cast<size_t>(t)] = 0.4f * static_cast<float>(std::sin(2 * kPi * t / 20.0));
  audio.rawF0.resize(static_cast<size_t>(frames));
  for (int t = 0; t < frames; ++t)
    audio.rawF0[static_cast<size_t>(t)] = midiToFreq(57.0f + vibrato[static_cast<size_t>(t)]);
  audio.cleanedF0 = audio.rawF0;
  audio.denseF0 = audio.rawF0;
  audio.f0 = audio.rawF0;
  audio.voicedMask.assign(static_cast<size_t>(frames), true);
  audio.vadMask.assign(static_cast<size_t>(frames), true);
  const auto noteOver = [&](int start, int end, bool keepsVibrato) {
    Note note(start, end, 59.0f);
    note.setSrcStartFrame(start);
    note.setSrcEndFrame(end);
    note.setOriginalMidiNote(57.0f);
    const std::vector<float> curve(vibrato.begin() + start, vibrato.begin() + end);
    note.setOriginalDeltaPitch(curve);
    note.setBakedDeltaPitch(keepsVibrato ? curve : std::vector<float>(curve.size(), 0.0f));
    return note;
  };
  project.addNote(noteOver(120, 170, true));
  project.addNote(noteOver(170, 220, false));
  PitchCurveProcessor::rebuildBaseFromNotes(project);
  const auto saved = recordingchange::fingerprint(source, kSourceRate);
  project.setRecordingFingerprint(saved);

  // Change frames 160-180 of the recording.
  const int from = static_cast<int>(160.0 * HOP_SIZE * kSourceRate / SAMPLE_RATE);
  const int to = static_cast<int>(180.0 * HOP_SIZE * kSourceRate / SAMPLE_RATE);
  for (int c = 0; c < 2; ++c)
    for (int i = from; i < to; ++i)
      source.setSample(c, i, source.getSample(c, i) * 0.9f);
  audio.sourceAudio.makeCopyOf(source);
  CHECK(recordingchange::hydrate(project));
  const float steady = midiToFreq(58.0f);
  const auto update = recordingchange::apply(
      project, saved,
      [&](const float *, int, int frames, std::vector<float> &f0, std::vector<bool> &voiced) {
        f0.assign(static_cast<size_t>(frames), steady);
        voiced.assign(static_cast<size_t>(frames), true);
        return true;
      });
  CHECK(!update.pitchFrames.empty());
  const Note *kept = nullptr;
  const Note *flattened = nullptr;
  for (const auto &note : project.getNotes())
    (note.getStartFrame() == 120 ? kept : flattened) = &note;
  CHECK(kept != nullptr && flattened != nullptr);
  int checkedKept = 0, checkedFlat = 0;
  float worstKept = 0.0f, worstFlat = 0.0f;
  for (int t = 120; t < 220; ++t)
  {
    if (freqToMidi(audio.denseF0[static_cast<size_t>(t)]) - 58.0f > 0.001f ||
        freqToMidi(audio.denseF0[static_cast<size_t>(t)]) - 58.0f < -0.001f)
      continue; // not a changed frame
    if (t < 170)
    {
      // Kept the vibrato: the line is now the steady new pitch plus the drag.
      worstKept = std::max(worstKept,
                           std::abs(kept->getBakedDeltaPitch()[static_cast<size_t>(t - 120)] -
                                    1.0f));
      ++checkedKept;
    }
    else
    {
      // Flattened: stays flat.
      worstFlat = std::max(worstFlat, std::abs(flattened->getBakedDeltaPitch()
                                                   [static_cast<size_t>(t - 170)]));
      ++checkedFlat;
    }
  }
  std::printf("  baked contours: kept vibrato off the new pitch by %.3f st (%d frames), "
              "flattened one moved %.3f st (%d frames)\n",
              worstKept, checkedKept, worstFlat, checkedFlat);
  CHECK(checkedKept > 5 && checkedFlat > 5);
  CHECK(worstKept < 0.02f);
  CHECK(worstFlat < 0.02f);
}
} // namespace

int main()
{
  juce::ScopedJuceInitialiser_GUI juce;
  changedBlocksAreFound();
  readerAndBufferAgree();
  linkedProjectRoundTrips();
  changedPitchMovesTheEditedLine();
  bakedContoursFollowAsFarAsTheyFollowedTheRecording();
  std::printf("RecordingChangeTests passed\n");
  return 0;
}
