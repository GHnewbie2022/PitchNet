// Spec Q5: playback is the recording at its own rate and channel count, with
// only the edited ranges replaced (SourceAudio.h).
#include "TestAssert.h"

#include "../Source/Audio/RecordingChange.h"
#include "../Source/Audio/SourceAudio.h"
#include "../Source/Audio/Synthesis/SourceKinds.h"
#include "../Source/Audio/Synthesis/WarpedSource.h"
#include "../Source/Audio/WarpLayer.h"
#include "../Source/Audio/WarpMap.h"
#include "../Source/Models/ProjectSerializer.h"
#include "../Source/Utils/AudioResampler.h"
#include "../Source/Utils/Constants.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr int kSourceRate = 48000;

// A 48 kHz stereo recording whose channels differ, and its 44.1 kHz mono
// analysis copy made the way the loaders make it.
AudioData makeRecording(double seconds)
{
  AudioData audio;
  const int samples = static_cast<int>(seconds * kSourceRate);
  audio.sourceAudio.setSize(2, samples);
  for (int i = 0; i < samples; ++i)
  {
    const double t = static_cast<double>(i) / kSourceRate;
    audio.sourceAudio.setSample(0, i, static_cast<float>(0.4 * std::sin(2 * kPi * 440 * t)));
    audio.sourceAudio.setSample(1, i, static_cast<float>(0.3 * std::sin(2 * kPi * 660 * t)));
  }
  audio.sourceSampleRate = kSourceRate;

  juce::AudioBuffer<float> mono(1, samples);
  for (int i = 0; i < samples; ++i)
    mono.setSample(0, i, 0.5f * (audio.sourceAudio.getSample(0, i) +
                                 audio.sourceAudio.getSample(1, i)));
  audio.originalWaveform = AudioResampler::resample(mono, kSourceRate, SAMPLE_RATE);
  audio.waveform.makeCopyOf(audio.originalWaveform);
  audio.sampleRate = SAMPLE_RATE;
  return audio;
}

double rmsDb(const std::vector<double> &values)
{
  double sum = 0.0;
  for (double v : values)
    sum += v * v;
  return 10.0 * std::log10(sum / std::max<size_t>(1, values.size()) + 1.0e-30);
}

void uneditedAudioIsTheRecordingItself()
{
  auto audio = makeRecording(1.0);
  const auto playback = sourceaudio::build(audio);
  CHECK(playback.sampleRate == kSourceRate);
  CHECK(playback.audio.getNumChannels() == 2);
  CHECK(playback.audio.getNumSamples() == audio.sourceAudio.getNumSamples());
  for (int channel = 0; channel < 2; ++channel)
    for (int i = 0; i < playback.audio.getNumSamples(); ++i)
      CHECK(playback.audio.getSample(channel, i) ==
            audio.sourceAudio.getSample(channel, i));
}

void onlyTheEditedRangeIsReplaced()
{
  auto audio = makeRecording(2.0);
  // The "edit": 0.8 to 1.2 s of the composite becomes an 880 Hz tone.
  const int editStart = static_cast<int>(0.8 * SAMPLE_RATE);
  const int editEnd = static_cast<int>(1.2 * SAMPLE_RATE);
  for (int i = editStart; i < editEnd; ++i)
    audio.waveform.setSample(
        0, i, static_cast<float>(0.25 * std::sin(2 * kPi * 880 * i / SAMPLE_RATE)));

  const auto playback = sourceaudio::build(audio);
  CHECK(playback.sampleRate == kSourceRate);

  const double margin = sourceaudio::kMarginSeconds;
  const int replacedFrom = static_cast<int>((0.8 - margin) * kSourceRate) - 2;
  const int replacedTo = static_cast<int>((1.2 + margin) * kSourceRate) + 2;
  int untouched = 0;
  for (int channel = 0; channel < 2; ++channel)
    for (int i = 0; i < playback.audio.getNumSamples(); ++i)
      if (i < replacedFrom || i >= replacedTo)
      {
        CHECK(playback.audio.getSample(channel, i) ==
              audio.sourceAudio.getSample(channel, i));
        ++untouched;
      }
  CHECK(untouched > 0);

  // Inside the edit, both channels carry the edit converted to 48 kHz.
  for (int channel = 0; channel < 2; ++channel)
  {
    std::vector<double> error;
    for (int i = static_cast<int>(0.85 * kSourceRate);
         i < static_cast<int>(1.15 * kSourceRate); ++i)
    {
      const double expected = 0.25 * std::sin(2 * kPi * 880 * i / kSourceRate);
      error.push_back(playback.audio.getSample(channel, i) - expected);
    }
    const double errorDb = rmsDb(error) - 20.0 * std::log10(0.25 / std::sqrt(2.0));
    std::printf("  channel %d: edit error %.1f dB\n", channel, errorDb);
    CHECK(errorDb < -60.0);
  }
}

void editedRangesAreWidenedAndJoined()
{
  std::vector<float> reference(1000, 0.0f), composite(1000, 0.0f);
  composite[100] = 1.0f;
  composite[130] = 1.0f; // within `join` of the first: one range
  composite[600] = 1.0f;
  const auto ranges = sourceaudio::editedRanges(composite.data(), 1000,
                                                reference.data(), 1000, 10, 40);
  CHECK(ranges.size() == 2);
  CHECK(ranges[0].first == 90 && ranges[0].second == 141);
  CHECK(ranges[1].first == 590 && ranges[1].second == 611);

  // Float noise is not an edit.
  composite.assign(1000, 0.0f);
  composite[500] = 1.0e-7f;
  CHECK(sourceaudio::editedRanges(composite.data(), 1000, reference.data(),
                                  1000, 10, 40)
            .empty());
}

void warpRendersAtTheRecordingsOwnRate()
{
  // Frame 40 is dragged to 50: 0-40 stretches, 40-80 compresses, and from
  // frame 80 on the recording plays unchanged.
  warp::WarpMap map(400.0);
  const int anchor = map.addAnchorAtDestination(40.0);
  map.addAnchorAtDestination(80.0);
  map.moveAnchor(anchor, 50.0);

  warp::WarpedSourceSettings settings;
  settings.sampleRate = kSourceRate;
  settings.hopSize = static_cast<double>(HOP_SIZE) * kSourceRate / SAMPLE_RATE;
  const int samples = static_cast<int>(std::ceil(400 * settings.hopSize));
  std::vector<float> source(static_cast<size_t>(samples));
  for (int i = 0; i < samples; ++i)
    source[static_cast<size_t>(i)] = static_cast<float>(std::sin(0.05 * i) * 0.5);
  std::vector<float> output(static_cast<size_t>(samples));
  warp::renderWarpedSource(source.data(), samples, map, output.data(), samples,
                           settings);

  // Past the hand-over after the last anchor (at most 0.3 s), the recording
  // plays unchanged, at the same place.
  const int from = static_cast<int>(std::ceil(80 * settings.hopSize)) +
                   static_cast<int>(0.4 * kSourceRate);
  CHECK(from < samples);
  for (int i = from; i < samples; ++i)
    CHECK(output[static_cast<size_t>(i)] == source[static_cast<size_t>(i)]);
}
// The plugin plays the host's audio with differingPieces laid over it. That
// has to be the standalone's playback, sample for sample, at any block size.
void piecesOverTheRecordingAreThePlayback()
{
  auto audio = makeRecording(2.0);
  for (int i = static_cast<int>(0.5 * SAMPLE_RATE);
       i < static_cast<int>(0.7 * SAMPLE_RATE); ++i)
    audio.waveform.setSample(0, i, 0.1f);
  for (int i = static_cast<int>(1.3 * SAMPLE_RATE);
       i < static_cast<int>(1.4 * SAMPLE_RATE); ++i)
    audio.waveform.setSample(0, i, -0.1f);

  // The recording kept apart from the edit, as the plugin keeps it.
  AudioData recording;
  recording.sourceAudio.makeCopyOf(audio.sourceAudio);
  recording.sourceSampleRate = audio.sourceSampleRate;
  const auto expected = sourceaudio::build(audio);
  const auto playback = sourceaudio::build(audio, audio.waveform, recording);
  CHECK(playback.sampleRate == expected.sampleRate);

  const auto pieces = sourceaudio::differingPieces(
      playback.audio, recording.sourceAudio, kSourceRate / 20);
  CHECK(pieces.size() == 2);

  for (int block : {64, 512, 1000})
  {
    juce::AudioBuffer<float> out(2, playback.audio.getNumSamples());
    for (int start = 0; start < out.getNumSamples(); start += block)
    {
      const int n = std::min(block, out.getNumSamples() - start);
      for (int c = 0; c < 2; ++c)
        out.copyFrom(c, start, recording.sourceAudio, c, start, n);
      sourceaudio::overlay(pieces, start, out, start, n);
    }
    for (int c = 0; c < 2; ++c)
      for (int i = 0; i < out.getNumSamples(); ++i)
        CHECK(out.getSample(c, i) == expected.audio.getSample(c, i));
  }

  // Nothing edited: nothing stored.
  auto untouched = makeRecording(1.0);
  CHECK(sourceaudio::differingPieces(sourceaudio::build(untouched).audio,
                                     untouched.sourceAudio, 100)
            .empty());
}

// One stretch, not two: with the recording held at its own rate, the
// reference every edit is blended onto is that stretch converted as the
// analysis copy was made, sample for sample. Stretched on its own from the
// analysis copy it sat at another phase, and the crossfade from an edit back
// to the recording dipped. Without the recording (the plug-in's project) the
// analysis copy is stretched as before.
void theReferenceIsThePlaybackStretch()
{
  auto audio = makeRecording(3.0);
  const int frames = audio.originalWaveform.getNumSamples() / HOP_SIZE;
  warp::WarpMap map(static_cast<double>(frames));
  const int anchor = map.addAnchorAtDestination(40.0);
  map.addAnchorAtDestination(80.0);
  map.moveAnchor(anchor, 50.0);
  audio.warpMap = map;

  warp::rebuildReference(audio);
  CHECK(audio.warpedSourceAudio.getNumSamples() == audio.sourceAudio.getNumSamples());
  const auto converted =
      recordingchange::analysisCopy(audio.warpedSourceAudio, audio.sourceSampleRate);
  CHECK(audio.warpedOriginal.getNumChannels() == audio.originalWaveform.getNumChannels());
  CHECK(audio.warpedOriginal.getNumSamples() == audio.originalWaveform.getNumSamples());
  CHECK(converted.getNumSamples() == audio.warpedOriginal.getNumSamples());
  for (int i = 0; i < converted.getNumSamples(); ++i)
    CHECK(audio.warpedOriginal.getSample(0, i) == converted.getSample(0, i));

  // Nothing edited: playback is the recording as the warp laid it out.
  audio.waveform.makeCopyOf(warp::reference(audio));
  const auto playback = sourceaudio::build(audio);
  for (int channel = 0; channel < 2; ++channel)
    for (int i = 0; i < playback.audio.getNumSamples(); ++i)
      CHECK(playback.audio.getSample(channel, i) ==
            audio.warpedSourceAudio.getSample(channel, i));

  // An edit in the stretched part: 20 ms outside it, where playback hands
  // back to the recording, the two sides are the same audio (they used to
  // differ by the phase of two separate stretches).
  const int editStart = static_cast<int>(0.30 * SAMPLE_RATE);
  const int editEnd = static_cast<int>(0.40 * SAMPLE_RATE);
  for (int i = editStart; i < editEnd; ++i)
    audio.waveform.setSample(0, i, 0.0f);
  const auto edited = sourceaudio::build(audio);
  std::vector<double> difference, level;
  const int from = static_cast<int>((0.40 + 0.002) * kSourceRate);
  const int to = static_cast<int>((0.40 + sourceaudio::kMarginSeconds - 0.001) * kSourceRate);
  for (int i = from; i < to; ++i)
  {
    const double mono = 0.5 * (audio.warpedSourceAudio.getSample(0, i) +
                               audio.warpedSourceAudio.getSample(1, i));
    const double played = 0.5 * (edited.audio.getSample(0, i) + edited.audio.getSample(1, i));
    difference.push_back(played - mono);
    level.push_back(mono);
  }
  const double errorDb = rmsDb(difference) - rmsDb(level);
  std::printf("  margin after an edit in a stretch: %.1f dB off the recording\n", errorDb);
  CHECK(errorDb < -40.0);

  AudioData withoutRecording;
  withoutRecording.originalWaveform.makeCopyOf(audio.originalWaveform);
  withoutRecording.sampleRate = SAMPLE_RATE;
  withoutRecording.warpMap = map;
  warp::rebuildReference(withoutRecording);
  CHECK(withoutRecording.warpedSourceAudio.getNumSamples() == 0);
  CHECK(withoutRecording.warpedOriginal.getNumSamples() ==
        audio.originalWaveform.getNumSamples());
}

// A stretched project saved before the one stretch (format 7 and earlier)
// keeps the reference its edits were rendered onto - the analysis copy
// stretched on its own - until the warp is laid out again, which renders
// every edit anew. The file says which reference it was saved with.
void olderStretchedProjectsKeepTheirReference()
{
  auto audio = makeRecording(3.0);
  const int frames = audio.originalWaveform.getNumSamples() / HOP_SIZE;
  warp::WarpMap map(static_cast<double>(frames));
  const int anchor = map.addAnchorAtDestination(40.0);
  map.addAnchorAtDestination(80.0);
  map.moveAnchor(anchor, 50.0);
  audio.warpMap = map;

  AudioData separate;
  separate.originalWaveform.makeCopyOf(audio.originalWaveform);
  separate.sampleRate = SAMPLE_RATE;
  separate.warpMap = map;
  warp::rebuildReference(separate);

  audio.warpReferenceFromRecording = false;
  warp::rebuildReference(audio);
  CHECK(audio.warpedOriginal.getNumSamples() == separate.warpedOriginal.getNumSamples());
  for (int i = 0; i < audio.warpedOriginal.getNumSamples(); ++i)
    CHECK(audio.warpedOriginal.getSample(0, i) == separate.warpedOriginal.getSample(0, i));

  Project project;
  project.getAudioData() = audio;
  auto json = ProjectSerializer::toJson(project);
  CHECK(static_cast<int>(json.getProperty("formatVersion", 0)) == ProjectSerializer::FORMAT_VERSION);
  CHECK(!static_cast<bool>(json.getProperty("warp", juce::var())
                               .getProperty("referenceFromRecording", true)));
  CHECK(static_cast<int>(json.getProperty("warp", juce::var()).getProperty("stretch", 0)) == 2);
  Project reopened;
  CHECK(ProjectSerializer::fromJson(reopened, json));
  CHECK(!reopened.getAudioData().warpReferenceFromRecording);
  CHECK(!reopened.getAudioData().warpLayoutPending);

  // A format 8 file was stretched with one setting for everything: stretched,
  // it is laid out again as it is opened (the stretch by kind, WarpedSource.h).
  json.getDynamicObject()->setProperty("formatVersion", 8);
  json.getProperty("warp", juce::var()).getDynamicObject()->removeProperty("stretch");
  Project eight;
  CHECK(ProjectSerializer::fromJson(eight, json));
  CHECK(eight.getAudioData().warpLayoutPending);

  // A format 7 file has no such word: stretched, it was rendered onto the
  // separate stretch; unstretched, its first layout is the one stretch.
  json.getDynamicObject()->setProperty("formatVersion", 7);
  json.getProperty("warp", juce::var()).getDynamicObject()->removeProperty("referenceFromRecording");
  Project older;
  CHECK(ProjectSerializer::fromJson(older, json));
  CHECK(!older.getAudioData().warpReferenceFromRecording);
  Project plain;
  auto plainJson = ProjectSerializer::toJson(plain);
  plainJson.getDynamicObject()->setProperty("formatVersion", 7);
  Project plainReopened;
  CHECK(ProjectSerializer::fromJson(plainReopened, plainJson));
  CHECK(plainReopened.getAudioData().warpReferenceFromRecording);
  CHECK(!plainReopened.getAudioData().warpLayoutPending); // nothing stretched

  // Laid out again, the project is on the one stretch.
  warp::layOut(project);
  CHECK(project.getAudioData().warpReferenceFromRecording);
  const auto &laidOut = project.getAudioData();
  const auto converted =
      recordingchange::analysisCopy(laidOut.warpedSourceAudio, laidOut.sourceSampleRate);
  CHECK(converted.getNumSamples() == laidOut.warpedOriginal.getNumSamples());
  for (int i = 0; i < converted.getNumSamples(); ++i)
    CHECK(laidOut.warpedOriginal.getSample(0, i) == converted.getSample(0, i));
}

// The plug-in keeps the host's audio apart from the project and stretches it
// channel by channel for playback. Its analysis copy is channel 0 converted;
// given that channel at the host's rate, the project's reference is the same
// stretch converted the same way, so an edit hands back to the host's audio
// in phase there too.
void thePluginsReferenceIsTheHostsStretch()
{
  const auto standalone = makeRecording(3.0);
  AudioData recording; // as PluginProcessor::buildPlaybackPieces keeps it
  recording.sourceAudio.makeCopyOf(standalone.sourceAudio);
  recording.sourceSampleRate = kSourceRate;

  auto channel0 = std::make_shared<juce::AudioBuffer<float>>(1, recording.sourceAudio.getNumSamples());
  channel0->copyFrom(0, 0, recording.sourceAudio, 0, 0, recording.sourceAudio.getNumSamples());

  AudioData edit; // the plug-in's project: no sourceAudio of its own
  edit.originalWaveform = AudioResampler::resample(*channel0, kSourceRate, SAMPLE_RATE);
  edit.sampleRate = SAMPLE_RATE;
  edit.analysisSource = channel0;
  edit.analysisSourceRate = kSourceRate;
  const int frames = edit.originalWaveform.getNumSamples() / HOP_SIZE;
  warp::WarpMap map(static_cast<double>(frames));
  const int anchor = map.addAnchorAtDestination(40.0);
  map.addAnchorAtDestination(80.0);
  map.moveAnchor(anchor, 50.0);
  edit.warpMap = map;
  recording.warpMap = map;
  // The project's analysis, which the host's audio does not carry: a breath
  // and a fry onset inside the stretch. Each is stretched its own way
  // (SourceKinds.h), and the plug-in hands the same kinds to both.
  edit.breathSibilance.assign(static_cast<size_t>(frames), breathsibilance::None);
  for (int frame = 18; frame < 30; ++frame)
    edit.breathSibilance[static_cast<size_t>(frame)] = breathsibilance::Breath;
  edit.rawF0.assign(static_cast<size_t>(frames), 0.0f);
  for (int frame = 54; frame < 58; ++frame)
    edit.rawF0[static_cast<size_t>(frame)] = 100.0f;
  for (int frame = 58; frame < 78; ++frame)
    edit.rawF0[static_cast<size_t>(frame)] = 230.0f;
  const auto kinds = warp::sourceKinds(edit);
  CHECK(!kinds.empty());

  warp::rebuildReference(edit);
  sourceaudio::rebuildWarped(recording, {});
  juce::AudioBuffer<float> allVoice;
  allVoice.makeCopyOf(recording.warpedSourceAudio);
  sourceaudio::rebuildWarped(recording, kinds);
  CHECK(recording.stretchKinds == kinds);
  bool kindsHeard = false;
  for (int i = 0; i < allVoice.getNumSamples() && !kindsHeard; ++i)
    kindsHeard = allVoice.getSample(0, i) != recording.warpedSourceAudio.getSample(0, i);
  CHECK(kindsHeard);
  CHECK(edit.warpedSourceAudio.getNumSamples() == 0);
  CHECK(edit.warpedOriginal.getNumSamples() == edit.originalWaveform.getNumSamples());

  // The reference is channel 0 of what playback stretched, converted.
  juce::AudioBuffer<float> played(1, recording.warpedSourceAudio.getNumSamples());
  played.copyFrom(0, 0, recording.warpedSourceAudio, 0, 0, played.getNumSamples());
  const auto converted = AudioResampler::resample(played, kSourceRate, SAMPLE_RATE);
  CHECK(converted.getNumSamples() == edit.warpedOriginal.getNumSamples());
  for (int i = 0; i < converted.getNumSamples(); ++i)
    CHECK(edit.warpedOriginal.getSample(0, i) == converted.getSample(0, i));

  // An edit in the stretched part hands back to the host's audio in phase.
  edit.waveform.makeCopyOf(warp::reference(edit));
  for (int i = static_cast<int>(0.30 * SAMPLE_RATE); i < static_cast<int>(0.40 * SAMPLE_RATE); ++i)
    edit.waveform.setSample(0, i, 0.0f);
  const auto playback = sourceaudio::build(edit, edit.waveform, recording);
  std::vector<double> difference, level;
  for (int i = static_cast<int>(0.402 * kSourceRate);
       i < static_cast<int>((0.40 + sourceaudio::kMarginSeconds - 0.001) * kSourceRate); ++i)
  {
    const double host = recording.warpedSourceAudio.getSample(0, i);
    difference.push_back(playback.audio.getSample(0, i) - host);
    level.push_back(host);
  }
  const double errorDb = rmsDb(difference) - rmsDb(level);
  std::printf("  plug-in, margin after an edit in a stretch: %.1f dB off the host's audio\n", errorDb);
  CHECK(errorDb < -40.0);

  // At the host's own rate there is nothing to convert: the analysis copy is
  // that audio, and its stretch is the playback's.
  AudioData sameRate;
  sameRate.originalWaveform.makeCopyOf(edit.originalWaveform);
  sameRate.sampleRate = SAMPLE_RATE;
  sameRate.warpMap = map;
  warp::rebuildReference(sameRate);
  CHECK(sameRate.warpedOriginal.getNumSamples() == edit.originalWaveform.getNumSamples());
}
} // namespace

int main()
{
  uneditedAudioIsTheRecordingItself();
  onlyTheEditedRangeIsReplaced();
  editedRangesAreWidenedAndJoined();
  warpRendersAtTheRecordingsOwnRate();
  piecesOverTheRecordingAreThePlayback();
  theReferenceIsThePlaybackStretch();
  olderStretchedProjectsKeepTheirReference();
  thePluginsReferenceIsTheHostsStretch();
  std::printf("SourceAudioTests passed\n");
  return 0;
}
