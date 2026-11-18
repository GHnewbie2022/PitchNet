// Spec Q5: playback is the recording at its own rate and channel count, with
// only the edited ranges replaced (SourceAudio.h).
#include "TestAssert.h"

#include "../Source/Audio/SourceAudio.h"
#include "../Source/Audio/Synthesis/WarpedSource.h"
#include "../Source/Audio/WarpMap.h"
#include "../Source/Utils/AudioResampler.h"
#include "../Source/Utils/Constants.h"

#include <cmath>
#include <cstdio>
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
} // namespace

int main()
{
  uneditedAudioIsTheRecordingItself();
  onlyTheEditedRangeIsReplaced();
  editedRangesAreWidenedAndJoined();
  warpRendersAtTheRecordingsOwnRate();
  piecesOverTheRecordingAreThePlayback();
  std::printf("SourceAudioTests passed\n");
  return 0;
}
