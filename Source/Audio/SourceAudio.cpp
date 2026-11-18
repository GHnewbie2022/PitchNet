#include "SourceAudio.h"

#include "Synthesis/WarpedSource.h"
#include "WarpLayer.h"
#include "../Utils/AudioResampler.h"
#include "../Utils/Constants.h"
#include "../Utils/SHA256Utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>

namespace sourceaudio
{
namespace
{
// Differences smaller than this (about -120 dBFS) are float noise from
// recomputing the same value, not an edit.
constexpr float kDifferenceThreshold = 1.0e-6f;
// Extra 44.1 kHz audio converted on each side of a range, so r8brain's filter
// has settled before the samples that are used.
constexpr int kResamplePad = 2048;

// Smallest step of 44.1 kHz samples that lands on a whole sample at `rate`.
int alignmentStep(int rate)
{
  return SAMPLE_RATE / std::gcd(SAMPLE_RATE, rate);
}

long long toSourceSample(long long analysisSample, int rate)
{
  return analysisSample * rate / SAMPLE_RATE;
}
} // namespace

bool load(AudioData &audioData, const juce::File &file)
{
  audioData.sourceAudio.setSize(0, 0);
  audioData.sourceSampleRate = 0;
  audioData.warpedSourceAudio.setSize(0, 0);

  juce::AudioFormatManager formats;
  formats.registerBasicFormats();
  std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
  if (reader == nullptr || reader->lengthInSamples <= 0 ||
      reader->numChannels <= 0 || reader->sampleRate <= 0.0 ||
      reader->lengthInSamples > std::numeric_limits<int>::max())
    return false;

  const int channels = static_cast<int>(reader->numChannels);
  const int samples = static_cast<int>(reader->lengthInSamples);
  audioData.sourceAudio.setSize(channels, samples);
  if (!reader->read(&audioData.sourceAudio, 0, samples, 0, true, true))
  {
    audioData.sourceAudio.setSize(0, 0);
    return false;
  }
  audioData.sourceSampleRate = static_cast<int>(std::lround(reader->sampleRate));
  rebuildWarped(audioData);
  return true;
}

bool attachRecording(Project &project, const juce::File &projectFile,
                     const juce::File &chosen)
{
  auto &audioData = project.getAudioData();
  const double expectedSeconds =
      audioData.originalWaveform.getNumSamples() /
      static_cast<double>(std::max(1, audioData.sampleRate));

  auto matches = [&](const juce::File &candidate)
  {
    if (!candidate.existsAsFile())
      return false;
    if (project.getAudioSha256().isNotEmpty() &&
        !SHA256Utils::fileSHA256(candidate).equalsIgnoreCase(
            project.getAudioSha256()))
      return false;
    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader(
        formats.createReaderFor(candidate));
    if (reader == nullptr || reader->sampleRate <= 0.0)
      return false;
    const double seconds = reader->lengthInSamples / reader->sampleRate;
    return std::abs(seconds - expectedSeconds) < 0.05;
  };

  juce::Array<juce::File> candidates;
  if (chosen != juce::File{})
    candidates.add(chosen);
  else
  {
    candidates.add(project.getFilePath());
    for (const auto *extension : {".wav", ".aif", ".aiff", ".flac"})
      candidates.add(projectFile.withFileExtension(extension));
  }
  for (const auto &candidate : candidates)
  {
    if (!matches(candidate) || !load(audioData, candidate))
      continue;
    project.setFilePath(candidate);
    if (project.getAudioSha256().isEmpty())
      project.setAudioSha256(SHA256Utils::fileSHA256(candidate));
    // Saved projects keep no warped reference; without it every warped
    // range would read as an edit.
    if (!audioData.warpMap.isIdentity() &&
        audioData.warpedOriginal.getNumSamples() == 0)
      warp::rebuildReference(audioData);
    return true;
  }
  return false;
}

void rebuildWarped(AudioData &audioData)
{
  const auto &source = audioData.sourceAudio;
  const int rate = audioData.sourceSampleRate;
  if (audioData.warpMap.isIdentity() || source.getNumSamples() <= 0 || rate <= 0)
  {
    audioData.warpedSourceAudio.setSize(0, 0);
    return;
  }

  warp::WarpedSourceSettings settings;
  settings.sampleRate = rate;
  settings.hopSize = static_cast<double>(HOP_SIZE) * rate / SAMPLE_RATE;
  const int samples = source.getNumSamples();
  audioData.warpedSourceAudio.setSize(source.getNumChannels(), samples, false,
                                      true, false);
  for (int channel = 0; channel < source.getNumChannels(); ++channel)
    warp::renderWarpedSource(source.getReadPointer(channel), samples,
                             audioData.warpMap,
                             audioData.warpedSourceAudio.getWritePointer(channel),
                             samples, settings);
}

std::vector<std::pair<int, int>> editedRanges(const float *composite,
                                              int compositeSamples,
                                              const float *reference,
                                              int referenceSamples, int margin,
                                              int join)
{
  std::vector<std::pair<int, int>> ranges;
  if (composite == nullptr || compositeSamples <= 0)
    return ranges;

  const auto differs = [&](int i)
  {
    const float expected =
        reference != nullptr && i < referenceSamples ? reference[i] : 0.0f;
    return std::abs(composite[i] - expected) > kDifferenceThreshold;
  };

  for (int i = 0; i < compositeSamples;)
  {
    if (!differs(i))
    {
      ++i;
      continue;
    }
    int end = i + 1;
    while (end < compositeSamples && differs(end))
      ++end;
    const int start = std::max(0, i - margin);
    const int stop = std::min(compositeSamples, end + margin);
    if (!ranges.empty() && start - ranges.back().second < join)
      ranges.back().second = stop;
    else
      ranges.emplace_back(start, stop);
    i = end;
  }
  return ranges;
}

Playback build(const AudioData &audioData) { return build(audioData, audioData); }

Playback build(const AudioData &audioData, const AudioData &recording)
{
  return build(audioData, audioData.waveform, recording);
}

Playback build(const AudioData &audioData,
               const juce::AudioBuffer<float> &composite,
               const AudioData &recording)
{
  Playback result;
  const int rate = recording.sourceSampleRate;
  const auto &source = !audioData.warpMap.isIdentity() &&
                               recording.warpedSourceAudio.getNumSamples() > 0
                           ? recording.warpedSourceAudio
                           : recording.sourceAudio;
  if (source.getNumSamples() <= 0 || rate <= 0 || composite.getNumSamples() <= 0)
  {
    result.audio.makeCopyOf(composite);
    result.sampleRate = audioData.sampleRate;
    return result;
  }

  result.audio.makeCopyOf(source);
  result.sampleRate = rate;

  const auto &reference = warp::reference(audioData);
  const bool hasReference =
      &reference != &audioData.waveform && reference.getNumSamples() > 0;
  const int compositeSamples = composite.getNumSamples();
  const int margin = static_cast<int>(kMarginSeconds * SAMPLE_RATE);
  const int fadeAnalysis = static_cast<int>(kFadeSeconds * SAMPLE_RATE);
  const auto ranges = editedRanges(
      composite.getReadPointer(0), compositeSamples,
      hasReference ? reference.getReadPointer(0) : nullptr,
      hasReference ? reference.getNumSamples() : 0, margin,
      2 * margin + fadeAnalysis);

  const int step = alignmentStep(rate);
  const int outputSamples = result.audio.getNumSamples();
  const int fade = std::max(1, static_cast<int>(kFadeSeconds * rate));

  for (const auto &[start, end] : ranges)
  {
    // Convert a step-aligned stretch around the range, so sample k of the
    // conversion is exactly source sample toSourceSample(first) + k.
    const int first = std::max(0, (start - kResamplePad) / step * step);
    int last = end + kResamplePad;
    last = (last + step - 1) / step * step;
    const int length = last - first;
    juce::AudioBuffer<float> piece(1, length);
    piece.clear();
    const int available = std::min(last, compositeSamples) - first;
    if (available > 0)
      piece.copyFrom(0, 0, composite, 0, first, available);
    const int convertedLength =
        static_cast<int>(toSourceSample(length, rate));
    const auto converted = AudioResampler::resample(
        piece, SAMPLE_RATE, static_cast<double>(rate), convertedLength);
    const int offset = static_cast<int>(toSourceSample(first, rate));

    const int from = static_cast<int>(toSourceSample(start, rate));
    const int to = std::min(outputSamples,
                            static_cast<int>(toSourceSample(end, rate)));
    const bool fadeIn = start > 0;
    const bool fadeOut = end < compositeSamples;
    const float *edit = converted.getReadPointer(0);
    for (int sample = std::max(0, from); sample < to; ++sample)
    {
      const int index = sample - offset;
      if (index < 0 || index >= converted.getNumSamples())
        continue;
      float amount = 1.0f;
      if (fadeIn && sample < from + fade)
        amount = static_cast<float>(sample - from + 1) / (fade + 1);
      if (fadeOut && sample >= to - fade)
        amount = std::min(amount, static_cast<float>(to - sample) / (fade + 1));
      for (int channel = 0; channel < result.audio.getNumChannels(); ++channel)
      {
        float *out = result.audio.getWritePointer(channel);
        out[sample] += amount * (edit[index] - out[sample]);
      }
    }
  }
  return result;
}

std::vector<Piece> differingPieces(const juce::AudioBuffer<float> &playback,
                                   const juce::AudioBuffer<float> &recording,
                                   int join)
{
  std::vector<Piece> pieces;
  const int samples = playback.getNumSamples();
  const int channels = playback.getNumChannels();
  if (samples <= 0 || channels <= 0)
    return pieces;

  const auto differs = [&](int i)
  {
    for (int c = 0; c < channels; ++c)
    {
      const float expected =
          recording.getNumChannels() > 0 && i < recording.getNumSamples()
              ? recording.getSample(std::min(c, recording.getNumChannels() - 1), i)
              : 0.0f;
      if (playback.getSample(c, i) != expected)
        return true;
    }
    return false;
  };

  std::vector<std::pair<int, int>> runs;
  for (int i = 0; i < samples;)
  {
    if (!differs(i))
    {
      ++i;
      continue;
    }
    int end = i + 1;
    while (end < samples && differs(end))
      ++end;
    if (!runs.empty() && i - runs.back().second < join)
      runs.back().second = end;
    else
      runs.emplace_back(i, end);
    i = end;
  }

  for (const auto &[start, end] : runs)
  {
    Piece piece;
    piece.start = start;
    piece.audio.setSize(channels, end - start);
    for (int c = 0; c < channels; ++c)
      piece.audio.copyFrom(c, 0, playback, c, start, end - start);
    pieces.push_back(std::move(piece));
  }
  return pieces;
}

void overlay(const std::vector<Piece> &pieces, long long recordingStart,
             juce::AudioBuffer<float> &dest, int destStart, int length)
{
  if (length <= 0)
    return;
  const long long recordingEnd = recordingStart + length;
  for (const auto &piece : pieces)
  {
    const long long pieceEnd = piece.start + piece.audio.getNumSamples();
    const long long from = std::max(recordingStart, piece.start);
    const long long to = std::min(recordingEnd, pieceEnd);
    if (to <= from || piece.audio.getNumChannels() <= 0)
      continue;
    for (int c = 0; c < dest.getNumChannels(); ++c)
      dest.copyFrom(c, destStart + static_cast<int>(from - recordingStart),
                    piece.audio, std::min(c, piece.audio.getNumChannels() - 1),
                    static_cast<int>(from - piece.start),
                    static_cast<int>(to - from));
  }
}
} // namespace sourceaudio
