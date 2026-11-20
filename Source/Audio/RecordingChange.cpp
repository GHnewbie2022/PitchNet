#include "RecordingChange.h"

#include "BreathSibilanceAnalysis.h"
#include "NoteTiming.h"
#include "SourceAudio.h"
#include "WarpLayer.h"
#include "../Utils/AudioResampler.h"
#include "../Utils/Constants.h"
#include "../Utils/F0Smoother.h"
#include "../Utils/MelSpectrogram.h"
#include "../Utils/NoteAmplitude.h"
#include "../Utils/PitchCurveProcessor.h"
#include "../Utils/SHA256Utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace recordingchange
{
namespace
{
// Analysis samples a change in the recording reaches into after conversion
// to 44.1 kHz and into the mel (r8brain's filter, half the FFT window).
constexpr int kSpreadSamples = 2048;
// Audio on each side of a changed part given to the pitch detector, so it
// judges the part with the context the whole-file analysis had.
constexpr double kContextSeconds = 2.0;
// Runs of the composite closer than this are saved as one piece.
constexpr int kPieceJoin = 64;
// Frames around a pitch change over which a committed contour is compared
// with the analysed curve (about a quarter of a second).
constexpr int kFollowReach = 20;

// How much of the analysed curve's shape `pairs` (analysed, committed) keep:
// the regression slope, 0 to 1. With too little shape to tell, all of it
// when the two differ by a constant, else none.
float shareFollowed(const std::vector<std::pair<double, double>> &pairs)
{
  if (pairs.size() < 3)
    return 1.0f;
  double meanA = 0.0, meanB = 0.0;
  for (const auto &[a, b] : pairs)
  {
    meanA += a;
    meanB += b;
  }
  meanA /= static_cast<double>(pairs.size());
  meanB /= static_cast<double>(pairs.size());
  double covariance = 0.0, variance = 0.0, offsetSpread = 0.0;
  for (const auto &[a, b] : pairs)
  {
    covariance += (a - meanA) * (b - meanB);
    variance += (a - meanA) * (a - meanA);
    const double offset = (b - meanB) - (a - meanA);
    offsetSpread += offset * offset;
  }
  const double n = static_cast<double>(pairs.size());
  if (variance / n < 1.0e-3) // under about 0.03 semitones of shape
    return std::sqrt(offsetSpread / n) < 0.02 ? 1.0f : 0.0f;
  return static_cast<float>(std::clamp(covariance / variance, 0.0, 1.0));
}

void addRange(std::vector<std::pair<int, int>> &ranges, int start, int end)
{
  if (end <= start)
    return;
  if (!ranges.empty() && start <= ranges.back().second)
    ranges.back().second = std::max(ranges.back().second, end);
  else
    ranges.emplace_back(start, end);
}

bool overlaps(const std::vector<std::pair<int, int>> &ranges, int start, int end)
{
  for (const auto &[from, to] : ranges)
    if (start < to && end > from)
      return true;
  return false;
}

void computeMel(AudioData &audioData)
{
  const auto &original = audioData.originalWaveform;
  MelSpectrogram mel(audioData.sampleRate > 0 ? audioData.sampleRate : SAMPLE_RATE,
                     N_FFT, HOP_SIZE, NUM_MELS, FMIN, FMAX);
  audioData.melSpectrogram = mel.compute(original.getReadPointer(0), original.getNumSamples());
  // As ProjectSerializer: frames ahead of the timeline offset are silent.
  const int offsetFrames = juce::jlimit(
      0, static_cast<int>(audioData.melSpectrogram.size()),
      static_cast<int>(std::llround(audioData.timelineOffsetSeconds * audioData.sampleRate /
                                    static_cast<double>(HOP_SIZE))));
  for (int i = 0; i < offsetFrames; ++i)
    std::fill(audioData.melSpectrogram[static_cast<size_t>(i)].begin(),
              audioData.melSpectrogram[static_cast<size_t>(i)].end(), 0.0f);
}
} // namespace

namespace
{
// FNV-1a over the samples' bits, channel by channel.
std::uint32_t hashBlock(const juce::AudioBuffer<float> &audio, int from, int to)
{
  std::uint32_t hash = 2166136261u;
  for (int channel = 0; channel < audio.getNumChannels(); ++channel)
  {
    const float *data = audio.getReadPointer(channel);
    for (int i = from; i < to; ++i)
    {
      std::uint32_t bits = 0;
      std::memcpy(&bits, data + i, sizeof bits);
      for (int b = 0; b < 4; ++b)
      {
        hash ^= (bits >> (8 * b)) & 0xFFu;
        hash *= 16777619u;
      }
    }
  }
  return hash;
}
} // namespace

recording::Fingerprint fingerprint(juce::AudioFormatReader &reader)
{
  recording::Fingerprint result;
  if (reader.numChannels <= 0 || reader.lengthInSamples <= 0 || reader.sampleRate <= 0.0)
    return result;
  result.sampleRate = static_cast<int>(std::lround(reader.sampleRate));
  result.channels = static_cast<int>(reader.numChannels);
  result.samples = reader.lengthInSamples;
  result.blockSamples = recording::kBlockSamples;
  constexpr int kBlocksPerRead = 64;
  juce::AudioBuffer<float> chunk(result.channels, kBlocksPerRead * result.blockSamples);
  for (juce::int64 at = 0; at < result.samples; at += chunk.getNumSamples())
  {
    const int length =
        static_cast<int>(std::min<juce::int64>(chunk.getNumSamples(), result.samples - at));
    chunk.clear();
    if (!reader.read(&chunk, 0, length, at, true, true))
      return {};
    for (int from = 0; from < length; from += result.blockSamples)
      result.hashes.push_back(hashBlock(chunk, from, std::min(length, from + result.blockSamples)));
  }
  return result;
}

recording::Fingerprint fingerprint(const juce::AudioBuffer<float> &source, int sampleRate)
{
  recording::Fingerprint result;
  const int channels = source.getNumChannels();
  const int samples = source.getNumSamples();
  if (channels <= 0 || samples <= 0 || sampleRate <= 0)
    return result;
  result.sampleRate = sampleRate;
  result.channels = channels;
  result.samples = samples;
  result.blockSamples = recording::kBlockSamples;
  const int blocks = (samples + result.blockSamples - 1) / result.blockSamples;
  result.hashes.resize(static_cast<size_t>(blocks));
  for (int block = 0; block < blocks; ++block)
  {
    const int from = block * result.blockSamples;
    result.hashes[static_cast<size_t>(block)] =
        hashBlock(source, from, std::min(samples, from + result.blockSamples));
  }
  return result;
}

std::vector<std::pair<juce::int64, juce::int64>>
changedSamples(const recording::Fingerprint &before, const recording::Fingerprint &after)
{
  std::vector<std::pair<juce::int64, juce::int64>> ranges;
  if (!before.sameShape(after))
    return ranges;
  // Blocks of another size cannot be compared: all of it changed.
  if (before.blockSamples != after.blockSamples || before.hashes.size() != after.hashes.size())
  {
    ranges.emplace_back(0, after.samples);
    return ranges;
  }
  const auto block = static_cast<juce::int64>(after.blockSamples);
  for (size_t i = 0; i < after.hashes.size(); ++i)
  {
    if (before.hashes[i] == after.hashes[i])
      continue;
    const juce::int64 from = static_cast<juce::int64>(i) * block;
    const juce::int64 to = std::min(after.samples, from + block);
    if (!ranges.empty() && ranges.back().second == from)
      ranges.back().second = to;
    else
      ranges.emplace_back(from, to);
  }
  return ranges;
}

juce::AudioBuffer<float> analysisCopy(const juce::AudioBuffer<float> &source, int sampleRate)
{
  const int samples = source.getNumSamples();
  juce::AudioBuffer<float> mono(1, std::max(0, samples));
  if (samples <= 0 || source.getNumChannels() <= 0)
    return mono;
  if (source.getNumChannels() == 1)
    mono.copyFrom(0, 0, source, 0, 0, samples);
  else
  {
    const float *left = source.getReadPointer(0);
    const float *right = source.getReadPointer(1);
    float *out = mono.getWritePointer(0);
    for (int i = 0; i < samples; ++i)
      out[i] = (left[i] + right[i]) * 0.5f;
  }
  if (sampleRate != SAMPLE_RATE)
    return AudioResampler::resample(mono, sampleRate, SAMPLE_RATE);
  return mono;
}

std::vector<recording::Piece> editPieces(const AudioData &audioData)
{
  std::vector<recording::Piece> pieces;
  const auto &composite = audioData.waveform;
  const auto &reference = warp::reference(audioData);
  if (composite.getNumChannels() <= 0 || composite.getNumSamples() <= 0)
    return pieces;
  const float *edited = composite.getReadPointer(0);
  const int samples = composite.getNumSamples();
  const bool hasReference = &reference != &composite && reference.getNumChannels() > 0;
  const float *recorded = hasReference ? reference.getReadPointer(0) : nullptr;
  const int recordedSamples = hasReference ? reference.getNumSamples() : 0;
  const auto differs = [&](int i) {
    return recorded == nullptr || i >= recordedSamples || edited[i] != recorded[i];
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
    if (!runs.empty() && i - runs.back().second < kPieceJoin)
      runs.back().second = end;
    else
      runs.emplace_back(i, end);
    i = end;
  }
  for (const auto &[start, end] : runs)
  {
    recording::Piece piece;
    piece.start = start;
    piece.samples.assign(edited + start, edited + end);
    pieces.push_back(std::move(piece));
  }
  return pieces;
}

bool adoptRecording(Project &project)
{
  auto &audioData = project.getAudioData();
  if (audioData.sourceAudio.getNumSamples() <= 0 || audioData.sourceSampleRate <= 0)
    return false;
  auto fresh = analysisCopy(audioData.sourceAudio, audioData.sourceSampleRate);
  auto &composite = audioData.waveform;
  if (fresh.getNumSamples() != audioData.originalWaveform.getNumSamples() ||
      composite.getNumChannels() <= 0 || composite.getNumSamples() <= 0)
    return false;
  if (!audioData.warpMap.isIdentity() && audioData.warpedOriginal.getNumSamples() == 0)
    warp::rebuildReference(audioData);
  juce::AudioBuffer<float> before;
  before.makeCopyOf(warp::reference(audioData));
  // The rendered edits, as playback tells them from the recording.
  const auto edited = sourceaudio::editedRanges(
      composite.getReadPointer(0), composite.getNumSamples(), before.getReadPointer(0),
      before.getNumSamples(), 0, kPieceJoin);
  audioData.originalWaveform = std::move(fresh);
  warp::rebuildReference(audioData);
  const auto &after = warp::reference(audioData);
  int from = 0;
  const auto follow = [&](int start, int end) {
    end = std::min({end, composite.getNumSamples(), after.getNumSamples()});
    for (int channel = 0; channel < composite.getNumChannels() && end > start; ++channel)
      composite.copyFrom(channel, start, after, std::min(channel, after.getNumChannels() - 1),
                         start, end - start);
  };
  for (const auto &[start, end] : edited)
  {
    follow(from, start);
    from = end;
  }
  follow(from, composite.getNumSamples());
  return true;
}

bool hydrate(Project &project)
{
  auto &audioData = project.getAudioData();
  if (audioData.sourceAudio.getNumSamples() <= 0 || audioData.sourceSampleRate <= 0)
    return false;
  audioData.sampleRate = SAMPLE_RATE;
  audioData.originalWaveform = analysisCopy(audioData.sourceAudio, audioData.sourceSampleRate);
  if (audioData.originalWaveform.getNumSamples() <= 0)
    return false;
  computeMel(audioData);
  // The map takes the mel's length; then the reference at both rates.
  warp::rebuildReference(audioData);
  const auto &reference = warp::reference(audioData);
  audioData.waveform.makeCopyOf(reference);
  auto &pieces = project.getLinkedEditPieces();
  for (const auto &piece : pieces)
  {
    const int from = std::max(0, piece.start);
    const int to = std::min(audioData.waveform.getNumSamples(),
                            piece.start + static_cast<int>(piece.samples.size()));
    for (int channel = 0; channel < audioData.waveform.getNumChannels(); ++channel)
      for (int i = from; i < to; ++i)
        audioData.waveform.setSample(channel, i,
                                     piece.samples[static_cast<size_t>(i - piece.start)]);
  }
  pieces.clear();
  pieces.shrink_to_fit();
  audioData.warpLayoutPending = false;

  for (auto &note : project.getNotes())
    NoteAmplitude::update(note, audioData);
  audioData.breathSibilance.clear();
  breathsibilance::ensure(audioData);
  // As ProjectSerializer::loadFromFile, now that the frames are known.
  PitchCurveProcessor::refitUntouchedCurves(project, 0, audioData.getNumFrames());
  return true;
}

Update apply(Project &project, const recording::Fingerprint &saved, const Detect &detect)
{
  const auto &audioData = project.getAudioData();
  return apply(project, saved, fingerprint(audioData.sourceAudio, audioData.sourceSampleRate),
               detect);
}

Update apply(Project &project, const recording::Fingerprint &saved,
             const recording::Fingerprint &now, const Detect &detect)
{
  Update update;
  auto &audioData = project.getAudioData();
  const auto changed = changedSamples(saved, now);
  project.setRecordingFingerprint(now);
  const int frames = audioData.getNumFrames();
  const auto &original = audioData.originalWaveform;
  if (changed.empty() || frames <= 0 || original.getNumSamples() <= 0)
    return update;

  // The changed blocks in analysis frames, with what the conversion and the
  // mel spread them over.
  const double toAnalysis = static_cast<double>(SAMPLE_RATE) / now.sampleRate;
  for (const auto &[from, to] : changed)
  {
    const double first = static_cast<double>(from) * toAnalysis - kSpreadSamples;
    const double last = static_cast<double>(to) * toAnalysis + kSpreadSamples;
    addRange(update.audioFrames,
             std::max(0, static_cast<int>(std::floor(first / HOP_SIZE))),
             std::min(frames, static_cast<int>(std::ceil(last / HOP_SIZE))));
  }

  // The pitch there detected again, each part with context around it.
  const size_t pitchFrames = std::min({audioData.rawF0.size(), audioData.voicedMask.size(),
                                       audioData.cleanedF0.size(), audioData.denseF0.size(),
                                       static_cast<size_t>(frames)});
  auto raw = audioData.rawF0;
  auto voiced = audioData.voicedMask;
  const int context = static_cast<int>(kContextSeconds * SAMPLE_RATE / HOP_SIZE);
  for (const auto &[from, to] : update.audioFrames)
  {
    const int windowStart = std::max(0, from - context);
    const int windowEnd = std::min(frames, to + context);
    const int sampleStart = windowStart * HOP_SIZE;
    const int sampleEnd = std::min(original.getNumSamples(), windowEnd * HOP_SIZE);
    std::vector<float> f0;
    std::vector<bool> isVoiced;
    if (sampleEnd <= sampleStart || !detect ||
        !detect(original.getReadPointer(0, sampleStart), sampleEnd - sampleStart,
                windowEnd - windowStart, f0, isVoiced))
    {
      update.detected = false;
      continue;
    }
    for (int t = from; t < to && static_cast<size_t>(t) < pitchFrames; ++t)
    {
      const auto local = static_cast<size_t>(t - windowStart);
      if (local < f0.size() && local < isVoiced.size())
      {
        raw[static_cast<size_t>(t)] = f0[local];
        voiced[static_cast<size_t>(t)] = isVoiced[local];
      }
    }
  }

  // The later analysis steps on the old and the new detection: what differs
  // is what the change did (the stored values are kept everywhere else).
  const auto oldClean = F0Smoother::removeOutliers(audioData.rawF0, 1.5f);
  const auto oldDense = PitchCurveProcessor::interpolateWithUvMask(oldClean, audioData.voicedMask);
  const auto newClean = F0Smoother::removeOutliers(raw, 1.5f);
  const auto newDense = PitchCurveProcessor::interpolateWithUvMask(newClean, voiced);
  std::vector<float> shift(static_cast<size_t>(frames), 0.0f); // semitones, new - old
  std::vector<bool> pitchChanged(static_cast<size_t>(frames), false);
  for (size_t t = 0; t < pitchFrames && t < oldDense.size() && t < newDense.size(); ++t)
  {
    if (raw[t] == audioData.rawF0[t] && voiced[t] == audioData.voicedMask[t] &&
        newClean[t] == oldClean[t] && newDense[t] == oldDense[t])
      continue;
    pitchChanged[t] = true;
    if (audioData.denseF0[t] > 0.0f && newDense[t] > 0.0f)
      shift[t] = freqToMidi(newDense[t]) - freqToMidi(audioData.denseF0[t]);
    audioData.cleanedF0[t] = newClean[t];
    audioData.denseF0[t] = newDense[t];
  }
  audioData.rawF0 = std::move(raw);
  audioData.voicedMask = std::move(voiced);
  for (int t = 0; t < frames; ++t)
    if (pitchChanged[static_cast<size_t>(t)])
      addRange(update.pitchFrames, t, t + 1);

  // Energy voicing as the analysis computes it, where the audio changed.
  if (audioData.vadMask.size() >= static_cast<size_t>(frames))
    for (const auto &[from, to] : update.audioFrames)
      for (int t = from; t < to; ++t)
      {
        const int start = t * HOP_SIZE;
        const int end = std::min(start + HOP_SIZE, original.getNumSamples());
        if (start >= end)
        {
          audioData.vadMask[static_cast<size_t>(t)] = false;
          continue;
        }
        float sum = 0.0f;
        for (int i = start; i < end; ++i)
          sum += original.getSample(0, i) * original.getSample(0, i);
        audioData.vadMask[static_cast<size_t>(t)] =
            std::sqrt(sum / static_cast<float>(end - start)) > 0.008f;
      }
  audioData.breathSibilance.clear();
  breathsibilance::ensure(audioData);

  // Every note's analysed curve takes the difference, read where each value
  // sits on the recording.
  const auto shiftAt = [&](double source) {
    const double clamped = std::clamp(source, 0.0, static_cast<double>(frames - 1));
    const auto left = static_cast<size_t>(std::floor(clamped));
    const auto right = std::min(left + 1, static_cast<size_t>(frames - 1));
    const float amount = static_cast<float>(clamped - static_cast<double>(left));
    return shift[left] * (1.0f - amount) + shift[right] * amount;
  };
  // Frames near a pitch change, where a committed contour is compared with
  // the analysed curve.
  std::vector<bool> nearChange(static_cast<size_t>(frames), false);
  for (const auto &[from, to] : update.pitchFrames)
    for (int t = std::max(0, from - kFollowReach); t < std::min(frames, to + kFollowReach); ++t)
      nearChange[static_cast<size_t>(t)] = true;
  for (auto &note : project.getNotes())
  {
    if (note.isRest() ||
        !overlaps(update.pitchFrames, note.getSrcStartFrame() - 1, note.getSrcEndFrame() + 1))
      continue;
    const int srcStart = note.getSrcStartFrame();
    const int srcLength = note.getSrcDurationFrames();
    const auto map = notetiming::forNote(audioData, note);
    // A curve as long as the note plays one value per timeline frame; any
    // other length is spread over the note's recorded frames (NoteTiming.h).
    const auto sourceOf = [&](int i, int length) {
      return length == note.getDurationFrames()
                 ? map.sourceAt(static_cast<double>(note.getStartFrame() + i) + 0.5) - 0.5
                 : srcStart + (i + 0.5) * srcLength / static_cast<double>(length) - 0.5;
    };
    const auto valueAt = [&](const std::vector<float> &curve, double source) {
      const auto length = static_cast<int>(curve.size());
      const double index =
          length == note.getDurationFrames()
              ? map.destinationAt(source + 0.5) - 0.5 - note.getStartFrame()
              : (source + 0.5 - srcStart) * length / static_cast<double>(srcLength) - 0.5;
      const double clamped = std::clamp(index, 0.0, static_cast<double>(length - 1));
      const auto left = static_cast<size_t>(std::floor(clamped));
      const auto right = std::min(left + 1, static_cast<size_t>(length - 1));
      const float amount = static_cast<float>(clamped - static_cast<double>(left));
      return curve[left] * (1.0f - amount) + curve[right] * amount;
    };
    const auto patched = [&](std::vector<float> curve, float share) {
      const auto length = static_cast<int>(curve.size());
      for (int i = 0; i < length; ++i)
        curve[static_cast<size_t>(i)] += share * shiftAt(sourceOf(i, length));
      return curve;
    };

    // A committed contour (a tool's result, or a split keeping the sung
    // pitch) has no edit left to apply again. It takes the difference as far
    // as it followed the recording around the change: its share of the
    // analysed curve's shape there. A kept contour takes all of it, a
    // flattened vibrato none (else the old pitch would come back mirrored),
    // a halved one half.
    float follows = 1.0f;
    if (note.hasBakedDeltaPitch() && note.hasOriginalDeltaPitch())
    {
      const auto &baked = note.getBakedDeltaPitch();
      const auto &analysed = note.getOriginalDeltaPitch();
      std::vector<std::pair<double, double>> pairs; // (analysed, baked)
      const auto length = static_cast<int>(baked.size());
      for (int i = 0; i < length; ++i)
      {
        const double source = sourceOf(i, length);
        const auto frame = static_cast<int>(std::lround(source));
        if (frame >= 0 && frame < frames && nearChange[static_cast<size_t>(frame)])
          pairs.emplace_back(valueAt(analysed, source), baked[static_cast<size_t>(i)]);
      }
      follows = shareFollowed(pairs);
    }
    const bool scratchIsBaked =
        note.hasBakedDeltaPitch() && note.getDeltaPitch() == note.getBakedDeltaPitch();
    if (note.hasBakedDeltaPitch())
      note.setBakedDeltaPitch(patched(note.getBakedDeltaPitch(), follows));
    if (!note.getDeltaPitch().empty())
      note.setDeltaPitch(patched(note.getDeltaPitch(), scratchIsBaked ? follows : 1.0f));
    if (note.hasOriginalDeltaPitch())
      note.setOriginalDeltaPitch(patched(note.getOriginalDeltaPitch(), 1.0f));
    auto spans = note.getAnalysisSpans();
    for (auto &span : spans)
    {
      const auto length = static_cast<int>(span.originalDelta.size());
      for (int i = 0; i < length; ++i)
        span.originalDelta[static_cast<size_t>(i)] += shiftAt(
            span.first + (i + 0.5) * (span.second - span.first) / static_cast<double>(length) - 0.5);
    }
    note.setAnalysisSpans(std::move(spans));
    auto recorded = note.getF0Values();
    if (static_cast<int>(recorded.size()) == srcLength)
    {
      for (int i = 0; i < srcLength; ++i)
        if (const int t = srcStart + i; t >= 0 && t < frames && pitchChanged[static_cast<size_t>(t)])
          recorded[static_cast<size_t>(i)] = audioData.denseF0[static_cast<size_t>(t)];
      note.setF0Values(std::move(recorded));
    }
  }
  // A drawn curve's line is the user's own pitch and stays. The pitch it
  // replaced, given back when an anchor is moved or removed, was the
  // recording's: it takes the difference (per timeline frame).
  for (auto &curve : project.getDrawnCurves())
    for (size_t i = 0; i < curve.underDiff.size(); ++i)
      curve.underDiff[i] += shiftAt(
          audioData.warpMap.sourceAt(static_cast<double>(curve.underStart) + i + 0.5) - 0.5);
  PitchCurveProcessor::rebuildBaseFromNotes(project);
  PitchCurveProcessor::refitUntouchedCurves(project, 0, frames);

  // Where the audio or the pitch changed, on the timeline.
  const auto &map = audioData.warpMap;
  std::vector<std::pair<int, int>> timeline;
  {
    std::vector<std::pair<int, int>> all = update.audioFrames;
    all.insert(all.end(), update.pitchFrames.begin(), update.pitchFrames.end());
    std::sort(all.begin(), all.end());
    for (const auto &[from, to] : all)
      addRange(timeline, static_cast<int>(std::floor(map.destinationAt(from))),
               static_cast<int>(std::ceil(map.destinationAt(to))));
  }
  // Renders there were made from the old audio: back to the recording, and
  // every edited note they cover is rendered again.
  auto &composite = audioData.waveform;
  const auto &reference = warp::reference(audioData);
  if (composite.getNumSamples() > 0 && &reference != &composite &&
      reference.getNumSamples() > 0)
  {
    std::vector<std::pair<int, int>> stale;
    for (const auto &piece : editPieces(audioData))
    {
      const int end = piece.start + static_cast<int>(piece.samples.size());
      if (!overlaps(timeline, piece.start / HOP_SIZE, end / HOP_SIZE + 1))
        continue;
      const int to = std::min(end, reference.getNumSamples());
      for (int channel = 0; channel < composite.getNumChannels(); ++channel)
        composite.copyFrom(channel, piece.start, reference,
                           std::min(channel, reference.getNumChannels() - 1), piece.start,
                           std::max(0, to - piece.start));
      stale.emplace_back(piece.start / HOP_SIZE, end / HOP_SIZE + 1);
    }
    for (const auto &range : stale)
      timeline.push_back(range);
    std::sort(timeline.begin(), timeline.end());
  }
  for (auto &note : project.getNotes())
    if (!note.isRest() && overlaps(timeline, note.getStartFrame(), note.getEndFrame()) &&
        (!note.isNeutralForOriginalWaveform() || note.hasRenderedEdit()))
    {
      note.markDirty();
      note.markSynthDirty();
      ++update.notesToRender;
    }
  project.setModified(true);
  return update;
}

juce::String describe(const recording::Fingerprint &shape)
{
  return juce::String(shape.sampleRate) + " Hz, " + juce::String(shape.channels) + " ch, " +
         juce::String(static_cast<double>(shape.samples) / std::max(1, shape.sampleRate), 3) +
         " s (" + juce::String(shape.samples) + ")";
}

recording::Fingerprint shapeOf(const juce::File &file)
{
  recording::Fingerprint shape;
  juce::AudioFormatManager formats;
  formats.registerBasicFormats();
  std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
  if (reader == nullptr || reader->sampleRate <= 0.0)
    return shape;
  shape.sampleRate = static_cast<int>(std::lround(reader->sampleRate));
  shape.channels = static_cast<int>(reader->numChannels);
  shape.samples = reader->lengthInSamples;
  return shape;
}

Opened openLinked(Project &project, const juce::File &file, const Detect &detect)
{
  Opened opened;
  const auto saved = project.getRecordingFingerprint();
  opened.found = shapeOf(file);
  if (opened.found.sampleRate <= 0 || opened.found.samples <= 0)
    return opened;
  if (!saved.sameShape(opened.found))
  {
    opened.result = Opened::Result::otherShape;
    return opened;
  }
  auto &audioData = project.getAudioData();
  if (!sourceaudio::load(audioData, file) || !hydrate(project))
    return opened;
  project.setFilePath(file);
  opened.result = Opened::Result::opened;
  const auto sha = SHA256Utils::fileSHA256(file);
  if (sha.isNotEmpty() && sha.equalsIgnoreCase(project.getAudioSha256()))
    return opened;
  // Another file content: compare block by block. A file rewritten with the
  // same samples (only its header or markers changed) changes nothing.
  opened.update = apply(project, saved, detect);
  opened.changed = !opened.update.audioFrames.empty();
  project.setAudioSha256(sha);
  project.setModified(true);
  return opened;
}
} // namespace recordingchange
