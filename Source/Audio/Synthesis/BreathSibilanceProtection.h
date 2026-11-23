#pragma once

#include "../BreathSibilanceDetector.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace breathsibilance
{
/**
 * Keep detected breaths and sibilants on the original side of the
 * synthesizer's blend mask (per sample, 1 = synthesized, 0 = original).
 *
 * They carry no pitch to correct, so the recording is kept at any run length
 * rather than regenerated: the vocoder reproduces an "s" closely but not
 * exactly (about 1 dB duller above 3.5 kHz on vocadito, see
 * docs/breath-sibilance-detection.md).
 *
 * `mask` covers frames [startFrame, startFrame + numFrames) at hopSize
 * samples per frame; `labels` is indexed by absolute frame. A label run is
 * aligned to its frames' centres (frame k covers k +- 1/2 hop, as analysed)
 * and both fades lie inside it, so the crossfade only mixes noise with
 * noise - never the original voice with the shifted one, which combs. A run
 * cut by the range continues beyond it and is not faded at that end.
 */
inline void protectNoiseInBlendMask(std::vector<float> &mask,
                                    const std::vector<std::uint8_t> &labels,
                                    int startFrame, int numFrames, int hopSize,
                                    int fadeSamples = 256)
{
  if (labels.empty() || numFrames <= 0 || hopSize <= 0)
    return;

  const int numSamples = static_cast<int>(mask.size());
  const int halfHop = hopSize / 2;
  int frame = 0;
  while (frame < numFrames)
  {
    if (!isNoise(labelAt(labels, startFrame + frame)))
    {
      ++frame;
      continue;
    }
    const int runStart = frame;
    while (frame < numFrames && isNoise(labelAt(labels, startFrame + frame)))
      ++frame;
    const int runEnd = frame;

    const bool fadeIn = runStart > 0;
    const bool fadeOut = runEnd < numFrames;
    const int first =
        fadeIn ? std::clamp(runStart * hopSize - halfHop, 0, numSamples) : 0;
    const int last =
        fadeOut ? std::clamp(runEnd * hopSize - halfHop, 0, numSamples)
                : numSamples;
    const int length = last - first;
    if (length <= 0)
      continue;

    const int fade = std::min(fadeSamples, length / 4);
    for (int sample = first; sample < last; ++sample)
    {
      float keepOriginal = 1.0f;
      if (fadeIn && fade > 0 && sample < first + fade)
        keepOriginal = static_cast<float>(sample - first + 1) /
                       static_cast<float>(fade + 1);
      if (fadeOut && fade > 0 && sample >= last - fade)
        keepOriginal =
            std::min(keepOriginal, static_cast<float>(last - sample) /
                                       static_cast<float>(fade + 1));
      mask[static_cast<size_t>(sample)] *= 1.0f - keepOriginal;
    }
  }
}

/**
 * What an edited vowel leaves under the sibilant after it.
 *
 * A sibilant keeps the recording (above), and the recording there is not
 * the consonant alone. The vowel before it ends under the consonant's onset
 * - the detector grows a run into frames the pitch detector still calls
 * voiced (BreathSibilanceDetector.cpp, step 5) - and the room goes on
 * ringing with it under the rest: on 煙幕, recorded at home, a line on the
 * vowel's last pitch runs all through the "x" of 挑選 and of 呼吸, falling
 * some 365 dB a second, and in the middle of 84 of the 137 runs measured it
 * stands 12 dB or more over the gaps beside it. None of that follows a pitch
 * edit, so after one it sits off the pitch the vowel now has: 呼 moved 2
 * semitones left 50 ms 2.8 semitones away as it ended, heard as a wrong
 * note; 挑 moved 64 cents left a line seen in a spectrogram and not heard.
 * The vocoder does no better with those frames: that far under the vowel
 * its output follows the mel, which is the recording's.
 *
 * So after a vowel rendered with an edit, the harmonics of the pitch it was
 * recorded ending on are pressed down to the noise around them through the
 * run (HarmonicFloor.h), up to kVoiceTailTopHz, where the detector's high
 * band starts. From the run's first frame - its label begins some 25 ms
 * after the consonant's noise does, so nothing is held back there - to
 * kVoiceTailEndInsetSeconds before the consonant's end: the run's end, or
 * the first of the voiced frames it ends with, which are the next vowel's
 * (25 ms before the run's end left 20 ms of the line standing). A
 * sibilant is noise; what goes is not part of it. Chosen by ear and in a
 * spectral editor, 2026-11-23, over cutting the whole band under the
 * crossover (a hole in the consonant's noise) and over giving it to the
 * vocoder or to R3.
 *
 * Only there: an unedited note keeps every sample, and so does a run whose
 * vowel was not edited - its tail is on the pitch the vowel still has.
 */
constexpr double kVoiceTailTopHz = 3500.0;
constexpr double kVoiceTailEndInsetSeconds = 0.005;
constexpr int kVoiceTailReachFrames = 3; // a vowel ends this close before the run, or is not its vowel

struct VoiceTail
{
  int runStart = 0;     // the run up to the voiced frames it ends with,
  int runEnd = 0;       // frames [runStart, runEnd)
  int voiceFrame = 0;   // the vowel's last voiced frame
  float pitchHz = 0.0f; // the pitch it was recorded ending on
};

/**
 * The sibilant runs that reach into [startFrame, endFrame) and follow a
 * voice: one whose opening frames are voiced, or one that starts within
 * kVoiceTailReachFrames of a voiced frame. `labelAt`, `voicedAt` and
 * `pitchAt` (Hz, 0 where there is none) take any frame and answer None /
 * false / 0 outside the audio. The pitch is the median of the vowel's last
 * three voiced frames. A run with no unvoiced frame holds no consonant to
 * keep apart from a voice, and gives none.
 */
template <typename LabelAt, typename VoicedAt, typename PitchAt>
std::vector<VoiceTail> voiceTailsUnderSibilants(LabelAt &&labelAt, VoicedAt &&voicedAt,
                                                PitchAt &&pitchAt, int startFrame,
                                                int endFrame)
{
  std::vector<VoiceTail> tails;
  int frame = startFrame;
  while (frame < endFrame)
  {
    if (labelAt(frame) != Sibilance)
    {
      ++frame;
      continue;
    }
    // A run the range cuts started before it.
    int runStart = frame;
    while (labelAt(runStart - 1) == Sibilance)
      --runStart;
    int runEnd = frame;
    while (labelAt(runEnd) == Sibilance)
      ++runEnd;
    frame = runEnd;

    int voicedEnd = runStart;
    while (voicedEnd < runEnd && voicedAt(voicedEnd))
      ++voicedEnd;
    if (voicedEnd == runEnd)
      continue;
    // Voiced frames the run ends with are the next vowel starting.
    while (runEnd > voicedEnd && voicedAt(runEnd - 1))
      --runEnd;
    int voiceFrame = voicedEnd - 1;
    if (voicedEnd == runStart)
    {
      voiceFrame = runStart - 1;
      while (voiceFrame >= runStart - kVoiceTailReachFrames && !voicedAt(voiceFrame))
        --voiceFrame;
      if (voiceFrame < runStart - kVoiceTailReachFrames)
        continue;
    }
    float pitches[3];
    int count = 0;
    for (int k = voiceFrame; count < 3 && voicedAt(k); --k)
      if (const float hz = pitchAt(k); hz > 0.0f)
        pitches[count++] = hz;
    if (count == 0)
      continue;
    std::sort(pitches, pitches + count);
    tails.push_back({runStart, runEnd, voiceFrame, pitches[count / 2]});
  }
  return tails;
}

/** The frames [from, to) of a run that a pass writes, and the pitch to take. */
struct VoiceTailSpan
{
  int from = 0;
  int to = 0;
  bool toRunEnd = false; // `to` is where the run ends, not where the pass stops
  float pitchHz = 0.0f;
};

/**
 * How much of the tail goes, per sample, over `span` in `numSamples` samples
 * that start at frame `startFrame`. A run sits on its frames' centres, as in
 * protectNoiseInBlendMask. In over 12 ms centred on the span's first sample;
 * out over 12 ms centred kVoiceTailEndInsetSeconds before the consonant's
 * end (on the span's end where the pass stops short of it).
 */
inline std::vector<float> voiceTailWeight(const VoiceTailSpan &span, int startFrame,
                                          int numSamples, int hopSize, int sampleRate)
{
  std::vector<float> weight(static_cast<size_t>(std::max(0, numSamples)), 0.0f);
  const double half = 0.006 * sampleRate;
  const double first = static_cast<double>(span.from - startFrame) * hopSize - hopSize / 2;
  const double end = static_cast<double>(span.to - startFrame) * hopSize - hopSize / 2 -
                     (span.toRunEnd ? kVoiceTailEndInsetSeconds * sampleRate : 0.0);
  const auto raisedCosine = [](double x)
  { return 0.5 - 0.5 * std::cos(3.14159265358979323846 * std::clamp(x, 0.0, 1.0)); };
  const int begin = std::max(0, static_cast<int>(std::floor(first - half)));
  const int stop = std::min(numSamples, static_cast<int>(std::ceil(end + half)));
  for (int i = begin; i < stop; ++i)
    weight[static_cast<size_t>(i)] = static_cast<float>(
        std::min(raisedCosine((i - (first - half)) / (2.0 * half)),
                 raisedCosine(((end + half) - i) / (2.0 * half))));
  return weight;
}

} // namespace breathsibilance
