#pragma once

#include "../BreathSibilanceDetector.h"

#include <algorithm>
#include <cstdint>
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

} // namespace breathsibilance
