#pragma once

#include "../WarpMap.h"

#include <vector>

/**
 * The recording laid out on the timeline by the warp map - the layer every
 * other kind of processing is blended onto (spec Q1):
 *
 *   not stretched  the recording itself, moved to where the map puts it
 *   stretched      signalsmith-stretch, reading the recording through the map
 *
 * Neighbouring stretched segments are processed as one continuous run whose
 * rate changes at the anchors, so an anchor between two stretched segments
 * is not a seam. A run hands over to the copied recording inside unstretched
 * audio next to it, with a 10 ms crossfade placed where the recording is
 * quietest; there the stretch plays at rate 1, so both sides carry the same
 * audio and the fade cannot comb.
 *
 * The random seed is fixed: past a ratio of 2 signalsmith randomises phases,
 * and the same edit must always render the same audio.
 */
namespace warp
{
struct WarpedSourceSettings
{
  double sampleRate = 44100.0;
  int hopSize = 512;
  // Unstretched audio rendered with each run so it can hand over to the copy.
  double padSeconds = 0.06;
  double fadeSeconds = 0.010;
  long seed = 20261115L;
};

struct StretchedRun
{
  int startSample = 0; // timeline samples [startSample, endSample)
  int endSample = 0;
};

/**
 * Renders timeline samples [0, outputSamples) of `source` (mono, at the
 * map's hop) laid out by `map`. Returns the stretched runs, padding
 * included, so a caller can tell which samples are not plain copies.
 */
std::vector<StretchedRun> renderWarpedSource(const float *source,
                                             int sourceSamples,
                                             const WarpMap &map,
                                             float *output, int outputSamples,
                                             const WarpedSourceSettings &settings = {});
} // namespace warp
