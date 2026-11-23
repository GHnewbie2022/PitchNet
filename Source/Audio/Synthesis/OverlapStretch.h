#pragma once

#include <functional>

/**
 * A stretch that moves the recording's own waveform: short windows of it
 * laid over each other in time (WSOLA).
 *
 * For vocal fry (WarpedSource.h). Its pulses come apart, 8 to 25 ms from one
 * to the next, and a stretch that rebuilds phases in the spectrum cannot
 * tell them from each other inside one window: signalsmith turned the end of
 * a phrase into a buzz at every window tried (35, 55, 80, 120 ms). Laid over
 * each other, each pulse stays the pulse that was recorded; there are just
 * more or fewer of them. (Where two windows hold pulses at different
 * spacings, one can come out twice at part of its level.)
 *
 * Every hop a window of the recording is taken from where the map says that
 * time plays, moved by up to the search either way to where it best goes on
 * from the window before - the one that would have followed it in the
 * recording. At a ratio of 1 that is the recording itself.
 *
 * Not for breath: noise laid over a copy of itself a few ms off combs (heard
 * as phasiness at 1.10x with 32 ms windows, as a flutter with 64 ms), see
 * NoiseStretch.h. Not for plain voice either, where signalsmith with a 35 ms
 * window was preferred.
 */
namespace warp
{
struct OverlapStretchSettings
{
  double sampleRate = 44100.0;
  double windowSeconds = 0.032;
  double hopSeconds = 0.012;
  double searchSeconds = 0.008;
};

/**
 * Timeline samples [first, last) of `source` into `output` (last - first
 * samples). `inputAt(timeline sample)` is the recording's sample that plays
 * there.
 *
 * The windows start a window's length before `first`, so `first` is fully
 * covered. `alignTo`, if given, is another rendering of the same timeline,
 * `alignSamples` samples from timeline sample `alignFirst`: the first window
 * is placed where it matches that rendering best, so the two start out in
 * phase and a crossfade between them at `first` does not cancel. Where it
 * does not cover the first window, that window sits where the map puts it.
 */
void overlapStretch(const float *source, int sourceSamples,
                    const std::function<double(double)> &inputAt, int first,
                    int last, float *output,
                    const OverlapStretchSettings &settings,
                    const float *alignTo = nullptr, int alignFirst = 0,
                    int alignSamples = 0);
} // namespace warp
