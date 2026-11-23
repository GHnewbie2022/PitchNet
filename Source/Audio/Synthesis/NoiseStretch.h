#pragma once

#include <cstdint>
#include <functional>

/**
 * Breath and unvoiced sibilance stretched by making their noise again.
 *
 * Noise has no phase to keep, and every stretch that tries to keep one
 * leaves a mark in it: signalsmith with a 35 ms window was heard as plainly
 * phasey on an inhale, with 120 ms as slightly so, and windows of the
 * recording laid over each other comb (OverlapStretch.h). So only the shape
 * of the recording's spectrum is taken, where the map says each moment
 * plays, and the phases are drawn at random. The colour and the loudness of
 * the breath follow the recording; its waveform does not. Anything pitched
 * in it - a trace of voice, a click - comes out as noise too.
 *
 * STFT, Hann analysis and synthesis, hop a quarter of the window. The
 * window is the power of two nearest to windowSeconds (1024 samples at 44.1
 * and 48 kHz). Windows with unrelated phases add in power, not amplitude:
 * overlapped N / hop times and divided by the summed squared windows, the
 * output carries hop / N of the input's power whatever the window's shape,
 * so it is scaled by sqrt(N / hop) = 2.
 *
 * The phases come from std::mt19937's raw output (the same on every
 * platform; the standard's distributions are not), seeded by the caller:
 * the same stretch always renders the same audio.
 */
namespace warp
{
struct NoiseStretchSettings
{
  double sampleRate = 44100.0;
  double windowSeconds = 0.0213;
  std::uint32_t seed = 0;
};

/** The window, in samples, at this rate. */
int noiseStretchWindow(const NoiseStretchSettings &settings);

/**
 * Timeline samples [first, last) of `source` into `output` (last - first
 * samples). `inputAt(timeline sample)` is the recording's sample that plays
 * there.
 */
void noiseStretch(const float *source, int sourceSamples,
                  const std::function<double(double)> &inputAt, int first,
                  int last, float *output, const NoiseStretchSettings &settings);
} // namespace warp
