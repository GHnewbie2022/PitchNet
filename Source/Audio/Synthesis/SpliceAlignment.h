#pragma once

#include <vector>

/**
 * Lining a vocoder render up with the audio it is crossfaded into.
 *
 * The vocoder rebuilds phase on its own, so at a splice the render can sit at
 * any phase against the recording - near the opposite phase the crossfade
 * cancels and the level dips (heard as a short notch where an edited note
 * meets an unedited one). Before the crossfade, the render is moved by a
 * small time shift (at most kMaxLagSamples, about 1 ms) and rotated in phase
 * (every frequency turned by the same angle, which moves nothing in time),
 * the pair that matches the recording best across the fade. Shift and
 * rotation together fit the phase difference as a line over frequency, which
 * neither does alone.
 *
 * A span with two splices gets its own pair at each, and between them the
 * shift and the angle glide from one to the other, so the render stays
 * continuous; the glide is slow enough that the pitch change it implies is a
 * few cents at most.
 */
namespace splicealign
{
constexpr int kMaxLagSamples = 40;

struct Alignment
{
  double lag = 0.0;   // render sample n is taken from n - lag
  double angle = 0.0; // radians
};

/** The imaginary part of the analytic signal of `x` (an FFT Hilbert transform). */
std::vector<float> hilbert(const std::vector<float> &x);

/**
 * The shift and rotation of `render` (with `renderHilbert`) that best match
 * `existing` over [centre - fadeHalf, centre + fadeHalf), weighted towards
 * the middle of the fade, where both sides are heard. Identity when the
 * window does not fit or either side is silent.
 */
Alignment find(const float *existing, const std::vector<float> &render,
               const std::vector<float> &renderHilbert, int centre,
               int fadeHalf);

/** The rendered sample at `n` shifted and rotated by `alignment`. */
float sample(const std::vector<float> &render,
             const std::vector<float> &renderHilbert, double n,
             const Alignment &alignment);
} // namespace splicealign
