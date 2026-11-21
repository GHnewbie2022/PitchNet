#pragma once

#include "../../JuceHeader.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <vector>

/**
 * Spec Q12, the third way to render an edited note ("vocoder + recorded
 * highs"): the vocoder's render below 10 kHz, the recording above it.
 *
 * The vocoder turns the harmonics above about 10 kHz that follow the glottal
 * cycle into noise, and a voice with that fine rasp in it ("麻感") comes out
 * loose and electronic (煙幕 02:48.124: the coherent part of 10.5-15 kHz
 * down 10.2 dB). Put back from the recording, unshifted, the texture is
 * there again and the pitch edit is not heard to disagree with it up to 3
 * semitones (listening tests 2026-11-21 and 22, also on stretched notes,
 * where the recording is the one the warp layer laid out).
 *
 * The split is a complementary STFT mask (Hann analysis and synthesis, hop
 * a quarter of the window), as in the listening tests' bandmix.py: with a
 * weight of 1 the result is the render below the crossover and the
 * recording above it, and the two always sum back.
 */
namespace highband
{
constexpr double kCrossoverHz = 10000.0;
constexpr double kHalfWidthHz = 100.0; // raised-cosine transition, each side
constexpr int kFftOrder = 11;
constexpr int kFftSize = 1 << kFftOrder;
constexpr int kHop = kFftSize / 4;

/**
 * `render` takes the recording's band above the crossover where `weight` is
 * 1 (in between, that share of it): render += weight * highpass(recording -
 * render). All three are per sample and cover the same span. False if
 * cancelled.
 */
inline bool restore(std::vector<float> &render, const std::vector<float> &recording,
                    const std::vector<float> &weight, int sampleRate,
                    const std::atomic<bool> *cancel = nullptr)
{
  const int length = static_cast<int>(
      std::min({render.size(), recording.size(), weight.size()}));
  if (length <= 0 || sampleRate <= 0)
    return true;
  int first = -1, last = -1;
  for (int i = 0; i < length; ++i)
    if (weight[static_cast<size_t>(i)] > 0.0f)
    {
      if (first < 0)
        first = i;
      last = i;
    }
  if (first < 0)
    return true;

  std::vector<float> window(static_cast<size_t>(kFftSize));
  for (int i = 0; i < kFftSize; ++i)
    window[static_cast<size_t>(i)] = static_cast<float>(
        0.5 - 0.5 * std::cos(2.0 * juce::MathConstants<double>::pi * i / kFftSize));
  // The share of each bin that comes from the recording.
  std::vector<float> high(static_cast<size_t>(kFftSize), 0.0f);
  for (int k = 0; k < kFftSize; ++k)
  {
    const double hz = static_cast<double>(std::min(k, kFftSize - k)) * sampleRate / kFftSize;
    const double x = std::clamp((hz - (kCrossoverHz - kHalfWidthHz)) / (2.0 * kHalfWidthHz),
                                0.0, 1.0);
    const double s = std::sin(x * juce::MathConstants<double>::halfPi);
    high[static_cast<size_t>(k)] = static_cast<float>(s * s);
  }

  // Whole frames from a window before the first weighted sample to a window
  // past the last, so every weighted sample is covered by four frames.
  const int from = first - kFftSize;
  const int to = last + kFftSize;
  const int span = to - from;
  std::vector<float> sum(static_cast<size_t>(span + kFftSize), 0.0f);
  std::vector<float> norm(sum.size(), 0.0f);
  juce::dsp::FFT fft(kFftOrder);
  std::vector<juce::dsp::Complex<float>> time(static_cast<size_t>(kFftSize)),
      spectrum(static_cast<size_t>(kFftSize));
  for (int start = 0; start < span; start += kHop)
  {
    if (cancel != nullptr && cancel->load())
      return false;
    for (int i = 0; i < kFftSize; ++i)
    {
      const int n = from + start + i;
      const float difference =
          n >= 0 && n < length ? recording[static_cast<size_t>(n)] - render[static_cast<size_t>(n)]
                               : 0.0f;
      time[static_cast<size_t>(i)] = {difference * window[static_cast<size_t>(i)], 0.0f};
    }
    fft.perform(time.data(), spectrum.data(), false);
    for (int k = 0; k < kFftSize; ++k)
      spectrum[static_cast<size_t>(k)] *= high[static_cast<size_t>(k)];
    fft.perform(spectrum.data(), time.data(), true);
    for (int i = 0; i < kFftSize; ++i)
    {
      const auto at = static_cast<size_t>(start + i);
      sum[at] += time[static_cast<size_t>(i)].real() * window[static_cast<size_t>(i)];
      norm[at] += window[static_cast<size_t>(i)] * window[static_cast<size_t>(i)];
    }
  }
  for (int i = first; i <= last; ++i)
  {
    const float w = weight[static_cast<size_t>(i)];
    if (w <= 0.0f)
      continue;
    const auto at = static_cast<size_t>(i - from);
    render[static_cast<size_t>(i)] += w * sum[at] / std::max(norm[at], 1.0e-9f);
  }
  return true;
}
} // namespace highband
