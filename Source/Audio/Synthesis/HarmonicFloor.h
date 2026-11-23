#pragma once

#include "../../JuceHeader.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <vector>

/**
 * A voice's harmonics pressed down to the noise around them.
 *
 * What an edited vowel leaves under the consonant after it is the recording,
 * on the recorded pitch (BreathSibilanceProtection.h). That pitch is known,
 * so its harmonics are: around each multiple of it, up to `topHz`, the
 * spectrum is brought down to the level of the gaps on either side - the
 * median of the bins between two harmonics, interpolated across the band -
 * never up, and the phases are left alone. The consonant's own noise in the
 * band stays (about 3 dB lower: its peaks over the median go too), as when a
 * harmonic is marked in a spectral editor and attenuated to its surroundings.
 *
 * STFT, Hann analysis and synthesis, hop an eighth of the window. A band is
 * at least the Hann main lobe wide, and the window is the shortest that
 * still leaves gaps between the bands: 46 ms from 237 Hz up at 44.1 kHz, 93
 * or 186 ms under it. A pitch too low for that (under about 35 Hz) is left.
 */
namespace harmonicfloor
{
constexpr double kBandShare = 0.2;    // half a band, as a share of the pitch
constexpr double kMainLobeBins = 2.2; // half the Hann main lobe, with a margin
constexpr int kMinOrder = 11;
constexpr int kMaxOrder = 13;

/**
 * Where `weight` is 1 (in between, that share of it), `audio` loses the
 * harmonics of `pitchHz` under `topHz` to the level around them. Both are
 * per sample and cover the same span. False if cancelled.
 */
inline bool flatten(std::vector<float> &audio, const std::vector<float> &weight,
                    int sampleRate, double pitchHz, double topHz,
                    const std::atomic<bool> *cancel = nullptr)
{
  const int length = static_cast<int>(std::min(audio.size(), weight.size()));
  if (length <= 0 || sampleRate <= 0 || pitchHz <= 0.0)
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

  // The shortest window whose bins tell a band from the gap beside it.
  int order = kMinOrder;
  while (order < kMaxOrder &&
         kBandShare * pitchHz < kMainLobeBins * sampleRate / static_cast<double>(1 << order))
    ++order;
  const int size = 1 << order;
  const int hop = size / 8;
  const double binHz = static_cast<double>(sampleRate) / size;
  const double half = std::max(kBandShare * pitchHz, kMainLobeBins * binHz);
  if (pitchHz - 2.0 * half < 2.0 * binHz)
    return true;

  std::vector<float> window(static_cast<size_t>(size));
  for (int i = 0; i < size; ++i)
    window[static_cast<size_t>(i)] = static_cast<float>(
        0.5 - 0.5 * std::cos(2.0 * juce::MathConstants<double>::pi * i / size));

  const int from = first - size;
  const int to = last + size;
  const int span = to - from;
  std::vector<float> sum(static_cast<size_t>(span + size), 0.0f);
  std::vector<float> norm(sum.size(), 0.0f);
  juce::dsp::FFT fft(order);
  std::vector<juce::dsp::Complex<float>> time(static_cast<size_t>(size)),
      spectrum(static_cast<size_t>(size));
  const int bins = size / 2;
  std::vector<float> magnitude(static_cast<size_t>(bins + 1));
  std::vector<float> gap;
  // The median level of the bins in [lowHz, highHz); 0 if there are none.
  const auto level = [&](double lowHz, double highHz)
  {
    const int a = std::max(1, static_cast<int>(std::ceil(lowHz / binHz)));
    const int b = std::min(bins, static_cast<int>(std::ceil(highHz / binHz)));
    if (b <= a)
      return 0.0f;
    gap.assign(magnitude.begin() + a, magnitude.begin() + b);
    std::nth_element(gap.begin(), gap.begin() + gap.size() / 2, gap.end());
    return gap[gap.size() / 2];
  };

  for (int start = 0; start < span; start += hop)
  {
    if (cancel != nullptr && cancel->load())
      return false;
    for (int i = 0; i < size; ++i)
    {
      const int n = from + start + i;
      const float sample = n >= 0 && n < length ? audio[static_cast<size_t>(n)] : 0.0f;
      time[static_cast<size_t>(i)] = {sample * window[static_cast<size_t>(i)], 0.0f};
    }
    fft.perform(time.data(), spectrum.data(), false);
    for (int k = 0; k <= bins; ++k)
      magnitude[static_cast<size_t>(k)] = std::abs(spectrum[static_cast<size_t>(k)]);

    for (int harmonic = 1; harmonic * pitchHz - half < topHz; ++harmonic)
    {
      const double low = harmonic * pitchHz - half;
      const double high = harmonic * pitchHz + half;
      float below = level(std::max(60.0, (harmonic - 1) * pitchHz + half), low);
      float above = level(high + binHz, (harmonic + 1) * pitchHz - half);
      if (below <= 0.0f && above <= 0.0f)
        continue;
      if (below <= 0.0f)
        below = above;
      if (above <= 0.0f)
        above = below;
      const double logBelow = std::log(static_cast<double>(below) + 1.0e-12);
      const double logAbove = std::log(static_cast<double>(above) + 1.0e-12);
      const int a = std::max(1, static_cast<int>(std::ceil(low / binHz)));
      const int b = std::min(bins - 1, static_cast<int>(std::floor(high / binHz)));
      for (int k = a; k <= b; ++k)
      {
        const double across = std::clamp((k * binHz - low) / (high - low), 0.0, 1.0);
        const double floor = std::exp(logBelow + across * (logAbove - logBelow));
        const float gain = static_cast<float>(
            std::min(1.0, floor / (static_cast<double>(magnitude[static_cast<size_t>(k)]) + 1.0e-12)));
        spectrum[static_cast<size_t>(k)] *= gain;
        spectrum[static_cast<size_t>(size - k)] *= gain;
      }
    }

    fft.perform(spectrum.data(), time.data(), true);
    for (int i = 0; i < size; ++i)
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
    auto &sample = audio[static_cast<size_t>(i)];
    sample += w * (sum[at] / std::max(norm[at], 1.0e-9f) - sample);
  }
  return true;
}
} // namespace harmonicfloor
