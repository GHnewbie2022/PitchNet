#include "NoiseStretch.h"

#include "../../JuceHeader.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace warp
{
namespace
{
int orderOf(const NoiseStretchSettings &settings)
{
  const double samples = std::max(16.0, settings.windowSeconds * settings.sampleRate);
  return std::clamp(static_cast<int>(std::lround(std::log2(samples))), 4, 16);
}
} // namespace

int noiseStretchWindow(const NoiseStretchSettings &settings)
{
  return 1 << orderOf(settings);
}

void noiseStretch(const float *source, int sourceSamples,
                  const std::function<double(double)> &inputAt, int first,
                  int last, float *output, const NoiseStretchSettings &settings)
{
  const int count = last - first;
  if (count <= 0 || output == nullptr)
    return;
  std::fill(output, output + count, 0.0f);
  if (source == nullptr || sourceSamples <= 0)
    return;

  const int order = orderOf(settings);
  const int size = 1 << order;
  const int hop = size / 4;
  const int bins = size / 2;
  constexpr double kTwoPi = 2.0 * 3.14159265358979323846;

  std::vector<float> shape(static_cast<size_t>(size));
  for (int i = 0; i < size; ++i)
    shape[static_cast<size_t>(i)] =
        static_cast<float>(0.5 - 0.5 * std::cos(kTwoPi * i / size));

  juce::dsp::FFT fft(order);
  std::vector<juce::dsp::Complex<float>> time(static_cast<size_t>(size)),
      spectrum(static_cast<size_t>(size));
  std::vector<float> sum(static_cast<size_t>(count), 0.0f);
  std::vector<float> norm(static_cast<size_t>(count), 0.0f);
  std::mt19937 random(settings.seed);
  const auto turn = [&random]()
  { return kTwoPi * (static_cast<double>(random()) / 4294967296.0); };

  for (int start = first - size; start < last; start += hop)
  {
    const long long from =
        std::llround(inputAt(static_cast<double>(start) + 0.5 * size)) - size / 2;
    for (int i = 0; i < size; ++i)
    {
      const long long position = from + i;
      const float sample = position >= 0 && position < sourceSamples
                               ? source[static_cast<size_t>(position)]
                               : 0.0f;
      time[static_cast<size_t>(i)] = {sample * shape[static_cast<size_t>(i)], 0.0f};
    }
    fft.perform(time.data(), spectrum.data(), false);

    // The same magnitudes on new phases. Every window draws all its phases,
    // silent or not, so what one window holds never moves another's.
    for (int k = 1; k < bins; ++k)
    {
      const double magnitude = std::abs(spectrum[static_cast<size_t>(k)]);
      const double angle = turn();
      const juce::dsp::Complex<float> value(
          static_cast<float>(magnitude * std::cos(angle)),
          static_cast<float>(magnitude * std::sin(angle)));
      spectrum[static_cast<size_t>(k)] = value;
      spectrum[static_cast<size_t>(size - k)] = std::conj(value);
    }
    spectrum[0] = {0.0f, 0.0f}; // no offset in a breath
    const float top = std::abs(spectrum[static_cast<size_t>(bins)]);
    spectrum[static_cast<size_t>(bins)] = {turn() < 0.5 * kTwoPi ? top : -top, 0.0f};
    fft.perform(spectrum.data(), time.data(), true);

    for (int i = 0; i < size; ++i)
    {
      const int at = start + i - first;
      if (at < 0 || at >= count)
        continue;
      const float w = shape[static_cast<size_t>(i)];
      sum[static_cast<size_t>(at)] += w * time[static_cast<size_t>(i)].real();
      norm[static_cast<size_t>(at)] += w * w;
    }
  }

  const float gain = static_cast<float>(std::sqrt(static_cast<double>(size) / hop));
  for (int i = 0; i < count; ++i)
    output[i] = gain * sum[static_cast<size_t>(i)] /
                std::max(norm[static_cast<size_t>(i)], 1.0e-6f);
}
} // namespace warp
