#include "SpliceAlignment.h"

#include "../../JuceHeader.h"

#include <algorithm>
#include <cmath>

namespace splicealign
{
std::vector<float> hilbert(const std::vector<float> &x)
{
  const int n = static_cast<int>(x.size());
  std::vector<float> result(x.size(), 0.0f);
  if (n < 2)
    return result;

  int order = 1;
  while ((1 << order) < n)
    ++order;
  const int size = 1 << order;
  juce::dsp::FFT fft(order);
  std::vector<juce::dsp::Complex<float>> time(static_cast<size_t>(size)),
      spectrum(static_cast<size_t>(size));
  for (int i = 0; i < n; ++i)
    time[static_cast<size_t>(i)] = {x[static_cast<size_t>(i)], 0.0f};
  fft.perform(time.data(), spectrum.data(), false);

  // Analytic signal: keep DC and Nyquist, double the positive frequencies,
  // drop the negative ones.
  for (int k = 1; k < size / 2; ++k)
    spectrum[static_cast<size_t>(k)] *= 2.0f;
  for (int k = size / 2 + 1; k < size; ++k)
    spectrum[static_cast<size_t>(k)] = {0.0f, 0.0f};
  fft.perform(spectrum.data(), time.data(), true);
  for (int i = 0; i < n; ++i)
    result[static_cast<size_t>(i)] = time[static_cast<size_t>(i)].imag();
  return result;
}

float sample(const std::vector<float> &render,
             const std::vector<float> &renderHilbert, double n,
             const Alignment &alignment)
{
  const int count = static_cast<int>(render.size());
  if (count == 0)
    return 0.0f;
  const double position =
      std::clamp(n - alignment.lag, 0.0, static_cast<double>(count - 1));
  const int left = static_cast<int>(position);
  const int right = std::min(count - 1, left + 1);
  const float fraction = static_cast<float>(position - left);
  const auto at = [&](const std::vector<float> &v)
  {
    return v[static_cast<size_t>(left)] +
           fraction * (v[static_cast<size_t>(right)] - v[static_cast<size_t>(left)]);
  };
  return static_cast<float>(at(render) * std::cos(alignment.angle) +
                            at(renderHilbert) * std::sin(alignment.angle));
}

Alignment find(const float *existing, const std::vector<float> &render,
               const std::vector<float> &renderHilbert, int centre,
               int fadeHalf)
{
  return find(existing, render, renderHilbert, centre, fadeHalf, kMaxLagSamples);
}

Alignment find(const float *existing, const std::vector<float> &render,
               const std::vector<float> &renderHilbert, int centre,
               int fadeHalf, int maxLag)
{
  Alignment best;
  const int count = static_cast<int>(render.size());
  const int first = centre - fadeHalf;
  const int last = centre + fadeHalf;
  if (existing == nullptr || fadeHalf <= 0 || maxLag < 0 ||
      first - maxLag < 0 || last + maxLag > count)
    return best;

  double existingEnergy = 0.0, renderEnergy = 0.0;
  for (int i = first; i < last; ++i)
  {
    existingEnergy += static_cast<double>(existing[i]) * existing[i];
    renderEnergy += static_cast<double>(render[static_cast<size_t>(i)]) *
                    render[static_cast<size_t>(i)];
  }
  if (existingEnergy < 1.0e-8 || renderEnergy < 1.0e-8)
    return best;

  // For a given shift the best angle has a closed form: with
  // a = sum w*x*e and b = sum w*h*e, x*cos + h*sin matches e best at
  // atan2(b, a), and the match there is hypot(a, b).
  double bestScore = -1.0;
  for (int lag = -maxLag; lag <= maxLag; ++lag)
  {
    double a = 0.0, b = 0.0;
    for (int i = first; i < last; ++i)
    {
      const double t = static_cast<double>(i - first) / (2 * fadeHalf);
      const double blend = t * t * (3.0 - 2.0 * t);
      const double weight = 4.0 * blend * (1.0 - blend);
      const auto source = static_cast<size_t>(i - lag);
      a += weight * render[source] * existing[i];
      b += weight * renderHilbert[source] * existing[i];
    }
    const double score = std::hypot(a, b);
    // Prefer the smaller shift unless a larger one is clearly better.
    if (score > bestScore * 1.001 ||
        (score >= bestScore * 0.999 && std::abs(lag) < std::abs(best.lag)))
    {
      bestScore = std::max(bestScore, score);
      best.lag = lag;
      best.angle = std::atan2(b, a);
    }
  }
  return best;
}
} // namespace splicealign
