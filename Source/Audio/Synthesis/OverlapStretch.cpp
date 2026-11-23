#include "OverlapStretch.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace warp
{
void overlapStretch(const float *source, int sourceSamples,
                    const std::function<double(double)> &inputAt, int first,
                    int last, float *output,
                    const OverlapStretchSettings &settings,
                    const float *alignTo, int alignFirst, int alignSamples)
{
  const int count = last - first;
  if (count <= 0 || output == nullptr)
    return;
  std::fill(output, output + count, 0.0f);
  if (source == nullptr || sourceSamples <= 0)
    return;

  const int window =
      std::max(4, static_cast<int>(settings.windowSeconds * settings.sampleRate) / 2 * 2);
  const int hop = std::clamp(static_cast<int>(settings.hopSeconds * settings.sampleRate),
                             1, window / 2);
  const int search = std::max(0, static_cast<int>(settings.searchSeconds * settings.sampleRate));

  std::vector<float> shape(static_cast<size_t>(window));
  for (int i = 0; i < window; ++i)
    shape[static_cast<size_t>(i)] = static_cast<float>(
        0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * i / window));

  const auto sourceAt = [&](long long position)
  {
    return position >= 0 && position < sourceSamples
               ? source[static_cast<size_t>(position)]
               : 0.0f;
  };

  std::vector<float> want(static_cast<size_t>(window));
  std::vector<float> sum(static_cast<size_t>(count), 0.0f);
  std::vector<float> weight(static_cast<size_t>(count), 0.0f);

  // How well the recording from `start` matches `want`: the normalised
  // cross-correlation, -2 where either is silent.
  const auto match = [&](long long start, double wantEnergy)
  {
    double dot = 0.0, energy = 0.0;
    for (int i = 0; i < window; ++i)
    {
      const double s = sourceAt(start + i);
      dot += s * want[static_cast<size_t>(i)];
      energy += s * s;
    }
    if (energy <= 1.0e-18 || wantEnergy <= 1.0e-18)
      return -2.0;
    return dot / std::sqrt(energy * wantEnergy);
  };

  bool havePrevious = false;
  long long previous = 0; // where the window before was taken from
  for (int start = first - window; start < last; start += hop)
  {
    const long long nominal =
        std::llround(inputAt(static_cast<double>(start) + 0.5 * window)) - window / 2;

    // What this window should look like: the recording going on from the
    // window before, or to begin with the other rendering here.
    bool haveWant = false;
    if (havePrevious)
    {
      for (int i = 0; i < window; ++i)
        want[static_cast<size_t>(i)] = sourceAt(previous + hop + i);
      haveWant = true;
    }
    else if (alignTo != nullptr && start >= alignFirst &&
             start + window <= alignFirst + alignSamples)
    {
      std::copy(alignTo + (start - alignFirst), alignTo + (start - alignFirst) + window,
                want.begin());
      haveWant = true;
    }

    long long taken = nominal;
    if (haveWant && search > 0)
    {
      double wantEnergy = 0.0;
      for (const float value : want)
        wantEnergy += static_cast<double>(value) * value;
      // Nearest first, and a further one has to be better: where nothing
      // tells them apart the window stays where the map puts it.
      double best = match(nominal, wantEnergy);
      for (int distance = 1; distance <= search; ++distance)
        for (const int offset : {distance, -distance})
        {
          const double value = match(nominal + offset, wantEnergy);
          if (value > best)
          {
            best = value;
            taken = nominal + offset;
          }
        }
    }

    for (int i = 0; i < window; ++i)
    {
      const int at = start + i - first;
      if (at < 0 || at >= count)
        continue;
      const float w = shape[static_cast<size_t>(i)];
      sum[static_cast<size_t>(at)] += w * sourceAt(taken + i);
      weight[static_cast<size_t>(at)] += w;
    }
    previous = taken;
    havePrevious = true;
  }

  for (int i = 0; i < count; ++i)
    output[i] = sum[static_cast<size_t>(i)] /
                std::max(weight[static_cast<size_t>(i)], 1.0e-3f);
}
} // namespace warp
