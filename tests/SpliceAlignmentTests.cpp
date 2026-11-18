// A render that sits near the opposite phase of the recording must not dip
// when crossfaded into it once aligned (SpliceAlignment.h).
#include "TestAssert.h"

#include "../Source/Audio/Synthesis/SpliceAlignment.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kRate = 44100.0;

// A voice-like tone: 590 Hz with falling harmonics, each harmonic at the
// phase `phaseOf(k)`.
template <class Phase>
std::vector<float> voice(int samples, Phase phaseOf)
{
  std::vector<float> out(static_cast<size_t>(samples));
  for (int i = 0; i < samples; ++i)
  {
    double v = 0.0;
    for (int k = 1; k <= 8; ++k)
      v += std::sin(2 * kPi * 590.0 * k * i / kRate + phaseOf(k)) / k;
    out[static_cast<size_t>(i)] = static_cast<float>(0.2 * v);
  }
  return out;
}

double minLevelDb(const std::vector<float> &x, int from, int to)
{
  constexpr int w = 64;
  double lowest = 1.0e9;
  for (int i = from; i + w <= to; i += 16)
  {
    double e = 0.0;
    for (int j = i; j < i + w; ++j)
      e += static_cast<double>(x[static_cast<size_t>(j)]) * x[static_cast<size_t>(j)];
    lowest = std::min(lowest, 10.0 * std::log10(e / w + 1e-20));
  }
  return lowest;
}

std::vector<float> crossfade(const std::vector<float> &a,
                             const std::vector<float> &b, int centre, int half)
{
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i)
  {
    double t = std::clamp((static_cast<double>(i) - (centre - half)) / (2 * half), 0.0, 1.0);
    t = t * t * (3 - 2 * t);
    out[i] = static_cast<float>(a[i] + t * (b[i] - a[i]));
  }
  return out;
}

void antiPhaseSpliceDoesNotDip()
{
  const int samples = 8192, centre = 4096, half = 512;
  const auto recording = voice(samples, [](int) { return 0.0; });
  // The render: opposite phase plus a small delay-like drift per harmonic.
  const auto render = voice(samples, [](int k) { return kPi + 0.15 * k; });

  const double before = minLevelDb(crossfade(recording, render, centre, half),
                                   centre - half, centre + half);

  const auto h = splicealign::hilbert(render);
  const auto a = splicealign::find(recording.data(), render, h, centre, half);
  std::vector<float> aligned(render.size());
  for (int i = 0; i < samples; ++i)
    aligned[static_cast<size_t>(i)] = splicealign::sample(render, h, i, a);
  const double after = minLevelDb(crossfade(recording, aligned, centre, half),
                                  centre - half, centre + half);
  const double level = minLevelDb(recording, centre - half, centre + half);

  std::printf("  lag %.0f angle %.1f deg; dip %.1f dB -> %.1f dB (level %.1f)\n",
              a.lag, a.angle * 180 / kPi, before, after, level);
  CHECK(before < level - 10.0);
  CHECK(after > level - 1.0);
  CHECK(std::abs(a.lag) <= splicealign::kMaxLagSamples);
}

void alignedRenderMatchesInPhaseRecordingUnchanged()
{
  const auto recording = voice(4096, [](int) { return 0.0; });
  const auto h = splicealign::hilbert(recording);
  const auto a = splicealign::find(recording.data(), recording, h, 2048, 512);
  CHECK(a.lag == 0.0);
  CHECK(std::abs(a.angle) < 0.01);
}
} // namespace

int main()
{
  antiPhaseSpliceDoesNotDip();
  alignedRenderMatchesInPhaseRecordingUnchanged();
  std::printf("SpliceAlignmentTests passed\n");
  return 0;
}
