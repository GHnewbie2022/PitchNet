// Pitch from three detectors: a lone outlier is outvoted, shared detail
// stays, voicing goes by majority, and a late detector is read earlier.
#include "../Source/Audio/PitchEnsemble.h"
#include "TestAssert.h"
#include <cmath>
#include <iostream>

using pitchensemble::Curve;

namespace
{
float semis(float hz) { return 12.0f * std::log2(hz / 440.0f) + 69.0f; }
float hertz(float st) { return 440.0f * std::exp2((st - 69.0f) / 12.0f); }

Curve curveOf(const std::vector<float>& semitones)
{
  Curve c;
  for (float st : semitones)
  {
    c.f0.push_back(st > 0.0f ? hertz(st) : 0.0f);
    c.voiced.push_back(st > 0.0f);
  }
  return c;
}
} // namespace

int main()
{
  const auto near = [](float a, float b) { return std::abs(a - b) < 1.0e-3f; };

  // Frame 1: one detector dips by 1.4 st; the median keeps the other two.
  // Frame 2: all three share a bend; the middle value stays.
  {
    const auto a = curveOf({70.0f, 70.0f, 69.5f});
    const auto b = curveOf({70.0f, 70.1f, 69.6f});
    const auto c = curveOf({70.0f, 68.6f, 69.4f});
    const auto out = pitchensemble::combine({a, b, c});
    CHECK(near(semis(out.f0[1]), 70.0f));
    CHECK(near(semis(out.f0[2]), 69.5f));
  }
  // Voicing by majority: one voiced of three is unvoiced, two is voiced
  // (their mean when they agree).
  {
    const auto a = curveOf({60.0f, 60.0f});
    const auto b = curveOf({0.0f, 60.4f});
    const auto c = curveOf({0.0f, 0.0f});
    const auto out = pitchensemble::combine({a, b, c});
    CHECK(!out.voiced[0]);
    CHECK(out.voiced[1] && near(semis(out.f0[1]), 60.2f));
  }
  // Two voiced that disagree: the one nearer the frame before.
  {
    const auto a = curveOf({62.0f, 62.1f});
    const auto b = curveOf({62.0f, 74.1f});
    const auto c = curveOf({62.0f, 0.0f});
    const auto out = pitchensemble::combine({a, b, c});
    CHECK(near(semis(out.f0[1]), 62.1f));
  }
  // A detector reporting 10 ms late is read 10 ms later.
  {
    std::vector<float> source(100, 0.0f);
    for (int i = 0; i < 100; ++i)
      source[static_cast<size_t>(i)] = hertz(60.0f + 0.1f * static_cast<float>(i));
    const auto onTime = pitchensemble::resample(source, 0.010, 0.0, 50, 0.010);
    const auto late = pitchensemble::resample(source, 0.010, 0.010, 50, 0.010);
    CHECK(near(semis(late.f0[20]), semis(onTime.f0[21])));
  }
  std::cout << "Pitch ensemble passed\n";
}
