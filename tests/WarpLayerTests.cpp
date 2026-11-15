// Warp layer: the time map (WarpMap.h) and the recording laid out by it
// (WarpedSource.h). JUCE-free.
#include "TestAssert.h"

#include "../Source/Audio/Synthesis/WarpedSource.h"
#include "../Source/Audio/WarpMap.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kRate = 44100.0;
constexpr int kHop = 512;

bool near(double a, double b, double tolerance = 1.0e-9)
{
  return std::abs(a - b) <= tolerance;
}

void identityMapsEveryPositionToItself()
{
  warp::WarpMap map(100.0);
  CHECK(map.isIdentity());
  CHECK(near(map.destinationAt(37.25), 37.25));
  CHECK(near(map.sourceAt(37.25), 37.25));
  // An anchor that has not moved changes nothing.
  CHECK(map.addAnchorAtDestination(40.0) == 0);
  CHECK(map.isIdentity());
  CHECK(near(map.sourceAt(12.0), 12.0));
}

void draggingAnAnchorStretchesBothNeighboursOppositely()
{
  warp::WarpMap map(100.0);
  const int left = map.addAnchorAtDestination(20.0);
  const int right = map.addAnchorAtDestination(60.0);
  CHECK(left == 0 && right == 1);
  const int middle = map.addAnchorAtDestination(40.0);
  CHECK(middle == 1);
  map.moveAnchor(middle, 50.0);
  // [20, 40) -> [20, 50): 150%, [40, 60) -> [50, 60): 50%.
  CHECK(near(map.ratioAtDestination(30.0), 1.5));
  CHECK(near(map.ratioAtDestination(55.0), 0.5));
  CHECK(near(map.ratioAtDestination(10.0), 1.0));
  CHECK(near(map.ratioAtDestination(80.0), 1.0));
  CHECK(near(map.destinationAt(30.0), 35.0));
  CHECK(near(map.sourceAt(35.0), 30.0));
  CHECK(near(map.sourceAt(55.0), 50.0));
  // Outside the two dragged segments nothing moves.
  CHECK(near(map.destinationAt(10.0), 10.0));
  CHECK(near(map.destinationAt(75.0), 75.0));
  CHECK(map.isValid());

  // Removing the dragged anchor re-stretches [20, 60) evenly.
  CHECK(map.removeAnchor(middle));
  CHECK(map.isIdentity());
}

// The warp tool places an anchor before it is dragged; until then the sound
// must not be laid out again (warp::apply uses this check).
void pinningAnchorsKeepsTheMapping()
{
  warp::WarpMap stretched(100.0);
  stretched.moveAnchor(stretched.addAnchorAtDestination(40.0), 50.0);
  auto pinned = stretched;
  CHECK(pinned.addAnchorAtDestination(25.0) >= 0);
  CHECK(pinned.addAnchorAtDestination(70.0) >= 0);
  CHECK(pinned != stretched);
  CHECK(pinned.sameMappingAs(stretched));
  CHECK(stretched.sameMappingAs(pinned));

  // Deleting a pinned anchor keeps the mapping; deleting the moved one
  // re-stretches its neighbours.
  auto unpinned = pinned;
  CHECK(unpinned.removeAnchor(unpinned.findAnchorAtDestination(25.0)));
  CHECK(unpinned.sameMappingAs(stretched));
  auto removed = pinned;
  CHECK(removed.removeAnchor(removed.findAnchorAtDestination(50.0)));
  CHECK(!removed.sameMappingAs(stretched));

  // Moving a pinned anchor changes it.
  auto moved = pinned;
  moved.moveAnchor(moved.findAnchorAtDestination(70.0), 72.0);
  CHECK(!moved.sameMappingAs(pinned));
  CHECK(!warp::WarpMap(100.0).sameMappingAs(warp::WarpMap(90.0)));
}

// The piano roll draws audio laid out for one map where another puts it.
void relayoutMapsOneTimelineOntoAnother()
{
  warp::WarpMap laidOut(100.0);
  laidOut.moveAnchor(laidOut.addAnchorAtDestination(40.0), 50.0);
  warp::WarpMap shown(100.0);
  shown.moveAnchor(shown.addAnchorAtDestination(60.0), 70.0);

  const auto breaks = warp::WarpMap::relayout(laidOut, shown);
  auto at = [&](double shownFrame)
  {
    for (size_t i = 0; i + 1 < breaks.size(); ++i)
      if (shownFrame <= breaks[i + 1].destination)
        return breaks[i].source + (shownFrame - breaks[i].destination) *
                                      (breaks[i + 1].source - breaks[i].source) /
                                      (breaks[i + 1].destination - breaks[i].destination);
    return shownFrame;
  };
  CHECK(near(breaks.front().destination, 0.0) && near(breaks.back().destination, 100.0));
  // Every shown position reads the laid-out audio of the same recorded frame.
  for (double frame = 0.0; frame <= 100.0; frame += 2.5)
    CHECK(near(laidOut.sourceAt(at(frame)), shown.sourceAt(frame), 1.0e-9));

  // Same map: the identity.
  for (const auto &b : warp::WarpMap::relayout(shown, shown))
    CHECK(near(b.source, b.destination));
}

void dragIsLimitedToTheHardLimits()
{
  warp::WarpMap map(1000.0);
  map.addAnchorAtDestination(100.0);
  const int index = map.addAnchorAtDestination(200.0);
  map.addAnchorAtDestination(300.0);
  // Left segment [100, 200) may grow to 800% but the right one may only
  // shrink to 12.5% of 100 frames: 300 - 12.5 = 287.5.
  CHECK(near(map.moveAnchor(index, 10000.0), 287.5));
  // Squeezing the left segment stops at 12.5% as well.
  CHECK(near(map.moveAnchor(index, -10000.0), 112.5));
  CHECK(map.isValid());

  // Short segments stop at one frame.
  warp::WarpMap shortMap(100.0);
  shortMap.addAnchorAtDestination(10.0);
  const int k = shortMap.addAnchorAtDestination(14.0);
  shortMap.addAnchorAtDestination(50.0);
  CHECK(near(shortMap.moveAnchor(k, 0.0), 11.0));
  CHECK(shortMap.isValid());

  // Anchors closer than a frame are refused, and the ends cannot be pinned.
  CHECK(shortMap.addAnchorAtDestination(11.5) == -1);
  CHECK(shortMap.addAnchorAtDestination(0.0) == -1);
  CHECK(shortMap.addAnchorAtDestination(100.0) == -1);
}

std::vector<float> tones(const std::vector<double> &frequencies,
                         int samplesEach)
{
  std::vector<float> signal;
  double phase = 0.0;
  for (double f : frequencies)
    for (int i = 0; i < samplesEach; ++i)
    {
      signal.push_back(static_cast<float>(0.5 * std::sin(phase)));
      phase += 2.0 * kPi * f / kRate;
    }
  return signal;
}

// Dominant frequency (Goertzel scan, 10 Hz steps) of a window.
double dominantFrequency(const std::vector<float> &signal, int start,
                         int length)
{
  double best = 0.0, bestPower = -1.0;
  for (double f = 100.0; f <= 2000.0; f += 10.0)
  {
    const double w = 2.0 * kPi * f / kRate;
    double s1 = 0.0, s2 = 0.0;
    for (int i = 0; i < length; ++i)
    {
      const double s = signal[static_cast<size_t>(start + i)] +
                       2.0 * std::cos(w) * s1 - s2;
      s2 = s1;
      s1 = s;
    }
    const double power = s1 * s1 + s2 * s2 - 2.0 * std::cos(w) * s1 * s2;
    if (power > bestPower)
    {
      bestPower = power;
      best = f;
    }
  }
  return best;
}

void identityRenderIsTheRecording()
{
  const auto source = tones({300.0, 600.0, 900.0}, 44100);
  warp::WarpMap map(static_cast<double>(source.size()) / kHop);
  map.addAnchorAtDestination(50.0);
  std::vector<float> out(source.size());
  const auto runs = warp::renderWarpedSource(
      source.data(), static_cast<int>(source.size()), map, out.data(),
      static_cast<int>(out.size()));
  CHECK(runs.empty());
  for (size_t i = 0; i < source.size(); ++i)
    CHECK(out[i] == source[i]);
}

void stretchedSegmentsPlayTheRightRecordingAtTheRightTime()
{
  // Five one-second tones; frames are 512 samples.
  const auto source = tones({300.0, 500.0, 700.0, 900.0, 1100.0}, 44100);
  const double second = kRate / kHop;
  warp::WarpMap map(static_cast<double>(source.size()) / kHop);
  map.addAnchorAtSource(1.0 * second);
  const int k = map.addAnchorAtSource(2.0 * second);
  map.addAnchorAtSource(3.0 * second);
  // Tone 2 (500 Hz) stretched to 1.5 s, tone 3 (700 Hz) squeezed to 0.5 s.
  map.moveAnchor(k, 2.5 * second);
  CHECK(near(map.ratioAtDestination(1.2 * second), 1.5));
  CHECK(near(map.ratioAtDestination(2.7 * second), 0.5));

  std::vector<float> out(source.size());
  const auto runs = warp::renderWarpedSource(
      source.data(), static_cast<int>(source.size()), map, out.data(),
      static_cast<int>(out.size()));
  CHECK(runs.size() == 1);

  const int window = 4096;
  auto at = [&](double seconds)
  {
    return dominantFrequency(out, static_cast<int>(seconds * kRate) - window / 2,
                             window);
  };
  CHECK(near(at(0.5), 300.0, 10.0));
  CHECK(near(at(1.3), 500.0, 10.0));
  CHECK(near(at(2.2), 500.0, 10.0));  // still tone 2: it now ends at 2.5 s
  CHECK(near(at(2.75), 700.0, 10.0)); // tone 3 squeezed into 2.5 .. 3.0 s
  CHECK(near(at(3.5), 900.0, 10.0));
  CHECK(near(at(4.5), 1100.0, 10.0));

  // Away from the run the recording is copied sample for sample.
  for (int i = 0; i < static_cast<int>(0.8 * kRate); ++i)
    CHECK(out[static_cast<size_t>(i)] == source[static_cast<size_t>(i)]);
  for (int i = static_cast<int>(3.2 * kRate);
       i < static_cast<int>(source.size()); ++i)
    CHECK(out[static_cast<size_t>(i)] == source[static_cast<size_t>(i)]);

  // The stretched audio keeps its level (no fade dip, no doubling).
  double rms = 0.0;
  const int from = static_cast<int>(1.2 * kRate), to = static_cast<int>(2.3 * kRate);
  for (int i = from; i < to; ++i)
    rms += out[static_cast<size_t>(i)] * out[static_cast<size_t>(i)];
  rms = std::sqrt(rms / (to - from));
  CHECK(near(rms, 0.5 / std::sqrt(2.0), 0.05));

  // Same edit, same audio.
  std::vector<float> again(source.size());
  warp::renderWarpedSource(source.data(), static_cast<int>(source.size()), map,
                           again.data(), static_cast<int>(again.size()));
  CHECK(again == out);
}

void handOverIsAlignedWithTheCopy()
{
  // Around the hand-over the stretch runs at rate 1, so it must line up with
  // the copied recording or the crossfade would comb.
  const auto source = tones({440.0, 440.0, 440.0, 440.0}, 44100);
  const double second = kRate / kHop;
  warp::WarpMap map(static_cast<double>(source.size()) / kHop);
  map.addAnchorAtSource(1.5 * second);
  const int k = map.addAnchorAtSource(2.0 * second);
  map.addAnchorAtSource(2.5 * second);
  map.moveAnchor(k, 2.1 * second);
  std::vector<float> out(source.size());
  warp::renderWarpedSource(source.data(), static_cast<int>(source.size()), map,
                           out.data(), static_cast<int>(out.size()));
  double error = 0.0, energy = 0.0;
  const int from = static_cast<int>((1.5 - 0.05) * kRate);
  const int to = static_cast<int>((1.5 - 0.01) * kRate);
  for (int i = from; i < to; ++i)
  {
    const double d = out[static_cast<size_t>(i)] - source[static_cast<size_t>(i)];
    error += d * d;
    energy += source[static_cast<size_t>(i)] * source[static_cast<size_t>(i)];
  }
  std::printf("hand-over residual %.1f dB\n", 10.0 * std::log10(error / energy));
  CHECK(error / energy < 0.01); // better than -20 dB
}
} // namespace

int main()
{
  identityMapsEveryPositionToItself();
  draggingAnAnchorStretchesBothNeighboursOppositely();
  pinningAnchorsKeepsTheMapping();
  relayoutMapsOneTimelineOntoAnother();
  dragIsLimitedToTheHardLimits();
  identityRenderIsTheRecording();
  stretchedSegmentsPlayTheRightRecordingAtTheRightTime();
  handOverIsAlignedWithTheCopy();
  std::printf("WarpLayerTests passed\n");
  return 0;
}
