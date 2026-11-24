// The stretch by what the recording holds (WarpedSource.h): fry by its own
// waveform (OverlapStretch.h), breath by its noise made again
// (NoiseStretch.h), which frame is which (SourceKinds.h), and the three laid
// together by renderWarpedSource.
#include "TestAssert.h"

#include "../Source/Audio/Synthesis/NoiseStretch.h"
#include "../Source/Audio/Synthesis/OverlapStretch.h"
#include "../Source/Audio/Synthesis/SourceKinds.h"
#include "../Source/Audio/Synthesis/WarpedSource.h"
#include "../Source/Audio/WarpMap.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kRate = 44100.0;
constexpr int kHop = 512;

// Uniform in [-1, 1), the same everywhere.
struct Noise
{
  std::uint32_t state = 12345u;
  float next()
  {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state >> 8) / 8388608.0f - 1.0f;
  }
};

std::vector<float> voice(int samples, double f0 = 220.0)
{
  std::vector<float> out(static_cast<size_t>(samples));
  for (int i = 0; i < samples; ++i)
  {
    double v = 0.0;
    for (int k = 1; k <= 8; ++k)
      v += std::sin(2 * kPi * f0 * k * i / kRate + 0.7 * k) / k;
    out[static_cast<size_t>(i)] = static_cast<float>(0.2 * v);
  }
  return out;
}

// Pulses 8 to 25 ms apart, each a damped ring; returns where they start.
std::vector<float> fry(int samples, std::vector<int> &starts)
{
  std::vector<float> out(static_cast<size_t>(samples), 0.0f);
  Noise gap;
  for (int at = 400; at + 400 < samples;)
  {
    starts.push_back(at);
    for (int i = 0; i < 300; ++i)
      out[static_cast<size_t>(at + i)] += static_cast<float>(
          0.5 * std::exp(-i / 40.0) * std::sin(2 * kPi * 900.0 * i / kRate));
    at += static_cast<int>(kRate * (0.0165 + 0.0085 * gap.next()));
  }
  return out;
}

double rmsDb(const float *x, int from, int to)
{
  double e = 0.0;
  for (int i = from; i < to; ++i)
    e += static_cast<double>(x[i]) * x[i];
  return 10.0 * std::log10(e / std::max(1, to - from) + 1.0e-20);
}

double correlation(const float *a, const float *b, int length)
{
  double dot = 0.0, ea = 0.0, eb = 0.0;
  for (int i = 0; i < length; ++i)
  {
    dot += static_cast<double>(a[i]) * b[i];
    ea += static_cast<double>(a[i]) * a[i];
    eb += static_cast<double>(b[i]) * b[i];
  }
  return dot / std::sqrt(ea * eb + 1.0e-30);
}

// ---- OverlapStretch ---------------------------------------------------------

void overlapAtUnityIsTheRecording()
{
  std::vector<int> starts;
  auto source = fry(44100, starts);
  const auto tone = voice(44100);
  for (size_t i = 0; i < source.size(); ++i)
    source[i] += tone[i];
  const int first = 8000, last = 30000;
  std::vector<float> out(static_cast<size_t>(last - first));
  warp::OverlapStretchSettings settings;
  warp::overlapStretch(
      source.data(), static_cast<int>(source.size()), [](double t) { return t; }, first,
      last, out.data(), settings);
  double worst = 0.0;
  for (int i = first; i < last; ++i)
    worst = std::max(worst, static_cast<double>(std::abs(
                                out[static_cast<size_t>(i - first)] -
                                source[static_cast<size_t>(i)])));
  std::printf("overlap at 1.00x: largest difference %.2e\n", worst);
  CHECK(worst < 1.0e-5);
}

void overlapKeepsEachPulse()
{
  std::vector<int> starts;
  const auto source = fry(44100, starts);
  const double ratio = 1.33;
  const int first = 4000, last = static_cast<int>(40000 * ratio);
  std::vector<float> out(static_cast<size_t>(last - first));
  warp::OverlapStretchSettings settings;
  const auto render = [&](std::vector<float> &to)
  {
    warp::overlapStretch(
        source.data(), static_cast<int>(source.size()),
        [ratio](double t) { return t / ratio; }, first, last, to.data(), settings);
  };
  render(out);

  // Every strong peak of the output is a recorded pulse, shape and all.
  const int length = 120; // 2.7 ms of the ring
  int strong = 0, kept = 0;
  for (int i = 1; i + length < static_cast<int>(out.size()); ++i)
  {
    const float v = out[static_cast<size_t>(i)];
    // The ring's first crest, 12 samples after a pulse starts.
    if (v < 0.3f || v < out[static_cast<size_t>(i - 1)] || v < out[static_cast<size_t>(i + 1)])
      continue;
    bool first_crest = true;
    for (int back = 1; back <= 30 && i - back >= 0; ++back)
      first_crest = first_crest && out[static_cast<size_t>(i - back)] <= v;
    if (!first_crest || i < 12)
      continue;
    ++strong;
    if (correlation(out.data() + i - 12, source.data() + starts[0], length) > 0.98)
      ++kept;
    i += 200;
  }
  const int inSource = static_cast<int>(std::count_if(
      starts.begin(), starts.end(),
      [&](int s) { return s >= first / ratio && s < last / ratio; }));
  // Where two windows hold pulses at different spacings, a pulse comes out
  // twice at part of its level: count those too.
  int faint = 0;
  for (int i = 31; i + 1 < static_cast<int>(out.size()); ++i)
  {
    const float v = out[static_cast<size_t>(i)];
    if (v < 0.08f || v >= 0.3f)
      continue;
    bool crest = v >= out[static_cast<size_t>(i + 1)];
    for (int back = 1; back <= 30; ++back)
      crest = crest && out[static_cast<size_t>(i - back)] <= v;
    if (!crest)
      continue;
    ++faint;
    i += 200;
  }
  std::printf("overlap at 1.33x: %d pulses in, %d out at full level (%d of them as "
              "recorded), %d fainter\n",
              inSource, strong, kept, faint);
  // None lost, those at full level are the recorded ones, and the fainter
  // doubles stay few per pulse. (These pulses are spaced at random, 8 to 25
  // ms; recorded fry is more regular from one pulse to the next.)
  CHECK(strong >= inSource);
  CHECK(kept >= strong * 9 / 10);
  CHECK(strong + faint < 3 * inSource);

  std::vector<float> again(out.size());
  render(again);
  CHECK(again == out);
}

void overlapStartsInPhaseWithTheOtherRendering()
{
  // The other rendering: the same voice 1.3 ms late, as a stretch that keeps
  // its own phase would have it.
  const auto source = voice(44100);
  const int late = 57;
  std::vector<float> rendering(source.size(), 0.0f);
  for (size_t i = late; i < source.size(); ++i)
    rendering[i] = source[i - late];
  const int first = 12000, last = 20000;
  warp::OverlapStretchSettings settings;
  const auto render = [&](bool aligned)
  {
    std::vector<float> out(static_cast<size_t>(last - first));
    warp::overlapStretch(
        source.data(), static_cast<int>(source.size()), [](double t) { return t; }, first,
        last, out.data(), settings, aligned ? rendering.data() : nullptr, 0,
        static_cast<int>(rendering.size()));
    return correlation(out.data(), rendering.data() + first, 882);
  };
  const double with = render(true), without = render(false);
  std::printf("overlap against a rendering 1.3 ms late: correlation %.3f aligned, %.3f not\n",
              with, without);
  CHECK(with > 0.99);
  CHECK(without < 0.9);
}

// ---- NoiseStretch -----------------------------------------------------------

std::vector<float> whiteNoise(int samples, float level = 0.1f)
{
  Noise noise;
  std::vector<float> out(static_cast<size_t>(samples));
  for (auto &v : out)
    v = level * noise.next();
  return out;
}

void noiseKeepsItsLevel()
{
  const auto source = whiteNoise(88200);
  for (const double ratio : {0.80, 1.10, 1.33})
  {
    const int first = 8000, last = static_cast<int>(70000 * ratio);
    std::vector<float> out(static_cast<size_t>(last - first));
    warp::NoiseStretchSettings settings;
    settings.seed = 7;
    warp::noiseStretch(
        source.data(), static_cast<int>(source.size()),
        [ratio](double t) { return t / ratio; }, first, last, out.data(), settings);
    const double in = rmsDb(source.data(), 8000, 70000);
    const double made = rmsDb(out.data(), 0, last - first);
    std::printf("noise at %.2fx: %.2f dB off the recording\n", ratio, made - in);
    CHECK(std::abs(made - in) < 0.5);
  }
}

// The level of `x` through a one-pole low-pass at `hz`: a crude band.
double lowPassedDb(const std::vector<float> &x, int from, int to, double hz)
{
  const double a = std::exp(-2 * kPi * hz / kRate);
  double state = 0.0, e = 0.0;
  for (int i = from; i < to; ++i)
  {
    state = a * state + (1.0 - a) * x[static_cast<size_t>(i)];
    e += state * state;
  }
  return 10.0 * std::log10(e / (to - from) + 1.0e-20);
}

void noiseKeepsItsColourAndFollowsItsLoudness()
{
  // Dull noise that turns loud half way.
  auto source = whiteNoise(88200);
  double state = 0.0;
  for (size_t i = 0; i < source.size(); ++i)
  {
    state = 0.9 * state + 0.1 * source[i];
    source[i] = static_cast<float>(state) * (i < 44100 ? 0.05f : 1.0f);
  }
  const double ratio = 1.33;
  const int first = 4000, last = static_cast<int>(84000 * ratio);
  std::vector<float> out(static_cast<size_t>(last - first));
  warp::NoiseStretchSettings settings;
  settings.seed = 7;
  warp::noiseStretch(
      source.data(), static_cast<int>(source.size()),
      [ratio](double t) { return t / ratio; }, first, last, out.data(), settings);

  const int step = static_cast<int>(44100 * ratio) - first;
  const int loudFrom = step + 4410, loudTo = static_cast<int>(out.size()) - 4410;
  for (const double hz : {300.0, 1500.0, 6000.0})
  {
    const double in = lowPassedDb(source, 48000, 84000, hz);
    const double made = lowPassedDb(out, loudFrom, loudTo, hz);
    std::printf("noise below %.0f Hz: %.2f dB off the recording\n", hz, made - in);
    CHECK(std::abs(made - in) < 1.0);
  }
  // Quiet until the step, loud a window after it.
  const double loud = rmsDb(out.data(), loudFrom, loudTo);
  const double before = rmsDb(out.data(), step - 3528, step - 1764);
  const double after = rmsDb(out.data(), step + 1764, step + 3528);
  std::printf("noise around the step: %.1f dB before, %.1f dB after (loud = 0)\n",
              before - loud, after - loud);
  CHECK(before - loud < -20.0);
  CHECK(std::abs(after - loud) < 1.5);
}

void noiseIsTheSameForTheSameSeed()
{
  const auto source = whiteNoise(44100);
  const auto render = [&](std::uint32_t seed)
  {
    std::vector<float> out(20000);
    warp::NoiseStretchSettings settings;
    settings.seed = seed;
    warp::noiseStretch(
        source.data(), static_cast<int>(source.size()), [](double t) { return t / 1.1; },
        5000, 25000, out.data(), settings);
    return out;
  };
  CHECK(render(3) == render(3));
  CHECK(render(3) != render(4));
  CHECK(warp::noiseStretchWindow({44100.0}) == 1024);
  CHECK(warp::noiseStretchWindow({48000.0}) == 1024);
  CHECK(warp::noiseStretchWindow({96000.0}) == 2048);
}

// ---- which frame is which -----------------------------------------------------

void kindsComeFromTheBreathLabelsAndTheFry()
{
  AudioData audio;
  audio.sampleRate = 44100;
  audio.rawF0.assign(100, 0.0f);
  audio.voicedMask.assign(100, false);
  audio.breathSibilance.assign(100, breathsibilance::None);
  CHECK(warp::sourceKinds(audio).empty());

  // A low run of 4 frames an octave and more under the voice it leads into.
  for (int i = 20; i < 24; ++i)
    audio.rawF0[static_cast<size_t>(i)] = 100.0f;
  for (int i = 24; i < 70; ++i)
    audio.rawF0[static_cast<size_t>(i)] = 230.0f;
  for (int i = 20; i < 70; ++i)
    audio.voicedMask[static_cast<size_t>(i)] = true;
  for (int i = 80; i < 90; ++i)
    audio.breathSibilance[static_cast<size_t>(i)] = breathsibilance::Breath;
  // Sibilance is noise where there is no pitch, the voice where there is.
  audio.breathSibilance[75] = breathsibilance::Sibilance;
  audio.breathSibilance[60] = breathsibilance::Sibilance;
  const auto kinds = warp::sourceKinds(audio);
  CHECK(kinds.size() == 100);
  const auto is = [&](int frame, warp::SourceKind kind)
  { return kinds[static_cast<size_t>(frame)] == static_cast<std::uint8_t>(kind); };
  CHECK(is(21, warp::SourceKind::Pulses) && is(23, warp::SourceKind::Pulses));
  CHECK(is(30, warp::SourceKind::Voice) && is(60, warp::SourceKind::Voice));
  CHECK(is(74, warp::SourceKind::Voice) && is(75, warp::SourceKind::Noise) &&
        is(76, warp::SourceKind::Voice));
  CHECK(is(80, warp::SourceKind::Noise) && is(89, warp::SourceKind::Noise));
  CHECK(is(90, warp::SourceKind::Voice));

  // Noise wins where the two meet.
  audio.breathSibilance[22] = breathsibilance::Breath;
  CHECK(warp::sourceKinds(audio)[22] == static_cast<std::uint8_t>(warp::SourceKind::Noise));
}

// ---- the three together -------------------------------------------------------

struct Layout
{
  warp::WarpMap map;
  double second = kRate / kHop;
};

// Seconds 1..2 stretched to 1.3 s, seconds 2..3 squeezed to 0.7 s.
Layout stretchedInTheMiddle(int samples)
{
  Layout layout{warp::WarpMap(static_cast<double>(samples) / kHop)};
  layout.map.addAnchorAtSource(1.0 * layout.second);
  const int k = layout.map.addAnchorAtSource(2.0 * layout.second);
  layout.map.addAnchorAtSource(3.0 * layout.second);
  layout.map.moveAnchor(k, 2.3 * layout.second);
  return layout;
}

std::vector<std::uint8_t> kindsBetween(int samples, double from, double to,
                                       warp::SourceKind kind,
                                       std::vector<std::uint8_t> kinds = {})
{
  kinds.resize(static_cast<size_t>(samples / kHop + 1),
               static_cast<std::uint8_t>(warp::SourceKind::Voice));
  for (int frame = static_cast<int>(std::ceil(from * kRate / kHop));
       frame < static_cast<int>(to * kRate / kHop); ++frame)
    kinds[static_cast<size_t>(frame)] = static_cast<std::uint8_t>(kind);
  return kinds;
}

std::vector<float> render(const std::vector<float> &source, const warp::WarpMap &map,
                          const std::vector<std::uint8_t> *kinds)
{
  std::vector<float> out(source.size());
  warp::WarpedSourceSettings settings;
  settings.kinds = kinds;
  warp::renderWarpedSource(source.data(), static_cast<int>(source.size()), map, out.data(),
                           static_cast<int>(out.size()), settings);
  return out;
}

// The lowest level of `x` over [from, to), two periods of 220 Hz at a time.
double lowestDb(const std::vector<float> &x, int from, int to)
{
  constexpr int window = 401;
  double lowest = 1.0e9;
  for (int i = from; i + window <= to; i += 50)
    lowest = std::min(lowest, rmsDb(x.data(), i, i + window));
  return lowest;
}

void eachKindIsLaidOverItsOwnFramesOnly()
{
  const int samples = 5 * 44100;
  auto source = voice(samples);
  // Breath from 1.40 to 1.60 s of the recording, fry from 2.30 to 2.55 s.
  Noise noise;
  for (int i = static_cast<int>(1.38 * kRate); i < static_cast<int>(1.62 * kRate); ++i)
    source[static_cast<size_t>(i)] = 0.05f * noise.next();
  std::vector<int> starts;
  const auto pulses = fry(samples, starts);
  for (int i = static_cast<int>(2.28 * kRate); i < static_cast<int>(2.57 * kRate); ++i)
    source[static_cast<size_t>(i)] = pulses[static_cast<size_t>(i)];

  const auto layout = stretchedInTheMiddle(samples);
  const auto kinds = kindsBetween(
      samples, 2.30, 2.55, warp::SourceKind::Pulses,
      kindsBetween(samples, 1.40, 1.60, warp::SourceKind::Noise));
  const auto plain = render(source, layout.map, nullptr);
  const auto laid = render(source, layout.map, &kinds);
  CHECK(render(source, layout.map, &kinds) == laid); // same edit, same audio

  const auto at = [&](double sourceSeconds)
  {
    return static_cast<int>(layout.map.destinationAt(sourceSeconds * layout.second) * kHop);
  };
  // 30 ms clear of the breath, and of the fry with its 40 ms of reach, the
  // rendering is signalsmith's, sample for sample.
  const int margin = static_cast<int>(0.030 * kRate), reach = static_cast<int>(0.040 * kRate);
  int same = 0, changed = 0;
  for (int i = 0; i < samples; ++i)
  {
    const bool nearBreath = i > at(1.40) - margin && i < at(1.60) + margin;
    const bool nearFry = i > at(2.30) - margin && i < at(2.55) + margin + reach;
    if (nearBreath || nearFry)
    {
      changed += plain[static_cast<size_t>(i)] != laid[static_cast<size_t>(i)];
      continue;
    }
    CHECK(plain[static_cast<size_t>(i)] == laid[static_cast<size_t>(i)]);
    ++same;
  }
  std::printf("kinds: %d samples as signalsmith left them, %d laid over\n", same, changed);
  CHECK(changed > static_cast<int>(0.3 * kRate));

  // The breath keeps its level; the fry is the recorded pulses.
  const double breath = rmsDb(laid.data(), at(1.45), at(1.55));
  std::printf("kinds: breath %.2f dB off the recording\n",
              breath - rmsDb(source.data(), static_cast<int>(1.45 * kRate),
                             static_cast<int>(1.55 * kRate)));
  CHECK(std::abs(breath - rmsDb(source.data(), static_cast<int>(1.45 * kRate),
                                static_cast<int>(1.55 * kRate))) < 1.0);
  double best = 0.0;
  const int length = 120;
  const int pulse = *std::find_if(starts.begin(), starts.end(), [](int s)
                                  { return s > static_cast<int>(2.35 * kRate); });
  for (int i = at(2.33); i + length < at(2.52); ++i)
    best = std::max(best, correlation(laid.data() + i, source.data() + pulse, length));
  std::printf("kinds: a recorded pulse found in the fry at correlation %.3f\n", best);
  CHECK(best > 0.98);

  // The same again: nothing is kept from one rendering to the next.
  CHECK(render(source, layout.map, &kinds) == laid);
}

void fryMeetsTheVoiceWithoutADip()
{
  // A held voice with a stretch of it called fry: whatever the join costs
  // shows as a dip in an otherwise level tone. Once inside one stretched
  // segment, once across the anchor between two of different ratios.
  const int samples = 5 * 44100;
  const auto source = voice(samples);
  const auto layout = stretchedInTheMiddle(samples);
  const auto plain = render(source, layout.map, nullptr);
  struct Case
  {
    const char *name;
    double from, to;
  };
  for (const Case &c : {Case{"inside one segment", 1.40, 1.70},
                        Case{"across an anchor", 1.80, 2.20}})
  {
    const auto kinds = kindsBetween(samples, c.from, c.to, warp::SourceKind::Pulses);
    const auto laid = render(source, layout.map, &kinds);
    const int from = static_cast<int>(layout.map.destinationAt(c.from * layout.second) * kHop) -
                     static_cast<int>(0.05 * kRate);
    const int to = static_cast<int>(layout.map.destinationAt(c.to * layout.second) * kHop) +
                   static_cast<int>(0.10 * kRate);
    const double before = lowestDb(plain, from, to), after = lowestDb(laid, from, to);
    float jump = 0.0f, plainJump = 0.0f;
    for (int i = from + 1; i < to; ++i)
    {
      jump = std::max(jump, std::abs(laid[static_cast<size_t>(i)] -
                                     laid[static_cast<size_t>(i - 1)]));
      plainJump = std::max(plainJump, std::abs(plain[static_cast<size_t>(i)] -
                                               plain[static_cast<size_t>(i - 1)]));
    }
    // It is another rendering there, not signalsmith's again.
    int changed = 0;
    for (int i = from; i < to; ++i)
      changed += std::abs(laid[static_cast<size_t>(i)] - plain[static_cast<size_t>(i)]) > 1.0e-3f;
    std::printf("fry %s: lowest level %.2f dB (signalsmith alone %.2f dB), "
                "largest step %.4f (%.4f), %d samples differ\n",
                c.name, after, before, jump, plainJump, changed);
    CHECK(changed > static_cast<int>(0.05 * kRate));
    CHECK(after > before - 1.0);
    CHECK(jump < 1.5f * plainJump);
  }
}

void aBreathCutByAnAnchorKeepsItsLevel()
{
  // A breath with the anchor at 1 s in the middle of it: the recording up to
  // the anchor, its noise made again past it. Two noises that have nothing
  // to do with each other meet there, and a plain crossfade between them
  // sags by 3 dB (heard as a flutter in the breath).
  const int samples = 5 * 44100;
  auto source = voice(samples);
  Noise noise;
  for (int i = static_cast<int>(0.58 * kRate); i < static_cast<int>(1.42 * kRate); ++i)
    source[static_cast<size_t>(i)] = 0.05f * noise.next();
  const auto layout = stretchedInTheMiddle(samples);
  const auto kinds = kindsBetween(samples, 0.60, 1.40, warp::SourceKind::Noise);
  const auto laid = render(source, layout.map, &kinds);

  const int anchor = static_cast<int>(1.0 * kRate), window = static_cast<int>(0.010 * kRate);
  const double level = rmsDb(source.data(), static_cast<int>(0.7 * kRate),
                             static_cast<int>(0.9 * kRate));
  double lowest = 1.0e9, highest = -1.0e9;
  for (int i = anchor - static_cast<int>(0.06 * kRate);
       i + window <= anchor + static_cast<int>(0.06 * kRate); i += window / 4)
  {
    const double db = rmsDb(laid.data(), i, i + window) - level;
    lowest = std::min(lowest, db);
    highest = std::max(highest, db);
  }
  // What the noise made again does by itself, well past the anchor.
  double lowestAway = 1.0e9, highestAway = -1.0e9;
  for (int i = anchor + static_cast<int>(0.10 * kRate);
       i + window <= anchor + static_cast<int>(0.40 * kRate); i += window / 4)
  {
    const double db = rmsDb(laid.data(), i, i + window) - level;
    lowestAway = std::min(lowestAway, db);
    highestAway = std::max(highestAway, db);
  }
  int changed = 0;
  for (int i = anchor; i < anchor + static_cast<int>(0.05 * kRate); ++i)
    changed += laid[static_cast<size_t>(i)] != source[static_cast<size_t>(i)];
  std::printf("breath across an anchor: 10 ms levels from %.2f to %.2f dB off the breath "
              "(%.2f to %.2f well past it)\n",
              lowest, highest, lowestAway, highestAway);
  CHECK(changed > static_cast<int>(0.04 * kRate)); // made again past the anchor
  // No further off than the noise made again is by itself (it was 3 to 4 dB
  // down with a plain crossfade at the anchor).
  CHECK(lowest > lowestAway - 0.5);
  CHECK(highest < highestAway + 0.5);
}

void aBreathIsTheSameNoiseWhateverElseMoves()
{
  const int samples = 6 * 44100;
  auto source = voice(samples);
  Noise noise;
  for (int i = static_cast<int>(1.38 * kRate); i < static_cast<int>(1.62 * kRate); ++i)
    source[static_cast<size_t>(i)] = 0.05f * noise.next();
  const auto kinds = kindsBetween(samples, 1.40, 1.60, warp::SourceKind::Noise);

  auto layout = stretchedInTheMiddle(samples);
  const auto one = render(source, layout.map, &kinds);
  // Another stretch four seconds on: a run of its own.
  layout.map.addAnchorAtSource(4.5 * layout.second);
  const int k = layout.map.addAnchorAtSource(5.0 * layout.second);
  layout.map.addAnchorAtSource(5.5 * layout.second);
  layout.map.moveAnchor(k, 5.1 * layout.second);
  const auto two = render(source, layout.map, &kinds);
  for (int i = 0; i < static_cast<int>(3.5 * kRate); ++i)
    CHECK(one[static_cast<size_t>(i)] == two[static_cast<size_t>(i)]);
  bool differsLater = false;
  for (int i = static_cast<int>(4.6 * kRate); i < static_cast<int>(5.4 * kRate); ++i)
    differsLater = differsLater || one[static_cast<size_t>(i)] != two[static_cast<size_t>(i)];
  CHECK(differsLater);
}

void nothingStretchedIsStillTheRecording()
{
  const int samples = 3 * 44100;
  const auto source = voice(samples);
  warp::WarpMap map(static_cast<double>(samples) / kHop);
  map.addAnchorAtDestination(50.0);
  const auto kinds = kindsBetween(samples, 0.5, 1.5, warp::SourceKind::Noise);
  CHECK(render(source, map, &kinds) == source);
}
} // namespace

int main()
{
  overlapAtUnityIsTheRecording();
  overlapKeepsEachPulse();
  overlapStartsInPhaseWithTheOtherRendering();
  noiseKeepsItsLevel();
  noiseKeepsItsColourAndFollowsItsLoudness();
  noiseIsTheSameForTheSameSeed();
  kindsComeFromTheBreathLabelsAndTheFry();
  eachKindIsLaidOverItsOwnFramesOnly();
  fryMeetsTheVoiceWithoutADip();
  aBreathCutByAnAnchorKeepsItsLevel();
  aBreathIsTheSameNoiseWhateverElseMoves();
  nothingStretchedIsStillTheRecording();
  std::printf("KindStretchTests passed\n");
  return 0;
}
