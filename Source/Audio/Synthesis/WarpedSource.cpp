#include "WarpedSource.h"

#include "NoiseStretch.h"
#include "OverlapStretch.h"
#include "signalsmith-stretch.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>

namespace warp
{
namespace
{
constexpr double kUnityTolerance = 1.0e-6;
// The voice: signalsmith's window and hop, in seconds (its default is 0.12
// and 0.03).
constexpr double kVoiceWindowSeconds = 0.035;
constexpr double kVoiceHopSeconds = 0.009;

// signalsmith reads inputs[channel][index]; this reads the recording from an
// absolute offset and is silent outside it.
struct SourceReader
{
  const float *samples = nullptr;
  long long count = 0;
  long long base = 0;

  float operator[](int index) const
  {
    const long long position = base + index;
    return position >= 0 && position < count
               ? samples[static_cast<size_t>(position)]
               : 0.0f;
  }
};

struct SourceInputs
{
  SourceReader reader;
  SourceReader &operator[](int) { return reader; }
};

struct BufferOutputs
{
  float *samples = nullptr;
  float *operator[](int) { return samples; }
};

/**
 * signalsmith-stretch over timeline samples [first, last), reading the
 * recording at inputAt(timeline sample). The rate may change from chunk to
 * chunk. Mirrors SignalsmithStretch::exact(): after outputSeek() the output
 * is aligned with the input it was given, and from then on the input is kept
 * inputLatency() + rate * outputLatency() ahead of the output.
 */
template <class InputAt>
void stretchRange(const float *source, int sourceSamples, InputAt inputAt,
                  int first, int last, float *output,
                  const WarpedSourceSettings &settings)
{
  const int count = last - first;
  if (count <= 0)
    return;

  signalsmith::stretch::SignalsmithStretch<float> stretch(settings.seed);
  stretch.configure(1, static_cast<int>(settings.sampleRate * kVoiceWindowSeconds),
                    static_cast<int>(settings.sampleRate * kVoiceHopSeconds));
  const int inputLatency = stretch.inputLatency();
  const int outputLatency = stretch.outputLatency();

  const double startInput = inputAt(static_cast<double>(first));
  const double startRate =
      (inputAt(static_cast<double>(first + outputLatency)) - startInput) /
      std::max(1, outputLatency);
  const long long base = static_cast<long long>(std::floor(startInput));
  const int seekLength = std::max(
      inputLatency, static_cast<int>(std::lround(
                        inputLatency + std::max(0.0, startRate) * outputLatency)));

  stretch.outputSeek(SourceInputs{{source, sourceSamples, base}}, seekLength);
  long long fed = base + seekLength;

  constexpr int kChunk = 256;
  for (int position = first; position < last;)
  {
    const int n = std::min(kChunk, last - position);
    const long long target = std::llround(
        inputAt(static_cast<double>(position + n + outputLatency)) +
        inputLatency);
    const int consume = static_cast<int>(std::max<long long>(0, target - fed));
    stretch.process(SourceInputs{{source, sourceSamples, fed}}, consume,
                    BufferOutputs{output + (position - first)}, n);
    fed += consume;
    position += n;
  }
}

// Start of the `length`-sample window in [from, to) where two renderings of
// the same timeline differ least, so a crossfade between them loses least.
// Scanned from `nearest` outward; a later window has to be clearly better to
// win, so where the two agree the fade stays where it was asked for.
// `a` holds timeline samples from `aFirst` on, `b` from `bFirst` on.
int leastDifferenceWindow(const float *a, int aFirst, const float *b, int bFirst,
                          int from, int to, int length, bool nearestIsEnd)
{
  if (to - from <= length)
    return from;
  constexpr int kStep = 32;
  auto difference = [&](int start)
  {
    double energy = 0.0;
    for (int i = start; i < start + length; ++i)
    {
      const double d = static_cast<double>(b[i - bFirst]) - a[i - aFirst];
      energy += d * d;
    }
    return energy;
  };
  const int lastStart = to - length;
  int best = nearestIsEnd ? lastStart : from;
  double bestEnergy = difference(best);
  for (int k = 1;; ++k)
  {
    const int start = nearestIsEnd ? lastStart - k * kStep : from + k * kStep;
    if (start < from || start > lastStart)
      break;
    const double energy = difference(start);
    if (energy < bestEnergy * 0.9)
    {
      bestEnergy = energy;
      best = start;
    }
  }
  return best;
}

// How alike two renderings are across `length` samples from `start`: their
// correlation with no lag, from 0 (unrelated - two noises) to 1 (the same).
double likeness(const float *a, int aFirst, const float *b, int bFirst, int start,
                int length)
{
  double ab = 0.0, aa = 0.0, bb = 0.0;
  for (int i = start; i < start + length; ++i)
  {
    const double x = a[i - aFirst], y = b[i - bFirst];
    ab += x * y;
    aa += x * x;
    bb += y * y;
  }
  if (aa <= 0.0 || bb <= 0.0)
    return 1.0;
  return std::clamp(ab / std::sqrt(aa * bb), 0.0, 1.0);
}

// What a crossfade `amount` of the way between two renderings has to be
// raised by to keep their level. Two of the same add up to it by themselves;
// two unrelated ones add in power, 3 dB short of it halfway - a breath that
// an anchor cuts in two (the recording up to it, its noise made again past
// it) was heard to flutter there.
double levelGain(double amount, double like)
{
  const double other = 1.0 - amount;
  return 1.0 / std::sqrt(amount * amount + other * other + 2.0 * like * amount * other);
}

// Where a run takes over from the copied recording, or hands back to it.
// `stretched` holds timeline samples from `stretchedFirst` on.
int handOverWindow(const float *copy, const float *stretched,
                   int stretchedFirst, int from, int to, int length,
                   bool nearestIsEnd)
{
  return leastDifferenceWindow(copy, 0, stretched, stretchedFirst, from, to, length,
                               nearestIsEnd);
}

struct Segment
{
  int start = 0;
  int end = 0;
  bool stretched = false;
};

// Timeline samples [start, end) that play one kind of the recording, from
// its frame `sourceFrame` on.
struct KindSpan
{
  int start = 0;
  int end = 0;
  SourceKind kind = SourceKind::Voice;
  int sourceFrame = 0;
};

// The breath and the fry inside the stretched segments of [from, to). A
// frame of the recording sits on its centre: frame f is [f - 0.5, f + 0.5).
std::vector<KindSpan> kindSpans(const WarpMap &map, double hop,
                                const std::vector<std::uint8_t> &kinds,
                                const std::vector<Segment> &segments, int from, int to)
{
  std::vector<KindSpan> spans;
  for (const auto &segment : segments)
  {
    const int a = std::max(segment.start, from);
    const int b = std::min(segment.end, to);
    if (!segment.stretched || b <= a)
      continue;
    const int firstFrame = std::max(
        0, static_cast<int>(std::floor(map.sourceAt(a / hop) + 0.5)));
    const int lastFrame = std::min(
        static_cast<int>(kinds.size()) - 1,
        static_cast<int>(std::floor(map.sourceAt(b / hop) + 0.5)));
    for (int frame = firstFrame; frame <= lastFrame; ++frame)
    {
      const auto kind = static_cast<SourceKind>(kinds[static_cast<size_t>(frame)]);
      if (kind != SourceKind::Noise && kind != SourceKind::Pulses)
        continue;
      const int start = std::clamp(
          static_cast<int>(std::llround(map.destinationAt(frame - 0.5) * hop)), a, b);
      const int end = std::clamp(
          static_cast<int>(std::llround(map.destinationAt(frame + 0.5) * hop)), a, b);
      if (end <= start)
        continue;
      if (!spans.empty() && spans.back().kind == kind && spans.back().end == start)
        spans.back().end = end;
      else
        spans.push_back({start, end, kind, frame});
    }
  }
  return spans;
}

// A breath that an anchor cuts in two. The run renders the unstretched side
// at rate 1, but within about half a window of the anchor signalsmith is
// already turning into the stretched side, and noise loses level while it
// does (2 dB over 15 ms, heard as a flutter in the breath). So the breath's
// own noise starts `lead` before the run's first stretched sample `from`, or
// goes on `lead` past its last `to` - as far as the breath itself goes, and
// no further than the run [first, last).
void extendBreathPastTheAnchors(std::vector<KindSpan> &spans, const WarpMap &map,
                                double hop, const std::vector<std::uint8_t> &kinds,
                                int from, int to, int first, int last, int lead)
{
  if (spans.empty())
    return;
  const auto isBreath = [&](int frame)
  {
    return frame >= 0 && frame < static_cast<int>(kinds.size()) &&
           static_cast<SourceKind>(kinds[static_cast<size_t>(frame)]) == SourceKind::Noise;
  };
  const auto sampleAt = [&](double frame)
  { return static_cast<int>(std::llround(map.destinationAt(frame) * hop)); };

  auto &head = spans.front();
  if (head.kind == SourceKind::Noise && head.start == from)
  {
    const int limit = std::max(first, from - lead);
    for (int frame = static_cast<int>(std::floor(map.sourceAt(from / hop) + 0.5));
         isBreath(frame) && head.start > limit; --frame)
      head.start = std::max(limit, std::min(head.start, sampleAt(frame - 0.5)));
  }
  auto &tail = spans.back();
  if (tail.kind == SourceKind::Noise && tail.end == to)
  {
    const int limit = std::min(last, to + lead);
    for (int frame = static_cast<int>(std::floor(map.sourceAt(to / hop) + 0.5));
         isBreath(frame) && tail.end < limit; ++frame)
      tail.end = std::min(limit, std::max(tail.end, sampleAt(frame + 0.5)));
  }
}

// Lays the fry and then the breath of `spans` over `rendered`, signalsmith's
// rendering of timeline samples [first, last).
void layKindsOver(const float *source, int sourceSamples,
                  const std::function<double(double)> &inputAt,
                  const std::vector<KindSpan> &spans, int first, int last,
                  std::vector<float> &rendered, const WarpedSourceSettings &settings)
{
  const int fade = std::max(2, static_cast<int>(settings.kindFadeSeconds *
                                                settings.sampleRate) / 2 * 2);
  const int half = fade / 2;
  const int reach = std::max(0, static_cast<int>(settings.kindReachSeconds *
                                                 settings.sampleRate));
  const auto raisedCosine = [](double x)
  { return 0.5 - 0.5 * std::cos(3.14159265358979323846 * std::clamp(x, 0.0, 1.0)); };
  std::vector<float> other;

  // `other`, timeline samples from `otherFirst`, takes over across `fade`
  // samples from `in` and hands back across `fade` samples from `out`. At
  // either end of the rendering there is nothing to fade from.
  const auto mix = [&](int otherFirst, int otherLast, int in, int out)
  {
    // How alike the two are across each fade, taken before either is mixed.
    const auto likenessAcross = [&](int from)
    {
      const int a = std::max(from, otherFirst), b = std::min(from + fade, otherLast);
      return b > a ? likeness(rendered.data(), first, other.data(), otherFirst, a, b - a) : 1.0;
    };
    const double likeIn = likenessAcross(in), likeOut = likenessAcross(out);
    for (int i = otherFirst; i < otherLast; ++i)
    {
      const double up = in <= first ? 1.0 : raisedCosine((i - in) / static_cast<double>(fade));
      const double down =
          out + fade >= last ? 1.0 : 1.0 - raisedCosine((i - out) / static_cast<double>(fade));
      const double amount = std::min(up, down);
      if (amount <= 0.0)
        continue;
      auto &sample = rendered[static_cast<size_t>(i - first)];
      const double mixed = sample + amount * (other[static_cast<size_t>(i - otherFirst)] - sample);
      sample = static_cast<float>(levelGain(amount, up <= down ? likeIn : likeOut) * mixed);
    }
  };

  for (size_t index = 0; index < spans.size(); ++index)
  {
    const auto &span = spans[index];
    if (span.kind != SourceKind::Pulses)
      continue;
    // The hand-over back may go past the fry, into the voice after it, but
    // not into what the next span lays down.
    const int limit = index + 1 < spans.size()
                          ? std::max(span.end + half, spans[index + 1].start - half)
                          : last;
    const int otherFirst = std::max(first, span.start - half);
    const int otherLast = std::min({last, limit, span.end + half + reach});
    if (otherLast <= otherFirst)
      continue;
    other.assign(static_cast<size_t>(otherLast - otherFirst), 0.0f);
    OverlapStretchSettings overlap;
    overlap.sampleRate = settings.sampleRate;
    overlapStretch(source, sourceSamples, inputAt, otherFirst, otherLast, other.data(),
                   overlap, rendered.data(), first, static_cast<int>(rendered.size()));
    const int out = leastDifferenceWindow(
        rendered.data(), first, other.data(), otherFirst,
        std::max(otherFirst, std::min(span.end - half, otherLast - fade)), otherLast, fade,
        false);
    mix(otherFirst, otherLast, span.start - half, out);
  }

  for (const auto &span : spans)
  {
    if (span.kind != SourceKind::Noise)
      continue;
    const int otherFirst = std::max(first, span.start - half);
    const int otherLast = std::min(last, span.end + half);
    if (otherLast <= otherFirst)
      continue;
    other.assign(static_cast<size_t>(otherLast - otherFirst), 0.0f);
    NoiseStretchSettings noise;
    noise.sampleRate = settings.sampleRate;
    // By where the breath starts in the recording: the same breath is the
    // same noise whatever else the map moves, and on every channel.
    noise.seed = static_cast<std::uint32_t>(settings.seed) ^
                 (static_cast<std::uint32_t>(span.sourceFrame) * 2654435761u);
    noiseStretch(source, sourceSamples, inputAt, otherFirst, otherLast, other.data(), noise);
    mix(otherFirst, otherLast, span.start - half, span.end - half);
  }
}
} // namespace

std::vector<StretchedRun> renderWarpedSource(const float *source,
                                             int sourceSamples,
                                             const WarpMap &map,
                                             float *output, int outputSamples,
                                             const WarpedSourceSettings &settings)
{
  std::vector<StretchedRun> runs;
  if (output == nullptr || outputSamples <= 0)
    return runs;
  std::fill(output, output + outputSamples, 0.0f);
  if (source == nullptr || sourceSamples <= 0)
    return runs;

  const double hop = settings.hopSize;
  const auto inputAt = [&](double timelineSample)
  { return map.sourceAt(timelineSample / hop) * hop; };
  const auto toSample = [&](double frame)
  {
    return std::clamp(static_cast<int>(std::llround(frame * hop)), 0,
                      outputSamples);
  };

  // 1. Every unstretched segment: the recording, moved.
  const auto points = map.points();
  std::vector<Segment> segments;
  for (size_t i = 0; i + 1 < points.size(); ++i)
  {
    const auto &a = points[i];
    const auto &b = points[i + 1];
    Segment segment{toSample(a.destination), toSample(b.destination),
                    std::abs(WarpMap::segmentRatio(a, b) - 1.0) >
                        kUnityTolerance};
    if (i + 2 == points.size())
      segment.end = outputSamples; // the track's tail, past the last frame
    if (segment.end <= segment.start)
      continue;
    segments.push_back(segment);
    if (segment.stretched)
      continue;
    const long long offset =
        std::llround((a.destination - a.source) * hop);
    for (int sample = segment.start; sample < segment.end; ++sample)
    {
      const long long from = sample - offset;
      if (from >= 0 && from < sourceSamples)
        output[sample] = source[static_cast<size_t>(from)];
    }
  }

  // 2. Stretched runs. Unstretched stretches too short to hold both hand-
  // overs stay inside the run.
  const int pad = std::max(1, static_cast<int>(settings.padSeconds *
                                               settings.sampleRate));
  const int fade = std::max(1, static_cast<int>(settings.fadeSeconds *
                                                settings.sampleRate));
  std::vector<StretchedRun> cores;
  for (const auto &segment : segments)
  {
    if (!segment.stretched)
      continue;
    if (!cores.empty() && segment.start - cores.back().endSample < 2 * pad + fade)
      cores.back().endSample = segment.end;
    else
      cores.push_back({segment.start, segment.end});
  }

  const int reach = std::max(pad, static_cast<int>(settings.handOverSeconds *
                                                   settings.sampleRate));
  std::vector<float> rendered;
  for (size_t c = 0; c < cores.size(); ++c)
  {
    const auto &core = cores[c];
    // Up to `reach` into the unstretched audio, but at most halfway to the
    // neighbouring run, so two runs never hand over in the same place.
    int first = std::max(0, core.startSample - reach);
    int last = std::min(outputSamples, core.endSample + reach);
    if (c > 0)
      first = std::max(first, (cores[c - 1].endSample + core.startSample) / 2);
    if (c + 1 < cores.size())
      last = std::min(last, (core.endSample + cores[c + 1].startSample) / 2);
    rendered.assign(static_cast<size_t>(last - first), 0.0f);
    stretchRange(source, sourceSamples, inputAt, first, last, rendered.data(),
                 settings);
    if (settings.kinds != nullptr && !settings.kinds->empty())
    {
      auto spans = kindSpans(map, hop, *settings.kinds, segments, core.startSample,
                             core.endSample);
      extendBreathPastTheAnchors(
          spans, map, hop, *settings.kinds, core.startSample, core.endSample, first, last,
          static_cast<int>(settings.sampleRate * kVoiceWindowSeconds));
      layKindsOver(source, sourceSamples, inputAt, spans, first, last, rendered, settings);
    }

    // Where the run takes over from the copy, and hands back to it.
    const bool fadeIn = first > 0;
    const bool fadeOut = last < outputSamples;
    const int inStart =
        fadeIn ? handOverWindow(output, rendered.data(), first, first,
                                core.startSample, fade, true)
               : first;
    const int outStart =
        fadeOut ? handOverWindow(output, rendered.data(), first,
                                 core.endSample, last, fade, false)
                : last;
    const auto likenessAcross = [&](int from)
    {
      const int length = std::min(fade, std::min(last, outputSamples) - from);
      return length > 0
                 ? likeness(output, 0, rendered.data(), first, from, length)
                 : 1.0;
    };
    const double likeIn = fadeIn ? likenessAcross(inStart) : 1.0;
    const double likeOut = fadeOut ? likenessAcross(outStart) : 1.0;
    for (int sample = inStart; sample < (fadeOut ? outStart + fade : last);
         ++sample)
    {
      if (sample >= outputSamples)
        break;
      float amount = 1.0f;
      double like = 1.0;
      if (fadeIn && sample < inStart + fade)
      {
        amount = static_cast<float>(sample - inStart + 1) /
                 static_cast<float>(fade + 1);
        like = likeIn;
      }
      if (fadeOut && sample >= outStart)
      {
        const float down = static_cast<float>(outStart + fade - sample) /
                           static_cast<float>(fade + 1);
        if (down < amount)
          like = likeOut;
        amount = std::min(amount, down);
      }
      const float stretched = rendered[static_cast<size_t>(sample - first)];
      output[sample] += amount * (stretched - output[sample]);
      if (amount < 1.0f)
        output[sample] = static_cast<float>(levelGain(amount, like) * output[sample]);
    }
    runs.push_back({first, last});
  }
  return runs;
}
} // namespace warp
