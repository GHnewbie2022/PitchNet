#include "WarpedSource.h"

#include "signalsmith-stretch.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace warp
{
namespace
{
constexpr double kUnityTolerance = 1.0e-6;

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
  stretch.presetDefault(1, static_cast<float>(settings.sampleRate));
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

// Start of the quietest `length`-sample window in buffer[from, to).
int quietestWindow(const float *buffer, int from, int to, int length)
{
  if (to - from <= length)
    return from;
  constexpr int kStep = 32;
  int best = from;
  double bestEnergy = std::numeric_limits<double>::infinity();
  for (int start = from; start + length <= to; start += kStep)
  {
    double energy = 0.0;
    for (int i = start; i < start + length; ++i)
      energy += static_cast<double>(buffer[i]) * buffer[i];
    if (energy < bestEnergy)
    {
      bestEnergy = energy;
      best = start;
    }
  }
  return best;
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

  const double hop = static_cast<double>(settings.hopSize);
  const auto inputAt = [&](double timelineSample)
  { return map.sourceAt(timelineSample / hop) * hop; };
  const auto toSample = [&](double frame)
  {
    return std::clamp(static_cast<int>(std::llround(frame * hop)), 0,
                      outputSamples);
  };

  // 1. Every unstretched segment: the recording, moved.
  const auto points = map.points();
  struct Segment
  {
    int start = 0;
    int end = 0;
    bool stretched = false;
  };
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

  std::vector<float> rendered;
  for (const auto &core : cores)
  {
    const int first = std::max(0, core.startSample - pad);
    const int last = std::min(outputSamples, core.endSample + pad);
    rendered.assign(static_cast<size_t>(last - first), 0.0f);
    stretchRange(source, sourceSamples, inputAt, first, last, rendered.data(),
                 settings);

    // Where the run takes over from the copy, and hands back to it.
    const bool fadeIn = first > 0;
    const bool fadeOut = last < outputSamples;
    const int inStart =
        fadeIn ? quietestWindow(output, first, core.startSample, fade) : first;
    const int outStart =
        fadeOut ? quietestWindow(output, core.endSample, last, fade) : last;
    for (int sample = inStart; sample < (fadeOut ? outStart + fade : last);
         ++sample)
    {
      if (sample >= outputSamples)
        break;
      float amount = 1.0f;
      if (fadeIn && sample < inStart + fade)
        amount = static_cast<float>(sample - inStart + 1) /
                 static_cast<float>(fade + 1);
      if (fadeOut && sample >= outStart)
        amount = std::min(amount, static_cast<float>(outStart + fade - sample) /
                                      static_cast<float>(fade + 1));
      const float stretched = rendered[static_cast<size_t>(sample - first)];
      output[sample] += amount * (stretched - output[sample]);
    }
    runs.push_back({first, last});
  }
  return runs;
}
} // namespace warp
