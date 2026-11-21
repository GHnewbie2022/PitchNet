#include "RubberBandShift.h"

#include <rubberband/RubberBandStretcher.h>

#include <algorithm>

namespace rubberbandshift
{
namespace
{
// The cepstral cutoff is put at 1/(margin x pitch): the harmonics' peak sits
// at 1/pitch, and a cutoff right at it still let them into the envelope.
// 1.5 against 1.25 on the 煙幕 phrase at 165.65 s shifted 3 semitones
// (2026-11-21): the envelope's error was the same (2.8 dB against 2.7, the
// vocoder 2.8, PSOLA 2.9), and shifted up every note stayed within 1.3 dB
// of the recording's level, where 1.25 left the high notes up to 2.9 dB
// under it (Rubber Band's own cutoff: up to 8.7 dB).
constexpr double kFormantMargin = 1.5;
// R3 analyses behind the input it was last given (its start pad is 2048
// samples at 44.1 kHz, its window 2048): the hint covers the recorded pitch
// from 140 ms back to 50 ms ahead of the block, past that on both sides.
constexpr double kHintBackSeconds = 0.14;
constexpr double kHintAheadSeconds = 0.05;
} // namespace

std::vector<float> shift(const std::vector<float> &source, const std::vector<float> &ratio,
                         int sampleRate, const std::atomic<bool> *cancel,
                         bool preserveFormant, const std::vector<float> *sourceHz)
{
  const auto length = source.size();
  if (length == 0 || ratio.size() != length || sampleRate <= 0)
    return {};
  if (sourceHz != nullptr && sourceHz->size() != length)
    sourceHz = nullptr;
  using Stretcher = RubberBand::RubberBandStretcher;
  // Real-time mode: the pitch scale may change between process calls, which
  // a time-varying edit needs (offline mode fixes it before study). High
  // consistency keeps those changes free of discontinuities, as the
  // library's own command line sets it for a pitch map; without it a
  // changing ratio left steps in the output (232.100 s and 232.115 s of the
  // 煙幕 dry take: 0.022 and 0.025, the recording 0.004 and 0.009).
  Stretcher stretcher(static_cast<size_t>(sampleRate), 1,
                      Stretcher::OptionProcessRealTime | Stretcher::OptionEngineFiner |
                          Stretcher::OptionPitchHighConsistency |
                          (preserveFormant ? Stretcher::OptionFormantPreserved
                                           : Stretcher::OptionFormantShifted));
  // The pitch scale follows the curve every 64 samples, as finely as the
  // command line follows a pitch map: coarser steps left larger edges.
  constexpr size_t kBlock = 64;
  stretcher.setMaxProcessSize(kBlock);
  stretcher.setPitchScale(std::clamp(static_cast<double>(ratio.front()), 0.25, 4.0));

  // Real-time mode leaves the start to the caller: pad the input with the
  // silence it asks for and drop the delay from the output.
  const size_t pad = stretcher.getPreferredStartPad();
  const size_t delay = stretcher.getStartDelay();
  std::vector<float> output;
  output.reserve(length + delay + kBlock);
  std::vector<float> chunk(kBlock, 0.0f);
  const auto collect = [&]() {
    for (int ready = stretcher.available(); ready > 0; ready = stretcher.available())
    {
      const size_t start = output.size();
      output.resize(start + static_cast<size_t>(ready));
      float *out[1] = {output.data() + start};
      output.resize(start + stretcher.retrieve(out, static_cast<size_t>(ready)));
    }
  };
  for (size_t done = 0; done < pad;)
  {
    const size_t n = std::min(kBlock, pad - done);
    std::fill(chunk.begin(), chunk.end(), 0.0f);
    const float *in[1] = {chunk.data()};
    stretcher.process(in, n, false);
    collect();
    done += n;
  }
  // The formant hint: the highest recorded pitch over the frames R3 may be
  // analysing when this block goes in (its analysis runs behind the input),
  // with a margin above it so the cepstral cutoff stays clear of the
  // harmonics' own peak.
  const auto hintAt = [&](size_t at) {
    const size_t back = static_cast<size_t>(kHintBackSeconds * sampleRate);
    const size_t ahead = static_cast<size_t>(kHintAheadSeconds * sampleRate);
    const size_t from = at > back ? at - back : 0;
    const size_t to = std::min(length, at + ahead);
    float highest = 0.0f;
    for (size_t i = from; i < to; i += 16)
      highest = std::max(highest, (*sourceHz)[i]);
    return static_cast<double>(highest) * kFormantMargin;
  };
  for (size_t at = 0; at < length; at += kBlock)
  {
    if (cancel != nullptr && cancel->load())
      return {};
    const size_t n = std::min(kBlock, length - at);
    stretcher.setPitchScale(std::clamp(static_cast<double>(ratio[at]), 0.25, 4.0));
    if (preserveFormant && sourceHz != nullptr)
      stretcher.setFormantPitchHint(hintAt(at));
    const float *in[1] = {source.data() + at};
    stretcher.process(in, n, false);
    collect();
  }
  // Flush with silence until the delayed tail is out.
  for (int guard = 0; output.size() < length + delay && guard < 4096; ++guard)
  {
    std::fill(chunk.begin(), chunk.end(), 0.0f);
    const float *in[1] = {chunk.data()};
    stretcher.process(in, kBlock, false);
    collect();
  }
  if (output.size() < length + delay)
    output.resize(length + delay, 0.0f);
  return std::vector<float>(output.begin() + static_cast<std::ptrdiff_t>(delay),
                            output.begin() + static_cast<std::ptrdiff_t>(delay + length));
}
} // namespace rubberbandshift
