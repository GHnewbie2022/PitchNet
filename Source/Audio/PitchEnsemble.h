#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

/**
 * Pitch from several detectors at once. Each detector's F0 is put on the
 * analysis frame grid (its own frame time and time offset), then every frame
 * takes the median of the detectors that hear a pitch there: one detector
 * that slips (an octave jump, a dip at a note start) is outvoted by the two
 * that agree, and where all agree the detail they share stays.
 */
namespace pitchensemble
{

// Detector order RMVPE, FCPE, SwiftF0: seconds per frame (160 / 16 kHz,
// 160 / 16 kHz, 256 / 16 kHz) and how late each reports against FCPE, in
// seconds. FCPE's window is centred 80 samples (5 ms) after its frame time,
// RMVPE's on it, so RMVPE reads 5 ms late. WarpRender --ensemble-lag measured
// 0-9 ms over five recordings (about 9 ms sung, 0 ms whispered); 5 ms and
// 9 ms tie over all of them. SwiftF0 measured 0 ms.
inline constexpr double kFrameTime[3] = {0.010, 0.010, 0.016};
inline constexpr double kOffset[3] = {0.005, 0.0, 0.0};

/** One detector's F0 (Hz, 0 = unvoiced) on the analysis grid. */
struct Curve
{
  std::vector<float> f0;
  std::vector<bool> voiced;
};

/**
 * Resamples a detector's F0 onto `targetFrames` frames of `targetFrameTime`
 * seconds. `offsetSeconds` is how late the detector reports: its value for
 * time t is read at t + offsetSeconds. Voicing comes from the nearest source
 * frame, pitch is interpolated in the log domain between voiced frames.
 */
inline Curve resample(const std::vector<float>& source, double sourceFrameTime,
                      double offsetSeconds, int targetFrames,
                      double targetFrameTime)
{
  Curve out;
  out.f0.assign(static_cast<size_t>(std::max(0, targetFrames)), 0.0f);
  out.voiced.assign(out.f0.size(), false);
  const int count = static_cast<int>(source.size());
  if (count == 0 || sourceFrameTime <= 0.0)
    return out;

  for (int i = 0; i < targetFrames; ++i)
  {
    const double pos =
        std::max(0.0, (i * targetFrameTime + offsetSeconds) / sourceFrameTime);
    const int left = std::clamp(static_cast<int>(std::floor(pos)), 0, count - 1);
    const int right = std::min(left + 1, count - 1);
    const int nearest =
        std::clamp(static_cast<int>(std::llround(pos)), 0, count - 1);
    const double frac = std::clamp(pos - std::floor(pos), 0.0, 1.0);

    // Voicing on its own, so a voiced neighbour never voices this frame.
    if (source[static_cast<size_t>(nearest)] <= 0.0f)
      continue;
    out.voiced[static_cast<size_t>(i)] = true;

    const float l = source[static_cast<size_t>(left)];
    const float r = source[static_cast<size_t>(right)];
    if (l > 0.0f && r > 0.0f && left != right)
      out.f0[static_cast<size_t>(i)] = static_cast<float>(
          std::exp(std::log(l) * (1.0 - frac) + std::log(r) * frac));
    else
      out.f0[static_cast<size_t>(i)] = source[static_cast<size_t>(nearest)];
  }
  return out;
}

/**
 * Combines the curves frame by frame:
 *   voicing   the majority of the curves (with two curves, either one)
 *   3 voiced  the median
 *   2 voiced  their mean when within `agreeSemitones`; otherwise the one
 *             nearer the previous frame's result
 *   1 voiced  that one (only reachable with fewer than three curves)
 * Pitch is compared and averaged in semitones. Empty curves are skipped.
 */
inline Curve combine(const std::vector<Curve>& curves,
                     float agreeSemitones = 1.0f)
{
  std::vector<const Curve*> used;
  size_t frames = 0;
  for (const auto& c : curves)
    if (!c.f0.empty())
    {
      used.push_back(&c);
      frames = std::max(frames, c.f0.size());
    }

  Curve out;
  out.f0.assign(frames, 0.0f);
  out.voiced.assign(frames, false);
  if (used.empty())
    return out;
  if (used.size() == 1)
    return *used.front();

  const size_t majority = used.size() == 2 ? 1 : used.size() / 2 + 1;
  auto semis = [](float hz) { return 12.0f * std::log2(hz / 440.0f) + 69.0f; };
  auto hertz = [](float st) { return 440.0f * std::exp2((st - 69.0f) / 12.0f); };

  float previous = -1.0f; // last voiced result, semitones
  std::vector<float> values;
  for (size_t i = 0; i < frames; ++i)
  {
    values.clear();
    for (const auto* c : used)
      if (i < c->f0.size() && c->voiced[i] && c->f0[i] > 0.0f)
        values.push_back(semis(c->f0[i]));
    if (values.empty() || values.size() < majority)
      continue;

    float result = 0.0f;
    std::sort(values.begin(), values.end());
    if (values.size() >= 3)
    {
      const size_t mid = values.size() / 2;
      result = values.size() % 2 == 1
                   ? values[mid]
                   : 0.5f * (values[mid - 1] + values[mid]);
    }
    else if (values.size() == 2)
    {
      if (values[1] - values[0] <= agreeSemitones || previous < 0.0f)
        result = 0.5f * (values[0] + values[1]);
      else
        result = std::abs(values[0] - previous) <= std::abs(values[1] - previous)
                     ? values[0]
                     : values[1];
    }
    else
    {
      result = values[0];
    }

    out.voiced[i] = true;
    out.f0[i] = hertz(result);
    previous = result;
  }
  return out;
}

} // namespace pitchensemble
