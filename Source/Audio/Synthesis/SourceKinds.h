#pragma once

#include "WarpedSource.h"
#include "../BreathSibilanceDetector.h"
#include "../../Models/Project.h"
#include "../../Utils/FryDetector.h"

#include <cstdint>
#include <vector>

/**
 * What each frame of the recording is stretched with (WarpedSource.h): the
 * one place that decides it, for every caller of renderWarpedSource - the
 * standalone's playback and reference, and the plug-in's, whose copy of the
 * host's audio holds no analysis of its own and takes the project's.
 *
 *   Noise   frames labelled breath, and frames labelled sibilance that the
 *           pitch detector found no pitch in (BreathSibilanceDetector.h).
 *           Sibilance went with the voice at first and was heard to turn
 *           phasey (the q of qiang at 1.33x); as noise it did not. A
 *           sibilant frame with a pitch still holds the voice and stays
 *           with it.
 *   Pulses  vocal fry as the pitch curve shows it (FryDetector.h): the low
 *           run at the end of a phrase (fry::detect) and the lead, the low
 *           run and the frames up to the voice as a word starts
 *           (fry::onsets). The frames themselves, not what a note on Auto
 *           keeps of the recording around them (fry::kept): the stretch was
 *           listened to with these.
 *   Voice   everything else
 *
 * Noise wins where the two meet. A voice with a lot of breath in it (a
 * sighed word) is Voice: 35 ms, 120 ms and noise above 2 kHz were heard as
 * much the same there. Empty when every frame is Voice, so a
 * recording with nothing to tell apart compares equal to one never analysed.
 */
namespace warp
{
inline std::vector<std::uint8_t> sourceKinds(const AudioData &audioData)
{
  const size_t frames =
      std::max(audioData.breathSibilance.size(), audioData.rawF0.size());
  std::vector<std::uint8_t> kinds(frames, static_cast<std::uint8_t>(SourceKind::Voice));
  bool any = false;
  const auto mark = [&](const fry::Ranges &ranges)
  {
    for (const auto &[from, to] : ranges)
      for (int frame = std::max(0, from); frame < to && frame < static_cast<int>(frames);
           ++frame)
      {
        kinds[static_cast<size_t>(frame)] = static_cast<std::uint8_t>(SourceKind::Pulses);
        any = true;
      }
  };
  mark(fry::detect(audioData));
  mark(fry::onsets(audioData.rawF0));
  for (size_t frame = 0; frame < audioData.breathSibilance.size(); ++frame)
  {
    const auto label = audioData.breathSibilance[frame];
    const bool voiced = frame < audioData.voicedMask.size() && audioData.voicedMask[frame];
    if (label == breathsibilance::Breath ||
        (label == breathsibilance::Sibilance && !voiced))
    {
      kinds[frame] = static_cast<std::uint8_t>(SourceKind::Noise);
      any = true;
    }
  }
  if (!any)
    kinds.clear();
  return kinds;
}
} // namespace warp
