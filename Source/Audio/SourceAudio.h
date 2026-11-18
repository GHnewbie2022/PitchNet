#pragma once

#include "../JuceHeader.h"
#include "../Models/Project.h"

#include <utility>
#include <vector>

/**
 * The recording at its own sample rate and channel count, and the audio
 * playback and export are built from (spec Q5).
 *
 * PitchNet analyses and renders at 44.1 kHz mono (Constants.h): the vocoder
 * and the models are tied to it. Playing that copy back means every sample,
 * edited or not, is resampled twice. Instead:
 *
 *   A  Only edited ranges are replaced. Wherever the 44.1 kHz composite still
 *      equals its reference (the recording laid out by the warp map), the
 *      recording itself plays - no conversion at all.
 *   B  Edited ranges are converted to the recording's rate once, with r8brain.
 *   C  The warp layer renders the recording at its own rate as well
 *      (warpedSourceAudio), so stretched but unedited audio is never resampled.
 *
 * Edited ranges are found by comparing the composite with its reference, not
 * by asking the synthesizer what it rendered: whatever changed the composite
 * (pitch, formant, note gain), the comparison sees it. Each range is widened
 * by kMarginSeconds of unchanged audio and crossfaded there, inside the
 * margin, where both sides carry the same recording.
 *
 * The rendered edit is mono; a stereo recording gets it on every channel
 * inside edited ranges and keeps its own channels everywhere else.
 */
namespace sourceaudio
{
constexpr double kMarginSeconds = 0.020;
constexpr double kFadeSeconds = 0.010;

/** Reads `file` at its own rate and channel count into audioData. */
bool load(AudioData &audioData, const juce::File &file);

/** Rebuilds warpedSourceAudio from sourceAudio and warpMap. */
void rebuildWarped(AudioData &audioData);

struct Playback
{
  juce::AudioBuffer<float> audio;
  int sampleRate = 0;
};

/**
 * The project's audio for playback or export: the recording at its own rate
 * with the edited ranges replaced (A, B, C). Without sourceAudio (plugin
 * mode, or no file) it is the 44.1 kHz composite itself.
 */
Playback build(const AudioData &audioData);

/**
 * Sample ranges [start, end) where `composite` differs from `reference`,
 * each widened by `margin` and merged when they come closer than `join`.
 * Exposed for tests.
 */
std::vector<std::pair<int, int>> editedRanges(const float *composite,
                                              int compositeSamples,
                                              const float *reference,
                                              int referenceSamples, int margin,
                                              int join);
} // namespace sourceaudio
