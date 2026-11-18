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

/**
 * For a project opened from a .pitchnet that embeds its audio: find the
 * recording and load it (load()). Tried in order: `chosen` alone if given,
 * else the saved path, then a recording beside `projectFile` with its name.
 * A candidate must be as long as the analysed copy (and match the saved
 * fingerprint, if any). On success the project remembers the file and its
 * fingerprint, and a warped project's reference is rebuilt so build() can
 * tell edits from the warp.
 */
bool attachRecording(Project &project, const juce::File &projectFile,
                     const juce::File &chosen = {});

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
 * The same, with the edit (waveform, warpMap, the reference inputs) taken
 * from `edit` and the recording (sourceAudio, sourceSampleRate,
 * warpedSourceAudio) from `recording`. The plugin keeps the recording apart
 * from the project, read from the ARA source.
 */
Playback build(const AudioData &edit, const AudioData &recording);

/** The same, with `composite` in place of edit.waveform. */
Playback build(const AudioData &edit, const juce::AudioBuffer<float> &composite,
               const AudioData &recording);

/** A run of playback samples that differ from the recording. */
struct Piece
{
  long long start = 0; // in recording samples
  juce::AudioBuffer<float> audio;
};

/**
 * The runs where `playback` differs from `recording` in any channel, exactly
 * (no threshold: everything else is the recording, bit for bit). Runs closer
 * than `join` samples become one. This is what the plugin stores and plays:
 * the host's own audio with these pieces laid over it.
 */
std::vector<Piece> differingPieces(const juce::AudioBuffer<float> &playback,
                                   const juce::AudioBuffer<float> &recording,
                                   int join);

/**
 * Overwrites dest[destStart, destStart + length), which holds the recording
 * from recording sample `recordingStart`, with the pieces that cover it.
 * Channel c takes piece channel min(c, channels - 1). Allocates nothing.
 */
void overlay(const std::vector<Piece> &pieces, long long recordingStart,
             juce::AudioBuffer<float> &dest, int destStart, int length);

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
