#pragma once

#include "../JuceHeader.h"
#include "../Models/Project.h"
#include "../Models/RecordingFingerprint.h"

#include <functional>
#include <utility>
#include <vector>

/**
 * Q11: a project that links its recording (RecordingFingerprint.h), and the
 * recording edited outside PitchNet (RX) between two sessions.
 *
 * On open the recording is read and everything derived from it is rebuilt
 * (hydrate). When its fingerprint differs from the saved one, the blocks that
 * changed are analysed again (apply): the pitch is detected again there with
 * some context around it, every note's analysed curve takes the difference
 * between the new and the old pitch (so the edits on it - centre, tilt,
 * vibrato, drift - apply to the new pitch), untouched notes are fitted to the
 * new pitch, and every edited note over the changed audio is queued for
 * rendering from the new audio. A contour a tool committed (baked) has no
 * edit left to apply again: it takes the difference as far as it followed
 * the recording around the change (a flattened vibrato stays flat). A drawn
 * curve's line stays; the pitch under it (given back when it is undone)
 * takes the difference.
 * A recording of a different rate, channel count or length is refused by the
 * caller: every edit is placed by frame.
 */
namespace recordingchange
{
/** The recording's fingerprint: shape and one hash per block (all channels). */
recording::Fingerprint fingerprint(const juce::AudioBuffer<float> &source, int sampleRate);

/** The same, read from `reader` block by block (the plug-in's host audio,
 *  without holding all of it at once). */
recording::Fingerprint fingerprint(juce::AudioFormatReader &reader);

/** Source sample ranges [start, end) whose blocks differ (same shape). */
std::vector<std::pair<juce::int64, juce::int64>>
changedSamples(const recording::Fingerprint &before, const recording::Fingerprint &after);

/** What the import analyses: 44.1 kHz mono, channels 0 and 1 averaged
 *  (EditorController::loadAudioFileAsync). */
juce::AudioBuffer<float> analysisCopy(const juce::AudioBuffer<float> &source, int sampleRate);

/** The composite's rendered edits: the runs where it differs from the
 *  reference (the recording laid out by the warp map), exactly. */
std::vector<recording::Piece> editPieces(const AudioData &audioData);

/**
 * For a project file that links its recording, once the recording is in
 * audioData.sourceAudio: the analysis copy, the mel, the warped reference,
 * the composite (the reference with the saved pieces laid over it), note
 * amplitudes and breath labels, and untouched curves fitted to the pitch.
 */
bool hydrate(Project &project);

/**
 * An embedded project linked to its recording for the first time: the
 * analysis copy read from the recording replaces the stored one (saved as
 * text, rounded to 1e-6), the reference is rebuilt from it, and the
 * composite outside its rendered edits follows the new reference. Else the
 * stretched reference, which the warp renders anew from a slightly different
 * input, no longer equals the composite where nothing is edited, and the
 * linked file would keep that part as an edit. False (nothing changed) when
 * the recording's copy is not as long as the stored one.
 */
bool adoptRecording(Project &project);

/** Pitch of `numSamples` analysis samples, `frames` analysis frames long:
 *  f0 (Hz, 0 = unvoiced) and voicing, as the analysis detects it. */
using Detect = std::function<bool(const float *samples, int numSamples, int frames,
                                  std::vector<float> &f0, std::vector<bool> &voiced)>;

struct Update
{
  // Source frames whose audio changed, and whose pitch changed.
  std::vector<std::pair<int, int>> audioFrames;
  std::vector<std::pair<int, int>> pitchFrames;
  int notesToRender = 0;
  bool detected = true; // every changed part could be analysed again
};

/**
 * The hydrated project's recording differs from `saved`: analyse the changed
 * blocks again and queue the edited notes there for rendering (see above).
 * The project's fingerprint becomes the recording's. `saved` must have the
 * same shape.
 */
Update apply(Project &project, const recording::Fingerprint &saved, const Detect &detect);

/** The same with the recording's fingerprint given (the plug-in, whose
 *  project holds no sourceAudio: the host owns the recording). */
Update apply(Project &project, const recording::Fingerprint &saved,
             const recording::Fingerprint &now, const Detect &detect);

/** "48000 Hz, 2 ch, 248.875 s (11946000)": a shape for messages. */
juce::String describe(const recording::Fingerprint &shape);

/** The recording's shape as the file reads (rate, channels, length). */
recording::Fingerprint shapeOf(const juce::File &file);

struct Opened
{
  enum class Result
  {
    opened,     // as saved, or analysed again where the recording changed
    otherShape, // another rate, channel count or length: refused
    unreadable
  };
  Result result = Result::unreadable;
  recording::Fingerprint found; // the file's shape (otherShape)
  bool changed = false;         // the recording differs from the saved one
  Update update;
};

/**
 * A project that links its recording, opened with `file`: refused when the
 * file's shape differs from the saved fingerprint; else read, hydrated, and
 * when its content differs, analysed again where it changed (apply). The
 * project then names the file and its SHA-256. The App's open and the
 * offline tools go through this alike.
 */
Opened openLinked(Project &project, const juce::File &file, const Detect &detect);
} // namespace recordingchange
