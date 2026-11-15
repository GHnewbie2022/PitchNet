#pragma once

#include "WarpMap.h"
#include "../Models/Project.h"

#include <cstdint>

/**
 * The warp layer's glue to the project (the map itself is WarpMap.h).
 *
 * Time lives in the warp layer only. Notes keep the source frames they were
 * detected at; their timeline frames are derived from the map, so moving an
 * anchor moves the notes, their pitch curves and their audio together.
 *
 * Audio: the recording laid out by the map (AudioData::warpedOriginal, see
 * WarpedSource.h) is the reference everything else is blended onto. Where a
 * note's pitch is unchanged the composite is that reference; where it is
 * changed the vocoder renders the note from the mel and pitch read through
 * the same map, and its unvoiced stretches, breaths and sibilants fall back
 * to the reference - the recording where it is not stretched, signalsmith
 * where it is.
 */
namespace warp
{
/** Keeps the map spanning the analysis grid. */
void fitToProject(AudioData &audioData);

/** Derives every note's timeline frames from its source frames. */
void syncNotes(Project &project);

/** Records which map the composite is laid out for, the first time the map
 *  is about to change (until then the composite matches warpMap). Call
 *  before writing warpMap. */
void rememberCompositeMap(AudioData &audioData);

/** True when the composite is laid out for a different map than warpMap:
 *  the piano roll then draws it re-laid out (relayout()). */
bool compositeLagsMap(const AudioData &audioData);

/** Sets a new map: notes follow, the pitch curves are rebuilt, edited notes
 *  are queued for rendering and the composite is marked to be laid out. */
void apply(Project &project, const WarpMap &map);

/** The recording laid out on the timeline: warpedOriginal, or the
 *  recording itself while nothing is warped. */
const juce::AudioBuffer<float> &reference(const AudioData &audioData);

/** Rebuilds warpedOriginal from originalWaveform and the map. */
void rebuildReference(AudioData &audioData);

/**
 * Lays the composite (AudioData::waveform) out again after the map changed:
 * the reference with every gain edit applied, and every note whose pitch is
 * edited queued for rendering at its new place. Clears warpLayoutPending.
 */
void layOut(Project &project);

/** Analysis data of the audio that plays at timeline frame `frame`. */
bool voicedAt(const AudioData &audioData, int frame);
std::uint8_t labelAt(const AudioData &audioData, int frame);

/** Source frame (whole) that plays at timeline frame `frame`. */
int sourceFrameAt(const AudioData &audioData, int frame);
} // namespace warp
