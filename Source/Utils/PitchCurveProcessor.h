#pragma once

#include "../Models/Project.h"
#include <vector>

namespace PitchCurveProcessor
{
    /**
     * Linearly interpolate pitch through unvoiced regions using the uv mask.
     * Returns a dense pitch (Hz) array with the same length as the input.
     */
    std::vector<float> interpolateWithUvMask(const std::vector<float>& pitchHz,
                                             const std::vector<bool>& uvMask);

    /**
     * Rebuild base pitch (midi) from current notes and keep existing delta.
     * Ensures base/delta are dense and aligned to the project frame count,
     * then composes audioData.f0 (without applying the uv mask).
     */
    void rebuildBaseFromNotes(Project& project);

    /**
     * Notes nobody edited the pitch of sing what was recorded, so their curve
     * is the recorded pitch (denseF0) over the base pitch every note would
     * have unedited. A split or merge moves that base near a boundary and the
     * old curve then shows a spike there. An edited neighbour does not change
     * the curve: the note's edge follows the base's bend toward it smoothly.
     * Stretched notes are fitted per timeline frame through the warp. Refit
     * the curves of such notes reaching into [start, end), rebuild, and
     * return the start frames of the notes changed. With apply false nothing
     * is changed: the notes whose curve is not the fit are only reported.
     * With warpChanged (the warp map has just changed, warp::apply) the
     * analysed contour of edited notes the warp lays out is fitted too, and
     * that of notes it has put back on their recorded frames: their edits
     * stay on top of a contour that follows the recording as it now plays.
     * Notes with a baked or drawn contour keep it.
     */
    std::vector<int> refitUntouchedCurves(Project& project, int start, int end,
                                          bool apply = true, bool warpChanged = false);

    /**
     * For the notes the warp laid out off their recorded frames (stretched or
     * moved), the pitch (midi, one value per frame of [start, end), 0 outside
     * such notes) each would have with no pitch edit, laid out the way its
     * edited curve is: the unedited base plus its analysed contour through
     * the same note map. An edited note's curve keeps its stored contour,
     * spread over the note's own frames (NoteTiming.h), which sits a fraction
     * of a frame off the recording as the track's map plays it; this is the
     * curve to measure the edit against, so that only the edit is left
     * (spec Q12: Rubber Band's pitch ratio on a stretched note).
     */
    std::vector<float> uneditedPitchOnMovedNotes(Project& project, int start, int end);

    /** Drop every pitch edit on the note: it sings what the analysis gave it. */
    void resetNoteToUnedited(Note& note);

    // Cache the linear drift response once for interactive display previews.
    std::vector<float> getDriftPreviewResponse(const Project& project, const Note& note);

    /**
     * Rebuild delta pitch and f0 only for specific notes (partial rebuild).
     * Skips basePitch regeneration (assumes positions unchanged).
     * Much faster than rebuildBaseFromNotes() when only transformation
     * parameters change on a few notes.
     */
    void rebuildDeltaForNotes(Project& project, const std::vector<Note*>& affectedNotes);

    /**
     * Lightweight rebuild for interactive note drag.
     * Regenerates basePitch from ALL notes (needed for correct pitch display),
     * but only processes deltaPitch for the specified affected notes instead
     * of iterating every note.  Recomposes f0 only for the affected range.
     * Much faster than rebuildBaseFromNotes() during interactive drag.
     */
    void rebuildBaseFromNotesForDrag(Project& project, const std::vector<Note*>& affectedNotes);

    /**
     * Rebuild base and delta from a source pitch (Hz). This is used after
     * detection/segmentation or when we need to recompute delta from edited
     * curves. The uv mask is kept as-is; the composed f0 omits uv masking.
     */
    void rebuildCurvesFromSource(Project& project,
                                 const std::vector<float>& sourcePitchHz);

    /**
     * Compose f0 (Hz) from base + delta + optional global offset.
     * When applyUvMask is true, frames marked unvoiced are forced to 0 for
     * synthesis; when false the curve stays dense for UI display.
     */
    std::vector<float> composeF0(const Project& project,
                                 bool applyUvMask,
                                 float globalPitchOffset = 0.0f);

    /**
     * Convenience to update audioData.f0 in-place using composeF0.
     */
    void composeF0InPlace(Project& project,
                          bool applyUvMask,
                          float globalPitchOffset = 0.0f);
} // namespace PitchCurveProcessor

