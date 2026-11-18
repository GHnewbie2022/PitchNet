#pragma once

#include "UndoableAction.h"
#include "../Models/Note.h"
#include "../Models/Project.h"
#include "../Models/NoteRef.h"
#include "../Utils/PitchCurveProcessor.h"
#include "../Utils/TransformParams.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

struct PitchShaperNoteState
{
    NoteRef note;
    TransformParams params;
    std::vector<float> bakedDeltaPitch;
};

/**
 * Commits an anchor-curve preview into each note's editable baked contour and
 * resets its pitch-tool transforms. The immutable analysis contour is never
 * changed; the previous baked contour and transforms are restored by undo/redo.
 */
class PitchShaperAction final : public UndoableAction
{
public:
    PitchShaperAction(Project* projectToEdit,
                      std::vector<PitchShaperNoteState> beforeStates,
                      std::vector<PitchShaperNoteState> afterStates,
                      std::vector<DrawnCurve> curvesBeforeEdit = {},
                      std::vector<DrawnCurve> curvesAfterEdit = {})
        : project(projectToEdit), before(std::move(beforeStates)),
          after(std::move(afterStates)),
          curvesBefore(std::move(curvesBeforeEdit)),
          curvesAfter(std::move(curvesAfterEdit))
    {
    }

    void undo() override { apply(before, curvesBefore); }
    void redo() override { apply(after, curvesAfter); }
    juce::String getName() const override { return "Apply Pitch Shaper"; }

private:
    void apply(const std::vector<PitchShaperNoteState>& states,
               const std::vector<DrawnCurve>& curves)
    {
        if (project)
            project->getDrawnCurves() = curves;

        int minFrame = std::numeric_limits<int>::max();
        int maxFrame = std::numeric_limits<int>::min();

        for (const auto& state : states)
        {
            auto* note = state.note.get();
            if (!note)
                continue;

            state.params.applyToNote(*note);
            note->setBakedDeltaPitch(state.bakedDeltaPitch);
            note->markDirty();
            note->markSynthDirty();
            minFrame = std::min(minFrame, note->getStartFrame());
            maxFrame = std::max(maxFrame, note->getEndFrame());
        }

        if (!project || minFrame > maxFrame)
            return;

        PitchCurveProcessor::rebuildBaseFromNotes(*project);
        project->markNoteEditDirtyRange(minFrame, maxFrame);
    }

    Project* project = nullptr;
    std::vector<PitchShaperNoteState> before;
    std::vector<PitchShaperNoteState> after;
    std::vector<DrawnCurve> curvesBefore;
    std::vector<DrawnCurve> curvesAfter;
};
