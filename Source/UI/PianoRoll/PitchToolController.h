#pragma once

#include "../../JuceHeader.h"
#include "../../Models/Note.h"
#include "../../Models/Project.h"
#include "../../Undo/SynthChoiceAction.h"
#include "../../Undo/UndoActions.h"
#include "../../Utils/FryDetector.h"
#include "../../Utils/PitchToolOperations.h"
#include "../../Utils/TransformParams.h"
#include "CoordinateMapper.h"
#include "PitchToolHandles.h"
#include "VisualWaveformEnvelope.h"
#include <functional>
#include <memory>
#include <utility>
#include <vector>

/**
 * Handles mouse interaction with pitch tool handles.
 */
class PitchToolController {
public:
  PitchToolController();

  /**
   * Handle mouse down on pitch tool handle.
   * @return true if event was handled, false otherwise
   */
  bool mouseDown(const juce::MouseEvent& e,
                 const PitchToolHandles& handles,
                 const std::vector<Note*>& selectedNotes,
                 const CoordinateMapper& mapper);

  /**
   * Handle mouse drag on active handle.
   * @return true if event was handled, false otherwise
   */
  bool mouseDrag(const juce::MouseEvent& e,
                 std::vector<Note*>& selectedNotes,
                 const CoordinateMapper& mapper);

  /**
   * Handle mouse up (commit operation to undo stack).
   * @return true if event was handled, false otherwise
   */
  bool mouseUp(const juce::MouseEvent& e,
               PitchUndoManager* undoManager,
               std::function<void(int, int)> onRangeChanged);

  /**
   * Check if currently dragging a handle.
   */
  bool isDragging() const { return dragging; }
  /**
   * While the amplitude or sibilant balance handle is dragged: the gain the
   * drag applies to `forProject`'s rendered audio, per sample - exactly what
   * the edit will apply when the drag ends (projectgain::changeRatio), edges
   * included. Null when there is nothing to preview.
   */
  const VisualWaveformEnvelope::GainCurve* getGainPreview(
      const Project& forProject) const;
  bool isDraggingSibilantBalance() const
  {
    return dragging &&
           activeHandleType == PitchToolHandles::HandleType::SibilantBalance;
  }
  bool isDraggingVibrato() const
  {
    return dragging && activeHandleType == PitchToolHandles::HandleType::Vibrato;
  }
  bool isDraggingTilt() const
  {
    return dragging &&
           (activeHandleType == PitchToolHandles::HandleType::TiltLeft ||
            activeHandleType == PitchToolHandles::HandleType::TiltRight);
  }
  /** A tilt, rotation or pivot drag: the pivot line is drawn through it. */
  bool isDraggingTiltOrPivot() const
  {
    return dragging &&
           (activeHandleType == PitchToolHandles::HandleType::TiltLeft ||
            activeHandleType == PitchToolHandles::HandleType::TiltRight ||
            activeHandleType == PitchToolHandles::HandleType::TiltPivot);
  }
  /** One of a note's green squares (the stretches kept as recorded by hand,
   *  Note::setKeptRange). */
  static bool isKeepHandle(PitchToolHandles::HandleType type)
  {
    return type == PitchToolHandles::HandleType::KeepHead ||
           type == PitchToolHandles::HandleType::KeepTail;
  }
  bool isDraggingKeep() const { return dragging && isKeepHandle(activeHandleType); }
  PitchToolHandles::HandleType getActiveHandleType() const
  {
    return activeHandleType;
  }
  Note* getActiveHandleNote() const { return activeHandleNote; }
  const std::vector<Note*>& getAffectedNotes() const { return affectedNotes; }

  /**
   * Cancel current drag operation.
   */
  void cancel();

  /**
   * Set project reference (required for accessing audioData.deltaPitch).
   */
  void setProject(Project* proj) { project = proj; }

  /**
   * Callback fired when pitch is edited (for triggering repaint).
   */
  std::function<void()> onPitchEdited;
  std::function<void()> onAmplitudeEdited;
  /**
   * A green square was dragged and let go: the change, to be applied and
   * put in the history as a choice of the synthesis menu is (the note is
   * handed over as it was before the drag; SynthChoiceAction).
   */
  std::function<void(std::unique_ptr<UndoableAction>)> onSynthChoice;

private:
  Project* project = nullptr;
  bool dragging = false;
  // Tilt with Alt held: rotate about the pivot instead of tilting one side.
  bool rotating = false;
  // Alt + click on the pivot: put it back in the middle.
  bool pivotReset = false;
  PitchToolHandles::HandleType activeHandleType = PitchToolHandles::HandleType::None;
  Note* activeHandleNote = nullptr;
  std::vector<Note*> affectedNotes;
  std::vector<TransformParams> originalParams;
  juce::Point<float> dragStartPos;
  // A drag of a green square: the note's choice before it, where its two
  // squares stood (frames of the recording), and the x (world) of the edge
  // the dragged one stood on.
  SynthChoice keepBefore;
  fry::Shown keepStart;
  float keepStartX = 0.0f;
  // getGainPreview's curve, worked out once per drag step.
  mutable VisualWaveformEnvelope::GainCurve gainPreview;
  mutable bool gainPreviewValid = false;

  void applyOperation(std::vector<Note*>& notes,
                      PitchToolHandles::HandleType type,
                      float dragDeltaX,
                      float dragDeltaY,
                      const CoordinateMapper& mapper);
  void dragKeep(float dragDeltaX, const CoordinateMapper& mapper);
};
