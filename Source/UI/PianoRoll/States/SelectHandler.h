#pragma once

#include "InteractionHandler.h"
#include "../../../Models/Note.h"
#include "../../../Models/NoteRef.h"
#include "../../../Undo/F0FrameEdit.h"

#include <limits>
#include <utility>
#include <vector>

/**
 * Handles selection, note dragging (single + multi), box selection,
 * delta scale/offset drags, pitch tool handle interactions, and double-click snap.
 * The most complex handler — covers the Select edit mode.
 */
class SelectHandler : public InteractionHandler {
public:
  explicit SelectHandler(PianoRollComponent &owner);

  bool mouseDown(const juce::MouseEvent &e, float worldX,
                 float worldY) override;
  bool mouseDrag(const juce::MouseEvent &e, float worldX,
                 float worldY) override;
  bool mouseUp(const juce::MouseEvent &e, float worldX,
               float worldY) override;
  void mouseMove(const juce::MouseEvent &e, float worldX,
                 float worldY) override;
  void mouseDoubleClick(const juce::MouseEvent &e, float worldX,
                        float worldY) override;
  bool isActive() const override;
  void cancel() override;

  // ── Rendering-state getters (used by PianoRollComponent draw code) ──
  bool getIsDeltaScaleDragging() const { return isDeltaScaleDragging; }
  const std::vector<Note *> &getDeltaScaleTargetNotes() const {
    return deltaScaleTargetNotes;
  }
  float getDeltaScaleFactor() const { return deltaScaleFactor; }

  bool getIsDeltaOffsetDragging() const { return isDeltaOffsetDragging; }
  const std::vector<Note *> &getDeltaOffsetTargetNotes() const {
    return deltaOffsetTargetNotes;
  }
  float getDeltaOffsetSemitones() const { return deltaOffsetSemitones; }

  bool isSingleNoteDragging() const { return isDragging; }
  /** A note drag started from the fine pitch handle (cents, no snapping). */
  bool isFinePitchDragging() const { return isDragging && fineDrag; }
  Note *getDraggedNote() const { return draggedNote; }

  enum class ResizeEdge { None, Left, Right };
  bool isHoveringNoteEdge() const
  {
    return hoveredEdge != ResizeEdge::None && hoveredEdgeNote != nullptr;
  }
  bool isResizingNoteEdge() const { return isResizingNote; }
  bool hitsResizeEdge(float worldX, float worldY) const;
  void clearEdgeHover()
  {
    hoveredEdgeNote = nullptr;
    hoveredEdge = ResizeEdge::None;
  }

private:
  std::pair<Note *, ResizeEdge> findResizeEdgeAt(float worldX, float worldY) const;
  /** Rebuild base pitch, fire edited/finished callbacks, and repaint. */
  void rebuildAndNotify();

  /** Shared init logic for deltaScale / deltaOffset mouseDown. */
  bool initDeltaDrag(float worldY,
                     bool &isDraggingOut,
                     float &dragStartYOut,
                     std::vector<Note *> &targetNotesOut,
                     std::vector<F0FrameEdit> &editsOut,
                     int &minFrameOut,
                     int &maxFrameOut);

  /** Start dragging one note's pitch, as from a click on the note. */
  void startSingleNoteDrag(Note *note, float worldY, bool fine);

  void prepareDragBasePreview();
  void applyDragBasePreview(float pitchOffsetSemitones);
  void restoreDragBasePreview();
  void restoreResizeInitialNotes(bool rebuildBase = false);

  // Notes already selected when a Cmd box selection began: the box adds to
  // them instead of replacing them.
  std::vector<Note *> boxBaseSelection;

  // Single note drag state
  bool isDragging = false;
  bool fineDrag = false;
  Note *draggedNote = nullptr;
  float dragStartY = 0.0f;
  float originalPitchOffset = 0.0f;
  float originalMidiNote = 60.0f;
  float boundaryF0Start = 0.0f;
  float boundaryF0End = 0.0f;
  std::vector<float> originalF0Values;
  float lastDragPitchOffset = 0.0f;
  int dragPreviewStartFrame = -1;
  int dragPreviewEndFrame = -1;
  std::vector<float> dragPreviewWeights;
  std::vector<float> dragBasePitchSnapshot;
  std::vector<float> dragF0Snapshot;

  // Delta pitch scale drag state
  bool isDeltaScaleDragging = false;
  float deltaScaleDragStartY = 0.0f;
  float deltaScaleFactor = 1.0f;
  int deltaScaleMinFrame = std::numeric_limits<int>::max();
  int deltaScaleMaxFrame = std::numeric_limits<int>::min();
  std::vector<Note *> deltaScaleTargetNotes;
  std::vector<F0FrameEdit> deltaScaleEdits;

  // Delta pitch offset drag state
  bool isDeltaOffsetDragging = false;
  float deltaOffsetDragStartY = 0.0f;
  float deltaOffsetSemitones = 0.0f;
  int deltaOffsetMinFrame = std::numeric_limits<int>::max();
  int deltaOffsetMaxFrame = std::numeric_limits<int>::min();
  std::vector<Note *> deltaOffsetTargetNotes;
  std::vector<F0FrameEdit> deltaOffsetEdits;

  // Note edge resize state
  Note *hoveredEdgeNote = nullptr;
  ResizeEdge hoveredEdge = ResizeEdge::None;
  bool isResizingNote = false;
  NoteRef resizedNoteRef;
  ResizeEdge activeResizeEdge = ResizeEdge::None;
  Note originalResizedNote;
  std::vector<Note> resizeInitialNotes;
  int resizeMinSrcFrame = 0;
  int resizeMaxSrcFrame = 0;
  int currentResizeSrcFrame = 0;
};
