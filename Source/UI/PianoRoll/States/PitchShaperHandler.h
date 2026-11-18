#pragma once

#include "InteractionHandler.h"
#include "../../../Models/DrawnCurve.h"
#include "../../../Utils/TransformParams.h"

#include <algorithm>
#include <utility>
#include <vector>

/**
 * The Pitch Shaper tool, RePitch Shaper style: anchors stay on the pitch line
 * (DrawnCurve, saved with the project) and can be dragged again later.
 *
 *   click on a note        place an anchor; anchors placed one after another
 *                          make one curve, smoothed between them
 *   click in empty space,  end the curve (it stays); the next click starts a
 *   Esc, another tool      new one - or, inside a drawn curve, edits it
 *   drag an anchor         reshape its curve
 *   Alt + click an anchor  delete it (double-click and right-click too); the
 *                          last one gone gives the pitch back
 *
 * Every gesture is applied at once and is one undo step.
 */
class PitchShaperHandler final : public InteractionHandler
{
public:
  explicit PitchShaperHandler(PianoRollComponent& owner);

  bool mouseDown(const juce::MouseEvent& e, float worldX,
                 float worldY) override;
  bool mouseDrag(const juce::MouseEvent& e, float worldX,
                 float worldY) override;
  bool mouseUp(const juce::MouseEvent& e, float worldX,
               float worldY) override;
  void mouseMove(const juce::MouseEvent& e, float worldX,
                 float worldY) override;
  void mouseDoubleClick(const juce::MouseEvent& e, float worldX,
                        float worldY) override;
  void draw(juce::Graphics& g) override;
  bool isActive() const override;
  void cancel() override;

  /** Alt pressed or released: the eraser cursor over an anchor. */
  void modifiersChanged(const juce::ModifierKeys& mods);

  bool hasAnchors() const { return !anchors.empty(); }
  bool isHoveringAnchor() const { return hovered.curveId >= 0; }
  /** An anchor (of any curve) at this world point. */
  bool hasAnchorAt(float worldX, float worldY) const
  {
    return hitTest(worldX, worldY).curveId >= 0;
  }
  bool isPointerOverPitchRegion() const { return pointerOverPitchRegion; }
  bool isPitchRegionAtWorldX(float worldX) const;
  bool removeAnchorAt(float worldX, float worldY);
  const std::vector<float>& getOriginalMidiCurve() const { return originalMidiCurve; }
  const std::vector<float>& getPreviewMidiCurve() const { return previewMidiCurve; }
  // Timeline frames the preview replaces, first to last anchor inclusive;
  // {0, -1} (empty) with no anchors.
  std::pair<int, int> getDrawnFrameRange() const
  {
    if (anchors.empty())
      return {0, -1};
    return {anchors.front().frame, anchors.back().frame};
  }
  void clearHover();
  // Ends the current curve: its anchors stay on the line, what they drew
  // stays (every step is already in the undo history). The next click
  // starts a new curve.
  void finish();

private:
  struct Anchor
  {
    int id = 0;
    int frame = 0;
    float midi = 60.0f;
  };

  // An anchor under the pointer: of the curve being edited (anchorId) or of
  // a stored one (frame).
  struct Hit
  {
    int curveId = -1;
    int anchorId = -1;
    int frame = -1;
  };

  struct OriginalNoteState
  {
    Note* note = nullptr;
    TransformParams params;
    std::vector<float> bakedDeltaPitch;
  };

  int frameFromWorldX(float worldX) const;
  float midiFromWorldY(float worldY) const;
  juce::Point<float> pointAt(int frame, float midi) const;
  float sungPitchAt(int frame) const;
  Hit hitTest(float worldX, float worldY) const;
  Anchor* findAnchorById(int id);
  bool frameBelongsToPitchNote(int frame) const;
  // Starts editing `curve` (or a new curve with nullptr).
  void beginSession(const DrawnCurve* curve);
  // Takes stored curves reaching into [start, end] into the one edited.
  void absorbCurvesIn(int start, int end);
  void extendRange(int start, int end);
  std::pair<int, int> affectedRange() const;
  void sortAnchors();
  void rebuildPreview();
  void updateAffectedNotePositions();
  void applyPreviewToProject();
  void restoreOriginalNoteStates();
  void removeAnchor(int anchorId);
  void clearState();
  void notifyStateChanged();
  void updateCursor(const juce::ModifierKeys& mods);
  // Puts what changed since the last step into the undo history, one step
  // per gesture (an anchor placed, dragged or removed).
  bool commitStep();

  std::vector<Anchor> anchors;
  int activeCurveId = -1;
  std::vector<int> absorbedCurveIds;
  // Frames the curve controls: its anchors' span and every frame it ever
  // covered (those get the pitch they had before back when it shrinks).
  int rangeStart = -1;
  int rangeEnd = -1;
  std::vector<OriginalNoteState> originalNoteStates;
  // The notes and curves as the undo history last saw them.
  std::vector<OriginalNoteState> committedNoteStates;
  std::vector<DrawnCurve> committedCurves;
  std::vector<float> originalMidiCurve;
  std::vector<float> previewMidiCurve;
  int nextAnchorId = 1;
  Hit hovered;
  int activeAnchorId = -1;
  bool dragging = false;
  bool dragMoved = false;
  bool pointerOverPitchRegion = false;
  float lastWorldX = 0.0f;
  float lastWorldY = 0.0f;
};
