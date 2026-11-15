#pragma once

#include "InteractionHandler.h"
#include "../../../Audio/WarpMap.h"

#include <utility>
#include <vector>

/**
 * Dragging note boundaries: the note-layer shortcut into the warp layer
 * (spec sections 4 and 5). It lives inside the Warp tool (WarpHandler),
 * which hands it a gesture that starts on a boundary.
 */
class TimingHandler final : public InteractionHandler
{
public:
  explicit TimingHandler(PianoRollComponent& owner);

  /** True when a boundary can be grabbed here (on its notes' rows). */
  bool hitsBoundary(float worldX, float worldY) const;

  bool mouseDown(const juce::MouseEvent& e, float worldX,
                 float worldY) override;
  bool mouseDrag(const juce::MouseEvent& e, float worldX,
                 float worldY) override;
  bool mouseUp(const juce::MouseEvent& e, float worldX,
               float worldY) override;
  void mouseMove(const juce::MouseEvent& e, float worldX,
                 float worldY) override;
  void draw(juce::Graphics& g) override;
  bool isActive() const override { return dragging; }
  void cancel() override;

private:
  struct Boundary
  {
    float frame = 0.0f;
    Note* left = nullptr;
    Note* right = nullptr;
    Note* restCompanion = nullptr;
    float top = 0.0f;
    float bottom = 0.0f;
  };

  struct BoundaryKey
  {
    Note* left = nullptr;
    Note* right = nullptr;
  };

  // A dragged boundary's anchor: its source position, and where on the
  // timeline it was when the drag started.
  struct DragAnchor
  {
    double source = 0.0;
    double startDestination = 0.0;
  };

  std::vector<Boundary> buildBoundaries() const;
  int findBoundary(float worldX, float worldY,
                   const std::vector<Boundary>& boundaries) const;
  // Spec sections 4 and 5: the anchors a boundary drag works on.
  void prepareDragAnchors(const std::vector<Boundary>& boundaries);
  void applyPreviewDelta(float delta);
  void showMap(const warp::WarpMap& map);
  static BoundaryKey keyFor(const Boundary& boundary);
  static bool keysMatch(const BoundaryKey& a, const BoundaryKey& b);
  bool isSelected(const Boundary& boundary) const;
  void addSelected(const Boundary& boundary);
  void removeSelected(const Boundary& boundary);
  std::vector<Boundary> resolveSelection(
      const std::vector<Boundary>& boundaries) const;

  Boundary activeBoundary;
  std::vector<Boundary> dragBoundaries;
  std::vector<BoundaryKey> selectedBoundaries;
  warp::WarpMap beforeMap;
  warp::WarpMap dragMap;
  std::vector<DragAnchor> dragAnchors;
  float previewDelta = 0.0f;
  float hoveredBoundaryFrame = -1.0f;
  bool dragging = false;
};
