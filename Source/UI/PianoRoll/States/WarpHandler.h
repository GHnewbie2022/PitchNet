#pragma once

#include "InteractionHandler.h"
#include "../../../Audio/WarpMap.h"

#include <memory>

class TimingHandler;

/**
 * The warp layer's own tool (spec section 3), AudioWarp style.
 *
 *   click            place an anchor (and drag it straight away)
 *   drag an anchor   stretch the two segments next to it in opposite
 *                    directions; everything else stays put
 *   Alt + click      delete an anchor; its neighbours re-stretch evenly
 *   double-click a ratio in the strip at the top
 *                    type the segment's length ("1.57 s") or ratio ("112%")
 *
 * Ratios and times show two decimals. A ratio outside 0.5 .. 2 is drawn in
 * the warning colour; the hard limits (one frame, 12.5% .. 800%) stop the
 * anchor and say why. Anchors snap to the timeline grid when warp snapping
 * is on.
 */
class WarpHandler final : public InteractionHandler
{
public:
  explicit WarpHandler(PianoRollComponent &owner);
  ~WarpHandler() override;

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
  void draw(juce::Graphics &g) override;
  bool isActive() const override { return dragging || boundaryDragging; }
  void cancel() override;

  /** Alt pressed or released while hovering: update the cursor. */
  void modifiersChanged(const juce::ModifierKeys &mods);

  /** True when a double-click at this point belongs to the warp layer
   *  (the ratio strip), so the canvas should not seek the transport. */
  bool handlesDoubleClickAt(float worldX, float worldY) const;

  /** True over an anchor: Alt + click deletes it there. */
  bool isOverAnchor(float worldX, float worldY) const
  {
    return !inStrip(worldY) && anchorAt(worldX) >= 0;
  }

  /** Draws the anchors and segment ratios; also used by the Timing tool,
   *  whose boundary drags place anchors too. */
  static void drawLayer(juce::Graphics &g, PianoRollInteractionContext &context,
                        int hoveredAnchor, bool deleteHover,
                        int activeAnchor, bool showStrip);

  static constexpr float kStripHeight = 18.0f;

  /** The eraser cursor Alt shows over an anchor; the Pitch Shaper tool's anchors
   *  use it too. */
  static const juce::MouseCursor &eraserCursor();

private:
  enum class Limit
  {
    None,
    MinLength,
    MinRatio,
    MaxRatio,
    TooClose,
    Fixed
  };

  double frameAtX(float worldX) const;
  float xAtFrame(double frame) const;
  bool inStrip(float worldY) const;
  int anchorAt(float worldX) const;
  int segmentAt(float worldX) const;
  static Limit limitFor(const warp::WarpMap &map, int index, double requested,
                        bool &leftSide);
  void showLimit(Limit limit, bool leftSide, double frame);
  void clearLimit();
  void preview(const warp::WarpMap &map);
  void commit(const warp::WarpMap &before, const warp::WarpMap &after,
              const juce::String &name);
  void updateCursor(const juce::ModifierKeys &mods);
  void openEditor(int segment);
  void closeEditor(bool apply);
  void applyEditorText(int segment, const juce::String &text);

  warp::WarpMap beforeMap;
  warp::WarpMap dragBase;
  int dragIndex = -1;
  float downX = 0.0f;
  bool dragging = false;

  // Note boundaries (spec section 4): on a note's row, a note edge drags
  // like the old Timing tool - anchors at the edge and the fixed points
  // beside it - instead of placing a single anchor.
  std::unique_ptr<TimingHandler> boundaries;
  bool boundaryDragging = false;
  bool hoveredBoundary = false;
  bool moved = false;

  int hoveredAnchor = -1;
  float lastWorldX = -1.0f;
  float lastWorldY = -1.0f;
  bool pointerInCanvas = false;

  juce::String limitText;
  double limitFrame = 0.0;

  std::unique_ptr<juce::TextEditor> editor;
  int editorSegment = -1;
};
