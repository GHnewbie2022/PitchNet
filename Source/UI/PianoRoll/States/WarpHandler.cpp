#include "WarpHandler.h"

#include "../../PianoRollComponent.h"
#include "TimingHandler.h"
#include "../../Components/AppFont.h"
#include "../../../Audio/WarpLayer.h"
#include "../../../Undo/WarpAction.h"
#include "../../../Utils/Constants.h"
#include "../../../Utils/Localization.h"
#include "../../../Utils/PitchCurveProcessor.h"
#include "../../../Utils/UI/Theme.h"

#include <algorithm>
#include <cmath>

namespace
{
constexpr float kAnchorHitPixels = 5.0f;
constexpr float kDragThresholdPixels = 3.0f;

// Literal values: a namespace-scope copy of APP_COLOR_SECONDARY would depend
// on the other file's static initialisation order and could read it empty.
const juce::Colour kAnchorColour = juce::Colour(0xFF44D1DBu);
const juce::Colour kLongerTint = juce::Colour(0xFF44D1DBu);
const juce::Colour kShorterTint = juce::Colour(0xFFE06AA8u);

double framesToSecondsD(double frame)
{
  return frame * static_cast<double>(HOP_SIZE) / SAMPLE_RATE;
}

juce::String formatRatio(double ratio)
{
  return juce::String(ratio * 100.0, 2) + "%";
}

juce::String formatSeconds(double seconds)
{
  return juce::String(seconds, 2) + " s";
}

// Cursors, drawn at 2x. Placing: a warp anchor (a vertical line under a
// tab) with a "+". Over an anchor with Alt held: an eraser, as in Cubase.
juce::Image placeCursorImage()
{
  constexpr int size = 32;
  juce::Image image(juce::Image::ARGB, size, size, true);
  juce::Graphics g(image);
  auto outlined = [&](const juce::Path &path, juce::Colour fill)
  {
    g.setColour(juce::Colours::black.withAlpha(0.85f));
    g.strokePath(path, juce::PathStrokeType(4.0f));
    g.setColour(fill);
    g.strokePath(path, juce::PathStrokeType(2.0f));
  };

  juce::Path line;
  line.startNewSubPath(16.0f, 4.0f);
  line.lineTo(16.0f, 28.0f);
  outlined(line, juce::Colours::white);

  juce::Path tab;
  tab.addTriangle(10.0f, 3.0f, 22.0f, 3.0f, 16.0f, 10.0f);
  g.setColour(juce::Colours::black.withAlpha(0.85f));
  g.strokePath(tab, juce::PathStrokeType(2.0f));
  g.setColour(kAnchorColour);
  g.fillPath(tab);

  juce::Path plus;
  plus.startNewSubPath(25.0f, 18.0f);
  plus.lineTo(25.0f, 28.0f);
  plus.startNewSubPath(20.0f, 23.0f);
  plus.lineTo(30.0f, 23.0f);
  outlined(plus, juce::Colours::white);
  return image;
}

// A tilted eraser: white rubber tip at the bottom left (the hotspot), red
// body towards the top right.
juce::Image eraserCursorImage()
{
  constexpr int size = 32;
  juce::Image image(juce::Image::ARGB, size, size, true);
  juce::Graphics g(image);
  auto quad = [](juce::Point<float> a, juce::Point<float> b,
                 juce::Point<float> c, juce::Point<float> d)
  {
    juce::Path path;
    path.startNewSubPath(a);
    path.lineTo(b);
    path.lineTo(c);
    path.lineTo(d);
    path.closeSubPath();
    return path;
  };
  const juce::Point<float> a{3.5f, 21.5f}, b{10.5f, 28.5f}, c{28.5f, 10.5f},
      d{21.5f, 3.5f}, e{9.8f, 15.2f}, f{16.8f, 22.2f};
  const auto whole = quad(a, b, c, d);
  g.setColour(juce::Colours::black.withAlpha(0.9f));
  g.strokePath(whole, juce::PathStrokeType(3.0f));
  g.setColour(juce::Colour(0xFFF2F2F2u));
  g.fillPath(quad(a, b, f, e));
  g.setColour(juce::Colour(0xFFE5484Du));
  g.fillPath(quad(e, f, c, d));
  g.setColour(juce::Colours::black.withAlpha(0.6f));
  g.drawLine({e, f}, 1.5f);
  return image;
}

const juce::MouseCursor &placeCursor()
{
  static const juce::MouseCursor cursor(
      juce::ScaledImage(placeCursorImage(), 2.0), {8, 8});
  return cursor;
}

const juce::MouseCursor &deleteCursor()
{
  static const juce::MouseCursor cursor(
      juce::ScaledImage(eraserCursorImage(), 2.0), {3, 12});
  return cursor;
}
} // namespace

const juce::MouseCursor &WarpHandler::eraserCursor() { return deleteCursor(); }

WarpHandler::WarpHandler(PianoRollComponent &owner)
    : InteractionHandler(owner),
      boundaries(std::make_unique<TimingHandler>(owner))
{
}

WarpHandler::~WarpHandler() { closeEditor(false); }

double WarpHandler::frameAtX(float worldX) const
{
  const double seconds = worldX / std::max(1.0f, owner_.pixelsPerSecond);
  return seconds * SAMPLE_RATE / static_cast<double>(HOP_SIZE);
}

float WarpHandler::xAtFrame(double frame) const
{
  return static_cast<float>(framesToSecondsD(frame) * owner_.pixelsPerSecond);
}

bool WarpHandler::inStrip(float worldY) const
{
  const auto visible = owner_.getVisibleWorldArea();
  return worldY >= visible.getY() && worldY < visible.getY() + kStripHeight;
}

int WarpHandler::anchorAt(float worldX) const
{
  if (!owner_.project)
    return -1;
  const auto &anchors = owner_.project->getAudioData().warpMap.anchors();
  int best = -1;
  float bestDistance = kAnchorHitPixels;
  for (size_t i = 0; i < anchors.size(); ++i)
  {
    const float distance = std::abs(worldX - xAtFrame(anchors[i].destination));
    if (distance <= bestDistance)
    {
      best = static_cast<int>(i);
      bestDistance = distance;
    }
  }
  return best;
}

int WarpHandler::segmentAt(float worldX) const
{
  if (!owner_.project)
    return -1;
  const auto &map = owner_.project->getAudioData().warpMap;
  if (!(map.length() > 0.0))
    return -1;
  const double frame = frameAtX(worldX);
  const auto points = map.points();
  for (size_t i = 0; i + 1 < points.size(); ++i)
    if (frame >= points[i].destination && frame < points[i + 1].destination)
      return static_cast<int>(i);
  return -1;
}

/**
 * Which hard limit stops anchor `index` short of `requested`, and on which
 * side of it (WarpMap::destinationRange, taken apart).
 */
WarpHandler::Limit WarpHandler::limitFor(const warp::WarpMap &map, int index,
                                         double requested, bool &leftSide)
{
  const auto [low, high] = map.destinationRange(index);
  if (requested >= low - 1.0e-6 && requested <= high + 1.0e-6)
    return Limit::None;

  const auto points = map.points();
  const auto &left = points[static_cast<size_t>(index)];
  const auto &self = points[static_cast<size_t>(index) + 1];
  const auto &right = points[static_cast<size_t>(index) + 2];
  const double leftSource = self.source - left.source;
  const double rightSource = right.source - self.source;
  using namespace warp;

  if (requested < low)
  {
    // Moving left: the left segment shrinks, the right one grows.
    const double leftShortest =
        left.destination + std::max(kMinSegmentFrames, kMinRatio * leftSource);
    const double rightLongest = right.destination - kMaxRatio * rightSource;
    leftSide = leftShortest >= rightLongest;
    if (leftSide)
      return kMinSegmentFrames > kMinRatio * leftSource ? Limit::MinLength
                                                         : Limit::MinRatio;
    return Limit::MaxRatio;
  }

  // Moving right: the left segment grows, the right one shrinks.
  const double leftLongest = left.destination + kMaxRatio * leftSource;
  const double rightShortest =
      right.destination - std::max(kMinSegmentFrames, kMinRatio * rightSource);
  leftSide = leftLongest <= rightShortest;
  if (leftSide)
    return Limit::MaxRatio;
  return kMinSegmentFrames > kMinRatio * rightSource ? Limit::MinLength
                                                      : Limit::MinRatio;
}

void WarpHandler::showLimit(Limit limit, bool leftSide, double frame)
{
  juce::String reason;
  switch (limit)
  {
  case Limit::MinLength:
    reason = TR("warp.limit.min_length");
    break;
  case Limit::MinRatio:
    reason = TR("warp.limit.min_ratio");
    break;
  case Limit::MaxRatio:
    reason = TR("warp.limit.max_ratio");
    break;
  case Limit::TooClose:
    limitText = TR("warp.limit.too_close");
    limitFrame = frame;
    return;
  case Limit::Fixed:
    limitText = TR("warp.limit.fixed");
    limitFrame = frame;
    return;
  case Limit::None:
    clearLimit();
    return;
  }
  limitText = TR(leftSide ? "warp.limit.left" : "warp.limit.right")
                  .replace("{reason}", reason);
  limitFrame = frame;
}

void WarpHandler::clearLimit() { limitText.clear(); }

void WarpHandler::preview(const warp::WarpMap &map)
{
  auto &audioData = owner_.project->getAudioData();
  warp::rememberCompositeMap(audioData);
  audioData.warpMap = map;
  warp::syncNotes(*owner_.project);
  PitchCurveProcessor::rebuildBaseFromNotes(*owner_.project);
  owner_.invalidateBasePitchCache();
  owner_.repaint();
}

void WarpHandler::commit(const warp::WarpMap &before,
                         const warp::WarpMap &after, const juce::String &name)
{
  auto &audioData = owner_.project->getAudioData();
  if (after == before)
  {
    audioData.warpMap = before;
    owner_.repaint();
    return;
  }

  // apply() compares against the map in the project to decide whether the
  // sound has to be laid out again, so start from the map before the edit.
  audioData.warpMap = before;
  warp::apply(*owner_.project, after);
  owner_.invalidateBasePitchCache();
  if (owner_.undoManager)
    owner_.undoManager->addAction(
        std::make_unique<WarpAction>(*owner_.project, before, after, name));
  if (!before.sameMappingAs(after) && owner_.onPitchEditFinished)
    owner_.onPitchEditFinished();
  owner_.repaint();
}

bool WarpHandler::mouseDown(const juce::MouseEvent &e, float worldX,
                            float worldY)
{
  closeEditor(false);
  if (!owner_.isCanvasPoint(e) || !owner_.project)
    return false;
  clearLimit();
  // The ratio strip edits by double-click; a click there places nothing.
  if (inStrip(worldY))
  {
    owner_.repaint();
    return true;
  }

  auto &audioData = owner_.project->getAudioData();
  warp::fitToProject(audioData);
  auto map = audioData.warpMap;
  if (!(map.length() > 0.0))
    return true;

  int index = anchorAt(worldX);
  if (e.mods.isAltDown())
  {
    if (index >= 0)
    {
      auto after = map;
      after.removeAnchor(index);
      hoveredAnchor = -1;
      commit(map, after, "Delete Warp Anchor");
    }
    return true;
  }

  // A note edge (on its row) drags as a boundary: spec sections 4 and 5.
  if (index < 0 && boundaries->hitsBoundary(worldX, worldY) &&
      boundaries->mouseDown(e, worldX, worldY))
  {
    boundaryDragging = true;
    return true;
  }
  if (!e.mods.isShiftDown())
    boundaries->cancel(); // a click elsewhere drops the boundary selection

  beforeMap = map;
  if (index < 0)
  {
    const double frame = owner_.snapWarpFrame(frameAtX(worldX));
    index = map.addAnchorAtDestination(frame);
    if (index < 0)
    {
      showLimit(Limit::TooClose, false, frame);
      owner_.repaint();
      return true;
    }
    // Pins the recording where it already plays: nothing moves yet.
    audioData.warpMap = map;
  }

  dragBase = map;
  dragIndex = index;
  hoveredAnchor = index;
  downX = worldX;
  dragging = true;
  moved = false;
  owner_.repaint();
  return true;
}

bool WarpHandler::mouseDrag(const juce::MouseEvent &e, float worldX,
                            float worldY)
{
  if (boundaryDragging)
    return boundaries->mouseDrag(e, worldX, worldY);
  if (!dragging || !owner_.project)
    return false;
  if (!moved && std::abs(worldX - downX) < kDragThresholdPixels)
    return true;
  moved = true;

  const double requested = owner_.snapWarpFrame(frameAtX(worldX));
  auto map = dragBase;
  bool leftSide = false;
  const Limit limit = limitFor(map, dragIndex, requested, leftSide);
  const double placed = map.moveAnchor(dragIndex, requested);
  if (limit == Limit::None)
    clearLimit();
  else
    showLimit(limit, leftSide, placed);
  preview(map);
  if (owner_.onPitchEdited)
    owner_.onPitchEdited();
  return true;
}

bool WarpHandler::mouseUp(const juce::MouseEvent &e, float worldX,
                          float worldY)
{
  if (boundaryDragging)
  {
    boundaryDragging = false;
    boundaries->mouseUp(e, worldX, worldY);
    mouseMove(e, worldX, worldY);
    return true;
  }
  if (!dragging || !owner_.project)
    return false;
  dragging = false;
  const auto after = owner_.project->getAudioData().warpMap;
  const bool added = after.anchors().size() > beforeMap.anchors().size();
  commit(beforeMap, after,
         moved ? (added ? "Add and Move Warp Anchor" : "Move Warp Anchor")
               : "Add Warp Anchor");
  dragIndex = -1;
  mouseMove(e, worldX, worldY);
  return true;
}

void WarpHandler::mouseMove(const juce::MouseEvent &e, float worldX,
                            float worldY)
{
  lastWorldX = worldX;
  lastWorldY = worldY;
  pointerInCanvas = owner_.isCanvasPoint(e);
  const int next = pointerInCanvas && !inStrip(worldY) ? anchorAt(worldX) : -1;
  if (next != hoveredAnchor)
  {
    hoveredAnchor = next;
    owner_.repaint();
  }
  // An anchor wins over a note edge under it.
  hoveredBoundary = next < 0 && pointerInCanvas && !inStrip(worldY) &&
                    boundaries->hitsBoundary(worldX, worldY);
  boundaries->mouseMove(e, next < 0 ? worldX : -1.0e9f, worldY);
  updateCursor(e.mods);
}

void WarpHandler::modifiersChanged(const juce::ModifierKeys &mods)
{
  if (dragging || boundaryDragging)
    return;
  updateCursor(mods);
  owner_.repaint();
}

void WarpHandler::updateCursor(const juce::ModifierKeys &mods)
{
  if (!pointerInCanvas || !owner_.project)
    owner_.setMouseCursor(juce::MouseCursor::NormalCursor);
  else if (inStrip(lastWorldY))
    owner_.setMouseCursor(segmentAt(lastWorldX) >= 0
                              ? juce::MouseCursor::PointingHandCursor
                              : juce::MouseCursor::NormalCursor);
  else if (hoveredAnchor >= 0)
    owner_.setMouseCursor(mods.isAltDown()
                              ? deleteCursor()
                              : juce::MouseCursor(
                                    juce::MouseCursor::LeftRightResizeCursor));
  else if (hoveredBoundary && !mods.isAltDown())
    owner_.setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
  else
    // Alt alone changes nothing here; only over an anchor it is the eraser.
    owner_.setMouseCursor(placeCursor());
}

bool WarpHandler::handlesDoubleClickAt(float worldX, float worldY) const
{
  return inStrip(worldY) && segmentAt(worldX) >= 0;
}

void WarpHandler::mouseDoubleClick(const juce::MouseEvent &, float worldX,
                                   float worldY)
{
  if (handlesDoubleClickAt(worldX, worldY))
    openEditor(segmentAt(worldX));
}

// ---------------------------------------------------------------------------
// Typing a segment's length or ratio (spec section 3, "輸入目標長度").

void WarpHandler::openEditor(int segment)
{
  closeEditor(false);
  if (!owner_.project || segment < 0)
    return;
  const auto points = owner_.project->getAudioData().warpMap.points();
  if (segment + 1 >= static_cast<int>(points.size()))
    return;
  const auto &left = points[static_cast<size_t>(segment)];
  const auto &right = points[static_cast<size_t>(segment) + 1];

  editorSegment = segment;
  editor = std::make_unique<juce::TextEditor>();
  editor->setFont(AppFont::getFont(13.0f));
  editor->setJustification(juce::Justification::centred);
  editor->setColour(juce::TextEditor::backgroundColourId,
                    juce::Colour(0xFF101010u));
  editor->setColour(juce::TextEditor::textColourId, APP_COLOR_TEXT_PRIMARY);
  editor->setColour(juce::TextEditor::outlineColourId, kAnchorColour);
  editor->setColour(juce::TextEditor::focusedOutlineColourId, kAnchorColour);
  editor->setText(formatRatio(warp::WarpMap::segmentRatio(left, right)), false);
  editor->setTooltip(TR("warp.input.tooltip"));

  const auto visible = owner_.getVisibleWorldArea();
  const float x0 = std::max(xAtFrame(left.destination), visible.getX());
  const float x1 = std::min(xAtFrame(right.destination), visible.getRight());
  const float centre = (x0 + x1) * 0.5f;
  constexpr int width = 96;
  const auto origin =
      owner_.worldToComponent(centre - width * 0.5f, visible.getY());
  editor->setBounds(origin.x, origin.y, width, static_cast<int>(kStripHeight) + 4);

  auto *raw = editor.get();
  editor->onReturnKey = [this] { closeEditor(true); };
  editor->onEscapeKey = [this] { closeEditor(false); };
  editor->onFocusLost = [this] { closeEditor(false); };
  owner_.getComponent().addAndMakeVisible(raw);
  raw->grabKeyboardFocus();
  raw->selectAll();
}

void WarpHandler::closeEditor(bool apply)
{
  if (!editor)
    return;
  // Detach first: removing the editor fires focus-lost, which comes back here.
  auto closing = std::move(editor);
  closing->onFocusLost = nullptr;
  closing->onReturnKey = nullptr;
  closing->onEscapeKey = nullptr;
  const auto text = closing->getText();
  const int segment = editorSegment;
  editorSegment = -1;
  if (auto *parent = closing->getParentComponent())
    parent->removeChildComponent(closing.get());
  // Deleting the editor inside its own key callback is not safe.
  juce::MessageManager::callAsync([doomed = closing.release()] { delete doomed; });
  if (apply)
    applyEditorText(segment, text);
}

/**
 * "1.57 s" or "1.57s" sets the segment's length on the timeline; "112.35%"
 * or a bare number sets its ratio. The anchor at the segment's right end
 * moves, so the next segment absorbs the change - or, when the right end is
 * the end of the track, the anchor at its left end and the previous segment.
 */
void WarpHandler::applyEditorText(int segment, const juce::String &rawText)
{
  if (!owner_.project || segment < 0)
    return;
  auto &audioData = owner_.project->getAudioData();
  const auto before = audioData.warpMap;
  const auto points = before.points();
  if (segment + 1 >= static_cast<int>(points.size()))
    return;
  const auto &left = points[static_cast<size_t>(segment)];
  const auto &right = points[static_cast<size_t>(segment) + 1];
  const double sourceLength = right.source - left.source;

  const auto text = rawText.trim().toLowerCase();
  if (text.isEmpty())
    return;
  const double value = text.retainCharacters("0123456789.").getDoubleValue();
  if (!(value > 0.0))
    return;
  double targetLength = 0.0;
  if (text.endsWithChar('s'))
    targetLength = value * SAMPLE_RATE / static_cast<double>(HOP_SIZE);
  else
    targetLength = sourceLength * value / 100.0;

  // points[i] is anchors()[i - 1]; the two ends are not anchors.
  const int anchorCount = static_cast<int>(before.anchors().size());
  const int rightAnchor = segment;          // anchors() index of `right`
  const int leftAnchor = segment - 1;       // anchors() index of `left`
  auto after = before;
  int moving = -1;
  double requested = 0.0;
  if (rightAnchor < anchorCount)
  {
    moving = rightAnchor;
    requested = left.destination + targetLength;
  }
  else if (leftAnchor >= 0)
  {
    moving = leftAnchor;
    requested = right.destination - targetLength;
  }
  else
  {
    showLimit(Limit::Fixed, false, (left.destination + right.destination) * 0.5);
    owner_.repaint();
    return;
  }

  bool leftSide = false;
  const Limit limit = limitFor(after, moving, requested, leftSide);
  const double placed = after.moveAnchor(moving, requested);
  if (limit != Limit::None)
    showLimit(limit, leftSide, placed);
  commit(before, after, "Set Warp Segment");
}

// ---------------------------------------------------------------------------

void WarpHandler::drawLayer(juce::Graphics &g,
                            PianoRollInteractionContext &context,
                            int hoveredAnchor, bool deleteHover,
                            int activeAnchor, bool showStrip)
{
  if (!context.project)
    return;
  const auto &map = context.project->getAudioData().warpMap;
  if (!(map.length() > 0.0))
    return;

  const auto visible = context.getVisibleWorldArea();
  const float canvasHeight =
      (MAX_MIDI_NOTE - MIN_MIDI_NOTE + 1) * context.pixelsPerSemitone;
  const float pps = context.pixelsPerSecond;
  auto xAt = [pps](double frame)
  { return static_cast<float>(framesToSecondsD(frame) * pps); };
  const auto points = map.points();

  // Stretched segments get a faint tint: longer and shorter differ in hue.
  for (size_t i = 0; i + 1 < points.size(); ++i)
  {
    const double ratio = warp::WarpMap::segmentRatio(points[i], points[i + 1]);
    if (std::abs(ratio - 1.0) < 1.0e-4)
      continue;
    const float x0 = xAt(points[i].destination);
    const float x1 = xAt(points[i + 1].destination);
    if (x1 < visible.getX() || x0 > visible.getRight())
      continue;
    g.setColour((ratio > 1.0 ? kLongerTint : kShorterTint).withAlpha(0.07f));
    g.fillRect(x0, 0.0f, x1 - x0, canvasHeight);
  }

  const auto &anchors = map.anchors();
  const float stripBottom = visible.getY() + (showStrip ? kStripHeight : 0.0f);
  for (size_t i = 0; i < anchors.size(); ++i)
  {
    const float x = xAt(anchors[i].destination);
    if (x < visible.getX() - 8.0f || x > visible.getRight() + 8.0f)
      continue;
    const bool hot = static_cast<int>(i) == hoveredAnchor ||
                     static_cast<int>(i) == activeAnchor;
    const auto colour = hot && deleteHover ? APP_COLOR_ALERT_ERROR
                        : hot ? kAnchorColour.brighter(0.35f)
                              : kAnchorColour.withAlpha(showStrip ? 0.85f : 0.45f);
    g.setColour(colour);
    const float width = hot ? 2.0f : 1.0f;
    g.fillRect(x - width * 0.5f, 0.0f, width, canvasHeight);
    juce::Path tab;
    tab.addTriangle(x - 5.0f, stripBottom, x + 5.0f, stripBottom, x,
                    stripBottom + 7.0f);
    g.fillPath(tab);
  }

  if (!showStrip)
    return;

  // The strip: each segment's ratio, two decimals, the warning colour
  // outside 0.5 .. 2.
  const juce::Rectangle<float> strip(visible.getX(), visible.getY(),
                                     visible.getWidth(), kStripHeight);
  g.setColour(juce::Colour(0xE8161616u));
  g.fillRect(strip);
  g.setColour(juce::Colours::white.withAlpha(0.10f));
  g.fillRect(strip.getX(), strip.getBottom() - 1.0f, strip.getWidth(), 1.0f);
  const auto font = AppFont::getFont(11.5f);
  g.setFont(font);
  for (size_t i = 0; i + 1 < points.size(); ++i)
  {
    const float x0 = std::max(xAt(points[i].destination), strip.getX());
    const float x1 = std::min(xAt(points[i + 1].destination), strip.getRight());
    if (x1 - x0 < 1.0f)
      continue;
    const double ratio = warp::WarpMap::segmentRatio(points[i], points[i + 1]);
    const auto text = formatRatio(ratio);
    if (juce::GlyphArrangement::getStringWidth(font, text) + 8.0f > x1 - x0)
      continue;
    const bool warning = ratio < warp::kCleanMinRatio - 1.0e-9 ||
                         ratio > warp::kCleanMaxRatio + 1.0e-9;
    g.setColour(warning ? APP_COLOR_ALERT_WARNING
                : std::abs(ratio - 1.0) < 1.0e-4 ? APP_COLOR_TEXT_MUTED
                                                 : APP_COLOR_TEXT_PRIMARY);
    g.drawText(text, juce::Rectangle<float>(x0, strip.getY(), x1 - x0,
                                            strip.getHeight()),
               juce::Justification::centred, false);
  }

  // The hovered or dragged anchor's place on the timeline.
  const int tagged = activeAnchor >= 0 ? activeAnchor : hoveredAnchor;
  if (tagged >= 0 && tagged < static_cast<int>(anchors.size()) && !deleteHover)
  {
    const double frame = anchors[static_cast<size_t>(tagged)].destination;
    const auto text = formatSeconds(
        context.projectToTimeline(framesToSecondsD(frame)));
    const float width = juce::GlyphArrangement::getStringWidth(font, text) + 10.0f;
    const juce::Rectangle<float> tag(xAt(frame) + 6.0f, stripBottom + 2.0f,
                                     width, 16.0f);
    g.setColour(juce::Colour(0xE0101010u));
    g.fillRoundedRectangle(tag, 3.0f);
    g.setColour(APP_COLOR_TEXT_PRIMARY);
    g.drawText(text, tag, juce::Justification::centred, false);
  }
}

void WarpHandler::draw(juce::Graphics &g)
{
  const bool deleteHover =
      hoveredAnchor >= 0 && !dragging &&
      juce::ModifierKeys::currentModifiers.isAltDown();
  drawLayer(g, owner_, hoveredAnchor, deleteHover, dragging ? dragIndex : -1,
            true);
  boundaries->draw(g);

  if (limitText.isNotEmpty())
  {
    const auto visible = owner_.getVisibleWorldArea();
    const auto font = AppFont::getFont(12.0f);
    g.setFont(font);
    const float width =
        juce::GlyphArrangement::getStringWidth(font, limitText) + 16.0f;
    float x = xAtFrame(limitFrame) - width * 0.5f;
    x = juce::jlimit(visible.getX() + 4.0f,
                     std::max(visible.getX() + 4.0f,
                              visible.getRight() - width - 4.0f),
                     x);
    const juce::Rectangle<float> box(x, visible.getY() + kStripHeight + 22.0f,
                                     width, 20.0f);
    g.setColour(juce::Colour(0xF0161616u));
    g.fillRoundedRectangle(box, 4.0f);
    g.setColour(APP_COLOR_ALERT_ERROR);
    g.drawRoundedRectangle(box, 4.0f, 1.0f);
    g.setColour(APP_COLOR_TEXT_PRIMARY);
    g.drawText(limitText, box, juce::Justification::centred, false);
  }
}

void WarpHandler::cancel()
{
  closeEditor(false);
  boundaries->cancel();
  boundaryDragging = false;
  hoveredBoundary = false;
  if (dragging && owner_.project)
    preview(beforeMap);
  dragging = false;
  moved = false;
  dragIndex = -1;
  hoveredAnchor = -1;
  clearLimit();
  owner_.repaint();
}
