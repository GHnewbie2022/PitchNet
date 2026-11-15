#include "TimingHandler.h"

#include "../../PianoRollComponent.h"
#include "../../../Utils/Constants.h"
#include "../../../Audio/WarpLayer.h"
#include "../../../Undo/WarpAction.h"
#include "../../../Utils/PitchCurveProcessor.h"
#include "../../../Utils/UI/Theme.h"

#include <algorithm>
#include <cmath>
#include <limits>

TimingHandler::TimingHandler(PianoRollComponent& owner)
    : InteractionHandler(owner) {}

std::vector<TimingHandler::Boundary> TimingHandler::buildBoundaries() const
{
  std::vector<Note*> notes;
  if (!owner_.project)
    return {};

  for (auto& note : owner_.project->getNotes())
    if (!note.isRest())
      notes.push_back(&note);
  std::sort(notes.begin(), notes.end(), [](const Note* a, const Note* b)
  {
    return a->getStartFrame() < b->getStartFrame();
  });

  std::vector<Boundary> result;
  auto addEdge = [&](float frame, Note* left, Note* right)
  {
    auto existing = std::find_if(result.begin(), result.end(),
                                 [frame](const Boundary& b)
                                 { return std::abs(b.frame - frame) < 0.001f; });
    if (existing == result.end())
    {
      result.push_back({frame, left, right, nullptr, 0.0f, 0.0f});
      return;
    }
    if (left)
      existing->left = left;
    if (right)
      existing->right = right;
  };

  for (auto* note : notes)
  {
    addEdge(note->getVisualStartFrame(), nullptr, note);
    addEdge(note->getVisualEndFrame(), note, nullptr);
  }

  for (auto& boundary : result)
  {
    std::vector<Note*> drawnNotes;
    if (boundary.left)
      drawnNotes.push_back(boundary.left);
    if (boundary.right)
      drawnNotes.push_back(boundary.right);

    if (!boundary.left && boundary.right)
    {
      for (auto* note : notes)
        if (note->getVisualEndFrame() < boundary.frame &&
            (!boundary.restCompanion ||
             note->getVisualEndFrame() >
                 boundary.restCompanion->getVisualEndFrame()))
          boundary.restCompanion = note;
    }
    else if (boundary.left && !boundary.right)
    {
      for (auto* note : notes)
        if (note->getVisualStartFrame() > boundary.frame &&
            (!boundary.restCompanion ||
             note->getVisualStartFrame() <
                 boundary.restCompanion->getVisualStartFrame()))
          boundary.restCompanion = note;
    }

    float top = std::numeric_limits<float>::max();
    float bottom = std::numeric_limits<float>::lowest();
    for (auto* note : drawnNotes)
    {
      const float y = owner_.midiToY(note->getAdjustedMidiNote());
      top = std::min(top, y);
      bottom = std::max(bottom, y + owner_.pixelsPerSemitone);
    }
    boundary.top = top;
    boundary.bottom = bottom;
  }

  std::sort(result.begin(), result.end(), [](const Boundary& a,
                                             const Boundary& b)
  { return a.frame < b.frame; });
  return result;
}

// Boundaries are drawn on their notes' rows only (full-height lines belong to
// the warp anchors), so they are grabbed there too, with a little slack.
constexpr float kBoundaryRowSlack = 6.0f;

int TimingHandler::findBoundary(float worldX, float worldY,
                                const std::vector<Boundary>& boundaries) const
{
  constexpr float hitPadding = 6.0f;
  int best = -1;
  float bestDistance = hitPadding + 1.0f;
  for (size_t i = 0; i < boundaries.size(); ++i)
  {
    const auto& boundary = boundaries[i];
    if (worldY < boundary.top - kBoundaryRowSlack ||
        worldY > boundary.bottom + kBoundaryRowSlack)
      continue;
    const float x = static_cast<float>(framesToSeconds(boundary.frame) *
                                       owner_.pixelsPerSecond);
    const float distance = std::abs(worldX - x);
    if (distance <= hitPadding && distance < bestDistance)
    {
      best = static_cast<int>(i);
      bestDistance = distance;
    }
  }
  return best;
}

bool TimingHandler::hitsBoundary(float worldX, float worldY) const
{
  return owner_.project && findBoundary(worldX, worldY, buildBoundaries()) >= 0;
}

/**
 * Spec sections 4 and 5. Dragging a note boundary is a shortcut for the warp
 * layer: the boundary gets an anchor (or the anchor already there is used),
 * and each side gets a fixed point -
 *
 *   left  the start of the note left of the boundary, or across a gap the
 *         end of the previous note (the start of the track if there is none)
 *   right the end of the note right of the boundary, or across a gap the
 *         start of the next note (the end of the track)
 *
 * unless an anchor already lies between the boundary and that point: then
 * the nearest such anchor is the fixed point and nothing is added. Only the
 * two segments next to the dragged anchor stretch; everything else stays.
 */
void TimingHandler::prepareDragAnchors(const std::vector<Boundary>& boundaries)
{
  auto& audioData = owner_.project->getAudioData();
  warp::fitToProject(audioData);
  beforeMap = audioData.warpMap;
  dragMap = beforeMap;
  dragAnchors.clear();
  const double length = dragMap.length();

  auto anchorBetween = [&](double from, double to)
  {
    const double low = std::min(from, to);
    const double high = std::max(from, to);
    for (const auto& anchor : dragMap.anchors())
      if (anchor.destination > low + 1.0e-6 &&
          anchor.destination < high - 1.0e-6)
        return true;
    return false;
  };

  for (const auto& boundary : boundaries)
  {
    const double point = boundary.frame;
    double leftTarget = 0.0;
    if (boundary.left)
      leftTarget = boundary.left->getStartFrame();
    else if (boundary.restCompanion)
      leftTarget = boundary.restCompanion->getEndFrame();
    double rightTarget = length;
    if (boundary.right)
      rightTarget = boundary.right->getEndFrame();
    else if (boundary.restCompanion)
      rightTarget = boundary.restCompanion->getStartFrame();

    if (!anchorBetween(leftTarget, point))
      dragMap.addAnchorAtDestination(leftTarget);
    if (!anchorBetween(point, rightTarget))
      dragMap.addAnchorAtDestination(rightTarget);

    const int index = dragMap.addAnchorAtDestination(point);
    if (index < 0)
      continue;
    const auto& anchor = dragMap.anchors()[static_cast<size_t>(index)];
    dragAnchors.push_back({anchor.source, anchor.destination});
  }
}

void TimingHandler::showMap(const warp::WarpMap& map)
{
  auto& audioData = owner_.project->getAudioData();
  warp::rememberCompositeMap(audioData);
  audioData.warpMap = map;
  warp::syncNotes(*owner_.project);
  PitchCurveProcessor::rebuildBaseFromNotes(*owner_.project);
  owner_.invalidateBasePitchCache();
}

void TimingHandler::applyPreviewDelta(float delta)
{
  previewDelta = delta;
  auto map = dragMap;
  for (const auto& drag : dragAnchors)
    map.moveAnchor(map.findAnchorAtSource(drag.source),
                   drag.startDestination + delta);
  showMap(map);
  owner_.repaint();
}

TimingHandler::BoundaryKey TimingHandler::keyFor(const Boundary& boundary)
{
  return {boundary.left, boundary.right};
}

bool TimingHandler::keysMatch(const BoundaryKey& a, const BoundaryKey& b)
{
  return a.left == b.left && a.right == b.right;
}

bool TimingHandler::isSelected(const Boundary& boundary) const
{
  const auto key = keyFor(boundary);
  return std::any_of(selectedBoundaries.begin(), selectedBoundaries.end(),
                     [&](const BoundaryKey& selected)
                     { return keysMatch(selected, key); });
}

void TimingHandler::addSelected(const Boundary& boundary)
{
  if (!isSelected(boundary))
    selectedBoundaries.push_back(keyFor(boundary));
}

void TimingHandler::removeSelected(const Boundary& boundary)
{
  const auto key = keyFor(boundary);
  selectedBoundaries.erase(
      std::remove_if(selectedBoundaries.begin(), selectedBoundaries.end(),
                     [&](const BoundaryKey& selected)
                     { return keysMatch(selected, key); }),
      selectedBoundaries.end());
}

std::vector<TimingHandler::Boundary> TimingHandler::resolveSelection(
    const std::vector<Boundary>& boundaries) const
{
  std::vector<Boundary> result;
  for (const auto& boundary : boundaries)
    if (isSelected(boundary))
      result.push_back(boundary);
  return result;
}

bool TimingHandler::mouseDown(const juce::MouseEvent& e, float worldX,
                              float worldY)
{
  if (!owner_.isCanvasPoint(e))
    return false;

  const auto boundaries = buildBoundaries();
  const int index = findBoundary(worldX, worldY, boundaries);
  // Away from a boundary the Warp tool places an anchor instead.
  if (index < 0)
    return false;

  activeBoundary = boundaries[static_cast<size_t>(index)];
  if (e.mods.isShiftDown() && isSelected(activeBoundary))
  {
    removeSelected(activeBoundary);
    owner_.repaint();
    return true;
  }

  if (!e.mods.isShiftDown() && !isSelected(activeBoundary))
    selectedBoundaries.clear();
  addSelected(activeBoundary);
  dragBoundaries = resolveSelection(boundaries);
  prepareDragAnchors(dragBoundaries);
  previewDelta = 0.0f;
  dragging = true;
  owner_.repaint();
  return true;
}

bool TimingHandler::mouseDrag(const juce::MouseEvent&, float worldX, float)
{
  if (!dragging || !owner_.project)
    return false;

  const float seconds = worldX / std::max(1.0f, owner_.pixelsPerSecond);
  const float frame = static_cast<float>(owner_.snapWarpFrame(
      seconds * SAMPLE_RATE / static_cast<double>(HOP_SIZE)));
  applyPreviewDelta(frame - activeBoundary.frame);
  if (owner_.onPitchEdited)
    owner_.onPitchEdited();
  return true;
}

bool TimingHandler::mouseUp(const juce::MouseEvent&, float, float)
{
  if (!dragging || !owner_.project)
    return false;

  dragging = false;
  auto& audioData = owner_.project->getAudioData();
  const auto after = audioData.warpMap;
  const bool changed = std::any_of(
      dragAnchors.begin(), dragAnchors.end(), [&](const DragAnchor& drag)
      {
        const int index = after.findAnchorAtSource(drag.source);
        return index >= 0 &&
               std::abs(after.anchors()[static_cast<size_t>(index)].destination -
                        drag.startDestination) > 1.0e-6;
      });

  if (changed)
  {
    // The drag previewed the map in place; apply() compares against the map
    // before the drag to decide whether the sound has to be laid out again.
    audioData.warpMap = beforeMap;
    warp::apply(*owner_.project, after);
    owner_.invalidateBasePitchCache();
    if (owner_.undoManager)
      owner_.undoManager->addAction(
          std::make_unique<WarpAction>(*owner_.project, beforeMap, after));
  }
  else
  {
    // Nothing moved: drop the anchors this drag added.
    showMap(beforeMap);
  }
  if (changed && owner_.onPitchEditFinished)
    owner_.onPitchEditFinished();
  dragAnchors.clear();
  dragBoundaries.clear();
  owner_.repaint();
  return true;
}

void TimingHandler::mouseMove(const juce::MouseEvent& e, float worldX,
                              float worldY)
{
  // Hover only; the Warp tool decides the cursor.
  float nextHover = -1.0f;
  if (owner_.isCanvasPoint(e))
  {
    const auto boundaries = buildBoundaries();
    const int index = findBoundary(worldX, worldY, boundaries);
    if (index >= 0)
      nextHover = boundaries[static_cast<size_t>(index)].frame;
  }
  if (std::abs(nextHover - hoveredBoundaryFrame) > 0.001f)
  {
    hoveredBoundaryFrame = nextHover;
    owner_.repaint();
  }
}

void TimingHandler::draw(juce::Graphics& g)
{
  // Only the boundary under the pointer and the selected ones: the anchors
  // are the warp layer's full-height lines, and every note edge drawn at
  // once would bury them.
  const float canvasHeight =
      (MAX_MIDI_NOTE - MIN_MIDI_NOTE + 1) * owner_.pixelsPerSemitone;
  for (const auto& boundary : buildBoundaries())
  {
    const bool selected = isSelected(boundary);
    const bool hovered = std::abs(boundary.frame -
                                  hoveredBoundaryFrame) < 0.001f;
    if (!hovered && !selected)
      continue;
    const float x = static_cast<float>(framesToSeconds(boundary.frame) *
                                       owner_.pixelsPerSecond);
    g.setColour(juce::Colour(0xFFFFFFFFu));
    // On the notes' rows only.
    const float top = std::max(0.0f, boundary.top - 2.0f);
    const float bottom = std::min(canvasHeight, boundary.bottom + 2.0f);
    if (bottom > top)
      g.fillRect(x - 1.0f, top, 2.0f, bottom - top);
  }
}

void TimingHandler::cancel()
{
  if (dragging && owner_.project)
    showMap(beforeMap);
  dragging = false;
  dragAnchors.clear();
  dragBoundaries.clear();
  selectedBoundaries.clear();
  hoveredBoundaryFrame = -1.0f;
  owner_.repaint();
}
