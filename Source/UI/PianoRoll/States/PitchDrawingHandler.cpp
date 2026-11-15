#include "PitchDrawingHandler.h"
#include "WarpHandler.h"
#include "../../../Undo/PitchDrawingAction.h"
#include "../../../Utils/Constants.h"
#include "../../../Utils/PitchCurveProcessor.h"
#include "../../../Utils/TransformParams.h"
#include "../../../Utils/UI/Theme.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
bool sameCurves(const std::vector<DrawnCurve>& a, const std::vector<DrawnCurve>& b)
{
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i)
  {
    if (a[i].id != b[i].id || a[i].anchorFrames != b[i].anchorFrames ||
        a[i].underStart != b[i].underStart ||
        a[i].underDiff.size() != b[i].underDiff.size())
      return false;
    for (size_t k = 0; k < a[i].underDiff.size(); ++k)
      if (std::abs(a[i].underDiff[k] - b[i].underDiff[k]) > 0.0005f)
        return false;
  }
  return true;
}

void neutralTools(Note& note)
{
  note.setTiltLeft(0.0f);
  note.setTiltRight(0.0f);
  note.setVibrato(1.0f);
  note.setPitchDrift(1.0f);
  note.setSmoothLeftFrames(0);
  note.setSmoothRightFrames(0);
  note.setDeltaScale(1.0f);
  note.setDeltaOffset(0.0f);
  note.setFrozenTilt({});
}
} // namespace

PitchDrawingHandler::PitchDrawingHandler(PianoRollComponent& owner)
    : InteractionHandler(owner)
{
}

// ---------------------------------------------------------------- mouse

bool PitchDrawingHandler::mouseDown(const juce::MouseEvent& e, float worldX,
                                    float worldY)
{
  if (e.mods.isPopupMenu())
    return removeAnchorAt(worldX, worldY);

  if (!e.mods.isLeftButtonDown() || !owner_.project)
    return false;

  auto hit = hitTest(worldX, worldY);
  if (hit.curveId >= 0)
  {
    if (hit.curveId != activeCurveId)
    {
      finish();
      const auto& curves = owner_.project->getDrawnCurves();
      const auto stored = std::find_if(curves.begin(), curves.end(),
                                       [&hit](const DrawnCurve& c)
                                       { return c.id == hit.curveId; });
      if (stored == curves.end())
        return true;
      const DrawnCurve curve = *stored;
      beginSession(&curve);
      hit = hitTest(worldX, worldY);
      if (hit.anchorId < 0)
        return true;
    }
    if (e.mods.isAltDown())
    {
      removeAnchor(hit.anchorId);
      return true;
    }
    activeAnchorId = hit.anchorId;
    hovered = hit;
    dragging = true;
    dragMoved = false;
    owner_.repaint();
    return true;
  }

  // Alt is the eraser: it only acts on anchors.
  if (e.mods.isAltDown())
    return true;

  if (!isPitchRegionAtWorldX(worldX))
  {
    // A click where no note is ends the curve, so the next one starts anew.
    if (activeCurveId >= 0)
      finish();
    return false;
  }
  const int frame = frameFromWorldX(worldX);

  if (activeCurveId < 0)
  {
    // Inside a drawn curve the click edits it; elsewhere it starts one.
    const auto& curves = owner_.project->getDrawnCurves();
    const auto inside = std::find_if(curves.begin(), curves.end(),
                                     [frame](const DrawnCurve& c)
                                     { return c.spans(frame); });
    if (inside != curves.end())
    {
      const DrawnCurve curve = *inside;
      beginSession(&curve);
    }
    else
      beginSession(nullptr);
  }

  const float midi = midiFromWorldY(worldY);
  auto sameFrame = std::find_if(anchors.begin(), anchors.end(),
                                [frame](const Anchor& anchor)
                                { return anchor.frame == frame; });
  if (sameFrame != anchors.end())
  {
    sameFrame->midi = midi;
    activeAnchorId = sameFrame->id;
  }
  else
  {
    anchors.push_back({nextAnchorId++, frame, midi});
    activeAnchorId = anchors.back().id;
  }

  hovered = {activeCurveId, activeAnchorId, frame};
  dragging = true;
  dragMoved = false;
  sortAnchors();
  extendRange(frame, frame);
  rebuildPreview();
  applyPreviewToProject();
  notifyStateChanged();
  return true;
}

bool PitchDrawingHandler::mouseDrag(const juce::MouseEvent&, float worldX,
                                    float worldY)
{
  if (!dragging)
    return false;

  auto* anchor = findAnchorById(activeAnchorId);
  if (!anchor || !owner_.project)
    return false;

  int newFrame = frameFromWorldX(worldX);
  const auto current = std::find_if(anchors.begin(), anchors.end(),
                                    [this](const Anchor& item)
                                    { return item.id == activeAnchorId; });
  if (current != anchors.end())
  {
    if (current != anchors.begin())
      newFrame = std::max(newFrame, std::prev(current)->frame + 1);
    if (std::next(current) != anchors.end())
      newFrame = std::min(newFrame, std::next(current)->frame - 1);
  }

  newFrame = juce::jlimit(0,
                          static_cast<int>(owner_.project->getAudioData().f0.size()) - 1,
                          newFrame);
  if (frameBelongsToPitchNote(newFrame))
    anchor->frame = newFrame;
  anchor->midi = midiFromWorldY(worldY);

  sortAnchors();
  extendRange(newFrame, newFrame);
  rebuildPreview();
  dragMoved = true;
  owner_.repaint();
  return true;
}

bool PitchDrawingHandler::mouseUp(const juce::MouseEvent&, float, float)
{
  if (!dragging)
    return false;
  dragging = false;
  activeAnchorId = -1;
  if (dragMoved)
    applyPreviewToProject();
  dragMoved = false;
  commitStep();
  owner_.repaint();
  return true;
}

void PitchDrawingHandler::mouseMove(const juce::MouseEvent& e, float worldX,
                                    float worldY)
{
  lastWorldX = worldX;
  lastWorldY = worldY;
  const auto newHovered = hitTest(worldX, worldY);
  const bool newPointerOverPitchRegion = isPitchRegionAtWorldX(worldX);
  const bool changed = newHovered.curveId != hovered.curveId ||
                       newHovered.anchorId != hovered.anchorId ||
                       newHovered.frame != hovered.frame ||
                       newPointerOverPitchRegion != pointerOverPitchRegion;
  hovered = newHovered;
  pointerOverPitchRegion = newPointerOverPitchRegion;
  updateCursor(e.mods);
  if (changed)
    owner_.repaint();
}

void PitchDrawingHandler::mouseDoubleClick(const juce::MouseEvent&, float worldX,
                                           float worldY)
{
  removeAnchorAt(worldX, worldY);
}

void PitchDrawingHandler::modifiersChanged(const juce::ModifierKeys& mods)
{
  updateCursor(mods);
  if (hovered.curveId >= 0)
    owner_.repaint();
}

void PitchDrawingHandler::updateCursor(const juce::ModifierKeys& mods)
{
  if (hovered.curveId >= 0)
    owner_.setMouseCursor(mods.isAltDown()
                              ? WarpHandler::eraserCursor()
                              : juce::MouseCursor(juce::MouseCursor::DraggingHandCursor));
  else
    owner_.setMouseCursor(pointerOverPitchRegion && !mods.isAltDown()
                              ? juce::MouseCursor::CrosshairCursor
                              : juce::MouseCursor::NormalCursor);
}

bool PitchDrawingHandler::removeAnchorAt(float worldX, float worldY)
{
  auto hit = hitTest(worldX, worldY);
  if (hit.curveId < 0 || !owner_.project)
    return false;
  if (hit.curveId != activeCurveId)
  {
    finish();
    const auto& curves = owner_.project->getDrawnCurves();
    const auto stored = std::find_if(curves.begin(), curves.end(),
                                     [&hit](const DrawnCurve& c)
                                     { return c.id == hit.curveId; });
    if (stored == curves.end())
      return true;
    const DrawnCurve curve = *stored;
    beginSession(&curve);
    hit = hitTest(worldX, worldY);
  }
  if (hit.anchorId >= 0)
    removeAnchor(hit.anchorId);
  return true;
}

void PitchDrawingHandler::removeAnchor(int anchorId)
{
  anchors.erase(std::remove_if(anchors.begin(), anchors.end(),
                               [anchorId](const Anchor& anchor)
                               { return anchor.id == anchorId; }),
                anchors.end());
  hovered = {};
  activeAnchorId = -1;
  dragging = false;
  dragMoved = false;

  // With no anchor left the curve gives its frames their pitch back.
  rebuildPreview();
  applyPreviewToProject();
  commitStep();
  if (anchors.empty())
    clearState();
  updateCursor(juce::ModifierKeys::currentModifiers);
  notifyStateChanged();
}

void PitchDrawingHandler::clearHover()
{
  if (hovered.curveId < 0 || dragging)
    return;
  hovered = {};
  owner_.repaint();
}

// ---------------------------------------------------------------- drawing

void PitchDrawingHandler::draw(juce::Graphics& g)
{
  constexpr float radius = 5.0f;
  const bool altDown = juce::ModifierKeys::currentModifiers.isAltDown();
  const auto drawAnchor = [&](juce::Point<float> point, bool highlighted)
  {
    const auto circle = juce::Rectangle<float>(point.x - radius, point.y - radius,
                                               radius * 2.0f, radius * 2.0f);
    if (highlighted)
    {
      g.setColour(altDown && !dragging ? APP_COLOR_ALERT_ERROR
                                       : APP_COLOR_PITCH_CURVE);
      g.fillEllipse(circle);
      return;
    }
    g.setColour(APP_COLOR_BACKGROUND);
    g.fillEllipse(circle);
    g.setColour(APP_COLOR_PITCH_CURVE);
    g.drawEllipse(circle, 1.5f);
  };

  // Curves drawn earlier: their anchors sit on the pitch line.
  if (owner_.project)
    for (const auto& curve : owner_.project->getDrawnCurves())
    {
      if (curve.id == activeCurveId ||
          std::find(absorbedCurveIds.begin(), absorbedCurveIds.end(), curve.id) !=
              absorbedCurveIds.end())
        continue;
      for (const int frame : curve.anchorFrames)
        drawAnchor(pointAt(frame, sungPitchAt(frame)),
                   hovered.curveId == curve.id && hovered.frame == frame);
    }

  for (const auto& anchor : anchors)
    drawAnchor(pointAt(anchor.frame, anchor.midi),
               (hovered.curveId == activeCurveId && hovered.anchorId == anchor.id) ||
                   anchor.id == activeAnchorId);
}

bool PitchDrawingHandler::isActive() const
{
  return dragging || activeCurveId >= 0;
}

void PitchDrawingHandler::cancel()
{
  // Nothing drawn is ever pending: every step is applied and in the undo
  // history, so cancelling a curve just ends it.
  finish();
}

void PitchDrawingHandler::finish()
{
  if (activeCurveId < 0)
    return;
  if (dragging && dragMoved)
    applyPreviewToProject();
  commitStep();
  clearState();
}

void PitchDrawingHandler::clearState()
{
  anchors.clear();
  activeCurveId = -1;
  absorbedCurveIds.clear();
  rangeStart = rangeEnd = -1;
  originalNoteStates.clear();
  committedNoteStates.clear();
  committedCurves.clear();
  originalMidiCurve.clear();
  previewMidiCurve.clear();
  hovered = {};
  activeAnchorId = -1;
  dragging = false;
  dragMoved = false;
  notifyStateChanged();
}

// ---------------------------------------------------------------- session

void PitchDrawingHandler::beginSession(const DrawnCurve* curve)
{
  clearState();
  if (!owner_.project)
    return;
  auto& project = *owner_.project;
  const auto& audio = project.getAudioData();
  const size_t size = audio.f0.size();

  originalMidiCurve.assign(size, 0.0f);
  for (size_t i = 0; i < size; ++i)
    originalMidiCurve[i] = sungPitchAt(static_cast<int>(i));

  if (curve)
  {
    activeCurveId = curve->id;
    // Under a drawn curve, the original is the pitch it replaced.
    for (size_t k = 0; k < curve->underDiff.size(); ++k)
    {
      const int f = curve->underStart + static_cast<int>(k);
      if (f >= 0 && f < static_cast<int>(size))
        originalMidiCurve[static_cast<size_t>(f)] += curve->underDiff[k];
    }
    rangeStart = curve->underStart;
    rangeEnd = curve->underEnd() - 1;
    for (const int frame : curve->anchorFrames)
      anchors.push_back({nextAnchorId++, frame, sungPitchAt(frame)});
  }
  else
    activeCurveId = project.nextDrawnCurveId();

  previewMidiCurve.assign(size, 0.0f);
  for (size_t i = 0; i < size; ++i)
    previewMidiCurve[i] = sungPitchAt(static_cast<int>(i));

  for (auto& note : project.getNotes())
    if (!note.isRest())
      originalNoteStates.push_back(
          {&note, TransformParams::fromNote(note), note.getBakedDeltaPitch()});
  committedNoteStates = originalNoteStates;
  committedCurves = project.getDrawnCurves();
}

void PitchDrawingHandler::extendRange(int start, int end)
{
  rangeStart = rangeStart < 0 ? start : std::min(rangeStart, start);
  rangeEnd = rangeEnd < 0 ? end : std::max(rangeEnd, end);
  // Curves the edited one now reaches join it: their anchors become its own.
  absorbCurvesIn(rangeStart, rangeEnd);
}

void PitchDrawingHandler::absorbCurvesIn(int start, int end)
{
  if (!owner_.project)
    return;
  bool grew = true;
  while (grew)
  {
    grew = false;
    for (const auto& curve : committedCurves)
    {
      if (curve.id == activeCurveId ||
          std::find(absorbedCurveIds.begin(), absorbedCurveIds.end(), curve.id) !=
              absorbedCurveIds.end() ||
          !curve.overlaps(start, end + 1))
        continue;
      absorbedCurveIds.push_back(curve.id);
      for (const int frame : curve.anchorFrames)
        if (std::none_of(anchors.begin(), anchors.end(),
                         [frame](const Anchor& a) { return a.frame == frame; }))
          anchors.push_back({nextAnchorId++, frame, sungPitchAt(frame)});
      for (size_t k = 0; k < curve.underDiff.size(); ++k)
      {
        const int f = curve.underStart + static_cast<int>(k);
        if (f >= 0 && f < static_cast<int>(originalMidiCurve.size()) &&
            (f < rangeStart || f > rangeEnd))
          originalMidiCurve[static_cast<size_t>(f)] =
              sungPitchAt(f) + curve.underDiff[k];
      }
      rangeStart = std::min(rangeStart, curve.underStart);
      rangeEnd = std::max(rangeEnd, curve.underEnd() - 1);
      start = rangeStart;
      end = rangeEnd;
      grew = true;
    }
  }
  sortAnchors();
}

std::pair<int, int> PitchDrawingHandler::affectedRange() const
{
  int start = rangeStart, end = rangeEnd;
  if (!anchors.empty())
  {
    start = start < 0 ? anchors.front().frame : std::min(start, anchors.front().frame);
    end = std::max(end, anchors.back().frame);
  }
  return {start, end};
}

bool PitchDrawingHandler::commitStep()
{
  if (!owner_.project || activeCurveId < 0)
    return false;
  auto& project = *owner_.project;

  std::vector<PitchDrawingNoteState> before;
  std::vector<PitchDrawingNoteState> after;
  for (auto& committed : committedNoteStates)
  {
    if (!committed.note)
      continue;
    auto& note = *committed.note;
    const auto params = TransformParams::fromNote(note);
    if (params == committed.params &&
        note.getBakedDeltaPitch() == committed.bakedDeltaPitch)
      continue;
    before.push_back({&note, committed.params, committed.bakedDeltaPitch});
    after.push_back({&note, params, note.getBakedDeltaPitch()});
    committed.params = params;
    committed.bakedDeltaPitch = note.getBakedDeltaPitch();
  }

  // The curve as it stands, replacing itself and the curves it took in.
  std::vector<DrawnCurve> curves;
  for (const auto& curve : committedCurves)
    if (curve.id != activeCurveId &&
        std::find(absorbedCurveIds.begin(), absorbedCurveIds.end(), curve.id) ==
            absorbedCurveIds.end())
      curves.push_back(curve);
  if (!anchors.empty() && rangeStart >= 0)
  {
    DrawnCurve curve;
    curve.id = activeCurveId;
    for (const auto& anchor : anchors)
      curve.anchorFrames.push_back(anchor.frame);
    curve.underStart = rangeStart;
    for (int f = rangeStart; f <= rangeEnd; ++f)
      curve.underDiff.push_back(
          f < static_cast<int>(originalMidiCurve.size())
              ? originalMidiCurve[static_cast<size_t>(f)] - sungPitchAt(f)
              : 0.0f);
    curves.push_back(std::move(curve));
    std::sort(curves.begin(), curves.end(),
              [](const DrawnCurve& a, const DrawnCurve& b) { return a.id < b.id; });
  }

  if (before.empty() && sameCurves(curves, committedCurves))
    return false;

  if (owner_.undoManager)
    owner_.undoManager->addAction(std::make_unique<PitchDrawingAction>(
        &project, std::move(before), std::move(after), committedCurves, curves));
  project.getDrawnCurves() = curves;
  committedCurves = curves;
  absorbedCurveIds.clear();
  project.setModified(true);
  owner_.invalidateBasePitchCache();
  if (owner_.onPitchEdited)
    owner_.onPitchEdited();
  if (owner_.onPitchEditCommitted)
    owner_.onPitchEditCommitted();
  return true;
}

// ---------------------------------------------------------------- geometry

int PitchDrawingHandler::frameFromWorldX(float worldX) const
{
  if (!owner_.project || owner_.project->getAudioData().f0.empty())
    return 0;
  const int lastFrame =
      static_cast<int>(owner_.project->getAudioData().f0.size()) - 1;
  const int frame = static_cast<int>(std::round(
      secondsToFrames(static_cast<float>(owner_.xToTime(worldX)))));
  return juce::jlimit(0, lastFrame, frame);
}

float PitchDrawingHandler::midiFromWorldY(float worldY) const
{
  float midi = owner_.yToMidi(worldY - owner_.pixelsPerSemitone * 0.5f);
  if (owner_.project)
    midi -= owner_.project->getGlobalPitchOffset();
  return juce::jlimit(static_cast<float>(MIN_MIDI_NOTE),
                      static_cast<float>(MAX_MIDI_NOTE), midi);
}

juce::Point<float> PitchDrawingHandler::pointAt(int frame, float midi) const
{
  const float globalOffset = owner_.project
                                 ? owner_.project->getGlobalPitchOffset()
                                 : 0.0f;
  return {owner_.timeToX(framesToSeconds(frame)),
          owner_.midiToY(midi + globalOffset) + owner_.pixelsPerSemitone * 0.5f};
}

float PitchDrawingHandler::sungPitchAt(int frame) const
{
  if (!owner_.project || frame < 0)
    return 0.0f;
  const auto& audio = owner_.project->getAudioData();
  const auto i = static_cast<size_t>(frame);
  const float base = i < audio.basePitch.size() ? audio.basePitch[i] : 0.0f;
  const float delta = i < audio.deltaPitch.size() ? audio.deltaPitch[i] : 0.0f;
  return base + delta;
}

PitchDrawingHandler::Hit PitchDrawingHandler::hitTest(float worldX,
                                                      float worldY) const
{
  constexpr float hitRadius = 8.0f;
  const juce::Point<float> pointer(worldX, worldY);
  for (auto it = anchors.rbegin(); it != anchors.rend(); ++it)
    if (pointAt(it->frame, it->midi).getDistanceFrom(pointer) <= hitRadius)
      return {activeCurveId, it->id, it->frame};
  if (owner_.project)
    for (const auto& curve : owner_.project->getDrawnCurves())
    {
      if (curve.id == activeCurveId ||
          std::find(absorbedCurveIds.begin(), absorbedCurveIds.end(), curve.id) !=
              absorbedCurveIds.end())
        continue;
      for (const int frame : curve.anchorFrames)
        if (pointAt(frame, sungPitchAt(frame)).getDistanceFrom(pointer) <= hitRadius)
          return {curve.id, -1, frame};
    }
  return {};
}

PitchDrawingHandler::Anchor* PitchDrawingHandler::findAnchorById(int id)
{
  auto it = std::find_if(anchors.begin(), anchors.end(),
                         [id](const Anchor& anchor) { return anchor.id == id; });
  return it == anchors.end() ? nullptr : &*it;
}

bool PitchDrawingHandler::frameBelongsToPitchNote(int frame) const
{
  if (!owner_.project)
    return false;
  return std::any_of(owner_.project->getNotes().begin(),
                     owner_.project->getNotes().end(),
                     [frame](const Note& note)
                     {
                       return !note.isRest() && note.getStartFrame() <= frame &&
                              note.getEndFrame() > frame;
                     });
}

bool PitchDrawingHandler::isPitchRegionAtWorldX(float worldX) const
{
  if (!owner_.project)
    return false;

  const auto& f0 = owner_.project->getAudioData().f0;
  const int frame = static_cast<int>(std::round(
      secondsToFrames(static_cast<float>(owner_.xToTime(worldX)))));
  return frame >= 0 && frame < static_cast<int>(f0.size()) &&
         frameBelongsToPitchNote(frame);
}

void PitchDrawingHandler::sortAnchors()
{
  std::sort(anchors.begin(), anchors.end(),
            [](const Anchor& a, const Anchor& b) { return a.frame < b.frame; });
}

// ---------------------------------------------------------------- preview

void PitchDrawingHandler::rebuildPreview()
{
  previewMidiCurve = originalMidiCurve;
  if (previewMidiCurve.empty())
    return;

  if (anchors.size() == 1)
  {
    previewMidiCurve[static_cast<size_t>(anchors.front().frame)] =
        anchors.front().midi;
  }
  else
  {
    for (size_t segment = 1; segment < anchors.size(); ++segment)
    {
      const auto& left = anchors[segment - 1];
      const auto& right = anchors[segment];
      const int length = std::max(1, right.frame - left.frame);
      for (int frame = left.frame; frame <= right.frame; ++frame)
      {
        const float t = static_cast<float>(frame - left.frame) /
                        static_cast<float>(length);
        // Quintic smootherstep: flat at both anchors, with a steeper
        // transition through the middle than linear interpolation.
        const float shapedT = t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
        previewMidiCurve[static_cast<size_t>(frame)] =
            juce::jmap(shapedT, left.midi, right.midi);
      }
    }
  }

  updateAffectedNotePositions();
}

void PitchDrawingHandler::updateAffectedNotePositions()
{
  const auto [first, last] = affectedRange();
  if (!owner_.project || first < 0 || previewMidiCurve.empty())
    return;

  // Start from the exact pre-edit positions on every preview update, so
  // dragging an anchor or shrinking its range cannot accumulate movement.
  restoreOriginalNoteStates();

  const auto& audioData = owner_.project->getAudioData();
  for (auto& note : owner_.project->getNotes())
  {
    if (note.isRest() || note.getEndFrame() <= first || note.getStartFrame() > last)
      continue;

    const int start = std::max(0, note.getStartFrame());
    const int end = std::min(note.getEndFrame(),
                             static_cast<int>(previewMidiCurve.size()));
    double sum = 0.0;
    int count = 0;
    for (int frame = start; frame < end; ++frame)
    {
      const bool voiced =
          audioData.voicedMask.empty() ||
          (frame < static_cast<int>(audioData.voicedMask.size()) &&
           audioData.voicedMask[static_cast<size_t>(frame)]);
      if (!voiced || !std::isfinite(previewMidiCurve[static_cast<size_t>(frame)]))
        continue;
      sum += previewMidiCurve[static_cast<size_t>(frame)];
      ++count;
    }

    if (count > 0)
      note.setMidiNote(static_cast<float>(sum / static_cast<double>(count)));
  }

  if (owner_.onPitchEdited)
    owner_.onPitchEdited();
}

void PitchDrawingHandler::applyPreviewToProject()
{
  const auto [first, last] = affectedRange();
  if (!owner_.project || first < 0 || previewMidiCurve.empty())
    return;

  auto& project = *owner_.project;
  auto& audioData = project.getAudioData();
  std::vector<Note*> affected;
  for (auto& note : project.getNotes())
  {
    if (note.isRest() || note.getEndFrame() <= first || note.getStartFrame() > last)
      continue;
    // Keep the live note center calculated by updateAffectedNotePositions(),
    // but make the per-note transforms neutral before deriving the base curve
    // for the baked preview contour.
    neutralTools(note);
    affected.push_back(&note);
  }
  if (affected.empty())
    return;

  PitchCurveProcessor::rebuildBaseFromNotes(project);
  int dirtyStart = std::numeric_limits<int>::max();
  int dirtyEnd = std::numeric_limits<int>::min();
  for (auto* note : affected)
  {
    const int duration = std::max(0, note->getDurationFrames());
    std::vector<float> bakedDelta(static_cast<size_t>(duration), 0.0f);
    for (int i = 0; i < duration; ++i)
    {
      const int frame = note->getStartFrame() + i;
      if (frame >= 0 && frame < static_cast<int>(previewMidiCurve.size()) &&
          frame < static_cast<int>(audioData.basePitch.size()))
        bakedDelta[static_cast<size_t>(i)] =
            previewMidiCurve[static_cast<size_t>(frame)] -
            audioData.basePitch[static_cast<size_t>(frame)];
    }
    note->setBakedDeltaPitch(std::move(bakedDelta));
    note->markDirty();
    note->markSynthDirty();
    dirtyStart = std::min(dirtyStart, note->getStartFrame());
    dirtyEnd = std::max(dirtyEnd, note->getEndFrame());
  }
  PitchCurveProcessor::rebuildBaseFromNotes(project);

  // A note no anchor reaches any more, singing exactly the pitch it had
  // before this curve's edit: give it back its pre-edit state, so a note
  // nobody else edited plays the recording again rather than the vocoder.
  const int spanFirst = anchors.empty() ? std::numeric_limits<int>::max()
                                        : anchors.front().frame;
  const int spanLast = anchors.empty() ? std::numeric_limits<int>::min()
                                       : anchors.back().frame;
  bool restored = false;
  for (auto* note : affected)
  {
    if (note->getEndFrame() > spanFirst && note->getStartFrame() <= spanLast)
      continue;
    const auto keptParams = TransformParams::fromNote(*note);
    const auto keptBaked = note->getBakedDeltaPitch();
    const auto singsPreview = [&]()
    {
      PitchCurveProcessor::rebuildBaseFromNotes(project);
      for (int f = std::max(0, note->getStartFrame()); f < note->getEndFrame(); ++f)
        if (f < static_cast<int>(previewMidiCurve.size()) &&
            std::abs(sungPitchAt(f) - previewMidiCurve[static_cast<size_t>(f)]) >= 0.002f)
          return false;
      return true;
    };
    // First the analysed note (no edits at all), then its state when this
    // edit began.
    bool found = false;
    {
      TransformParams analysed;
      analysed.midiNote = note->getOriginalMidiNote();
      analysed.formantShift = keptParams.formantShift;
      analysed.volumeDb = keptParams.volumeDb;
      analysed.sibilantBalance = keptParams.sibilantBalance;
      analysed.applyToNote(*note);
      note->setBakedDeltaPitch({});
      found = std::abs(note->getPitchOffset()) < 0.001f && singsPreview();
    }
    if (!found)
    {
      const auto state = std::find_if(originalNoteStates.begin(), originalNoteStates.end(),
                                      [note](const OriginalNoteState& s)
                                      { return s.note == note; });
      if (state != originalNoteStates.end())
      {
        state->params.applyToNote(*note);
        note->setBakedDeltaPitch(state->bakedDeltaPitch);
        found = singsPreview();
      }
    }
    if (found)
      restored = true;
    else
    {
      keptParams.applyToNote(*note);
      note->setBakedDeltaPitch(keptBaked);
      PitchCurveProcessor::rebuildBaseFromNotes(project);
    }
  }
  if (restored)
    PitchCurveProcessor::rebuildBaseFromNotes(project);

  project.markNoteEditDirtyRange(dirtyStart, dirtyEnd);
  owner_.invalidateBasePitchCache();
  if (owner_.onPitchEdited)
    owner_.onPitchEdited();
  if (owner_.onPitchPreviewRenderRequested)
    owner_.onPitchPreviewRenderRequested();
}

void PitchDrawingHandler::restoreOriginalNoteStates()
{
  if (!owner_.project)
    return;
  for (const auto& state : originalNoteStates)
  {
    if (!state.note)
      continue;
    state.params.applyToNote(*state.note);
    state.note->setBakedDeltaPitch(state.bakedDeltaPitch);
  }
}

void PitchDrawingHandler::notifyStateChanged()
{
  owner_.repaint();
}
