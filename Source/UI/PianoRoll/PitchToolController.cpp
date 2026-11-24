#include "PitchToolController.h"
#include "../../Audio/NoteTiming.h"
#include "../../Utils/Constants.h"
#include "../../Utils/PitchToolOperations.h"
#include "../../Audio/ProjectGain.h"
#include "../../Undo/AmplitudeAction.h"
#include "../../Utils/PitchCurveProcessor.h"

#include <algorithm>
#include <climits>
#include <limits>

PitchToolController::PitchToolController()
{
}

const VisualWaveformEnvelope::GainCurve*
PitchToolController::getGainPreview(const Project& forProject) const
{
  const bool gainDrag =
      dragging && project == &forProject &&
      (activeHandleType == PitchToolHandles::HandleType::Amplitude ||
       activeHandleType == PitchToolHandles::HandleType::SibilantBalance);
  if (!gainDrag)
    return nullptr;
  if (!gainPreviewValid)
  {
    gainPreviewValid = true;
    gainPreview = {};
    // The dragged notes as they were when the drag began.
    std::vector<std::pair<const Note*, Note>> before;
    std::vector<const Note*> changed;
    for (size_t i = 0; i < affectedNotes.size() && i < originalParams.size(); ++i)
    {
      const Note* note = affectedNotes[i];
      if (!note || note->isRest() ||
          (note->getVolumeDb() == originalParams[i].volumeDb &&
           note->getSibilantBalance() == originalParams[i].sibilantBalance))
        continue;
      Note was = *note;
      was.setVolumeDb(originalParams[i].volumeDb);
      was.setSibilantBalance(originalParams[i].sibilantBalance);
      before.emplace_back(note, std::move(was));
      changed.push_back(note);
    }
    const auto& audioData = project->getAudioData();
    if (!before.empty())
    {
      // In analysis samples; the waveform is at its own rate. The notes
      // and the stray sibilants that belong to them.
      const auto [first, last] = projectgain::noteGainSpan(*project, changed);
      auto [start, gains] = projectgain::changeRatioAround(
          audioData, projectgain::regions(*project, before),
          projectgain::regions(*project), first, last);
      gainPreview.startSample = start;
      gainPreview.curveRate = SAMPLE_RATE;
      gainPreview.waveformRate = audioData.sampleRate;
      gainPreview.gains = std::move(gains);
    }
  }
  return gainPreview.gains.empty() ? nullptr : &gainPreview;
}

bool PitchToolController::mouseDown(const juce::MouseEvent& e,
                                    const PitchToolHandles& handles,
                                    const std::vector<Note*>& selectedNotes,
                                    const CoordinateMapper& mapper)
{
  const int hitIndex = handles.hitTest(e.position.x, e.position.y);
  if (hitIndex < 0) {
    return false;
  }

  const auto& handle = handles.getHandle(hitIndex);
  activeHandleType = handle.type;
  activeHandleNote = handle.note;

  // A green square: the stretch of this one note kept as recorded by hand.
  // The drag starts from where the squares stand - the user's own marks, or
  // on a note left to the detector what it decided - and nothing changes
  // until the square moves to another frame.
  if (isKeepHandle(handle.type))
  {
    if (!project || !handle.note || handle.note->isRest())
    {
      activeHandleType = PitchToolHandles::HandleType::None;
      activeHandleNote = nullptr;
      return false;
    }
    const auto& audio = project->getAudioData();
    thread_local fry::Cache found;
    keepBefore = SynthChoice::of(*handle.note);
    keepStart = fry::shownRange(*handle.note, project->getNotes(), audio.rawF0,
                                found.of(audio));
    const auto map = notetiming::forNoteOnScreen(audio, *handle.note);
    const int at = handle.type == PitchToolHandles::HandleType::KeepHead
                       ? keepStart.headUntil
                       : keepStart.tailFrom;
    keepStartX = mapper.timeToX(
        framesToSeconds(static_cast<float>(map.destinationAt(at))));
    affectedNotes.assign(1, handle.note);
    originalParams.clear();
    dragStartPos = e.position;
    dragging = true;
    gainPreviewValid = false;
    rotating = false;
    pivotReset = false;
    return true;
  }

  // The visible top controls belong to the hovered note. When it is part of a
  // multi-note selection, slopes and vibrato are group edits, so capture the
  // complete selection. Hovering an unselected note must still only affect
  // that note rather than an unrelated selection.
  affectedNotes.clear();
  const bool handleIsSelected =
      handle.note && std::find(selectedNotes.begin(), selectedNotes.end(),
                               handle.note) != selectedNotes.end();
  const bool isGroupEdit = handleIsSelected && selectedNotes.size() > 1 &&
      (handle.type == PitchToolHandles::HandleType::TiltLeft ||
       handle.type == PitchToolHandles::HandleType::TiltRight ||
       handle.type == PitchToolHandles::HandleType::Vibrato ||
       handle.type == PitchToolHandles::HandleType::PitchDrift ||
       handle.type == PitchToolHandles::HandleType::Formant ||
       handle.type == PitchToolHandles::HandleType::Amplitude ||
       handle.type == PitchToolHandles::HandleType::SibilantBalance);
  if (isGroupEdit || !handle.note)
    affectedNotes = selectedNotes;
  else
    affectedNotes.push_back(handle.note);
  if (affectedNotes.empty())
    return false;
  
  // Capture original transformation parameters (not curves)
  originalParams.clear();
  originalParams.reserve(affectedNotes.size());
  for (auto* note : affectedNotes)
  {
    if (note)
    {
      auto params = TransformParams::fromNote(*note);
      
      // Store baseline midiNote (without tilt mean shift)
      // This ensures tilt adjustments are absolute, not cumulative
      const float currentTiltMean = note->getTiltMean();
      params.midiNote = note->getMidiNote() - currentTiltMean;
      
      originalParams.push_back(params);
    }
    else
    {
      originalParams.emplace_back();
    }
  }

  dragStartPos = e.position;
  dragging = true;
  gainPreviewValid = false;
  rotating = e.mods.isAltDown();
  pivotReset = handle.type == PitchToolHandles::HandleType::TiltPivot &&
               e.mods.isAltDown();
  if (pivotReset)
    applyOperation(affectedNotes, activeHandleType, 0.0f, 0.0f, mapper);
  return true;
}

bool PitchToolController::mouseDrag(const juce::MouseEvent& e,
                                    std::vector<Note*>& selectedNotes,
                                    const CoordinateMapper& mapper)
{
  juce::ignoreUnused(selectedNotes);

  if (!dragging)
    return false;

  const float deltaX = e.position.x - dragStartPos.x;
  const float deltaY = e.position.y - dragStartPos.y;

  if (isKeepHandle(activeHandleType))
  {
    dragKeep(deltaX, mapper);
    return true;
  }
  if (pivotReset)
    return true;
  rotating = e.mods.isAltDown();
  applyOperation(affectedNotes, activeHandleType, deltaX, deltaY, mapper);
  gainPreviewValid = false;
  return true;
}

// A green square follows the pointer along the note, one frame of the
// recording at a time (through the note's time map: a stretched note's
// frames are wider on screen). The left one keeps from the note's start up
// to where it stands, the right one from where it stands to the note's end;
// neither passes the other, and the two meeting is all of the note.
void PitchToolController::dragKeep(float dragDeltaX, const CoordinateMapper& mapper)
{
  Note* note = activeHandleNote;
  if (!project || !note)
    return;
  const auto map = notetiming::forNoteOnScreen(project->getAudioData(), *note);
  const double frame = mapper.xToTime(keepStartX + dragDeltaX) * SAMPLE_RATE / HOP_SIZE;
  const int source = static_cast<int>(std::lround(map.sourceAt(frame)));
  const int start = note->getSrcStartFrame();
  const int end = std::max(start, note->getSrcEndFrame());
  int headUntil = keepStart.headUntil;
  int tailFrom = keepStart.tailFrom;
  if (activeHandleType == PitchToolHandles::HandleType::KeepHead)
    headUntil = std::clamp(source, start, tailFrom);
  else
    tailFrom = std::clamp(source, headUntil, end);

  // Back where the squares stood, the note is as it was: one left to the
  // detector is not marked by a drag that ends where it began.
  if (headUntil == keepStart.headUntil && tailFrom == keepStart.tailFrom)
    keepBefore.applyTo(*note);
  else
    note->setKeptRange(headUntil > start ? headUntil : Note::kKeepsNone,
                       tailFrom < end ? tailFrom : Note::kKeepsNone);
  // (The piano roll repaints after every drag step: the green fill and the
  // squares follow from the note.)
}

bool PitchToolController::mouseUp(const juce::MouseEvent& e,
                                  PitchUndoManager* undoManager,
                                  std::function<void(int, int)> onRangeChanged)
{
  juce::ignoreUnused(e);

  if (!dragging)
    return false;

  // A green square let go: one history step, shared with the synthesis menu
  // (SynthChoiceAction). The note goes back to how it was first - applying
  // the action is what changes it, so that it sees the note become whole (or
  // stop being so) and moves the pitch curve with it.
  if (isKeepHandle(activeHandleType))
  {
    Note* note = activeHandleNote;
    std::unique_ptr<UndoableAction> action;
    if (project && note)
    {
      const auto after = SynthChoice::of(*note);
      keepBefore.applyTo(*note);
      if (after != keepBefore)
        action = std::make_unique<SynthChoiceAction>(
            *project, std::vector<Note*>{note}, std::vector<SynthChoice>{keepBefore},
            std::vector<SynthChoice>{after}, "Set Kept Range");
    }
    dragging = false;
    activeHandleType = PitchToolHandles::HandleType::None;
    activeHandleNote = nullptr;
    affectedNotes.clear();
    originalParams.clear();
    if (action && onSynthChoice)
      onSynthChoice(std::move(action));
    else if (action)
    {
      action->redo();
      if (undoManager) undoManager->addAction(std::move(action));
    }
    return true;
  }

  if (activeHandleType == PitchToolHandles::HandleType::SibilantBalance && project)
  {
    std::vector<float> volumes, before, after;
    for (size_t i = 0; i < affectedNotes.size(); ++i)
    {
      volumes.push_back(originalParams[i].volumeDb);
      before.push_back(originalParams[i].sibilantBalance);
      after.push_back(affectedNotes[i] ? affectedNotes[i]->getSibilantBalance()
                                       : before.back());
    }
    auto action = gainedit::noteGains(*project, affectedNotes, volumes, volumes,
                                      before, after, onAmplitudeEdited,
                                      "Change Sibilant Balance");
    dragging = false;
    activeHandleType = PitchToolHandles::HandleType::None;
    activeHandleNote = nullptr;
    affectedNotes.clear();
    originalParams.clear();
    if (before != after)
    {
      action->redo();
      if (undoManager) undoManager->addAction(std::move(action));
    }
    return true;
  }

  if (activeHandleType == PitchToolHandles::HandleType::Amplitude && project)
  {
    std::vector<float> before, after;
    for (size_t i = 0; i < affectedNotes.size(); ++i)
    {
      before.push_back(originalParams[i].volumeDb);
      after.push_back(affectedNotes[i] ? affectedNotes[i]->getVolumeDb() : before.back());
    }
    auto action = std::make_unique<AmplitudeAction>(
        *project, affectedNotes, before, after, onAmplitudeEdited);
    dragging = false;
    activeHandleType = PitchToolHandles::HandleType::None;
    activeHandleNote = nullptr;
    affectedNotes.clear();
    originalParams.clear();
    if (before != after)
    {
      action->redo();
      if (undoManager) undoManager->addAction(std::move(action));
    }
    return true;
  }

  // The drag previews through rebuildDeltaForNotes(), which leaves basePitch
  // as it was. Commit through the same full rebuild PitchToolAction's undo and
  // redo use, so the edit renders from the curves its undo would reproduce.
  if (project)
  {
    for (auto* note : affectedNotes)
      if (note)
        note->markSynthDirty();
    PitchCurveProcessor::rebuildBaseFromNotes(*project);
  }

  // Capture new transformation parameters (not curves)
  std::vector<TransformParams> newParams;
  newParams.reserve(affectedNotes.size());
  for (auto* note : affectedNotes)
  {
    if (note)
      newParams.push_back(TransformParams::fromNote(*note));
    else
      newParams.emplace_back();
  }

  // Fix oldParams to store live midiNote (with tilt mean) for correct undo/redo
  // originalParams stores baseline midiNote (without tilt mean) for applyOperation,
  // but undo/redo needs the live midiNote value
  std::vector<TransformParams> undoOldParams = originalParams;
  for (size_t i = 0; i < undoOldParams.size(); ++i)
  {
    const float tiltMean = undoOldParams[i].tiltMean();
    undoOldParams[i].midiNote += tiltMean;
  }

  auto action = std::make_unique<PitchToolAction>(
      project, affectedNotes, undoOldParams, newParams, onRangeChanged);

  if (undoManager)
    undoManager->addAction(std::move(action));

  if (onRangeChanged)
  {
    for (auto* note : affectedNotes)
    {
      if (note)
        onRangeChanged(note->getStartFrame(), note->getEndFrame());
    }
  }

  dragging = false;
  activeHandleType = PitchToolHandles::HandleType::None;
  activeHandleNote = nullptr;
  affectedNotes.clear();
  originalParams.clear();
  return true;
}

void PitchToolController::applyOperation(std::vector<Note*>& notes,
                                         PitchToolHandles::HandleType type,
                                         float dragDeltaX,
                                         float dragDeltaY,
                                         const CoordinateMapper& mapper)
{
  if (!project)
    return;  // Cannot proceed without project reference

  const float pixelsPerSemitone = juce::jmax(1.0f, mapper.getPixelsPerSemitone());
  const float semitoneDelta = -dragDeltaY / pixelsPerSemitone;

  // Debug logging

  for (size_t i = 0; i < notes.size(); ++i)
  {
    auto* note = notes[i];
    if (!note || i >= originalParams.size())
      continue;

    // Restore original parameters before applying new transformation
    const auto& origParams = originalParams[i];
    if (type == PitchToolHandles::HandleType::Amplitude)
    {
      // Ten pixels per dB, independent of pitch zoom. Leave pitch state alone.
      note->setVolumeDb(juce::jlimit(-60.0f, 24.0f,
          std::round((origParams.volumeDb - dragDeltaY / 10.0f) * 10.0f) / 10.0f));
      continue;
    }
    if (type == PitchToolHandles::HandleType::SibilantBalance)
    {
      // One percent per pixel, up towards +100% (the rest of the note
      // quieter), down towards -100% (the sibilants quieter).
      note->setSibilantBalance(
          std::round((origParams.sibilantBalance - dragDeltaY / 100.0f) * 100.0f) /
          100.0f);
      continue;
    }
    origParams.applyToNote(*note);

    // Apply new transformation by updating the appropriate parameter
    switch (type)
    {
      case PitchToolHandles::HandleType::TiltLeft:
      {
        // Drag UP = positive semitoneDelta = left edge goes UP. With Alt
        // the note rotates about the pivot: the right side goes the other
        // way, staying on one straight line through the pivot.
        const float amount = semitoneDelta;
        note->setTiltLeft(origParams.tiltLeft + amount);
        const float pivot = origParams.tiltPivot;
        if (rotating && pivot > 0.01f)
          note->setTiltRight(origParams.tiltRight - amount * (1.0f - pivot) / pivot);
        
        // Calculate tilt mean shift
        const float newTiltMean = note->getTiltMean();
        note->setMidiNote(origParams.midiNote + newTiltMean);
        break;
      }
      case PitchToolHandles::HandleType::TiltRight:
      {
        const float amount = semitoneDelta;
        note->setTiltRight(origParams.tiltRight + amount);
        const float pivot = origParams.tiltPivot;
        if (rotating && pivot < 0.99f)
          note->setTiltLeft(origParams.tiltLeft - amount * pivot / (1.0f - pivot));
        
        // Calculate tilt mean shift
        const float newTiltMean = note->getTiltMean();
        note->setMidiNote(origParams.midiNote + newTiltMean);
        break;
      }
      case PitchToolHandles::HandleType::TiltPivot:
      {
        // Along the note, in proportion to its width on screen.
        const float noteWidth = juce::jmax(
            1.0f, mapper.timeToX(mapper.framesToSeconds(note->getEndFrame())) -
                      mapper.timeToX(mapper.framesToSeconds(note->getStartFrame())));
        // As in Cubase, moving the pivot leaves the curve as it is: the tilt
        // made about the old pivot is frozen into the note and only tilts
        // after this turn about the new place.
        if (std::abs(origParams.tiltLeft) > 0.001f ||
            std::abs(origParams.tiltRight) > 0.001f)
        {
          const size_t frames = static_cast<size_t>(std::max(2, note->getDurationFrames()));
          auto frozen = PitchToolOperations::tiltOffsets(
              frames, origParams.tiltLeft, origParams.tiltRight, origParams.tiltPivot);
          const auto& earlier = origParams.frozenTilt;
          for (size_t f = 0; f < frozen.size() && !earlier.empty(); ++f)
          {
            const double pos = earlier.size() == 1 ? 0.0
                : static_cast<double>(f) * static_cast<double>(earlier.size() - 1) /
                      static_cast<double>(frozen.size() - 1);
            const size_t lo = static_cast<size_t>(pos);
            const size_t hi = std::min(lo + 1, earlier.size() - 1);
            frozen[f] += earlier[lo] + (earlier[hi] - earlier[lo]) *
                                          static_cast<float>(pos - static_cast<double>(lo));
          }
          note->setFrozenTilt(std::move(frozen));
          note->setTiltLeft(0.0f);
          note->setTiltRight(0.0f);
        }
        note->setTiltPivot(pivotReset ? 0.5f
                                      : origParams.tiltPivot + dragDeltaX / noteWidth);
        note->setMidiNote(origParams.midiNote + note->getTiltMean());
        break;
      }
      case PitchToolHandles::HandleType::Formant:
      {
        note->setFormantShift(std::round((origParams.formantShift + semitoneDelta) * 10.0f) / 10.0f);
        note->setMidiNote(origParams.midiNote + origParams.tiltMean());
        break;
      }
      case PitchToolHandles::HandleType::PitchDrift:
      {
        note->setPitchDrift(origParams.pitchDrift - dragDeltaY / 100.0f);
        note->setMidiNote(origParams.midiNote + origParams.tiltMean());
        break;
      }
      case PitchToolHandles::HandleType::Vibrato:
      {
        // Additive accumulation from original value (consistent with tilt handles)
        // Drag UP increases modulation; drag DOWN decreases it.
        const float dragDelta = -dragDeltaY / 100.0f;
        const float newScale = origParams.vibrato + dragDelta;
        note->setVibrato(newScale);
        
        // Preserve tilt offset when adjusting vibrato.
        const float currentTiltMean = note->getTiltMean();
        note->setMidiNote(origParams.midiNote + currentTiltMean);
        break;
      }
      case PitchToolHandles::HandleType::SmoothLeft:
      {
        // Drag RIGHT (positive X) into note = more smoothing
        // 5px per additional frame, range [5, 50]
        const int transitionFrames = juce::jlimit(5, 50, 5 + static_cast<int>(dragDeltaX / 5.0f));
        note->setSmoothLeftFrames(transitionFrames);
        
        // Preserve tilt offset when adjusting smoothing
        const float currentTiltMean = note->getTiltMean();
        note->setMidiNote(origParams.midiNote + currentTiltMean);
        break;
      }
      case PitchToolHandles::HandleType::SmoothRight:
      {
        // Drag LEFT (negative X) into note = more smoothing
        // Use -dragDeltaX so LEFT drag increases frames
        const int transitionFrames = juce::jlimit(5, 50, 5 + static_cast<int>(-dragDeltaX / 5.0f));
        note->setSmoothRightFrames(transitionFrames);
        
        // Preserve tilt offset when adjusting smoothing
        const float currentTiltMean = note->getTiltMean();
        note->setMidiNote(origParams.midiNote + currentTiltMean);
        break;
      }
      case PitchToolHandles::HandleType::None:
      default:
        continue;  // Skip this note
    }

    note->markDirty();
  }

  if (type == PitchToolHandles::HandleType::Amplitude ||
      type == PitchToolHandles::HandleType::SibilantBalance)
  {
    if (onPitchEdited) onPitchEdited();
    return;
  }

  // NON-DESTRUCTIVE: Recompose only the affected notes' delta + f0
  // This is much faster than rebuildBaseFromNotes() which rebuilds everything
  PitchCurveProcessor::rebuildDeltaForNotes(*project, notes);
  
  // Mark dirty range for synthesis
  if (!notes.empty())
  {
    int minFrame = std::numeric_limits<int>::max();
    int maxFrame = std::numeric_limits<int>::min();
    for (const auto* note : notes)
    {
      minFrame = std::min(minFrame, note->getStartFrame());
      maxFrame = std::max(maxFrame, note->getEndFrame());
    }
    project->markNoteEditDirtyRange(minFrame, maxFrame);
  }

  // Trigger visual update
  if (onPitchEdited)
    onPitchEdited();
}

void PitchToolController::cancel()
{
  if (!dragging)
    return;

  if (isKeepHandle(activeHandleType) && activeHandleNote)
    keepBefore.applyTo(*activeHandleNote);

  // Restore original transformation parameters
  for (size_t i = 0; i < affectedNotes.size(); ++i)
  {
    if (i < originalParams.size() && affectedNotes[i])
    {
      const auto& params = originalParams[i];
      params.applyToNote(*affectedNotes[i]);
      
      // Restore midiNote with tilt mean (params.midiNote is baseline without tilt)
      const float tiltMean = params.tiltMean();
      affectedNotes[i]->setMidiNote(params.midiNote + tiltMean);
    }
  }

  dragging = false;
  activeHandleType = PitchToolHandles::HandleType::None;
  activeHandleNote = nullptr;
  affectedNotes.clear();
  originalParams.clear();
}
