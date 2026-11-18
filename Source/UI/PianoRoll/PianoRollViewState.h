#pragma once

#include "../../Models/Project.h"
#include "../../Utils/Constants.h"

#include <optional>

enum class EditMode
{
  Select = 0,
  // Value 1 belonged to the removed freehand tool.
  Split = 2,
  PitchDrawing = 3,
  // Value 4 belonged to the Timing tool, folded into Warp (note boundaries
  // drag there).
  // The warp layer (spec section 3): place, drag and delete warp anchors,
  // and drag note boundaries (spec section 4).
  Warp = 5
};

struct PianoRollViewState
{
  int hoveredPitchToolHandle = -1;

  float pixelsPerSecond = DEFAULT_PIXELS_PER_SECOND;
  float pixelsPerSemitone = DEFAULT_PIXELS_PER_SEMITONE;

  double cursorTime = 0.0;
  double scrollX = 0.0;
  double scrollY = 0.0;

  EditMode editMode = EditMode::Select;

  bool showDeltaPitch = true;
  bool showBasePitch = false;
  bool showSegmentsDebug = false;
  bool showGameValuesDebug = false;
  bool showNoteFramesDebug = false;
  bool showUvInterpolationDebug = false;
  bool showActualF0Debug = false;
  bool showCleanedF0Debug = false;
  bool showVocoderF0Debug = false;
  bool snapToSemitoneDrag = false;
  DragSnapMode dragSnapMode = DragSnapMode::Chromatic;

  int pitchReferenceHz = 440;
  TimelineDisplayMode timelineDisplayMode = TimelineDisplayMode::Beats;
  int timelineBeatNumerator = 4;
  int timelineBeatDenominator = 4;
  double timelineTempoBpm = 120.0;
  TimelineGridDivision timelineGridDivision = TimelineGridDivision::Quarter;
  bool timelineSnapCycle = false;
  // Warp anchors snap to the timeline grid (spec section 3).
  bool timelineSnapWarp = false;

  // Cycle (loop) range editing on the timeline. Disabled when the plugin
  // cannot drive the host transport (non-ARA plugin mode).
  bool cycleEditEnabled = true;
  // Mirrors the toolbar's loop button; the range is editable only while on.
  bool loopEditEnabled = false;

  ScaleMode selectedScaleMode = ScaleMode::Chromatic;
  int selectedScaleRootNote = 0;
  std::optional<int> previewScaleRootNote;
  std::optional<ScaleMode> previewScaleMode;

  bool cacheInvalidated = true;
};
