#pragma once

#include "../../JuceHeader.h"
#include "../../Models/Project.h"
#include "CoordinateMapper.h"

#include <utility>
#include <vector>

/**
 * Draws the main scrollable grid (horizontal pitch lines + vertical beat/time
 * markers) with scale-aware light and dark row backgrounds.
 *
 * Stateless beyond the CoordinateMapper pointer; the component computes scale
 * state and timeline timing values and passes them in as a Params struct.
 */
class GridRenderer
{
public:
  struct Params
  {
    ScaleMode scaleMode;
    int scaleRootNote;
    // A song that changes key (spec section 15): where each stretch starts
    // (content x) and its root, in order; the first runs from the left
    // edge. Empty: scaleRootNote throughout.
    std::vector<std::pair<float, int>> scaleSpans;
    float pitchAxisOffsetSemitones;
    TimelineDisplayMode timelineDisplayMode;
    // Pre-computed timing values from the component (depend on tempo + signature).
    double gridSeconds;
    double beatSeconds;
    double barSeconds;
    // Scrollable canvas duration. In ARA this includes one display bar after
    // the project's audio end.
    double timelineDuration;
    // Visible content sizes from the component.
    int componentWidth;
    int visibleContentWidth;
    int visibleContentHeight;
    bool drawRowBackgrounds = true;
    bool drawGridLines = true;
  };

  GridRenderer() = default;

  void setCoordinateMapper(CoordinateMapper *mapper) { coordMapper = mapper; }
  void setProject(Project *proj) { project = proj; }

  void draw(juce::Graphics &g, const Params &params);

private:
  CoordinateMapper *coordMapper = nullptr;
  Project *project = nullptr;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GridRenderer)
};
