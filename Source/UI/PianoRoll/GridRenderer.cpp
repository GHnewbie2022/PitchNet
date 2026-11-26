#include "GridRenderer.h"
#include "PianoRollViewHelpers.h"
#include "../../Utils/Constants.h"
#include "../../Utils/UI/Theme.h"

void GridRenderer::draw(juce::Graphics &g, const Params &params)
{
  if (!coordMapper)
    return;

  const float pixelsPerSecond = coordMapper->getPixelsPerSecond();
  const float pixelsPerSemitone = coordMapper->getPixelsPerSemitone();
  const double scrollX = coordMapper->getScrollX();
  const double scrollY = coordMapper->getScrollY();

  float duration = static_cast<float>(params.timelineDuration);
  if (duration <= 0.0f)
    duration = project ? project->getAudioData().getDuration()
                       : DEFAULT_EMPTY_PROJECT_DURATION_SECONDS;
  if (duration <= 0.0f)
    duration = DEFAULT_EMPTY_PROJECT_DURATION_SECONDS;
  // The timeline may start before project time zero (Track mode).
  const float startX = static_cast<float>(
      std::min(0.0, coordMapper->getViewStartSeconds()) * pixelsPerSecond);
  const float width =
      std::max(duration * pixelsPerSecond, static_cast<float>(params.componentWidth));
  const float height = (MAX_MIDI_NOTE - MIN_MIDI_NOTE + 1) * pixelsPerSemitone;

  // Only draw the visible area to avoid spending time on off-screen rows/columns.
  const float visibleStartX =
      juce::jlimit(startX, width, static_cast<float>(scrollX));
  const float visibleEndX = juce::jlimit(
      startX, width,
      visibleStartX + static_cast<float>(params.visibleContentWidth) + 2.0f);
  const float visibleTopY = juce::jlimit(0.0f, height, static_cast<float>(scrollY));
  const float visibleBottomY = juce::jlimit(
      0.0f, height,
      visibleTopY + static_cast<float>(params.visibleContentHeight) + pixelsPerSemitone);

  if (visibleEndX <= visibleStartX || visibleBottomY <= visibleTopY)
    return;

  const int visibleTopMidi = juce::jlimit(
      MIN_MIDI_NOTE, MAX_MIDI_NOTE,
      static_cast<int>(std::ceil(coordMapper->yToMidi(visibleTopY))));
  const int visibleBottomMidi = juce::jlimit(
      MIN_MIDI_NOTE, MAX_MIDI_NOTE,
      static_cast<int>(std::floor(coordMapper->yToMidi(visibleBottomY))));
  const int startMidi = juce::jlimit(MIN_MIDI_NOTE, MAX_MIDI_NOTE,
                                     visibleBottomMidi - 1);
  const int endMidi = juce::jlimit(MIN_MIDI_NOTE, MAX_MIDI_NOTE,
                                   visibleTopMidi + 1);

  // The stretches of one key across what is visible: one, unless the song
  // changes key (Params::scaleSpans).
  struct KeyRun
  {
    float x0, x1;
    int root;
  };
  std::vector<KeyRun> keyRuns;
  if (params.scaleSpans.empty())
  {
    keyRuns.push_back({visibleStartX, visibleEndX, params.scaleRootNote});
  }
  else
  {
    for (size_t i = 0; i < params.scaleSpans.size(); ++i)
    {
      const float x0 = i == 0 ? visibleStartX
                              : std::max(visibleStartX, std::round(params.scaleSpans[i].first));
      const float x1 = i + 1 < params.scaleSpans.size()
                           ? std::min(visibleEndX, std::round(params.scaleSpans[i + 1].first))
                           : visibleEndX;
      if (x1 > x0)
        keyRuns.push_back({x0, x1, params.scaleSpans[i].second});
    }
  }
  const bool scaleModeShown =
      params.scaleMode != ScaleMode::None && params.scaleMode != ScaleMode::Chromatic;
  const auto overlayOn = [&](const KeyRun &run) { return scaleModeShown && run.root >= 0; };
  bool showScaleOverlay = false;
  for (const auto &run : keyRuns)
    showScaleOverlay = showScaleOverlay || overlayOn(run);
  const auto whiteKeyRowColour = juce::Colour(0xFF1C1C1Bu);
  const auto blackKeyRowColour = juce::Colour(0xFF151515u);

  if (params.drawRowBackgrounds)
  {
    for (int midi = startMidi; midi <= endMidi; ++midi)
    {
      const int noteInOctave = (midi % 12 + 12) % 12;
      g.setColour(pianoRollView::isBlackKey(noteInOctave) ? blackKeyRowColour
                                                          : whiteKeyRowColour);
      const float y = coordMapper->midiToY(
          static_cast<float>(midi) + params.pitchAxisOffsetSemitones);
      g.fillRect(visibleStartX, y, visibleEndX - visibleStartX,
                 pixelsPerSemitone);
    }
  }

  if (params.drawRowBackgrounds && showScaleOverlay)
  {
    const auto inScaleRowColour = whiteKeyRowColour;
    const auto outOfScaleRowColour = blackKeyRowColour;

    for (const auto &run : keyRuns)
    {
      if (!overlayOn(run))
        continue;
      for (int midi = startMidi; midi <= endMidi; ++midi)
      {
        const int noteInOctave = (midi % 12 + 12) % 12;
        const auto toneState = pianoRollView::getScaleToneState(
            params.scaleMode, noteInOctave, run.root);
        g.setColour(toneState == pianoRollView::ScaleToneState::OutOfScale
                        ? outOfScaleRowColour
                        : inScaleRowColour);
        const float y = coordMapper->midiToY(
            static_cast<float>(midi) + params.pitchAxisOffsetSemitones);
        g.fillRect(run.x0, y, run.x1 - run.x0, pixelsPerSemitone);
      }
    }
  }

  if (!params.drawGridLines)
    return;

  // Horizontal pitch lines.
  for (int midi = startMidi; midi <= endMidi; ++midi)
  {
    const float y = coordMapper->midiToY(
        static_cast<float>(midi) + params.pitchAxisOffsetSemitones);
    const int noteInOctave = (midi % 12 + 12) % 12;

    for (const auto &run : keyRuns)
    {
      if (!overlayOn(run))
      {
        g.setColour(noteInOctave == 0 ? APP_COLOR_GRID_BAR : APP_COLOR_GRID);
      }
      else
      {
        const auto toneState = pianoRollView::getScaleToneState(
            params.scaleMode, noteInOctave, run.root);
        if (toneState == pianoRollView::ScaleToneState::Root)
          g.setColour(APP_COLOR_GRID_BAR);
        else if (toneState == pianoRollView::ScaleToneState::InScale)
          g.setColour(APP_COLOR_GRID);
        else
          g.setColour(APP_COLOR_GRID.darker(0.25f));
      }
      g.drawHorizontalLine(static_cast<int>(y), run.x0, run.x1);
    }
  }

  // Where the key changes: a line down the canvas, under the flag on the ruler.
  for (size_t i = 1; i < keyRuns.size(); ++i)
  {
    g.setColour(juce::Colour(0xFF6FB6FFu).withAlpha(0.45f));
    g.drawVerticalLine(static_cast<int>(keyRuns[i].x0), visibleTopY, visibleBottomY);
  }

  if (params.timelineDisplayMode == TimelineDisplayMode::Beats)
  {
    if (params.gridSeconds > 1.0e-6 && params.beatSeconds > 1.0e-6 &&
        params.barSeconds > 1.0e-6)
    {
      // Grid lines sit on HOST beats, like the ruler above them; the canvas
      // is in project time, so each host position is mapped back through the
      // display offset.
      const double visibleStartTime =
          coordMapper->projectToTimeline(visibleStartX / pixelsPerSecond);
      const double visibleEndTime =
          coordMapper->projectToTimeline(visibleEndX / pixelsPerSecond);
      const int firstGrid = std::max(
          0,
          static_cast<int>(std::floor(visibleStartTime / params.gridSeconds)) - 1);

      for (int i = firstGrid;; ++i)
      {
        const double time = static_cast<double>(i) * params.gridSeconds;
        if (time > visibleEndTime + params.gridSeconds)
          break;

        const float x = static_cast<float>(
            coordMapper->timelineToProject(time) * pixelsPerSecond);
        if (x < visibleStartX - 1.0f || x > visibleEndX + 1.0f)
          continue;

        const bool isBarLine = pianoRollView::isMultipleOf(time, params.barSeconds);
        if (isBarLine)
        {
          g.setColour(APP_COLOR_GRID_BAR);
        }
        else if (pianoRollView::isMultipleOf(time, params.beatSeconds))
        {
          g.setColour(APP_COLOR_GRID);
        }
        else
        {
          g.setColour(APP_COLOR_GRID.darker(0.2f));
        }
        if (isBarLine)
          g.fillRect(x - 0.5f, visibleTopY, 2.0f,
                     visibleBottomY - visibleTopY);
        else
          g.drawVerticalLine(static_cast<int>(x), visibleTopY, visibleBottomY);
      }
    }
  }
  else
  {
    float secondsPerTick;
    if (pixelsPerSecond >= 200.0f)
      secondsPerTick = 0.5f;
    else if (pixelsPerSecond >= 100.0f)
      secondsPerTick = 1.0f;
    else if (pixelsPerSecond >= 50.0f)
      secondsPerTick = 2.0f;
    else if (pixelsPerSecond >= 25.0f)
      secondsPerTick = 5.0f;
    else
      secondsPerTick = 10.0f;

    const float secondsPerLine = secondsPerTick * 2.0f;
    const float pixelsPerLine = secondsPerLine * pixelsPerSecond;
    if (pixelsPerLine > 1.0e-4f)
    {
      g.setColour(APP_COLOR_GRID);
      const double startTime =
          coordMapper->projectToTimeline(visibleStartX / pixelsPerSecond);
      const double endTime =
          coordMapper->projectToTimeline(visibleEndX / pixelsPerSecond);
      const double firstTime = std::floor(startTime / secondsPerLine) * secondsPerLine;
      for (double time = firstTime; time <= endTime; time += secondsPerLine) {
        const float x = coordMapper->timeToX(coordMapper->timelineToProject(time));
        if (x >= visibleStartX)
          g.drawVerticalLine(static_cast<int>(x), visibleTopY, visibleBottomY);
      }
    }
  }
}
