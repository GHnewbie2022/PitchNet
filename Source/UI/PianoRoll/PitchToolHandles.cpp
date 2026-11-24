#include "PitchToolHandles.h"
#include "../SynthEngineColours.h"
#include "BinaryData.h"

#include <algorithm>
#include <limits>

namespace {

juce::Image loadIcon(const char* resourceName)
{
  int size = 0;
  const auto* data = BinaryData::getNamedResource(resourceName, size);
  return data ? juce::ImageFileFormat::loadFrom(data, size) : juce::Image();
}

} // namespace

PitchToolHandles::PitchToolHandles()
    : leftTiltIcon(loadIcon("ltilt_png")),
      vibratoIcon(loadIcon("vibrato_png")),
      driftIcon(loadIcon("drift_png")),
      centIcon(loadIcon("cent_png")),
      formantIcon(loadIcon("formant_png")),
      amplitudeIcon(loadIcon("amplitude_png")),
      sibilantIcon(loadIcon("sibilant_png")),
      rightTiltIcon(loadIcon("rtilt_png")) {
  // Initialize (currently empty, but reserve space)
  handles.reserve(20);  // Typical max handles for multi-note selection
}

void PitchToolHandles::updateHandles(const std::vector<Note*>& selectedNotes,
                                     const CoordinateMapper& mapper,
                                     const juce::Rectangle<float>& hoverBounds,
                                     const KeepMarks& keep) {
  handles.clear();
  pivotGuide = {};

  if (selectedNotes.empty())
    return;

  int minStartFrame = std::numeric_limits<int>::max();
  int maxEndFrame = std::numeric_limits<int>::min();
  float maxMidi = std::numeric_limits<float>::lowest();

  for (const auto* note : selectedNotes) {
    if (!note)
      continue;

    minStartFrame = std::min(minStartFrame, note->getStartFrame());
    maxEndFrame = std::max(maxEndFrame, note->getEndFrame());
    maxMidi = std::max(maxMidi, note->getMidiNote());
  }

  if (minStartFrame == std::numeric_limits<int>::max())
    return;

  const float leftX = mapper.timeToX(mapper.framesToSeconds(minStartFrame));
  const float rightX = mapper.timeToX(mapper.framesToSeconds(maxEndFrame));
  const float centerX = (leftX + rightX) * 0.5f;
  const float topY = mapper.midiToY(maxMidi);
  auto* targetNote = selectedNotes.front();
  // Use the actual rendered hover bounds when supplied by the piano roll.
  // Fall back to the note body plus its normal hover padding otherwise.
  constexpr float hoverPadding = 4.0f;
  const float fallbackWidth =
      std::max(rightX - leftX, 4.0f) + hoverPadding * 2.0f;
  const float layoutWidth = hoverBounds.isEmpty()
      ? std::max(fallbackWidth, buttonGroupWidth)
      : hoverBounds.getWidth();
  const float groupLeft = hoverBounds.isEmpty()
      ? centerX - layoutWidth * 0.5f
      : hoverBounds.getX();
  const float groupTop = hoverBounds.isEmpty()
      ? topY - hoverPadding - buttonHeight - 7.0f
      : hoverBounds.getY() - buttonHeight - 7.0f;
  const float slotWidth = (layoutWidth - buttonGap * 2.0f) / 3.0f;
  const float bottom = hoverBounds.isEmpty()
      ? topY + mapper.getPixelsPerSemitone() + hoverPadding
      : hoverBounds.getBottom();
  const float bottomRowY = bottom + 7.0f;
  const auto addButtonHandle = [this, targetNote, groupLeft, slotWidth](
      HandleType type, int column, float y)
  {
    const float x = groupLeft + (slotWidth + buttonGap) * column;
    addHandle(type, x + slotWidth * 0.5f, y + buttonHeight * 0.5f,
              targetNote);
    handles.back().bounds = {x, y, slotWidth, buttonHeight};
  };

  // Pitch above the note, level and tone below it.
  addButtonHandle(HandleType::PitchDrift, 0, groupTop);
  addButtonHandle(HandleType::Vibrato, 1, groupTop);
  addButtonHandle(HandleType::FinePitch, 2, groupTop);
  addButtonHandle(HandleType::SibilantBalance, 0, bottomRowY);
  addButtonHandle(HandleType::Amplitude, 1, bottomRowY);
  addButtonHandle(HandleType::Formant, 2, bottomRowY);

  // Cubase's tilt: squares on the body's top corners tilt one side about the
  // pivot (Alt rotates the whole note about it); the diamond above is the
  // pivot itself.
  if (targetNote)
  {
    constexpr float corner = cornerSize, diamond = 11.0f;
    const float bodyTop = hoverBounds.isEmpty() ? topY - hoverPadding : hoverBounds.getY();
    const float bodyLeft = hoverBounds.isEmpty() ? leftX - hoverPadding : hoverBounds.getX();
    const float bodyRight = hoverBounds.isEmpty() ? rightX + hoverPadding : hoverBounds.getRight();
    addHandle(HandleType::TiltLeft, bodyLeft, bodyTop, targetNote);
    handles.back().bounds = {bodyLeft, bodyTop, corner, corner};
    addHandle(HandleType::TiltRight, bodyRight, bodyTop, targetNote);
    handles.back().bounds = {bodyRight - corner, bodyTop, corner, corner};

    const float noteLeft = mapper.timeToX(mapper.framesToSeconds(targetNote->getStartFrame()));
    const float noteRight = mapper.timeToX(mapper.framesToSeconds(targetNote->getEndFrame()));
    const float pivotX = noteLeft + (noteRight - noteLeft) * targetNote->getTiltPivot();
    const float diamondY = groupTop - diamond - 3.0f;
    addHandle(HandleType::TiltPivot, pivotX, diamondY, targetNote);
    handles.back().bounds = {pivotX - diamond * 0.5f, diamondY, diamond, diamond};
    pivotGuide = {pivotX, diamondY + diamond, pivotX, bottom};

    // The stretches kept as recorded by hand: a square on the body's lower
    // edge at each of the two places, centred on it - with nothing kept, on
    // the note's own two corners, under the tilt squares. Where the two
    // would overlap they stand side by side about the middle instead, so
    // that they touch when the stretches meet (all of the note is kept) and
    // each can still be taken hold of.
    if (keep.shown)
    {
      const float y = bottom - corner;
      float headLeft = keep.headX - corner * 0.5f;
      float tailLeft = keep.tailX - corner * 0.5f;
      if (tailLeft < headLeft + corner)
      {
        const float middle = (keep.headX + keep.tailX) * 0.5f;
        headLeft = middle - corner;
        tailLeft = middle;
      }
      addHandle(HandleType::KeepHead, headLeft, y, targetNote);
      handles.back().bounds = {headLeft, y, corner, corner};
      addHandle(HandleType::KeepTail, tailLeft, y, targetNote);
      handles.back().bounds = {tailLeft, y, corner, corner};
    }
  }
}

void PitchToolHandles::draw(juce::Graphics& g) const {
  for (int i = 0; i < static_cast<int>(handles.size()); ++i) {
    const auto& handle = handles[i];

    const bool hovered = i == hoveredHandleIndex;
    const auto bounds = handle.bounds;
    if (handle.type == HandleType::TiltLeft || handle.type == HandleType::TiltRight)
    {
      g.setColour(hovered ? juce::Colour(0xFF8A8A8Au) : juce::Colour(0xFF2E2E2Du));
      g.fillRect(bounds);
      g.setColour(juce::Colour(0xFF9A9A9Au));
      g.drawRect(bounds, 1.0f);
      continue;
    }
    if (handle.type == HandleType::KeepHead || handle.type == HandleType::KeepTail)
    {
      // The green of what they keep (NoteRenderer's fill), dark-edged so
      // they stand out on it.
      const auto green = synthEngineColour::keptFry();
      g.setColour(hovered ? green.brighter(0.6f) : green);
      g.fillRect(bounds);
      g.setColour(juce::Colour(0xFF1E3A0Au));
      g.drawRect(bounds, 1.0f);
      continue;
    }
    if (handle.type == HandleType::TiltPivot)
    {
      juce::Path diamond;
      const auto c = bounds.getCentre();
      const float r = bounds.getWidth() * 0.5f;
      diamond.addQuadrilateral(c.x, c.y - r, c.x + r, c.y, c.x, c.y + r, c.x - r, c.y);
      g.setColour(hovered ? juce::Colour(0xFFCFCFCFu) : juce::Colour(0xFF9A9A9Au));
      g.fillPath(diamond);
      continue;
    }
    g.setColour(hovered ? juce::Colour(0xFF3B3B3Au)
                        : juce::Colour(0xFF2E2E2Du));

    g.fillRect(bounds);

    const auto& icon = getIconForType(handle.type);
    if (icon.isValid())
    {
      const float iconWidth = icon.getWidth() * 0.5f;
      const float iconHeight = icon.getHeight() * 0.5f;
      g.drawImage(icon, bounds.getCentreX() - iconWidth * 0.5f,
                  bounds.getCentreY() - iconHeight * 0.5f + 1.0f,
                  iconWidth, iconHeight, 0, 0, icon.getWidth(),
                  icon.getHeight());
    }
  }
}

const juce::Image&
PitchToolHandles::getIconForType(HandleType type) const {
  switch (type) {
    case HandleType::TiltLeft:
      return leftTiltIcon;
    case HandleType::PitchDrift:
      return driftIcon;
    case HandleType::FinePitch:
      return centIcon;
    case HandleType::Vibrato:
      return vibratoIcon;
    case HandleType::Formant:
      return formantIcon;
    case HandleType::Amplitude:
      return amplitudeIcon;
    case HandleType::SibilantBalance:
      return sibilantIcon;
    case HandleType::TiltRight:
      return rightTiltIcon;
    default:
      return vibratoIcon;
  }
}

int PitchToolHandles::hitTest(float worldX, float worldY, float tolerance) const {
  for (int i = 0; i < static_cast<int>(handles.size()); ++i) {
    if (handles[i].bounds.contains(worldX, worldY)) {
      return i;
    }
  }
  for (int i = 0; i < static_cast<int>(handles.size()); ++i) {
    if (handles[i].bounds.expanded(tolerance * 0.25f).contains(worldX, worldY)) {
      return i;
    }
  }
  return -1;
}

bool PitchToolHandles::containsLayoutPoint(float worldX, float worldY) const
{
  // Combine handles within each row, not across the note between rows.
  // The tilt squares and the pivot stand alone: a row through the corner
  // squares would cover the note's top edge.
  const auto standsAlone = [](HandleType type)
  {
    return type == HandleType::TiltLeft || type == HandleType::TiltRight ||
           type == HandleType::TiltPivot || type == HandleType::KeepHead ||
           type == HandleType::KeepTail;
  };
  for (const auto& handle : handles)
  {
    if (standsAlone(handle.type))
    {
      if (handle.bounds.contains(worldX, worldY))
        return true;
      continue;
    }
    auto rowBounds = handle.bounds;
    for (const auto& other : handles)
      if (!standsAlone(other.type) && other.bounds.getY() == handle.bounds.getY())
        rowBounds = rowBounds.getUnion(other.bounds);
    if (rowBounds.contains(worldX, worldY))
      return true;
  }
  return false;
}

juce::Rectangle<float> PitchToolHandles::getLayoutBounds() const
{
  if (handles.empty())
    return {};

  auto layoutBounds = handles.front().bounds;
  for (size_t i = 1; i < handles.size(); ++i)
    layoutBounds = layoutBounds.getUnion(handles[i].bounds);
  return layoutBounds;
}

void PitchToolHandles::addHandle(HandleType type, float worldX, float worldY, Note* note) {
  Handle h;
  h.type = type;
  h.note = note;
  h.color = getColorForType(type);
  
  h.bounds = juce::Rectangle<float>(worldX, worldY, 0.0f, 0.0f);
  
  handles.push_back(h);
}

juce::Colour PitchToolHandles::getColorForType(HandleType type) const {
  switch (type) {
    case HandleType::TiltLeft:
    case HandleType::TiltRight:
      return juce::Colours::orange;
      
    case HandleType::PitchDrift:
    case HandleType::FinePitch:
    case HandleType::Vibrato:
      return juce::Colours::mediumpurple;
      
    case HandleType::SmoothLeft:
    case HandleType::SmoothRight:
      return juce::Colours::cyan; // "Smooth" implies liquid/soft -> cyan/blue
      
    default:
      return juce::Colours::white;
  }
}
