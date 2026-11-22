#pragma once

#include "../JuceHeader.h"
#include "../Models/Note.h"

/**
 * Spec Q12: the colour that stands for a note's synthesis choice, on the
 * note's outline and on its entry in the synthesis menu. Only Rubber Band R3
 * has one: vocoder + recorded highs, what most notes are rendered with, is
 * unmarked, so a colour on screen means a note that differs from it. Pass
 * what the note is rendered with (fry::resolved), not a stored Auto.
 */
namespace synthEngineColour
{
inline bool has(Note::SynthEngine engine) { return engine == Note::SynthEngine::RubberBand; }

inline juce::Colour of(Note::SynthEngine engine)
{
  juce::ignoreUnused(engine);
  return juce::Colour(0xffffd60a); // yellow
}
} // namespace synthEngineColour
