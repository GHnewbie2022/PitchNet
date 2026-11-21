#pragma once

#include "../JuceHeader.h"
#include "../Models/Note.h"

/**
 * Spec Q12: the colour that stands for a note's synthesis choice, on the
 * note's outline and on its entry in the synthesis menu. The vocoder, the
 * default, has none.
 */
namespace synthEngineColour
{
inline bool has(Note::SynthEngine engine) { return engine != Note::SynthEngine::Vocoder; }

inline juce::Colour of(Note::SynthEngine engine)
{
  return engine == Note::SynthEngine::RubberBand ? juce::Colour(0xffffd60a)   // yellow
                                                 : juce::Colour(0xff40c8e0);  // cyan
}
} // namespace synthEngineColour
