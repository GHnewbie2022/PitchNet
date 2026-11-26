#pragma once

#include <vector>
#include "../../JuceHeader.h"

#include <cmath>

/**
 * The scale a key is confirmed by ear with (spec section 15): degrees 1 to 8
 * of the major scale, a sine each, played over the accompaniment. Heard
 * once, it tells whether the last note comes home, and all seven notes are
 * sung: the key a fifth up differs from the right one in its 7th degree
 * alone, which degrees 1 to 5 never reach.
 */
namespace keyfinder
{
constexpr double kScaleNoteSeconds = 0.4;
constexpr double kScaleFadeSeconds = 0.010;
constexpr float kScalePeak = 0.1259f; // -18 dBFS

/** The MIDI note of pitch class `root` whose octave lies around `aroundMidi`
 *  (the voice's average pitch): the scale's middle is nearest to it. */
inline int scaleRootNear(int root, double aroundMidi)
{
    const int pitchClass = (root % 12 + 12) % 12;
    const double wanted = aroundMidi - 6.0; // the scale spans root .. root + 12
    return pitchClass + 12 * static_cast<int>(std::lround((wanted - pitchClass) / 12.0));
}

/**
 * Degrees 1 to 8 from `rootMidi`, `tuneSemitones` off A = 440 (the
 * accompaniment's tuning, or the singer's offset without one). Mono; each
 * note starts and ends at 0.
 */
inline juce::AudioBuffer<float> scaleTone(int rootMidi, double tuneSemitones, double sampleRate,
                                          float peak = kScalePeak,
                                          std::vector<int> degrees = {0, 2, 4, 5, 7, 9, 11, 12})
{
    // `degrees`: the scale's notes in semitones from the root, up to the
    // octave (a mode other than major, with five to eight notes).
    const int notes = static_cast<int>(degrees.size());
    const int noteSamples = static_cast<int>(std::lround(kScaleNoteSeconds * sampleRate));
    const int fadeSamples = std::max(1, static_cast<int>(std::lround(kScaleFadeSeconds * sampleRate)));
    juce::AudioBuffer<float> tone(1, std::max(1, notes) * noteSamples);
    float* out = tone.getWritePointer(0);
    constexpr double twoPi = 6.283185307179586;
    for (int n = 0; n < notes; ++n)
    {
        const double hz = 440.0 * std::pow(2.0, (rootMidi + degrees[static_cast<size_t>(n)] + tuneSemitones - 69.0) / 12.0);
        for (int i = 0; i < noteSamples; ++i)
        {
            const int fromEdge = std::min(i, noteSamples - 1 - i);
            const double fade = fromEdge >= fadeSamples
                                    ? 1.0
                                    : 0.5 - 0.5 * std::cos(twoPi * 0.5 * fromEdge / fadeSamples);
            out[n * noteSamples + i] =
                static_cast<float>(peak * fade * std::sin(twoPi * hz * i / sampleRate));
        }
    }
    return tone;
}
} // namespace keyfinder
