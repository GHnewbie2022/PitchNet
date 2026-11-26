#pragma once

#include "../../JuceHeader.h"

#include <array>
#include <vector>

/**
 * What an accompaniment plays, by pitch class and over time (spec section 14):
 * the first half of finding the key of a song from its backing track.
 *
 * Counting the energy of FFT bins by the note each is nearest to does not
 * work: equally spaced bins give the twelve pitch classes unequal numbers of
 * bins (between 110 and 440 Hz F# got one more than F, the two notes that
 * tell C major from G major), and under 110 Hz, where a bass plays the
 * chords' roots, some classes got none. So:
 *
 *   - the audio is brought to 12 kHz mono, and read with long windows (8192
 *     samples, 1.46 Hz a bin): a semitone at 55 Hz is 3.3 Hz wide;
 *   - how far the recording is tuned from A = 440 is measured first, and the
 *     grid of notes is laid on that;
 *   - the spectrum is read on a logarithmic grid, three bins to a semitone,
 *     every bin weighted alike, and the middle bin of each semitone is what
 *     counts for its note;
 *   - what does not hold still in time (drums) is taken out, by the medians
 *     along time and along frequency (harmonic-percussive separation).
 *
 * The share of the energy a scale's seven notes hold is then the measure of
 * that scale (scaleShares): with 煙幕 C 92.1 %, F 89.1 %, G 83.1 %, the whole
 * song and every 10 s of it. Taking the harmonics out as well (a non-negative
 * fit of note spectra) and correlating with key profiles both read that song
 * as F, and are not done.
 *
 * Written after the Python it was tried in (試聽工具, key_auto.py): the same
 * steps and numbers, so that the two can be compared song for song.
 */
namespace keyfinder
{
struct Chroma
{
    double tuneSemitones = 0.0; // the recording's tuning from A = 440: -0.5 .. 0.5
    double frameSeconds = 0.0;  // from one frame to the next
    std::vector<std::array<float, 12>> frames; // pitch classes C .. B, drums taken out

    double duration() const { return frameSeconds * static_cast<double>(frames.size()); }
    bool empty() const { return frames.empty(); }
};

/** Reads `audio` (any rate; its channels are averaged). Empty when it is
 *  shorter than one window (0.68 s). */
Chroma analyse(const juce::AudioBuffer<float>& audio, double sampleRate);

/** The pitch classes summed over frames [first, last). */
std::array<double, 12> sum(const Chroma& chroma, int first, int last);

/** Of what `classes` hold, the share that lies on each major scale's seven
 *  notes, by the scale's root (0 = C). All 0 when there is nothing. */
std::array<double, 12> scaleShares(const std::array<double, 12>& classes);

inline bool inMajorScale(int pitchClass, int root)
{
    static constexpr bool member[12] = {true,  false, true,  false, true,  true,
                                        false, true,  false, true,  false, true};
    return member[((pitchClass - root) % 12 + 12) % 12];
}
} // namespace keyfinder
