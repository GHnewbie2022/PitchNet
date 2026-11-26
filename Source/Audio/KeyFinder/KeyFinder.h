#pragma once

#include "AccompanimentChroma.h"
#include "../../Models/Project.h"

#include <array>
#include <vector>

/**
 * The key of a song, from its accompaniment and from what is sung, for the
 * user to confirm by ear (spec sections 14 and 15).
 *
 * A key here is a major scale, named by its root: the seven notes are what
 * tuning needs, and a minor song is stored as its relative major (the user's
 * decision: always major).
 *
 * Neither source is enough alone. A melody on five notes (煙幕: C D E G A,
 * 96 % of the time) fits a scale and its two neighbours on the circle of
 * fifths alike, which the accompaniment tells apart; an accompaniment read
 * 10 s at a time leans to a neighbour here and there (六等星: B in 6 of 25
 * windows) where the voice is plain about it. So both are read over the same
 * windows and their evidence added:
 *
 *   accompaniment  the share of its energy on the scale's notes, in points
 *                  below the window's best scale (AccompanimentChroma.h);
 *   voice          how far the notes are from the scale's nearest note on
 *                  average, by their length, in cents above the window's
 *                  best scale, after the singer's overall offset is taken
 *                  out (the median distance of the notes from a semitone:
 *                  +18 cents in 煙幕, where taking it out brought the notes
 *                  that pick the guide's target from 553 to 570 of 574).
 *
 * A window weighs as much as it holds (the accompaniment's energy, the
 * length sung) against the song's average window. With no change of key the
 * sums over the windows are then the whole song's own share and distance,
 * which is what was measured: the accompaniment's first scale right in 3
 * songs of 3, the voice's in 178 of 180 recordings of CSD.
 *
 * A note whose centre the user moved by hand is where the user says the
 * tune is: it is taken where it was put, with no offset, and a scale that
 * lacks it is 100 cents off for that note. It has no part in the offset.
 *
 * The key of each window is then the path through the windows that costs
 * least, a change of key costing kChangeCost: a song that modulates (晚餐歌,
 * up a semitone for the last minute) is cut there, and two windows leaning to
 * a neighbour are not. Each stretch between changes gives its two likeliest
 * keys, the best first.
 */
namespace keyfinder
{
constexpr double kWindowSeconds = 10.0;
constexpr int kMinNoteFrames = 9;      // notes shorter than 0.1 s are not counted
constexpr double kPointsPerCent = 1.0; // a point of accompaniment against a cent of voice
/** What a change of key costs, in the same points: see KeyFinder.cpp. */
extern const double kChangeCost;

struct VocalNote
{
    double startSeconds = 0.0; // on the timeline
    double weight = 0.0;       // its length as sung, in frames
    double pitch = 0.0;        // MIDI: the centre as analysed, or as put by hand
    bool placed = false;       // put by hand
};

/** The project's notes as evidence: sounding ones of kMinNoteFrames or more. */
std::vector<VocalNote> vocalNotes(const Project& project);

/** The singer's overall offset from the semitones, in semitones (-0.5 .. 0.5):
 *  the median over the notes not placed by hand. 0 without any. */
double vocalOffset(const std::vector<VocalNote>& notes);

/** For each major scale, how far the notes are from its nearest note on
 *  average (semitones, by weight), `offset` taken out of the notes not placed
 *  by hand. False when no note is given. */
bool vocalFit(const std::vector<VocalNote>& notes, double offset, std::array<double, 12>& fit);

struct Window
{
    double startSeconds = 0.0;
    bool hasBacking = false, hasVoice = false;
    std::array<double, 12> backingShare{}; // 0..1, by root
    std::array<double, 12> voiceFit{};     // semitones, by root
    std::array<double, 12> cost{};         // points above the window's best
    int notes = 0;
    int key = -1;                          // the root on the cheapest path
};

struct Stretch
{
    double startSeconds = 0.0, endSeconds = 0.0;
    int key = 0;       // the likeliest root
    int second = 0;    // the next
    double lead = 0.0; // points a window, the second above the first
};

struct Result
{
    double vocalOffsetSemitones = 0.0;
    double tuneSemitones = 0.0; // the accompaniment's, 0 without one
    bool hasBacking = false;
    std::vector<Window> windows;
    std::vector<Stretch> stretches; // one or more when there is any evidence
    // Major, or the minor of the same seven notes (its tonic a minor third
    // below): how the song's pitch classes - the accompaniment's energy and
    // the length sung, each as a share - fit the two key profiles
    // (Krumhansl and Kessler), over the keys found. A stretch's `key` and
    // `second` stay the major roots of their seven notes either way.
    bool minor = false;
    double majorFit = 0.0, minorFit = 0.0; // the two correlations
};

/** `backing` may be null. Its time 0 is the timeline's. */
Result find(const std::vector<VocalNote>& notes, const Chroma* backing);

/** The same with another cost for a change of key: to measure with. */
Result find(const std::vector<VocalNote>& notes, const Chroma* backing, double changeCost);

inline Result find(const Project& project, const Chroma* backing)
{
    return find(vocalNotes(project), backing);
}
} // namespace keyfinder
