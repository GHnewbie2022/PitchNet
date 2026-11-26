// The key of a song for the wizard (spec sections 14 and 15), with the calls
// the App makes: keyfinder::analyse on the accompaniment, vocalNotes on the
// project, find on both, scaleTone for the scale that is heard.
#include "TestAssert.h"

#include "../Source/Audio/KeyFinder/KeyFinder.h"
#include "../Source/Audio/KeyFinder/ScaleTone.h"
#include "../Source/Utils/Constants.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kRate = 12000.0; // analyse() works at this rate: nothing to convert

// Triads of the major scale on `root` (MIDI), 2 s each: I IV V vi ii iii,
// `rounds` times. Each note five harmonics, a bass an octave under the root.
// `tuneSemitones` shifts everything; `hits` adds a burst of noise every 0.5 s.
juce::AudioBuffer<float> progression(int root, int rounds, double tuneSemitones = 0.0, bool hits = false)
{
    static const int chords[6][3] = {{0, 4, 7}, {5, 9, 12}, {7, 11, 14}, {9, 12, 16}, {2, 5, 9}, {4, 7, 11}};
    const int chordSamples = static_cast<int>(2.0 * kRate);
    juce::AudioBuffer<float> audio(1, 6 * rounds * chordSamples);
    audio.clear();
    float* out = audio.getWritePointer(0);
    const auto addNote = [&](int start, double midi, double level)
    {
        const double hz = 440.0 * std::pow(2.0, (midi + tuneSemitones - 69.0) / 12.0);
        for (int h = 1; h <= 5; ++h)
            for (int i = 0; i < chordSamples; ++i)
            {
                const double fade = std::min(1.0, std::min(i, chordSamples - 1 - i) / 240.0);
                out[start + i] += static_cast<float>(level / h * fade *
                                                     std::sin(2.0 * kPi * hz * h * i / kRate));
            }
    };
    for (int r = 0; r < rounds; ++r)
        for (int c = 0; c < 6; ++c)
        {
            const int start = (r * 6 + c) * chordSamples;
            for (const int degree : chords[c])
                addNote(start, root + degree, 0.2);
            addNote(start, root + chords[c][0] - 12, 0.3);
        }
    if (hits)
    {
        juce::Random random(7);
        for (int at = 0; at + 360 < audio.getNumSamples(); at += static_cast<int>(0.5 * kRate))
            for (int i = 0; i < 360; ++i)
                out[at + i] += 0.8f * (random.nextFloat() * 2.0f - 1.0f) * (1.0f - i / 360.0f);
    }
    return audio;
}

int best(const std::array<double, 12>& values, bool highest)
{
    int at = 0;
    for (int r = 1; r < 12; ++r)
        if (highest ? values[static_cast<size_t>(r)] > values[static_cast<size_t>(at)]
                    : values[static_cast<size_t>(r)] < values[static_cast<size_t>(at)])
            at = r;
    return at;
}

std::array<double, 12> wholeShares(const keyfinder::Chroma& chroma)
{
    return keyfinder::scaleShares(keyfinder::sum(chroma, 0, static_cast<int>(chroma.frames.size())));
}

// A chroma made by hand: `windows` of 10 s, each with the major scale on
// keys[k] at full level and the other five classes at `floor`.
keyfinder::Chroma chromaOf(const std::vector<int>& keys, double floor = 0.05)
{
    keyfinder::Chroma chroma;
    chroma.frameSeconds = 1024.0 / 12000.0;
    const int perWindow = static_cast<int>(std::lround(keyfinder::kWindowSeconds / chroma.frameSeconds));
    for (const int key : keys)
        for (int f = 0; f < perWindow; ++f)
        {
            std::array<float, 12> frame{};
            for (int c = 0; c < 12; ++c)
                frame[static_cast<size_t>(c)] = keyfinder::inMajorScale(c, key) ? 1.0f : static_cast<float>(floor);
            chroma.frames.push_back(frame);
        }
    return chroma;
}

// Notes of 0.3 s on `pitches` in turn, every half second from `startSeconds`.
std::vector<keyfinder::VocalNote> sing(const std::vector<double>& pitches, int count, double startSeconds = 0.0)
{
    std::vector<keyfinder::VocalNote> notes;
    for (int i = 0; i < count; ++i)
    {
        keyfinder::VocalNote note;
        note.startSeconds = startSeconds + 0.5 * i;
        note.weight = 26.0;
        note.pitch = pitches[static_cast<size_t>(i) % pitches.size()];
        notes.push_back(note);
    }
    return notes;
}

void theAccompanimentsScaleIsRead()
{
    const auto chroma = keyfinder::analyse(progression(48, 2), kRate);
    CHECK(!chroma.empty());
    CHECK(std::abs(chroma.tuneSemitones) <= 0.02);
    const auto shares = wholeShares(chroma);
    CHECK(best(shares, true) == 0);
    CHECK(shares[0] > shares[5] + 0.02 && shares[0] > shares[7] + 0.02);
    std::printf("C major progression: tune %+.1f cents, C %.1f%%, F %.1f%%, G %.1f%%\n",
                100.0 * chroma.tuneSemitones, 100.0 * shares[0], 100.0 * shares[5], 100.0 * shares[7]);
}

void aRecordingTunedOffIsReadOnItsOwnGrid()
{
    const auto chroma = keyfinder::analyse(progression(48, 2, 0.34), kRate);
    CHECK(std::abs(chroma.tuneSemitones - 0.34) <= 0.02);
    const auto shares = wholeShares(chroma);
    CHECK(best(shares, true) == 0);
    // Read on the 440 grid it would be worse: the notes sit a third of a
    // semitone off the middle bins.
    const auto straight = wholeShares(keyfinder::analyse(progression(48, 2), kRate));
    CHECK(std::abs(shares[0] - straight[0]) < 0.02);
    std::printf("tuned +34 cents: measured %+.1f, C %.1f%%\n", 100.0 * chroma.tuneSemitones, 100.0 * shares[0]);
}

void drumsDoNotChangeTheScale()
{
    const auto plain = wholeShares(keyfinder::analyse(progression(50, 2), kRate));
    const auto hit = wholeShares(keyfinder::analyse(progression(50, 2, 0.0, true), kRate));
    CHECK(best(plain, true) == 2);
    CHECK(best(hit, true) == 2);
    std::printf("D major with hits: D %.1f%% (without %.1f%%)\n", 100.0 * hit[2], 100.0 * plain[2]);
}

void theSingersOffsetIsTakenOut()
{
    auto notes = sing({60.18, 62.18, 64.18, 65.18, 67.18, 69.18, 71.18}, 70);
    const double offset = keyfinder::vocalOffset(notes);
    CHECK(std::abs(offset - 0.18) <= 0.01);
    std::array<double, 12> fit{};
    CHECK(keyfinder::vocalFit(notes, offset, fit));
    CHECK(best(fit, false) == 0);
    CHECK(fit[0] < 1.0e-6);
    // Without it the notes are 18 cents from the scale.
    CHECK(keyfinder::vocalFit(notes, 0.0, fit));
    CHECK(std::abs(fit[0] - 0.18) < 1.0e-6);
    const auto result = keyfinder::find(notes, nullptr);
    CHECK(result.stretches.size() == 1 && result.stretches.front().key == 0);
}

void fiveNotesLeaveItToTheAccompaniment()
{
    const auto notes = sing({60.0, 62.0, 64.0, 67.0, 69.0}, 120);
    std::array<double, 12> fit{};
    CHECK(keyfinder::vocalFit(notes, keyfinder::vocalOffset(notes), fit));
    CHECK(std::abs(fit[0] - fit[5]) < 1.0e-9 && std::abs(fit[0] - fit[7]) < 1.0e-9);
    for (const int key : {0, 5, 7})
    {
        const auto chroma = chromaOf(std::vector<int>(6, key));
        const auto result = keyfinder::find(notes, &chroma);
        CHECK(result.stretches.size() == 1);
        CHECK(result.stretches.front().key == key);
    }
}

void aNotePutByHandIsOfTheKey()
{
    // Five notes, sung 18 cents high: C, F and G fit alike.
    auto notes = sing({60.18, 62.18, 64.18, 67.18, 69.18}, 120);
    auto result = keyfinder::find(notes, nullptr);
    const double offset = result.vocalOffsetSemitones;
    // An F put by hand, on the semitone: G major has no F.
    keyfinder::VocalNote placed;
    placed.startSeconds = 3.2;
    placed.weight = 40.0;
    placed.pitch = 65.0;
    placed.placed = true;
    notes.push_back(placed);
    result = keyfinder::find(notes, nullptr);
    CHECK(std::abs(result.vocalOffsetSemitones - offset) < 1.0e-12); // no part in the offset
    CHECK(result.stretches.size() == 1);
    CHECK(result.stretches.front().key != 7 && result.stretches.front().second != 7);
    std::array<double, 12> fit{};
    CHECK(keyfinder::vocalFit(notes, result.vocalOffsetSemitones, fit));
    CHECK(fit[0] < 1.0e-6 && fit[5] < 1.0e-6); // taken where it was put, not 18 cents under
    CHECK(fit[7] > 0.005);
    // ...and a B as well: F major has none. C is left.
    placed.startSeconds = 8.2;
    placed.pitch = 71.0;
    notes.push_back(placed);
    result = keyfinder::find(notes, nullptr);
    CHECK(result.stretches.front().key == 0);
}

void aSongThatModulatesIsCutThere()
{
    std::vector<int> keys(9, 8); // G# for 90 s
    keys.insert(keys.end(), 5, 9); // then A for 50 s
    const auto chroma = chromaOf(keys);
    // All seven notes of each key: sung alone they name it.
    auto notes = sing({68.0, 70.0, 72.0, 73.0, 75.0, 77.0, 79.0}, 170);
    const auto later = sing({69.0, 71.0, 73.0, 74.0, 76.0, 78.0, 80.0}, 90, 90.0);
    notes.insert(notes.end(), later.begin(), later.end());
    for (const auto* backing : {&chroma, static_cast<const keyfinder::Chroma*>(nullptr)})
    {
        const auto result = keyfinder::find(notes, backing);
        CHECK(result.stretches.size() == 2);
        CHECK(result.stretches[0].key == 8 && result.stretches[1].key == 9);
        CHECK(std::abs(result.stretches[1].startSeconds - 90.0) <= keyfinder::kWindowSeconds);
    }
}

void twoWindowsLeaningToANeighbourAreNotAChange()
{
    // F# throughout, but for two windows the accompaniment has E where F# has
    // E# (as B major does), a few points' worth: 六等星 at 0:00, 1:39, 2:49.
    auto chroma = chromaOf(std::vector<int>(14, 6));
    const int perWindow = static_cast<int>(chroma.frames.size()) / 14;
    for (int f = 5 * perWindow; f < 7 * perWindow; ++f)
    {
        chroma.frames[static_cast<size_t>(f)][5] = 0.25f; // E#
        chroma.frames[static_cast<size_t>(f)][4] = 0.55f; // E
    }
    const auto leaning = keyfinder::scaleShares(keyfinder::sum(chroma, 5 * perWindow, 6 * perWindow));
    CHECK(leaning[11] > leaning[6] + 0.02); // B ahead in those windows
    const auto notes = sing({66.0, 68.0, 70.0, 73.0, 75.0}, 270);
    auto result = keyfinder::find(notes, &chroma);
    CHECK(result.stretches.size() == 1 && result.stretches.front().key == 6);
    result = keyfinder::find(std::vector<keyfinder::VocalNote>{}, &chroma);
    CHECK(result.stretches.size() == 1 && result.stretches.front().key == 6);
    std::printf("two windows leaning: B %.1f%% against F# %.1f%% there, not cut\n", 100.0 * leaning[11],
                100.0 * leaning[6]);
}

void theScaleThatIsHeard()
{
    for (const double tune : {0.0, 0.34})
    {
        const double rate = 48000.0;
        const auto tone = keyfinder::scaleTone(60, tune, rate);
        const int noteSamples = static_cast<int>(std::lround(keyfinder::kScaleNoteSeconds * rate));
        CHECK(tone.getNumChannels() == 1 && tone.getNumSamples() == 8 * noteSamples);
        CHECK(std::abs(tone.getNumSamples() / rate - 3.2) < 1.0e-9);
        const float* x = tone.getReadPointer(0);
        static const int degrees[8] = {0, 2, 4, 5, 7, 9, 11, 12};
        for (int n = 0; n < 8; ++n)
        {
            CHECK(std::abs(x[n * noteSamples]) < 1.0e-6f);
            CHECK(std::abs(x[(n + 1) * noteSamples - 1]) < 1.0e-6f);
            // Rising zero crossings over the middle 0.3 s.
            const int from = n * noteSamples + static_cast<int>(0.05 * rate);
            const int to = n * noteSamples + static_cast<int>(0.35 * rate);
            double first = -1.0, last = -1.0;
            int crossings = 0;
            for (int i = from; i < to; ++i)
                if (x[i] <= 0.0f && x[i + 1] > 0.0f)
                {
                    const double at = i + x[i] / (x[i] - x[i + 1]);
                    if (first < 0.0)
                        first = at;
                    last = at;
                    ++crossings;
                }
            CHECK(crossings > 10);
            const double hz = (crossings - 1) * rate / (last - first);
            const double wanted = 440.0 * std::pow(2.0, (60 + degrees[n] + tune - 69.0) / 12.0);
            CHECK(std::abs(1200.0 * std::log2(hz / wanted)) < 1.0);
        }
        CHECK(tone.getMagnitude(0, tone.getNumSamples()) <= keyfinder::kScalePeak + 1.0e-6f);
        CHECK(tone.getMagnitude(0, tone.getNumSamples()) > 0.99f * keyfinder::kScalePeak);
    }
    // The octave around the voice: a scale on C for a voice around 64 starts
    // at 60, around 70 still at 60 (60..72), around 73 at 72.
    CHECK(keyfinder::scaleRootNear(0, 64.0) == 60);
    CHECK(keyfinder::scaleRootNear(0, 70.0) == 60);
    CHECK(keyfinder::scaleRootNear(0, 73.0) == 72);
    CHECK(keyfinder::scaleRootNear(9, 64.0) == 57);
}

void whichNotesOfAProjectCount()
{
    Project project;
    const auto add = [&](int start, int length, float midi) -> Note&
    {
        project.addNote(Note(start, start + length, midi));
        return project.getNotes().back();
    };
    add(0, 30, 60.2f);                               // as sung
    add(40, 30, 61.8f).setMidiNote(62.0f);           // dragged
    add(80, 30, 63.7f).setMidiNoteFromPitchCorrection(64.0f); // the Pitch Center macro alone
    add(120, 30, 65.3f).setTiltLeft(0.4f);           // a tilt alone
    add(160, 30, 66.6f).setPitchOffset(0.4f);        // offset
    add(200, 8, 70.0f);                              // too short
    add(220, 30, 72.0f).setRest(true);               // a rest
    const auto notes = keyfinder::vocalNotes(project);
    CHECK(notes.size() == 5);
    CHECK(!notes[0].placed && std::abs(notes[0].pitch - 60.2) < 1.0e-5);
    CHECK(notes[1].placed && std::abs(notes[1].pitch - 62.0) < 1.0e-5);
    CHECK(!notes[2].placed && std::abs(notes[2].pitch - 63.7) < 1.0e-5);
    CHECK(!notes[3].placed && std::abs(notes[3].pitch - 65.3) < 1.0e-5);
    CHECK(notes[4].placed && std::abs(notes[4].pitch - 67.0) < 1.0e-5);
    CHECK(std::abs(notes[1].startSeconds - 40.0 * HOP_SIZE / SAMPLE_RATE) < 1.0e-9);
    CHECK(notes[0].weight == 30.0);
}
} // namespace

int main()
{
    juce::ScopedJuceInitialiser_GUI juce;
    theAccompanimentsScaleIsRead();
    aRecordingTunedOffIsReadOnItsOwnGrid();
    drumsDoNotChangeTheScale();
    theSingersOffsetIsTakenOut();
    fiveNotesLeaveItToTheAccompaniment();
    aNotePutByHandIsOfTheKey();
    aSongThatModulatesIsCutThere();
    twoWindowsLeaningToANeighbourAreNotAChange();
    theScaleThatIsHeard();
    whichNotesOfAProjectCount();
    std::printf("KeyFinderTests: all passed\n");
    return 0;
}
