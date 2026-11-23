#pragma once

#include "../Models/Note.h"
#include "../Models/Project.h"
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

/**
 * Spec Q7 / Q12: vocal fry at the end of a phrase, and as a word starts
 * (onsets() below), found in the detector's own pitch curve
 * (AudioData::rawF0, one value per analysis frame, 0 where unvoiced, in the
 * recording's time).
 *
 * Fry there reads as a short voiced run about an octave under the voice it
 * follows, after which the voice does not come back. The vocoder turns it
 * into an artefact whatever pitch it is given, with the recording put back
 * above 10 kHz too. Rubber Band R3 keeps sparse pulses (11 in the recording,
 * 8 after it) and smears dense ones (31, 5 to 20 ms apart, down to 4), R2
 * with short windows too (18): only the recording itself was accepted on
 * both. So in a note left on "Auto" that stretch of the recording is kept
 * as it is, not moved in pitch (kept()), and the rest of the note is
 * rendered as usual.
 *
 * The rule, as it was tuned on speech (5 phrase ends, 4 of them visible in
 * the App's curve; none found in 3 sung projects of the same song):
 *   1. the voiced frames are cut into runs where the pitch steps 4
 *      semitones or more between neighbours;
 *   2. a run of at most kMaxRun frames whose median sits kDrop semitones or
 *      more under the end of the run before it (no more than kGap unvoiced
 *      frames apart, and longer than this one);
 *   3. no voiced frame within 4 semitones of that earlier pitch in the
 *      kTail frames after it: a sub-harmonic caught between two stretches
 *      of normal voice is a detection error, not fry;
 *   4. in the recording, the sound goes on after the run without a break
 *      (the 70 ms after it within kAfterLevelDb of the run, the first 40 ms
 *      too) and is not periodic there (autocorrelation under
 *      kAfterPeriodicity): the irregular pulses of fry go on past what the
 *      detector calls voiced. Two of the four runs rules 1-3 found in that
 *      speech were heard as plain voice: after one the voice went on
 *      periodic at its own pitch (the octave drop was the detector's), after
 *      the other the sound stopped, and a consonant followed 30 ms later.
 * Frame counts are for the App's 512-sample hop at 44.1 kHz (11.6 ms).
 */
namespace fry
{
constexpr int kMaxRun = 26;   // 300 ms
constexpr int kTail = 9;      // 100 ms
constexpr int kGap = 3;
constexpr int kReference = 7; // frames at the end of the run before
constexpr float kDrop = 7.0f;
constexpr float kStep = 4.0f;
constexpr int kMinFrames = 3; // of a note, for it to count as holding fry
constexpr double kAfterWindow = 0.040; // seconds; 4 of them, 10 ms apart
constexpr int kAfterWindows = 4;
constexpr float kAfterLevelDb = 12.0f;
constexpr float kAfterPeriodicity = 0.4f;

using Ranges = std::vector<std::pair<int, int>>; // [first, last) frames

/** Rule 1: the curve in semitones and its voiced runs, a frame part-way
 *  between two of them left out. */
struct Runs
{
    std::vector<float> semitones; // 12 log2(Hz), 0 where unvoiced
    std::vector<char> voicedFrames;
    Ranges runs;

    bool voiced(int i) const { return voicedFrames[static_cast<size_t>(i)] != 0; }
    float at(int i) const { return semitones[static_cast<size_t>(i)]; }
    float median(int from, int to) const
    {
        std::vector<float> values;
        for (int i = from; i < to; ++i)
            values.push_back(at(i));
        std::sort(values.begin(), values.end());
        const size_t n = values.size();
        return n % 2 == 1 ? values[n / 2] : 0.5f * (values[n / 2 - 1] + values[n / 2]);
    }
};

inline Runs runsOf(const std::vector<float>& rawF0)
{
    Runs curve;
    const int total = static_cast<int>(rawF0.size());
    curve.semitones.assign(rawF0.size(), 0.0f);
    curve.voicedFrames.assign(rawF0.size(), 0);
    for (int i = 0; i < total; ++i)
        if (rawF0[static_cast<size_t>(i)] > 0.0f)
        {
            curve.semitones[static_cast<size_t>(i)] = 12.0f * std::log2(rawF0[static_cast<size_t>(i)]);
            curve.voicedFrames[static_cast<size_t>(i)] = 1;
        }

    Ranges runs;
    int start = 0;
    for (int i = 1; i <= total; ++i)
    {
        const bool cut = i == total || curve.voiced(i) != curve.voiced(i - 1) ||
                         (curve.voiced(i) && std::abs(curve.at(i) - curve.at(i - 1)) >= kStep);
        if (!cut)
            continue;
        if (curve.voiced(start))
            runs.emplace_back(start, i);
        start = i;
    }

    // The curve is resampled to the App's frames, and a jump leaves one
    // frame part-way between the two pitches. Such a frame is not a run of
    // its own: it would stand between the fry and the voice next to it.
    for (size_t j = 0; j < runs.size(); ++j)
    {
        const auto [a, b] = runs[j];
        if (b - a == 1 && j > 0 && j + 1 < runs.size() && runs[j - 1].second == a &&
            runs[j + 1].first == b)
        {
            const float low = std::min(curve.at(a - 1), curve.at(b));
            const float high = std::max(curve.at(a - 1), curve.at(b));
            if (curve.at(a) >= low && curve.at(a) <= high)
                continue;
        }
        curve.runs.push_back(runs[j]);
    }
    return curve;
}

/** Rules 1-3: the runs the pitch curve alone points to. */
inline Ranges candidates(const std::vector<float>& rawF0)
{
    const int total = static_cast<int>(rawF0.size());
    const auto curve = runsOf(rawF0);
    const auto& kept = curve.runs;

    Ranges found;
    for (size_t j = 1; j < kept.size(); ++j)
    {
        const auto [a, b] = kept[j];
        const auto [previousStart, previousEnd] = kept[j - 1];
        if (b - a > kMaxRun || a - previousEnd > kGap || previousEnd - previousStart <= b - a)
            continue;
        const float reference =
            curve.median(std::max(previousStart, previousEnd - kReference), previousEnd);
        if (reference - curve.median(a, b) < kDrop)
            continue;
        bool comesBack = false;
        for (int i = b; i < std::min(total, b + kTail) && !comesBack; ++i)
            comesBack = curve.voiced(i) && std::abs(curve.at(i) - reference) <= kStep;
        if (!comesBack)
            found.emplace_back(a, b);
    }
    return found;
}

/**
 * Vocal fry as a word starts (spec Q12), read in the same curve: the mirror
 * image of rules 1 and 2. A voiced run of at most kOnsetMaxRun frames whose
 * median sits kDrop semitones or more under the start of the run after it
 * (no more than kGap frames apart, and longer than this one). Rule 3 is
 * left out: 6 of the 9 below have a voiced frame at the voice's own pitch
 * in the 100 ms before, in 4 of them a single frame as the voice catches
 * after a consonant. Rule 4 has no counterpart yet: nothing was found that
 * it would have had to turn down.
 *
 * Found in 煙幕 (sung, 249 s) 6 times and in the speech the rule above was
 * tuned on (120 s) 3 times, and all 9 were heard as fry starting a word.
 * What it misses is not known: fry the detector calls unvoiced leaves no
 * low run to find. The runs were 1 to 5 frames long; nothing between 6 and
 * kMaxRun frames turned up in either recording, so kOnsetMaxRun is set from
 * what a sung pickup note under a leap of a fifth can be mistaken for, not
 * from a case heard.
 *
 * The vocoder turns such an onset into an artefact as it does the end of a
 * phrase (the word at 230.3 s of 煙幕, stretched 1.10x: the recording kept
 * was heard as best, R3 as acceptable, the vocoder as worst), so notes left
 * on Auto keep the recording there: from the low run up to the voice after
 * it, and back from it over what leads into it, kOnsetLead frames at most
 * - unvoiced frames, frames as low as the run, and a voiced frame right
 * after an unvoiced one. The voice's own band (70 to 900 Hz) rises in the
 * recording 1 to 5 frames before the low run in the 6 sung ones, while the
 * detector still calls it unvoiced, and a run of unvoiced frames that short
 * is rendered like the voice (generateBlendMask). The walk back stops at
 * the voice before: fry between two words keeps that voice rendered.
 */
constexpr int kOnsetMaxRun = 9; // 100 ms
constexpr int kOnsetLead = 6;   // 70 ms
constexpr int kArriveWithin = 2;

/** Each onset as [first, last) frames to keep: its lead, the low run, and
 *  the frames up to the voice after it. */
inline Ranges onsets(const std::vector<float>& rawF0)
{
    const auto curve = runsOf(rawF0);
    const auto& kept = curve.runs;

    Ranges found;
    for (size_t j = 0; j + 1 < kept.size(); ++j)
    {
        const auto [a, b] = kept[j];
        const auto [nextStart, nextEnd] = kept[j + 1];
        if (b - a > kOnsetMaxRun || nextStart - b > kGap || nextEnd - nextStart <= b - a)
            continue;
        const float reference =
            curve.median(nextStart, std::min(nextEnd, nextStart + kReference));
        if (reference - curve.median(a, b) < kDrop)
            continue;
        int start = a;
        while (start > 0 && a - start < kOnsetLead)
        {
            const int i = start - 1;
            const bool low = curve.voiced(i) && reference - curve.at(i) >= kDrop;
            const bool voiceCatching = curve.voiced(i) && (i == 0 || !curve.voiced(i - 1));
            if (curve.voiced(i) && !low && !voiceCatching)
                break;
            start = i;
        }
        found.emplace_back(start, nextStart);
    }
    return found;
}

/** Rule 4 on one candidate: what follows it in the recording. */
inline bool goesOnIrregular(const float* samples, int numSamples, double sampleRate,
                            double samplesPerFrame, int firstFrame, int lastFrame)
{
    const auto level = [samples](int from, int to)
    {
        double mean = 0.0, power = 0.0;
        for (int i = from; i < to; ++i)
            mean += samples[i];
        mean /= std::max(1, to - from);
        for (int i = from; i < to; ++i)
            power += (samples[i] - mean) * (samples[i] - mean);
        return 10.0 * std::log10(power / std::max(1, to - from) + 1.0e-12);
    };
    const int runFrom = static_cast<int>(firstFrame * samplesPerFrame);
    const int runTo = static_cast<int>(lastFrame * samplesPerFrame);
    const int window = static_cast<int>(kAfterWindow * sampleRate);
    const int step = static_cast<int>(0.010 * sampleRate);
    if (runFrom < 0 || runTo <= runFrom ||
        runTo + (kAfterWindows - 1) * step + window > numSamples)
        return false;
    const double runLevel = level(runFrom, runTo);

    // The highest normalised autocorrelation between 70 and 400 Hz.
    const int shortLag = static_cast<int>(sampleRate / 400.0);
    const int longLag = std::min(window - 1, static_cast<int>(sampleRate / 70.0));
    std::vector<double> levels, peaks;
    std::vector<double> frame(static_cast<size_t>(window));
    for (int k = 0; k < kAfterWindows; ++k)
    {
        const int from = runTo + k * step;
        double mean = 0.0;
        for (int i = 0; i < window; ++i)
            mean += samples[from + i];
        mean /= window;
        double energy = 0.0;
        for (int i = 0; i < window; ++i)
        {
            frame[static_cast<size_t>(i)] = samples[from + i] - mean;
            energy += frame[static_cast<size_t>(i)] * frame[static_cast<size_t>(i)];
        }
        double peak = 0.0;
        for (int lag = shortLag; lag < longLag; ++lag)
        {
            double sum = 0.0;
            for (int i = 0; i + lag < window; ++i)
                sum += frame[static_cast<size_t>(i)] * frame[static_cast<size_t>(i + lag)];
            peak = std::max(peak, sum / (energy + 1.0e-12));
        }
        levels.push_back(10.0 * std::log10(energy / window + 1.0e-12));
        peaks.push_back(peak);
    }
    const auto middle = [](std::vector<double> values)
    {
        std::sort(values.begin(), values.end());
        const size_t n = values.size();
        return n % 2 == 1 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
    };
    // No break right after the run either: a pause and then a consonant's
    // noise is loud and not periodic too (9.79 s of the same speech).
    return levels.front() >= runLevel - kAfterLevelDb &&
           middle(levels) >= runLevel - kAfterLevelDb && middle(peaks) < kAfterPeriodicity;
}

/** Fry at the end of a phrase in this recording (rules 1-4). */
inline Ranges detect(const AudioData& audioData)
{
    Ranges found;
    const auto& recording = audioData.originalWaveform;
    if (recording.getNumChannels() < 1 || recording.getNumSamples() < 1)
        return found;
    const double samplesPerFrame = 512.0 * audioData.sampleRate / 44100.0;
    for (const auto& [a, b] : candidates(audioData.rawF0))
        if (goesOnIrregular(recording.getReadPointer(0), recording.getNumSamples(),
                            audioData.sampleRate, samplesPerFrame, a, b))
            found.emplace_back(a, b);
    return found;
}

/** detect() kept between calls while the recording and its pitch curve stay
 *  the same: for what runs on every repaint. Not for use across threads. */
class Cache
{
public:
    const Ranges& of(const AudioData& audioData)
    {
        double sum = 0.0;
        for (const float value : audioData.rawF0)
            sum += value;
        const int samples = audioData.originalWaveform.getNumSamples();
        if (!valid || sum != pitchSum || audioData.rawF0.size() != frames || samples != length)
        {
            ranges = detect(audioData);
            pitchSum = sum;
            frames = audioData.rawF0.size();
            length = samples;
            valid = true;
        }
        return ranges;
    }

private:
    Ranges ranges;
    double pitchSum = 0.0;
    size_t frames = 0;
    int length = 0;
    bool valid = false;
};

/** The note's stretch of the recording holds kMinFrames or more of fry. */
inline bool holds(const Note& note, const Ranges& ranges)
{
    int frames = 0;
    for (const auto& [a, b] : ranges)
        frames += std::max(0, std::min(b, note.getSrcEndFrame()) -
                                  std::max(a, note.getSrcStartFrame()));
    return frames >= kMinFrames;
}

/** A note kept whole (keepsWhole below) with no edit but its pitch plays
 *  the recording as a note with no edit does, and the synthesizer hands
 *  over to it the same way: lined up with the note before, as an edited
 *  note meets an unedited one. A plain fade there (6 ms, voice on voice,
 *  not lined up) was heard as less smooth. With a volume, formant or
 *  sibilance edit it is rendered, and kept through the blend mask. */
inline bool keepsWhole(const Note& note, const Ranges& ranges);
inline bool playsRecording(const Note& note, const Ranges& ranges)
{
    constexpr float kNone = 0.001f;
    return keepsWhole(note, ranges) && std::abs(note.getVolumeDb()) <= kNone &&
           std::abs(note.getFormantShift()) <= kNone &&
           std::abs(note.getSibilantBalance()) <= kNone;
}

/** How the note is rendered: its own choice, or on Auto the vocoder with
 *  the recording's highs. */
inline Note::SynthEngine resolved(const Note& note)
{
    return note.getSynthEngine() == Note::SynthEngine::Auto
               ? Note::SynthEngine::VocoderRecordedHighs
               : note.getSynthEngine();
}

/** The frames of the recording that notes on Auto keep as recorded: each
 *  run of fry such a note holds, on to the end of that note (the pulses go
 *  on past what the detector calls voiced, and it calls some of them voiced
 *  again further on: 165 ms inside one phrase end were rendered otherwise),
 *  and the unvoiced frames after that up to the next voiced one (kKeepAfter
 *  frames at most, a longer unvoiced run is kept anyway). Where the note
 *  has less than kWholeUnder frames of normal voice before its fry, the
 *  whole note is kept: moved 3 semitones, 38 ms of voice in front of fry
 *  that had some pitch of its own was heard to stand apart from it, and
 *  the whole note kept was preferred; with 156 ms of voice in front (fry
 *  with no pitch to it) moving the voice was preferred, the whole note
 *  kept stood apart from the words before it. Both were accepted either
 *  way, and at 1 semitone and less no difference was heard. The threshold
 *  rests on those 2 phrase ends alone. A note that plays the recording
 *  (playsRecording) is not listed: it is not rendered at all. A note with a
 *  choice of its own is rendered whole with it. */
constexpr int kKeepAfter = 24;
constexpr int kWholeUnder = 9; // 100 ms

/** The note is on Auto, holds fry, and has less than kWholeUnder frames of
 *  voice before it: all of it stays the recording. Its pitch is then the
 *  recorded one whatever the note is moved to, and the pitch curve takes it
 *  so (PitchCurveProcessor): the note before it glides to the pitch that
 *  plays. With the curve on the moved pitch the voice stayed 3 semitones up
 *  to the last 10 ms and stepped down at the handover, which was heard as
 *  the fry alone kept. */
inline bool keepsWhole(const Note& note, const Ranges& ranges)
{
    if (note.isRest() || note.getSynthEngine() != Note::SynthEngine::Auto)
        return false;
    for (const auto& [a, b] : ranges)
        if (holds(note, Ranges{{a, b}}) && a - note.getSrcStartFrame() < kWholeUnder)
            return true;
    return false;
}
template <typename Notes>
inline Ranges kept(const Notes& notes, const std::vector<float>& rawF0, const Ranges& ranges)
{
    Ranges found;
    const int total = static_cast<int>(rawF0.size());
    for (const auto& [a, b] : ranges)
    {
        bool onAuto = false;
        int start = a;
        int end = b;
        for (const auto& note : notes)
            if (!note.isRest() && note.getSynthEngine() == Note::SynthEngine::Auto &&
                holds(note, Ranges{{a, b}}) && !playsRecording(note, Ranges{{a, b}}))
            {
                onAuto = true;
                end = std::max(end, std::min(total, note.getSrcEndFrame()));
                if (keepsWhole(note, Ranges{{a, b}}))
                    start = std::min(start, std::max(0, note.getSrcStartFrame()));
            }
        if (!onAuto)
            continue;
        const int noteEnd = end;
        while (end < total && end < noteEnd + kKeepAfter &&
               rawF0[static_cast<size_t>(end)] <= 0.0f)
            ++end;
        found.emplace_back(start, end);
    }
    return found;
}

/**
 * Where the voice has arrived after an onset whose kept frames end at `end`.
 * The vocoder sings the detector's smoothed curve (denseF0), and across the
 * jump up from the fry that curve can reach the voice a frame after the
 * detector's own (rawF0): 337 against 424 Hz at 230.4 s of 煙幕, in 3 of
 * the 9 onsets. The first frame from `end` on where the two are within a
 * semitone, kArriveWithin frames on at most. Up to there the vocoder is
 * given the voice's pitch (IncrementalSynthesizer.cpp), so that it is not
 * still singing the fry where the recording hands over to it: heard at
 * 0.80x to 1.50x on 3 onsets, 11 of 12 accepted and the twelfth marked for
 * another word. Keeping one frame more of the recording instead was
 * accepted on one onset and not on another stretched 1.33x and 1.50x,
 * where the later crossfade dipped 5 to 7 dB; a 1.5 ms switch at the
 * quietest point and a crossfade moved up to 50 ms on were each turned
 * down somewhere (stretched, and moved 2 semitones).
 */
inline int voiceArrivesAt(const std::vector<float>& rawF0,
                          const std::vector<float>& denseF0, int end)
{
    int arrived = end;
    for (int k = 0; k < kArriveWithin; ++k)
    {
        const auto at = static_cast<size_t>(std::max(0, arrived));
        if (at >= rawF0.size() || at >= denseF0.size() || !(rawF0[at] > 0.0f) ||
            !(denseF0[at] > 0.0f) || 12.0f * std::log2(rawF0[at] / denseF0[at]) <= 1.0f)
            break;
        ++arrived;
    }
    return arrived;
}

/** The frames of the recording that notes on Auto keep as recorded for the
 *  fry a word starts with (onsets()): each onset a note on Auto reaches
 *  into, less the frames of notes with a choice of their own, which are
 *  rendered whole with it. */
template <typename Notes>
inline Ranges keptOnsets(const Notes& notes, const Ranges& ranges)
{
    Ranges found;
    for (const auto& [a, b] : ranges)
    {
        bool onAuto = false;
        Ranges pieces{{a, b}};
        for (const auto& note : notes)
        {
            if (note.isRest())
                continue;
            const int from = std::max(a, note.getSrcStartFrame());
            const int to = std::min(b, note.getSrcEndFrame());
            if (to <= from)
                continue;
            if (note.getSynthEngine() == Note::SynthEngine::Auto)
            {
                onAuto = true;
                continue;
            }
            Ranges left;
            for (const auto& [first, last] : pieces)
            {
                if (first < from)
                    left.emplace_back(first, std::min(last, from));
                if (last > to)
                    left.emplace_back(std::max(first, to), last);
            }
            pieces = std::move(left);
        }
        if (onAuto)
            found.insert(found.end(), pieces.begin(), pieces.end());
    }
    return found;
}
} // namespace fry
