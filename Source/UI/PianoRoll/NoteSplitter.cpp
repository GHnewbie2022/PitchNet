#include "NoteSplitter.h"
#include "../../Undo/NoteActions.h"
#include "../../Audio/NoteTiming.h"
#include "../../Utils/FryDetector.h"
#include "../../Utils/NoteAmplitude.h"
#include "../../Utils/PitchCurveProcessor.h"
#include "../../Utils/Constants.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
float calculatePitchCenter(const AudioData& audioData, int startFrame,
                           int endFrame, float fallback)
{
    const int f0Size = static_cast<int>(audioData.f0.size());
    const int start = std::clamp(startFrame, 0, f0Size);
    const int end = std::clamp(endFrame, start, f0Size);

    double midiSum = 0.0;
    int voicedCount = 0;
    for (int frame = start; frame < end; ++frame) {
        const bool voiced = audioData.voicedMask.empty() ||
                            (frame < static_cast<int>(audioData.voicedMask.size()) &&
                             audioData.voicedMask[static_cast<size_t>(frame)]);
        const float f0 = audioData.f0[static_cast<size_t>(frame)];
        if (voiced && f0 > 0.0f && std::isfinite(f0)) {
            midiSum += freqToMidi(f0);
            ++voicedCount;
        }
    }

    return voicedCount > 0
               ? static_cast<float>(midiSum / voicedCount)
               : fallback;
}

void rebasePitchDeviation(Note& note, float amount)
{
    if (std::abs(amount) < 0.0001f)
        return;

    if (note.hasOriginalDeltaPitch()) {
        auto delta = note.getOriginalDeltaPitch();
        for (auto& value : delta)
            value += amount;
        note.setOriginalDeltaPitch(std::move(delta));
    }

    if (note.hasDeltaPitch()) {
        auto delta = note.getDeltaPitch();
        for (auto& value : delta)
            value += amount;
        note.setDeltaPitch(std::move(delta));
    }

    if (note.hasBakedDeltaPitch()) {
        auto delta = note.getBakedDeltaPitch();
        for (auto& value : delta)
            value += amount;
        note.setBakedDeltaPitch(std::move(delta));
    }
}

// The pitch sung over [start, end): base + delta per frame. Empty when the
// curves do not cover it.
std::vector<float> sungPitch(const AudioData& audio, int start, int end)
{
    std::vector<float> pitch;
    for (int f = start; f < end; ++f)
    {
        const auto i = static_cast<size_t>(f);
        if (f < 0 || i >= audio.basePitch.size() || i >= audio.deltaPitch.size())
            return {};
        pitch.push_back(audio.basePitch[i] + audio.deltaPitch[i]);
    }
    return pitch;
}

// No pitch tool, drag or drawn curve on the note: it sings what the
// analysis gave it.
bool pitchUntouched(const Note& n)
{
    constexpr float eps = 0.001f;
    return std::abs(n.getMidiNote() - n.getOriginalMidiNote()) <= eps &&
           std::abs(n.getPitchOffset()) <= eps && !n.hasBakedDeltaPitch() &&
           std::abs(n.getTiltLeft()) <= eps && std::abs(n.getTiltRight()) <= eps &&
           std::abs(n.getVibrato() - 1.0f) <= eps &&
           std::abs(n.getPitchDrift() - 1.0f) <= eps &&
           n.getSmoothLeftFrames() == 0 && n.getSmoothRightFrames() == 0 &&
           std::abs(n.getDeltaScale() - 1.0f) <= eps &&
           std::abs(n.getDeltaOffset()) <= eps;
}

Note* noteStartingAt(Project& project, int start)
{
    for (auto& note : project.getNotes())
        if (!note.isRest() && note.getStartFrame() == start)
            return &note;
    return nullptr;
}

/**
 * A split or merge re-centres the notes it makes, the base pitch bends
 * differently at the new boundary, and pitch tools (drift, vibrato, tilt,
 * smoothing) work over each note's own length - so the notes no longer sing
 * what they did. When that happened, commit `before` (the pitch over
 * [start, end) before the edit) as each note's own contour, as a drawn curve
 * is, with its pitch tools at neutral. Notes are given by start frame.
 */
void keepSungPitch(Project& project, const std::vector<int>& starts, int start,
                   int end, const std::vector<float>& before)
{
    if (before.empty())
        return;
    const auto after = sungPitch(project.getAudioData(), start, end);
    bool changed = after.size() != before.size();
    for (size_t i = 0; i < after.size() && !changed; ++i)
        changed = std::abs(after[i] - before[i]) > 0.001f;
    if (!changed)
        return;

    // Neutral tools first: tilt also moves a note's base pitch, so the
    // contour is measured against the base each note will keep.
    for (const int noteStart : starts)
        if (auto* note = noteStartingAt(project, noteStart))
        {
            note->setTiltLeft(0.0f);
            note->setTiltRight(0.0f);
            note->setVibrato(1.0f);
            note->setPitchDrift(1.0f);
            note->setSmoothLeftFrames(0);
            note->setSmoothRightFrames(0);
            note->setDeltaScale(1.0f);
            note->setDeltaOffset(0.0f);
            note->setFrozenTilt({});
        }
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    const auto& base = project.getAudioData().basePitch;
    for (const int noteStart : starts)
        if (auto* note = noteStartingAt(project, noteStart))
        {
            std::vector<float> contour;
            for (int f = note->getStartFrame(); f < note->getEndFrame(); ++f)
                contour.push_back(before[static_cast<size_t>(f - start)] -
                                  base[static_cast<size_t>(f)]);
            note->setBakedDeltaPitch(contour);
            note->setDeltaPitch(contour);
        }
    PitchCurveProcessor::rebuildBaseFromNotes(project);
}

// Frames either side of a boundary the base pitch's smoothing reaches
// (an 80 ms window at the analysis frame rate, with room to spare).
constexpr int kBoundaryReach = 12;

/**
 * A split or merge re-centres notes, and the base pitch bends smoothly from
 * one centre to the next across each boundary - a bend the notes' curves,
 * shifted by a constant, did not follow: a spike at the boundary, in the
 * note itself and in its neighbour's first frames. Fold the difference into
 * the curves of the notes around [start, end) so they sing `before` again
 * (it covers [start - kBoundaryReach, end + kBoundaryReach)). Notes starting
 * at `skip` are left alone. Returns the neighbours' edits for undo; notes
 * starting at `own` (the edited notes, which the undo restores whole) are
 * changed but not listed.
 */
std::vector<NoteCurveEdit> smoothOverBoundaries(Project& project, int start, int end,
                                               const std::vector<float>& before,
                                               const std::vector<int>& own,
                                               const std::vector<int>& skip = {},
                                               bool bakedOnly = false)
{
    std::vector<NoteCurveEdit> edits;
    // Untouched notes sing the recording: refit them to it over the new base
    // (PitchCurveProcessor::refitUntouchedCurves). Edited notes keep the
    // pitch they sang through the compensation below.
    const int first = std::max(0, start - kBoundaryReach);
    if (before.empty())
        return edits;
    const int last = std::min(end + kBoundaryReach,
                              first + static_cast<int>(before.size()));

    if (!bakedOnly)
    {
        struct PriorCurve
        {
            int startFrame = 0;
            std::vector<float> original;
            std::vector<float> baked;
        };
        std::vector<PriorCurve> prior;
        for (const auto& note : project.getNotes())
            if (!note.isRest() && note.getEndFrame() > first &&
                note.getStartFrame() < last)
                prior.push_back({note.getStartFrame(),
                                 note.getOriginalDeltaPitch(),
                                 note.getBakedDeltaPitch()});
        for (const int changed : PitchCurveProcessor::refitUntouchedCurves(
                 project, first, last))
        {
            if (std::find(own.begin(), own.end(), changed) != own.end() ||
                std::find(skip.begin(), skip.end(), changed) != skip.end())
                continue;
            const auto was = std::find_if(prior.begin(), prior.end(),
                                          [changed](const PriorCurve& p)
                                          { return p.startFrame == changed; });
            for (const auto& note : project.getNotes())
                if (!note.isRest() && note.getStartFrame() == changed &&
                    was != prior.end())
                    edits.push_back({changed, was->original,
                                     note.getOriginalDeltaPitch(),
                                     was->baked, note.getBakedDeltaPitch()});
        }
    }
    for (int pass = 0; pass < 2; ++pass)
    {
        const auto& audio = project.getAudioData();
        bool changed = false;
        for (auto& note : project.getNotes())
        {
            const int noteStart = note.getStartFrame();
            if (note.isRest() || note.getEndFrame() <= first ||
                noteStart >= last ||
                std::find(skip.begin(), skip.end(), noteStart) != skip.end())
                continue;
            if (bakedOnly && !note.hasBakedDeltaPitch())
                continue;
            if (!audio.denseF0.empty() && note.isPitchUntouched())
                continue; // refitted to the recording above
            const bool baked = note.hasBakedDeltaPitch();
            const auto& storedCurve =
                baked ? note.getBakedDeltaPitch() : note.getOriginalDeltaPitch();
            if (storedCurve.empty())
                continue;
            auto curve = static_cast<int>(storedCurve.size()) == note.getDurationFrames()
                             ? storedCurve
                             : notetiming::curveOnNote(audio, note, storedCurve);
            if (static_cast<int>(curve.size()) != note.getDurationFrames())
                continue;
            bool touched = false;
            for (int f = std::max(noteStart, first);
                 f < std::min(note.getEndFrame(), last); ++f)
            {
                const auto k = static_cast<size_t>(f - first);
                const auto i = static_cast<size_t>(f);
                if (k >= before.size() || i >= audio.basePitch.size() ||
                    i >= audio.deltaPitch.size())
                    continue;
                const float diff = before[k] - (audio.basePitch[i] + audio.deltaPitch[i]);
                if (std::abs(diff) > 0.0001f)
                {
                    curve[static_cast<size_t>(f - noteStart)] += diff;
                    touched = true;
                }
            }
            if (!touched)
                continue;
            changed = true;
            const bool listed = std::find(own.begin(), own.end(), noteStart) == own.end();
            auto edit = std::find_if(edits.begin(), edits.end(),
                                     [noteStart](const NoteCurveEdit& e)
                                     { return e.startFrame == noteStart; });
            if (listed && edit == edits.end())
            {
                edits.push_back({noteStart, note.getOriginalDeltaPitch(), {},
                                 note.getBakedDeltaPitch(), {}});
                edit = std::prev(edits.end());
            }
            if (baked)
                note.setBakedDeltaPitch(curve);
            else
            {
                note.setOriginalDeltaPitch(curve);
                if (!note.getDeltaPitch().empty())
                    note.setDeltaPitch(curve);
            }
            if (listed)
            {
                edit->originalAfter = note.getOriginalDeltaPitch();
                edit->bakedAfter = note.getBakedDeltaPitch();
            }
        }
        if (!changed)
            break;
        PitchCurveProcessor::rebuildBaseFromNotes(project);
    }
    return edits;
}

using PitchCurveProcessor::resetNoteToUnedited;

/**
 * A split or merge works out the notes' unedited pitch (originalMidiNote,
 * originalDeltaPitch and their boundary compensation) in the unedited pitch:
 * every note's pitch edits and the drawn curves are put aside while it does,
 * so no drag or tool bends the base pitch the analysis is measured against.
 * The other notes get their edits back afterwards, keeping the analysed
 * curves the work gave them.
 */
struct UneditedSpace
{
    Project& project;
    std::vector<Note> saved;
    std::vector<DrawnCurve> curves;

    explicit UneditedSpace(Project& p)
        : project(p), saved(p.getNotes()), curves(p.getDrawnCurves())
    {
        for (auto& n : project.getNotes())
            if (!n.isRest())
                resetNoteToUnedited(n);
        project.getDrawnCurves().clear();
        PitchCurveProcessor::rebuildBaseFromNotes(project);
    }

    const Note* savedNote(std::uint64_t uid) const
    {
        for (const auto& n : saved)
            if (n.getUid() == uid)
                return &n;
        return nullptr;
    }

    /** Edits back on every note but `edited` (by uid), and `drawn` in place. */
    void restoreOthers(const std::vector<std::uint64_t>& edited,
                       std::vector<DrawnCurve> drawn)
    {
        for (auto& n : project.getNotes())
        {
            if (n.isRest() ||
                std::find(edited.begin(), edited.end(), n.getUid()) != edited.end())
                continue;
            const auto* was = savedNote(n.getUid());
            if (!was)
                continue;
            const auto original = n.getOriginalDeltaPitch();
            const bool followed = was->getDeltaPitch() == was->getOriginalDeltaPitch();
            n = *was;
            n.setOriginalDeltaPitch(original);
            if (followed)
                n.setDeltaPitch(original);
        }
        project.getDrawnCurves() = std::move(drawn);
        PitchCurveProcessor::rebuildBaseFromNotes(project);
    }

    /** What the whole edit changed in the other notes' curves, for undo. */
    std::vector<NoteCurveEdit> neighbourEdits(const std::vector<int>& own) const
    {
        std::vector<NoteCurveEdit> edits;
        for (const auto& n : project.getNotes())
        {
            if (n.isRest() ||
                std::find(own.begin(), own.end(), n.getStartFrame()) != own.end())
                continue;
            const auto* was = savedNote(n.getUid());
            if (!was || (was->getOriginalDeltaPitch() == n.getOriginalDeltaPitch() &&
                         was->getBakedDeltaPitch() == n.getBakedDeltaPitch()))
                continue;
            edits.push_back({n.getStartFrame(), was->getOriginalDeltaPitch(),
                             n.getOriginalDeltaPitch(), was->getBakedDeltaPitch(),
                             n.getBakedDeltaPitch()});
        }
        return edits;
    }
};

float sampleBasePitchAt(const std::vector<float>& basePitch, double framePos)
{
    if (basePitch.empty())
        return 0.0f;
    const double clamped = std::clamp(
        framePos, 0.0, static_cast<double>(basePitch.size() - 1));
    const size_t i0 = static_cast<size_t>(std::floor(clamped));
    const size_t i1 = std::min(i0 + 1, basePitch.size() - 1);
    const float frac = static_cast<float>(clamped - static_cast<double>(i0));
    return basePitch[i0] * (1.0f - frac) + basePitch[i1] * frac;
}

std::vector<float> curveAtSourceLength(const std::vector<float>& curve, int srcLen)
{
    if (srcLen <= 0 || curve.empty() || static_cast<int>(curve.size()) == srcLen)
        return curve;
    return CurveResampler::resampleLinear(curve, srcLen);
}

Note* findResizedNote(Project& project, uint64_t uid, int startFrame)
{
    auto& notes = project.getNotes();
    if (uid != 0)
    {
        for (auto& n : notes)
            if (n.getUid() == uid)
                return &n;
    }
    return noteStartingAt(project, startFrame);
}

void sortNotesByStartFrame(Project& project)
{
    auto& notes = project.getNotes();
    if (!std::is_sorted(notes.begin(), notes.end(),
                        [](const Note& a, const Note& b)
                        { return a.getStartFrame() < b.getStartFrame(); }))
    {
        std::stable_sort(notes.begin(), notes.end(),
                         [](const Note& a, const Note& b)
                         { return a.getStartFrame() < b.getStartFrame(); });
    }
}

void applyResizedNoteState(Project& project,
                           Note& note,
                           const Note& originalNote,
                           int newSrcStartFrame,
                           int newSrcEndFrame,
                           const std::vector<float>& baseBefore)
{
    const auto& audioData = project.getAudioData();
    const auto& map = audioData.warpMap;
    const int newStart = map.isIdentity()
        ? newSrcStartFrame
        : static_cast<int>(std::lround(map.destinationAt(newSrcStartFrame)));
    int newEnd = map.isIdentity()
        ? newSrcEndFrame
        : static_cast<int>(std::lround(map.destinationAt(newSrcEndFrame)));
    if (newSrcEndFrame > newSrcStartFrame)
        newEnd = std::max(newEnd, newStart + 1);

    const auto uid = originalNote.getUid();
    note = originalNote;
    note.setSrcStartFrame(newSrcStartFrame);
    note.setSrcEndFrame(newSrcEndFrame);
    note.setStartFrame(newStart);
    note.setEndFrame(newEnd);
    note.clearTimingPreview();

    const int newSrcLen = std::max(0, newSrcEndFrame - newSrcStartFrame);
    const auto& pitchSource = !audioData.denseF0.empty() ? audioData.denseF0 : audioData.f0;
    const int pitchSize = static_cast<int>(pitchSource.size());

    std::vector<float> f0Values(static_cast<size_t>(newSrcLen), 0.0f);
    for (int i = 0; i < newSrcLen; ++i)
    {
        const int s = newSrcStartFrame + i;
        if (s >= 0 && s < pitchSize)
            f0Values[static_cast<size_t>(i)] = pitchSource[static_cast<size_t>(s)];
    }
    note.setF0Values(std::move(f0Values));
    note.trimAnalysisSpans();
    NoteAmplitude::update(note, audioData);

    sortNotesByStartFrame(project);
    Note* resized = findResizedNote(project, uid, newStart);
    if (!resized)
        return;

    PitchCurveProcessor::rebuildBaseFromNotes(project);
    const auto& baseAfter = project.getAudioData().basePitch;

    const int oldSrcStart = originalNote.getSrcStartFrame();
    const int oldSrcEnd = originalNote.getSrcEndFrame();
    const int oldSrcLen = std::max(0, oldSrcEnd - oldSrcStart);
    const auto beforeMap = notetiming::forNote(audioData, originalNote);
    // A curve kept at the note's timeline length (boundary compensation
    // writes one) reads back per recorded frame through the warp, not
    // spread evenly: evenly would slide the contour along the note.
    const auto atSource = [&](const std::vector<float>& curve)
    {
        const int oldLen = originalNote.getDurationFrames();
        if (static_cast<int>(curve.size()) != oldLen || oldLen == oldSrcLen ||
            oldLen <= 0)
            return curveAtSourceLength(curve, oldSrcLen);
        std::vector<float> out(static_cast<size_t>(oldSrcLen));
        for (int i = 0; i < oldSrcLen; ++i)
        {
            const double pos = std::clamp(
                beforeMap.destinationAt(oldSrcStart + i + 0.5) - 0.5 -
                    originalNote.getStartFrame(),
                0.0, static_cast<double>(oldLen - 1));
            const int a = static_cast<int>(std::floor(pos));
            const int b = std::min(a + 1, oldLen - 1);
            const float t = static_cast<float>(pos - a);
            out[static_cast<size_t>(i)] = curve[static_cast<size_t>(a)] * (1.0f - t) +
                                          curve[static_cast<size_t>(b)] * t;
        }
        return out;
    };
    const std::vector<float> oldOrig = atSource(originalNote.getOriginalDeltaPitch());
    const std::vector<float> oldBaked =
        originalNote.hasBakedDeltaPitch()
            ? atSource(originalNote.getBakedDeltaPitch())
            : std::vector<float>{};
    // An unedited note sings the recording: its contour is read from it
    // over the new base, wherever the recording is voiced.
    const bool fromRecording = pitchUntouched(originalNote);

    const auto nowMap = notetiming::forNote(audioData, *resized);
    const int oldStart = originalNote.getStartFrame();
    const int oldEnd = originalNote.getEndFrame();
    const float userShift =
        originalNote.getAdjustedMidiNote() - originalNote.getOriginalMidiNote();

    std::vector<float> origDelta(static_cast<size_t>(newSrcLen), 0.0f);
    std::vector<float> baked(
        originalNote.hasBakedDeltaPitch() ? static_cast<size_t>(newSrcLen) : 0, 0.0f);

    for (int i = 0; i < newSrcLen; ++i)
    {
        const int s = newSrcStartFrame + i;
        const double dstAfterPos = std::clamp(
            nowMap.destinationAt(static_cast<double>(s) + 0.5) - 0.5,
            static_cast<double>(newStart),
            static_cast<double>(std::max(newStart, newEnd - 1)));
        const float bAfter = sampleBasePitchAt(baseAfter, dstAfterPos);

        double dstBeforePos = dstAfterPos;
        if (s >= oldSrcStart && s < oldSrcEnd && oldEnd > oldStart)
        {
            dstBeforePos = std::clamp(
                beforeMap.destinationAt(static_cast<double>(s) + 0.5) - 0.5,
                static_cast<double>(oldStart),
                static_cast<double>(oldEnd - 1));
        }
        const float bBefore =
            !baseBefore.empty() ? sampleBasePitchAt(baseBefore, dstBeforePos) : bAfter;

        const int oldIdx = s - oldSrcStart;
        const float recorded = resized->getF0Values()[static_cast<size_t>(i)];
        if (fromRecording && recorded > 0.0f)
        {
            origDelta[static_cast<size_t>(i)] = freqToMidi(recorded) - bAfter;
        }
        else if (s >= oldSrcStart && s < oldSrcEnd &&
            oldIdx >= 0 && oldIdx < static_cast<int>(oldOrig.size()))
        {
            origDelta[static_cast<size_t>(i)] =
                oldOrig[static_cast<size_t>(oldIdx)] - (bAfter - bBefore);
        }
        else
        {
            const float f0 = resized->getF0Values()[static_cast<size_t>(i)];
            origDelta[static_cast<size_t>(i)] =
                (f0 > 0.0f) ? (freqToMidi(f0) - (bAfter - userShift)) : 0.0f;
        }

        if (originalNote.hasBakedDeltaPitch())
        {
            if (s >= oldSrcStart && s < oldSrcEnd &&
                oldIdx >= 0 && oldIdx < static_cast<int>(oldBaked.size()))
            {
                baked[static_cast<size_t>(i)] =
                    oldBaked[static_cast<size_t>(oldIdx)] - (bAfter - bBefore);
            }
            else
            {
                baked[static_cast<size_t>(i)] = origDelta[static_cast<size_t>(i)];
            }
        }
    }

    resized->setOriginalDeltaPitch(std::move(origDelta));
    resized->setDeltaPitch({});
    if (originalNote.hasBakedDeltaPitch())
        resized->setBakedDeltaPitch(std::move(baked));
}

std::vector<NoteCurveEdit> compensateNeighbourBaseChange(
    Project& project,
    const std::vector<float>& baseBefore,
    int reachStart,
    int reachEnd,
    uint64_t skipUid,
    int skipStartFrame)
{
    std::vector<NoteCurveEdit> edits;
    const auto& audioData = project.getAudioData();
    const auto& baseAfter = audioData.basePitch;
    if (baseBefore.empty() || baseAfter.empty())
        return edits;

    for (auto& note : project.getNotes())
    {
        if (note.isRest() || note.getEndFrame() <= reachStart ||
            note.getStartFrame() >= reachEnd)
            continue;
        if ((skipUid != 0 && note.getUid() == skipUid) ||
            note.getStartFrame() == skipStartFrame)
            continue;
        if (note.getDurationFrames() <= 0 || note.getSrcDurationFrames() <= 0)
            continue;

        const auto origBefore = note.getOriginalDeltaPitch();
        const auto bakedBefore = note.getBakedDeltaPitch();
        auto origAfter = origBefore;
        auto bakedAfter = bakedBefore;
        bool changed = false;

        const auto nowMap = notetiming::forNote(audioData, note);
        const int noteStart = note.getStartFrame();
        const int noteLast = std::max(noteStart, note.getEndFrame() - 1);

        auto compensateCurve = [&](std::vector<float>& curve)
        {
            if (curve.empty())
                return;
            const int len = static_cast<int>(curve.size());
            const auto storedMap = notetiming::forStoredCurve(audioData, note, len);
            for (int j = 0; j < len; ++j)
            {
                const double dstPos =
                    (len == note.getDurationFrames())
                        ? static_cast<double>(noteStart + j)
                        : std::clamp(
                              nowMap.destinationAt(
                                  storedMap.sourceAt(static_cast<double>(j) + 0.5)) -
                                  0.5,
                              static_cast<double>(noteStart),
                              static_cast<double>(noteLast));
                const float diff = sampleBasePitchAt(baseAfter, dstPos) -
                                   sampleBasePitchAt(baseBefore, dstPos);
                if (std::abs(diff) > 1.0e-6f)
                {
                    curve[static_cast<size_t>(j)] -= diff;
                    changed = true;
                }
            }
        };

        compensateCurve(origAfter);
        if (note.hasBakedDeltaPitch())
            compensateCurve(bakedAfter);

        if (!changed)
            continue;

        note.setOriginalDeltaPitch(origAfter);
        if (!note.hasBakedDeltaPitch() &&
            note.getDeltaPitch().size() == origAfter.size())
            note.setDeltaPitch(origAfter);
        if (note.hasBakedDeltaPitch())
            note.setBakedDeltaPitch(bakedAfter);

        edits.push_back({note.getStartFrame(), origBefore, origAfter,
                         bakedBefore, bakedAfter});
    }

    return edits;
}
}

Note* NoteSplitter::findNoteAt(float x, float y) {
    if (!project || !coordMapper)
        return nullptr;

    float pixelsPerSecond = coordMapper->getPixelsPerSecond();
    float pixelsPerSemitone = coordMapper->getPixelsPerSemitone();

    for (auto& note : project->getNotes()) {
        if (note.isRest())
            continue;

        float noteX = framesToSeconds(note.getStartFrame()) * pixelsPerSecond;
        float noteW = framesToSeconds(note.getDurationFrames()) * pixelsPerSecond;
        float noteY = coordMapper->midiToY(note.getAdjustedMidiNote());
        float noteH = pixelsPerSemitone;

        if (x >= noteX && x < noteX + noteW && y >= noteY && y < noteY + noteH) {
            return &note;
        }
    }

    return nullptr;
}

bool NoteSplitter::splitNoteAtFrame(Note* note, int splitFrame, bool keepCentre) {
    if (!note || !project)
        return false;

    int startFrame = note->getStartFrame();
    int endFrame = note->getEndFrame();

    // Keep at least one analysis frame on each side of the split.
    if (splitFrame <= startFrame || splitFrame >= endFrame)
        return false;

    // Store original note data for undo
    Note originalNote = *note;
    // What the note sings now, which the split keeps (see keepSungPitch).
    const auto pitchBefore =
        pitchUntouched(originalNote)
            ? std::vector<float>{}
            : sungPitch(project->getAudioData(), startFrame, endFrame);
    // ...and around it, which the boundaries' smoothing must not bend.
    const int totalFrames =
        static_cast<int>(project->getAudioData().basePitch.size());
    const int aroundStart = std::max(0, startFrame - kBoundaryReach);
    const int aroundEnd = std::min(totalFrames, endFrame + kBoundaryReach);
    const auto pitchAround = sungPitch(project->getAudioData(), aroundStart, aroundEnd);
    const bool sourceUntouched = pitchUntouched(originalNote);
    const float userShift =
        originalNote.getAdjustedMidiNote() - originalNote.getOriginalMidiNote();

    // Split the analysis, not the edit: the halves' unedited centres and
    // curves come from the unedited pitch, the edit goes back on after.
    UneditedSpace unedited(*project);
    const auto uneditedAround =
        sungPitch(project->getAudioData(), aroundStart, aroundEnd);
    const float previousPitchCenter = note->getAdjustedMidiNote();

    // Create the second note (right part)
    Note secondNote;
    secondNote.setStartFrame(splitFrame);
    secondNote.setEndFrame(endFrame);
    const int srcStartFrame = note->getSrcStartFrame();
    const int srcEndFrame = note->getSrcEndFrame();
    const int sourceDuration = srcEndFrame - srcStartFrame;
    // The recorded frame the note plays at the split, through its time map:
    // in a stretched note, a split in the held vowel cuts the vowel.
    const auto& splitAudio = project->getAudioData();
    const int sourceSplitCandidate = static_cast<int>(std::lround(
        notetiming::forNote(splitAudio, *note).sourceAt(splitFrame)));
    const int srcSplitFrame = sourceDuration >= 2
                                  ? std::clamp(sourceSplitCandidate,
                                               srcStartFrame + 1,
                                               srcEndFrame - 1)
                                  : std::clamp(sourceSplitCandidate,
                                               srcStartFrame, srcEndFrame);
    secondNote.setSrcStartFrame(srcSplitFrame);
    secondNote.setSrcEndFrame(srcEndFrame);
    secondNote.setMidiNote(note->getMidiNote());
    // Both halves keep the macro source and strength of the original note.
    secondNote.setLastNonMacroMidiNote(note->getLastNonMacroMidiNote());
    secondNote.setMacroPitchCenter(note->getMacroPitchCenter());
    secondNote.setLyric(note->getLyric());
    secondNote.setPhoneme(note->getPhoneme());
    secondNote.setVolumeDb(note->getVolumeDb());
    secondNote.setSibilantBalance(note->getSibilantBalance());
    secondNote.setSelected(note->isSelected());
    secondNote.setTiltLeft(note->getTiltLeft());
    secondNote.setTiltRight(note->getTiltRight());
    secondNote.setTiltPivot(note->getTiltPivot());
    secondNote.setVibrato(note->getVibrato());
    secondNote.setPitchDrift(note->getPitchDrift());
    secondNote.setFormantShift(note->getFormantShift());
    secondNote.setSmoothLeftFrames(note->getSmoothLeftFrames());
    secondNote.setSmoothRightFrames(note->getSmoothRightFrames());
    secondNote.setDeltaScale(note->getDeltaScale());
    secondNote.setDeltaOffset(note->getDeltaOffset());
    secondNote.setPitchOffset(0.0f);
    // Both halves are rendered the way the note was (it used to be the left
    // half only: the right one of a Rubber Band note went back to Auto), and
    // the stretches kept by hand go with them. Those are frames of the
    // recording (Note::setKeptRange), so each half keeps what of them lies
    // in it with nothing to work out.
    secondNote.setSynthEngine(note->getSynthEngine());
    if (note->hasKeptRange())
        secondNote.setKeptRange(note->getKeptHeadUntil(), note->getKeptTailFrom());

    // Split originalDeltaPitch if available.
    // Each curve stored with the note divides where the recorded frame at the
    // split was when the curve was written: at the split for a curve written
    // at the note's current length, at the recorded frame for the analysed
    // contour of a stretched note.
    const auto curveSplit = [&](size_t length) {
        if (static_cast<int>(length) == endFrame - startFrame)
            return splitFrame - startFrame;
        return notetiming::curveIndexAtSource(splitAudio, *note,
                                              static_cast<int>(length),
                                              srcSplitFrame);
    };

    if (note->hasOriginalDeltaPitch()) {
        const auto& origDelta = note->getOriginalDeltaPitch();
        int splitOffset = curveSplit(origDelta.size());
        splitOffset = std::max(0, std::min(splitOffset, static_cast<int>(origDelta.size())));
        std::vector<float> leftDelta(origDelta.begin(), origDelta.begin() + splitOffset);
        std::vector<float> rightDelta(origDelta.begin() + splitOffset, origDelta.end());
        note->setOriginalDeltaPitch(std::move(leftDelta));
        secondNote.setOriginalDeltaPitch(std::move(rightDelta));
    } else {
        // Fallback: extract from global deltaPitch if originalDeltaPitch is missing
        auto& audioData = project->getAudioData();
        const int totalFrames = static_cast<int>(audioData.deltaPitch.size());
        if (totalFrames > 0) {
            int leftLen = splitFrame - startFrame;
            int rightLen = endFrame - splitFrame;
            if (leftLen > 0) {
                std::vector<float> leftDelta(static_cast<size_t>(leftLen));
                for (int i = 0; i < leftLen; ++i) {
                    int gIdx = startFrame + i;
                    if (gIdx >= 0 && gIdx < totalFrames)
                        leftDelta[static_cast<size_t>(i)] = audioData.deltaPitch[static_cast<size_t>(gIdx)];
                }
                note->setOriginalDeltaPitch(std::move(leftDelta));
            }
            if (rightLen > 0) {
                std::vector<float> rightDelta(static_cast<size_t>(rightLen));
                for (int i = 0; i < rightLen; ++i) {
                    int gIdx = splitFrame + i;
                    if (gIdx >= 0 && gIdx < totalFrames)
                        rightDelta[static_cast<size_t>(i)] = audioData.deltaPitch[static_cast<size_t>(gIdx)];
                }
                secondNote.setOriginalDeltaPitch(std::move(rightDelta));
            }
        }
    }

    // Keep per-note pitch data aligned with the new frame ranges.
    if (note->hasDeltaPitch()) {
        const auto& delta = note->getDeltaPitch();
        int splitOffset = curveSplit(delta.size());
        splitOffset = std::clamp(splitOffset, 0, static_cast<int>(delta.size()));
        std::vector<float> leftDelta(delta.begin(), delta.begin() + splitOffset);
        std::vector<float> rightDelta(delta.begin() + splitOffset, delta.end());
        note->setDeltaPitch(std::move(leftDelta));
        secondNote.setDeltaPitch(std::move(rightDelta));
    }

    if (note->hasBakedDeltaPitch()) {
        const auto& delta = note->getBakedDeltaPitch();
        int splitOffset = curveSplit(delta.size());
        splitOffset = std::clamp(splitOffset, 0, static_cast<int>(delta.size()));
        std::vector<float> leftDelta(delta.begin(), delta.begin() + splitOffset);
        std::vector<float> rightDelta(delta.begin() + splitOffset, delta.end());
        note->setBakedDeltaPitch(std::move(leftDelta));
        secondNote.setBakedDeltaPitch(std::move(rightDelta));
    }

    const auto& f0Values = note->getF0Values();
    if (!f0Values.empty()) {
        int splitOffset = curveSplit(f0Values.size());
        splitOffset = std::clamp(splitOffset, 0, static_cast<int>(f0Values.size()));
        std::vector<float> leftF0(f0Values.begin(), f0Values.begin() + splitOffset);
        std::vector<float> rightF0(f0Values.begin() + splitOffset, f0Values.end());
        note->setF0Values(std::move(leftF0));
        secondNote.setF0Values(std::move(rightF0));
    }

    // Recalculate each half's center from its own voiced pitch frames. Rebase
    // its deviation curve by the opposite amount so splitting is pitch-neutral.
    // (keepCentre: both stay on the note's, and nothing is rebased.)
    const auto& audioData = project->getAudioData();
    const float leftPitchCenter = keepCentre ? previousPitchCenter : calculatePitchCenter(
        audioData, startFrame, splitFrame, previousPitchCenter);
    const float rightPitchCenter = keepCentre ? previousPitchCenter : calculatePitchCenter(
        audioData, splitFrame, endFrame, previousPitchCenter);

    rebasePitchDeviation(*note, previousPitchCenter - leftPitchCenter);
    rebasePitchDeviation(secondNote, previousPitchCenter - rightPitchCenter);
    note->setMidiNote(leftPitchCenter);
    note->setOriginalMidiNote(leftPitchCenter);
    note->setPitchOffset(0.0f);
    secondNote.setMidiNote(rightPitchCenter);
    secondNote.setOriginalMidiNote(rightPitchCenter);
    secondNote.setPitchOffset(0.0f);

    // Modify the first note (left part)
    note->setEndFrame(splitFrame);
    note->setSrcEndFrame(srcSplitFrame);
    // Each half keeps the analysed notes it still overlaps.
    secondNote.setAnalysisSpans(note->getAnalysisSpans());
    note->trimAnalysisSpans();
    secondNote.trimAnalysisSpans();

    NoteAmplitude::update(*note, audioData);
    NoteAmplitude::update(secondNote, audioData);

    // Splitting is a structural edit, so both segments inherit the source
    // note's processing state rather than treating the new tail as clean.
    const bool sourceIsDirty = originalNote.isDirty();
    const bool sourceIsSynthDirty = originalNote.isSynthDirty();
    const bool sourceHasRenderedEdit = originalNote.hasRenderedEdit();
    note->setDirty(sourceIsDirty);
    note->setSynthDirty(sourceIsSynthDirty);
    note->setRenderedEdit(sourceHasRenderedEdit);
    secondNote.setDirty(sourceIsDirty);
    secondNote.setSynthDirty(sourceIsSynthDirty);
    secondNote.setRenderedEdit(sourceHasRenderedEdit);

    // Add the second note to project (this may move `note`: find both
    // halves by start frame from here on).
    project->addNote(secondNote);

    // Rebuild the drawn pitch curve from the new notes (pitch-neutral edit:
    // the audio already matches) and keep the unedited pitch seamless.
    PitchCurveProcessor::rebuildBaseFromNotes(*project);
    smoothOverBoundaries(*project, startFrame, endFrame, uneditedAround,
                         {startFrame, splitFrame});

    // Edits back on: the neighbours' own, and the source note's on both
    // halves (its drag over each half's unedited centre), keeping what it sang.
    unedited.restoreOthers({originalNote.getUid()}, unedited.curves);
    if (!sourceUntouched)
    {
        for (const int half : {startFrame, splitFrame})
            if (auto* n = noteStartingAt(*project, half))
            {
                n->setMidiNote(n->getOriginalMidiNote() + userShift);
                n->setPitchOffset(0.0f);
                n->setTiltLeft(originalNote.getTiltLeft());
                n->setTiltRight(originalNote.getTiltRight());
                n->setVibrato(originalNote.getVibrato());
                n->setPitchDrift(originalNote.getPitchDrift());
                n->setSmoothLeftFrames(originalNote.getSmoothLeftFrames());
                n->setSmoothRightFrames(originalNote.getSmoothRightFrames());
                n->setDeltaScale(originalNote.getDeltaScale());
                n->setDeltaOffset(originalNote.getDeltaOffset());
            }
        PitchCurveProcessor::rebuildBaseFromNotes(*project);
        keepSungPitch(*project, {startFrame, splitFrame}, startFrame, endFrame,
                      pitchBefore);
    }
    // Baked neighbours keep what they sang over the new base.
    smoothOverBoundaries(*project, startFrame, endFrame, pitchAround,
                         {startFrame, splitFrame}, {startFrame, splitFrame}, true);
    const auto neighbourEdits = unedited.neighbourEdits({startFrame, splitFrame});

    // Create undo action - don't pass callback to avoid lifetime issues
    // UI refresh is handled by UndoManager's onUndoRedo callback
    if (undoManager) {
        const Note* left = noteStartingAt(*project, startFrame);
        const Note* right = noteStartingAt(*project, splitFrame);
        if (left && right)
        {
            auto action = std::make_unique<NoteSplitAction>(
                project, originalNote, *left, *right, nullptr);
            action->setNeighbourEdits(neighbourEdits);
            undoManager->addAction(std::move(action));
        }
    }
    if (onNoteSplit)
        onNoteSplit();

    return true;
}

bool NoteSplitter::splitNoteAtX(Note* note, float x) {
    if (!note || !coordMapper)
        return false;

    // Convert X coordinate to frame
    float pixelsPerSecond = coordMapper->getPixelsPerSecond();
    double time = x / pixelsPerSecond;
    int frame = static_cast<int>(time * SAMPLE_RATE / HOP_SIZE);

    return splitNoteAtFrame(note, frame);
}

std::pair<Note*, Note*> NoteSplitter::findMergeableNotesAt(float x, float y)
{
    if (!project || !coordMapper)
        return {};

    auto &notes = project->getNotes();
    const float pixelsPerSecond = coordMapper->getPixelsPerSecond();
    const float pixelsPerSemitone = coordMapper->getPixelsPerSemitone();
    constexpr float boundaryHitPadding = 5.0f;

    for (size_t i = 0; i < notes.size(); ++i)
    {
        auto &left = notes[i];
        if (left.isRest())
            continue;

        for (size_t j = i + 1; j < notes.size(); ++j)
        {
            auto &right = notes[j];
            if (right.isRest())
                continue;

            // Expose merge whenever adjacent notes form a continuous timeline and
            // source frame boundary, including resized notes across chunks.
            if (left.getEndFrame() != right.getStartFrame() ||
                left.getSrcEndFrame() != right.getSrcStartFrame())
                break;

            const float boundaryX = 0.5f * framesToSeconds(
                left.getEndFrame() + right.getStartFrame()) * pixelsPerSecond;
            const float lineTop = std::min(
                coordMapper->midiToY(left.getAdjustedMidiNote()),
                coordMapper->midiToY(right.getAdjustedMidiNote()));
            const float lineBottom = std::max(
                coordMapper->midiToY(left.getAdjustedMidiNote()),
                coordMapper->midiToY(right.getAdjustedMidiNote())) + pixelsPerSemitone;

            if (std::abs(x - boundaryX) <= boundaryHitPadding &&
                y >= lineTop && y <= lineBottom)
                return {&left, &right};

            break;
        }
    }

    return {};
}

bool NoteSplitter::mergeNotes(Note *first, Note *second)
{
    if (!project || !first || !second || first->isRest() || second->isRest() ||
        first->getEndFrame() != second->getStartFrame() ||
        first->getSrcEndFrame() != second->getSrcStartFrame())
        return false;

    const Note firstEdited = *first;
    const Note secondEdited = *second;

    // Like gluing segments in Cubase's VariAudio, the merged note goes back
    // to the pitch the analysis gave it: the halves' pitch edits (drag,
    // drift, vibrato, tilt, smoothing, drawn curves) are dropped. Undo brings
    // them back.
    const bool reshaped =
        !(pitchUntouched(firstEdited) && pitchUntouched(secondEdited));
    const auto curvesBefore = project->getDrawnCurves();

    // The pitch around the merge: its neighbours must not bend with the base.
    const int mergeStart = firstEdited.getStartFrame();
    const int mergeEnd = secondEdited.getEndFrame();
    const int totalFrames =
        static_cast<int>(project->getAudioData().basePitch.size());
    const int aroundStart = std::max(0, mergeStart - kBoundaryReach);
    const int aroundEnd = std::min(totalFrames, mergeEnd + kBoundaryReach);
    const auto pitchAround =
        sungPitch(project->getAudioData(), aroundStart, aroundEnd);

    // Merge the analysis in the unedited pitch: no note's drag (the halves'
    // or a neighbour's) bends the base the merged curve is measured against.
    UneditedSpace unedited(*project);
    const Note firstNote = *first;
    const Note secondNote = *second;
    const auto uneditedAround =
        sungPitch(project->getAudioData(), aroundStart, aroundEnd);

    Note mergedNote = firstNote;
    mergedNote.setEndFrame(secondNote.getEndFrame());
    mergedNote.setSrcEndFrame(secondNote.getSrcEndFrame());
    {
        auto spans = firstNote.getAnalysisSpans();
        const auto& more = secondNote.getAnalysisSpans();
        spans.insert(spans.end(), more.begin(), more.end());
        mergedNote.setAnalysisSpans(std::move(spans));
    }
    // The stretches kept by hand (Note::setKeptRange). The merged note is
    // rendered the way its first half was. Where that keeps fry and either
    // half was marked by hand, the merged note is marked too: with what the
    // two halves show kept from its start on, and up to its end. A stretch
    // that ends up in the middle goes - there are two squares to say it with.
    if (fry::keepsFry(firstNote) &&
        (firstNote.hasKeptRange() || secondNote.hasKeptRange()))
    {
        const auto& data = project->getAudioData();
        const auto found = fry::find(data);
        const auto left =
            fry::shownRange(firstNote, project->getNotes(), data.rawF0, found);
        const auto right =
            fry::shownRange(secondNote, project->getNotes(), data.rawF0, found);
        const int start = firstNote.getSrcStartFrame();
        const int end = secondNote.getSrcEndFrame();
        const bool leftWhole = left.headUntil >= left.tailFrom;
        const bool rightWhole = right.headUntil >= right.tailFrom;
        const int headUntil =
            leftWhole ? (rightWhole ? end : right.headUntil) : left.headUntil;
        const int tailFrom =
            rightWhole ? (leftWhole ? start : left.tailFrom) : right.tailFrom;
        mergedNote.setKeptRange(headUntil > start ? headUntil : Note::kKeepsNone,
                                tailFrom < end ? tailFrom : Note::kKeepsNone);
    }

    NoteAmplitude::update(mergedNote, project->getAudioData());

    const bool warpedMerged =
        mergedNote.getStartFrame() != mergedNote.getSrcStartFrame() ||
        mergedNote.getDurationFrames() != mergedNote.getSrcDurationFrames();

    auto append = [](auto left, const auto &right)
    {
        left.insert(left.end(), right.begin(), right.end());
        return left;
    };
    mergedNote.setF0Values(append(firstNote.getF0Values(), secondNote.getF0Values()));

    // Restore a single pitch center over the merged duration. Each half may
    // have been rebased when it was split, so convert its deviations through
    // the current absolute pitch before expressing them around the new center.
    const float mergedPitchCenter = calculatePitchCenter(
        project->getAudioData(), mergedNote.getStartFrame(),
        mergedNote.getEndFrame(), firstNote.getAdjustedMidiNote());
    const auto mergeDeviation = [&](
                                  const std::vector<float> &firstValues,
                                  float firstCenter,
                                  const std::vector<float> &secondValues,
                                  float secondCenter)
    {
        const bool bothMatchSrc =
            static_cast<int>(firstValues.size()) == firstNote.getSrcDurationFrames() &&
            static_cast<int>(secondValues.size()) == secondNote.getSrcDurationFrames();
        const bool bothMatchDst =
            static_cast<int>(firstValues.size()) == firstNote.getDurationFrames() &&
            static_cast<int>(secondValues.size()) == secondNote.getDurationFrames();

        const std::vector<float> *leftPtr = &firstValues;
        const std::vector<float> *rightPtr = &secondValues;
        std::vector<float> leftResampled;
        std::vector<float> rightResampled;
        if (!bothMatchSrc && !bothMatchDst &&
            !firstValues.empty() && !secondValues.empty())
        {
            if (static_cast<int>(firstValues.size()) != firstNote.getDurationFrames())
            {
                leftResampled = notetiming::curveOnNote(
                    project->getAudioData(), firstNote, firstValues);
                leftPtr = &leftResampled;
            }
            if (static_cast<int>(secondValues.size()) != secondNote.getDurationFrames())
            {
                rightResampled = notetiming::curveOnNote(
                    project->getAudioData(), secondNote, secondValues);
                rightPtr = &rightResampled;
            }
        }

        std::vector<float> values;
        values.reserve(leftPtr->size() + rightPtr->size());
        for (const auto value : *leftPtr)
            values.push_back(value + firstCenter - mergedPitchCenter);
        for (const auto value : *rightPtr)
            values.push_back(value + secondCenter - mergedPitchCenter);
        return values;
    };
    mergedNote.setOriginalDeltaPitch(mergeDeviation(
        firstNote.getOriginalDeltaPitch(), firstNote.getOriginalMidiNote(),
        secondNote.getOriginalDeltaPitch(), secondNote.getOriginalMidiNote()));
    mergedNote.setDeltaPitch(mergeDeviation(
        firstNote.getDeltaPitch(), firstNote.getAdjustedMidiNote(),
        secondNote.getDeltaPitch(), secondNote.getAdjustedMidiNote()));
    mergedNote.setMidiNote(mergedPitchCenter);
    mergedNote.setOriginalMidiNote(mergedPitchCenter);
    mergedNote.setPitchOffset(0.0f);
    const bool isDirty = firstEdited.isDirty() || secondEdited.isDirty();
    const bool isSynthDirty = firstEdited.isSynthDirty() || secondEdited.isSynthDirty();
    mergedNote.setDirty(isDirty);
    mergedNote.setRenderedEdit(firstEdited.hasRenderedEdit() ||
                               secondEdited.hasRenderedEdit());
    if (isSynthDirty)
        mergedNote.markSynthDirty();
    else
        mergedNote.setSynthDirty(false);

    // Back to the analysis: the detected note itself when the merge made it
    // whole again, otherwise the halves' analysed curves joined above.
    const bool restoredAnalysis = mergedNote.restoreAnalysisPitch();
    if (!restoredAnalysis)
    {
        mergedNote.setTiltLeft(0.0f);
        mergedNote.setTiltRight(0.0f);
        mergedNote.setVibrato(1.0f);
        mergedNote.setPitchDrift(1.0f);
        mergedNote.setSmoothLeftFrames(0);
        mergedNote.setSmoothRightFrames(0);
        mergedNote.setDeltaScale(1.0f);
        mergedNote.setDeltaOffset(0.0f);
        mergedNote.setFrozenTilt({});
        mergedNote.setBakedDeltaPitch({});
        mergedNote.setDeltaPitch(mergedNote.getOriginalDeltaPitch());
        if (reshaped)
        {
            mergedNote.markDirty();
            mergedNote.markSynthDirty();
        }
    }
    else if (!reshaped)
    {
        // Untouched halves sang the analysis already: nothing to render.
        mergedNote.setDirty(isDirty);
        if (!isSynthDirty)
            mergedNote.setSynthDirty(false);
    }

    *first = mergedNote;
    project->removeNoteByStartFrame(secondNote.getStartFrame());

    // Rebuild the drawn pitch curve from the new notes and keep the unedited
    // pitch seamless. A note the merge made whole again sings its analysis
    // as it is.
    PitchCurveProcessor::rebuildBaseFromNotes(*project);
    smoothOverBoundaries(*project, mergeStart, mergeEnd, uneditedAround, {mergeStart},
                         restoredAnalysis ? std::vector<int>{mergeStart}
                                          : std::vector<int>{});

    // The neighbours get their edits back; drawn curves on the halves went
    // with their pitch edits. Baked neighbours keep what they sang.
    unedited.restoreOthers(
        {mergedNote.getUid()},
        reshaped ? drawncurve::without(curvesBefore, mergeStart, mergeEnd)
                 : curvesBefore);
    smoothOverBoundaries(*project, mergeStart, mergeEnd, pitchAround, {mergeStart},
                         {mergeStart}, true);
    const auto neighbourEdits = unedited.neighbourEdits({mergeStart});
    if (const auto *updatedMerged = noteStartingAt(*project, mergeStart))
        mergedNote = *updatedMerged; // refitted to the recording: what redo puts back

    if (undoManager)
    {
        auto action = std::make_unique<NoteMergeAction>(
            project, firstEdited, secondEdited, mergedNote, nullptr);
        action->setDrawnCurves(curvesBefore, project->getDrawnCurves());
        action->setNeighbourEdits(neighbourEdits);
        undoManager->addAction(std::move(action));
    }
    if (onNoteSplit)
        onNoteSplit();

    return true;
}

std::vector<NoteCurveEdit> NoteSplitter::removeNotes(const std::vector<Note>& removed,
                                                     std::vector<DrawnCurve> curvesAfter)
{
    if (!project || removed.empty())
        return {};
    int minStart = std::numeric_limits<int>::max();
    int maxEnd = std::numeric_limits<int>::min();
    for (const auto& n : removed)
    {
        minStart = std::min(minStart, n.getStartFrame());
        maxEnd = std::max(maxEnd, n.getEndFrame());
    }
    const int totalFrames =
        static_cast<int>(project->getAudioData().basePitch.size());
    const int aroundStart = std::max(0, minStart - kBoundaryReach);
    const int aroundEnd = std::min(totalFrames, maxEnd + kBoundaryReach);
    const auto pitchAround =
        sungPitch(project->getAudioData(), aroundStart, aroundEnd);

    // The removed notes' drags must not bend the neighbours' analysed pitch:
    // compensate in the unedited pitch, then put the edits back.
    UneditedSpace unedited(*project);
    const auto uneditedAround =
        sungPitch(project->getAudioData(), aroundStart, aroundEnd);
    for (const auto& n : removed)
        project->removeNoteByStartFrame(n.getStartFrame());
    PitchCurveProcessor::rebuildBaseFromNotes(*project);
    smoothOverBoundaries(*project, minStart, maxEnd, uneditedAround, {});

    unedited.restoreOthers({}, std::move(curvesAfter));
    smoothOverBoundaries(*project, minStart, maxEnd, pitchAround, {}, {}, true);
    return unedited.neighbourEdits({});
}

std::optional<NoteSplitter::SeparationGroup>
NoteSplitter::separationGroupFor(const Note& note) const
{
    if (!project || note.isRest() || note.getAnalysisSpans().empty())
        return std::nullopt;

    // Every current note that shares an analysed note with the group joins
    // it, until nothing more does: the pieces of a split note, the notes a
    // merge swallowed.
    auto spans = note.getAnalysisSpans();
    std::vector<const Note*> members{&note};
    const auto& notes = project->getNotes();
    for (bool grew = true; grew;)
    {
        grew = false;
        for (const auto& candidate : notes)
        {
            if (candidate.isRest() ||
                std::find(members.begin(), members.end(), &candidate) != members.end())
                continue;
            const auto& own = candidate.getAnalysisSpans();
            const bool shares = std::any_of(own.begin(), own.end(), [&](const auto& s)
                { return std::find(spans.begin(), spans.end(), s) != spans.end(); });
            if (!shares)
                continue;
            members.push_back(&candidate);
            spans.insert(spans.end(), own.begin(), own.end());
            grew = true;
        }
    }
    std::sort(spans.begin(), spans.end());
    spans.erase(std::unique(spans.begin(), spans.end()), spans.end());
    std::sort(members.begin(), members.end(), [](const Note* a, const Note* b)
              { return a->getSrcStartFrame() < b->getSrcStartFrame(); });

    SeparationGroup group;
    group.srcStart = members.front()->getSrcStartFrame();
    group.srcEnd = members.back()->getSrcEndFrame();
    for (size_t i = 0; i < members.size(); ++i)
    {
        // Only a run of touching notes can be merged and recut.
        if (i > 0 && (members[i - 1]->getSrcEndFrame() != members[i]->getSrcStartFrame() ||
                      members[i - 1]->getEndFrame() != members[i]->getStartFrame()))
            return std::nullopt;
        if (i > 0)
            group.currentCuts.push_back(members[i]->getSrcStartFrame());
        group.notes.push_back(*members[i]);
    }
    for (const auto& span : spans)
        for (const int edge : {span.first, span.second})
            if (edge > group.srcStart && edge < group.srcEnd)
                group.targetCuts.push_back(edge);
    std::sort(group.targetCuts.begin(), group.targetCuts.end());
    group.targetCuts.erase(std::unique(group.targetCuts.begin(), group.targetCuts.end()),
                           group.targetCuts.end());
    return group;
}

bool NoteSplitter::canRestoreSeparation(const Note& note) const
{
    const auto group = separationGroupFor(note);
    return group && group->currentCuts != group->targetCuts;
}

bool NoteSplitter::restoreSeparation(const SeparationGroup& group,
                                     std::vector<Note>& after)
{
    if (!project || group.notes.empty() || group.currentCuts == group.targetCuts)
        return false;

    // The steps below are one edit: no undo entries or callbacks of their own.
    const auto savedUndo = undoManager;
    const auto savedCallback = onNoteSplit;
    undoManager = nullptr;
    onNoteSplit = nullptr;

    auto noteAtSource = [this](int srcStart) -> Note*
    {
        for (auto& n : project->getNotes())
            if (!n.isRest() && n.getSrcStartFrame() == srcStart)
                return &n;
        return nullptr;
    };

    // 1. One note over the whole group.
    for (size_t i = 1; i < group.notes.size(); ++i)
    {
        Note* left = noteAtSource(group.srcStart);
        Note* right = noteAtSource(group.notes[i].getSrcStartFrame());
        if (!left || !right || !mergeNotes(left, right))
            break;
    }

    // 2. Cut it where the analysis did. The split works on the timeline, so
    //    each cut goes through the warp map and the source edges are then set
    //    to the analysed frame exactly.
    const auto& map = project->getAudioData().warpMap;
    for (const int cut : group.targetCuts)
    {
        Note* note = nullptr;
        for (auto& n : project->getNotes())
            if (!n.isRest() && n.getSrcStartFrame() < cut && n.getSrcEndFrame() > cut)
                note = &n;
        if (!note)
            continue;
        const int leftSource = note->getSrcStartFrame();
        const auto spans = note->getAnalysisSpans();
        const int at = static_cast<int>(std::lround(map.destinationAt(cut)));
        if (!splitNoteAtFrame(note, at))
            continue;
        Note* left = noteAtSource(leftSource);
        Note* right = nullptr;
        for (auto& n : project->getNotes())
            if (!n.isRest() && n.getStartFrame() == at && &n != left)
                right = &n;
        if (!left || !right)
            continue;
        left->setSrcEndFrame(cut);
        right->setSrcStartFrame(cut);
        left->setAnalysisSpans(spans);
        right->setAnalysisSpans(spans);
        left->trimAnalysisSpans();
        right->trimAnalysisSpans();
    }

    // 3. Each note that is a detected note again gets its analysed pitch
    //    back. (Merging and splitting re-centred notes on their mean f0 and
    //    folded pitch edits into the curves.)
    for (auto& n : project->getNotes())
    {
        if (n.isRest() || n.getSrcStartFrame() < group.srcStart ||
            n.getSrcEndFrame() > group.srcEnd)
            continue;
        // A detected note again: its whole analysed pitch comes back, as in
        // Cubase (pitch moves, curve edits and pitch tools are undone).
        if (n.restoreAnalysisPitch())
            continue;
        // Spans saved without their curve: only the centre.
        const auto& own = n.getAnalysisSpans();
        if (own.size() != 1 || own.front().first != n.getSrcStartFrame() ||
            own.front().second != n.getSrcEndFrame() || !(own.front().midi > 0.0f))
            continue;
        const float centre = n.getAdjustedMidiNote();
        const float analysed = own.front().midi;
        rebasePitchDeviation(n, centre - analysed);
        n.setMidiNote(analysed);
        n.setOriginalMidiNote(analysed);
        n.setPitchOffset(0.0f);
    }

    undoManager = savedUndo;
    onNoteSplit = savedCallback;

    // Step 3 moved the centres again: refit the untouched notes in and
    // around the group to the recording over the base as it ends up.
    PitchCurveProcessor::rebuildBaseFromNotes(*project);
    {
        int first = std::numeric_limits<int>::max();
        int last = std::numeric_limits<int>::min();
        for (const auto& n : project->getNotes())
            if (!n.isRest() && n.getSrcStartFrame() >= group.srcStart &&
                n.getSrcEndFrame() <= group.srcEnd)
            {
                first = std::min(first, n.getStartFrame());
                last = std::max(last, n.getEndFrame());
            }
        if (first <= last)
            PitchCurveProcessor::refitUntouchedCurves(*project, first - kBoundaryReach,
                                                      last + kBoundaryReach);
    }

    after.clear();
    for (const auto& n : project->getNotes())
        if (!n.isRest() && n.getSrcStartFrame() >= group.srcStart &&
            n.getSrcEndFrame() <= group.srcEnd)
            after.push_back(n);
    return true;
}

bool NoteSplitter::previewResizeNote(Note* note, const Note& originalNote,
                                     int newSrcStartFrame, int newSrcEndFrame)
{
    if (!project || !note || note->isRest() || newSrcEndFrame <= newSrcStartFrame)
        return false;

    const auto uid = originalNote.getUid();
    *note = originalNote;
    sortNotesByStartFrame(*project);
    PitchCurveProcessor::rebuildBaseFromNotes(*project);
    const auto baseBefore = project->getAudioData().basePitch;

    Note* activeNote = findResizedNote(*project, uid, originalNote.getStartFrame());
    if (!activeNote)
        return false;

    applyResizedNoteState(*project, *activeNote, originalNote,
                          newSrcStartFrame, newSrcEndFrame, baseBefore);
    Note* resized = findResizedNote(*project, uid,
                                    project->getAudioData().warpMap.isIdentity()
                                        ? newSrcStartFrame
                                        : static_cast<int>(std::lround(
                                              project->getAudioData().warpMap.destinationAt(
                                                  newSrcStartFrame))));
    const int newStart = resized ? resized->getStartFrame() : originalNote.getStartFrame();
    const int newEnd = resized ? resized->getEndFrame() : originalNote.getEndFrame();
    const int spanStart = std::min(originalNote.getStartFrame(), newStart);
    const int spanEnd = std::max(originalNote.getEndFrame(), newEnd);
    const int reachStart = std::max(0, spanStart - kBoundaryReach);
    const int reachEnd = spanEnd + kBoundaryReach;

    compensateNeighbourBaseChange(*project, baseBefore, reachStart, reachEnd, uid, newStart);
    PitchCurveProcessor::rebuildBaseFromNotes(*project);
    PitchCurveProcessor::refitUntouchedCurves(*project, reachStart, reachEnd);
    return true;
}

bool NoteSplitter::commitResizeNote(Note* note, const Note& originalNote,
                                    int newSrcStartFrame, int newSrcEndFrame)
{
    if (!project || !note || note->isRest() || newSrcEndFrame <= newSrcStartFrame)
        return false;

    const auto uid = originalNote.getUid();
    const auto& map = project->getAudioData().warpMap;
    const int newStart = map.isIdentity()
        ? newSrcStartFrame
        : static_cast<int>(std::lround(map.destinationAt(newSrcStartFrame)));
    int newEnd = map.isIdentity()
        ? newSrcEndFrame
        : static_cast<int>(std::lround(map.destinationAt(newSrcEndFrame)));
    if (newSrcEndFrame > newSrcStartFrame)
        newEnd = std::max(newEnd, newStart + 1);

    // Restore original state first so baseBefore matches the pre-resize layout.
    *note = originalNote;
    sortNotesByStartFrame(*project);
    PitchCurveProcessor::rebuildBaseFromNotes(*project);
    const auto baseBefore = project->getAudioData().basePitch;

    const int spanStart = std::min(originalNote.getStartFrame(), newStart);
    const int spanEnd = std::max(originalNote.getEndFrame(), newEnd);
    const int reachStart = std::max(0, spanStart - kBoundaryReach);
    const int reachEnd = spanEnd + kBoundaryReach;

    const auto curvesBefore = project->getDrawnCurves();
    auto curvesAfter = curvesBefore;
    if (newStart > originalNote.getStartFrame())
        curvesAfter = drawncurve::without(curvesAfter, originalNote.getStartFrame(), newStart);
    if (newEnd < originalNote.getEndFrame())
        curvesAfter = drawncurve::without(curvesAfter, newEnd, originalNote.getEndFrame());
    project->getDrawnCurves() = curvesAfter;

    Note* activeNote = findResizedNote(*project, uid, originalNote.getStartFrame());
    if (!activeNote)
        return false;

    applyResizedNoteState(*project, *activeNote, originalNote,
                          newSrcStartFrame, newSrcEndFrame, baseBefore);
    activeNote = findResizedNote(*project, uid, newStart);

    const bool needsResynthesis =
        !originalNote.isNeutralForOriginalWaveform() ||
        originalNote.hasRenderedEdit() ||
        std::abs(project->getFormantShift()) > 0.001f;
    if (needsResynthesis)
    {
        if (activeNote)
        {
            activeNote->markDirty();
            activeNote->markSynthDirty();
        }
        project->getAudioData().warpLayoutPending = true;
        project->markNoteEditDirtyRange(spanStart, spanEnd);
    }

    auto neighbourEdits = compensateNeighbourBaseChange(
        *project, baseBefore, reachStart, reachEnd, uid, newStart);
    PitchCurveProcessor::rebuildBaseFromNotes(*project);

    // Unedited notes, the resized one included, sing the recording over the
    // new base. Record what that changes in the neighbours for undo.
    {
        std::vector<Note> prior;
        for (const auto& n : project->getNotes())
            if (!n.isRest() && n.getUid() != uid)
                prior.push_back(n);
        for (const int changed :
             PitchCurveProcessor::refitUntouchedCurves(*project, reachStart, reachEnd))
        {
            const Note* now = noteStartingAt(*project, changed);
            if (!now || now->getUid() == uid)
                continue;
            auto edit = std::find_if(neighbourEdits.begin(), neighbourEdits.end(),
                                     [changed](const NoteCurveEdit& e)
                                     { return e.startFrame == changed; });
            if (edit != neighbourEdits.end())
            {
                edit->originalAfter = now->getOriginalDeltaPitch();
                edit->bakedAfter = now->getBakedDeltaPitch();
                continue;
            }
            const auto was = std::find_if(prior.begin(), prior.end(),
                                          [now](const Note& n)
                                          { return n.getUid() == now->getUid(); });
            if (was != prior.end())
                neighbourEdits.push_back({changed, was->getOriginalDeltaPitch(),
                                          now->getOriginalDeltaPitch(),
                                          was->getBakedDeltaPitch(),
                                          now->getBakedDeltaPitch()});
        }
    }
    activeNote = findResizedNote(*project, uid, newStart);

    const Note afterNote = activeNote ? *activeNote : originalNote;
    if (undoManager)
    {
        undoManager->addAction(std::make_unique<NoteResizeAction>(
            project, originalNote, afterNote,
            std::move(neighbourEdits),
            std::move(curvesBefore), std::move(curvesAfter),
            needsResynthesis, nullptr));
    }

    project->setModified(true);
    if (onNoteSplit)
        onNoteSplit();
    return true;
}
