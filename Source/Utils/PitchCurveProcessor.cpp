#include "PitchCurveProcessor.h"
#include "BasePitchCurve.h"
#include "PitchToolOperations.h"
#include "../Audio/NoteTiming.h"
#include "../Utils/Constants.h"
#include <algorithm>
#include <cmath>

namespace
{
    inline float safeFreqToMidi(float freq)
    {
        if (freq <= 0.0f)
            return 0.0f;
        return freqToMidi(freq);
    }

    void ensureSizes(AudioData& audioData, int totalFrames)
    {
        if (totalFrames <= 0)
            return;

        if (audioData.basePitch.size() != static_cast<size_t>(totalFrames))
            audioData.basePitch.assign(static_cast<size_t>(totalFrames), 0.0f);
        if (audioData.deltaPitch.size() != static_cast<size_t>(totalFrames))
            audioData.deltaPitch.assign(static_cast<size_t>(totalFrames), 0.0f);
    }

    const std::vector<float>& sourceDeltaFor(const Note& note)
    {
        return note.getActiveDeltaPitch();
    }

    PitchToolOperations::AdjacentNoteContext buildAdjacentContext(
        const std::vector<Note>& allNotes, const Note& targetNote)
    {
        PitchToolOperations::AdjacentNoteContext ctx;

        const Note* prevNote = nullptr;
        int prevNoteEndFrame = -1;
        for (const auto& candidate : allNotes) {
            if (&candidate == &targetNote || candidate.isRest())
                continue;
            const int candidateEnd = candidate.getEndFrame();
            if (candidateEnd <= targetNote.getStartFrame() && candidateEnd > prevNoteEndFrame) {
                prevNote = &candidate;
                prevNoteEndFrame = candidateEnd;
            }
        }

        const Note* nextNote = nullptr;
        int nextNoteStartFrame = std::numeric_limits<int>::max();
        for (const auto& candidate : allNotes) {
            if (&candidate == &targetNote || candidate.isRest())
                continue;
            const int candidateStart = candidate.getStartFrame();
            if (candidateStart >= targetNote.getEndFrame() && candidateStart < nextNoteStartFrame) {
                nextNote = &candidate;
                nextNoteStartFrame = candidateStart;
            }
        }

        if (prevNote) {
            const auto& prevDelta = sourceDeltaFor(*prevNote);
            if (!prevDelta.empty()) {
                ctx.hasLeft = true;
                ctx.leftBoundaryDelta = prevDelta.back();
            }
        }
        if (nextNote) {
            const auto& nextDelta = sourceDeltaFor(*nextNote);
            if (!nextDelta.empty()) {
                ctx.hasRight = true;
                ctx.rightBoundaryDelta = nextDelta.front();
            }
        }

        return ctx;
    }

    std::vector<BasePitchCurve::NoteSegment> collectNoteSegments(const std::vector<Note>& notes)
    {
        std::vector<BasePitchCurve::NoteSegment> segments;
        segments.reserve(notes.size());

        for (const auto& note : notes)
        {
            if (note.isRest())
                continue;

            BasePitchCurve::NoteSegment seg;
            seg.startFrame = note.getStartFrame();
            seg.endFrame = note.getEndFrame();
            // Base pitch already includes per-note offset
            seg.midiNote = note.getMidiNote() + note.getPitchOffset()
                         - note.getTiltMean();
            segments.push_back(seg);
        }

        // Ensure segments are sorted by start frame for stable generation
        std::sort(segments.begin(), segments.end(),
                  [](const auto& a, const auto& b) { return a.startFrame < b.startFrame; });
        return segments;
    }
} // namespace

namespace PitchCurveProcessor
{
    std::vector<float> interpolateWithUvMask(const std::vector<float>& pitchHz,
                                             const std::vector<bool>& uvMask)
    {
        if (pitchHz.empty())
            return {};

        const int n = static_cast<int>(pitchHz.size());
        std::vector<float> dense(pitchHz);

        auto isVoicedFrame = [&](int i) -> bool {
            if (i < 0 || i >= n)
                return false;

            const bool hasPitch = pitchHz[static_cast<size_t>(i)] > 0.0f;
            if (!hasPitch)
                return false;

            // If uvMask is provided, respect it (out-of-range treated as unvoiced).
            // If uvMask is missing, fall back to pitch presence.
            if (!uvMask.empty())
                return i < static_cast<int>(uvMask.size()) && uvMask[static_cast<size_t>(i)];
            return true;
        };

        int nextVoiced = -1;
        auto findNext = [&](int idx) -> int {
            for (int i = idx; i < n; ++i)
            {
                if (isVoicedFrame(i))
                    return i;
            }
            return -1;
        };

        int lastVoiced = -1;
        nextVoiced = findNext(0);

        for (int i = 0; i < n; ++i)
        {
            const bool voiced = isVoicedFrame(i);
            if (voiced)
            {
                lastVoiced = i;
                if (i == nextVoiced)
                    nextVoiced = findNext(i + 1);
                continue;
            }

            // Update next voiced lazily
            if (nextVoiced != -1 && nextVoiced < i)
                nextVoiced = findNext(i + 1);

            float prevVal = (lastVoiced >= 0) ? dense[static_cast<size_t>(lastVoiced)] : 0.0f;
            float nextVal = (nextVoiced >= 0) ? dense[static_cast<size_t>(nextVoiced)] : 0.0f;

            if (prevVal <= 0.0f && nextVal <= 0.0f)
            {
                dense[static_cast<size_t>(i)] = 0.0f;
                continue;
            }

            if (prevVal <= 0.0f)
            {
                dense[static_cast<size_t>(i)] = nextVal;
                continue;
            }
            if (nextVal <= 0.0f)
            {
                dense[static_cast<size_t>(i)] = prevVal;
                continue;
            }

            const float t = (nextVoiced > i) ? static_cast<float>(i - lastVoiced) /
                                               static_cast<float>(nextVoiced - lastVoiced)
                                             : 0.0f;
            const float logA = std::log(prevVal);
            const float logB = std::log(nextVal);
            dense[static_cast<size_t>(i)] = std::exp(logA * (1.0f - t) + logB * t);
        }

        return dense;
    }

    void rebuildCurvesFromSource(Project& project,
                                 const std::vector<float>& sourcePitchHz)
    {
        auto& audioData = project.getAudioData();
        const int totalFrames = static_cast<int>(sourcePitchHz.size());
        ensureSizes(audioData, totalFrames);

        // Ensure delta is built from a dense F0 source that includes UV
        // head/tail fill (same behavior expectation as F0 interpolation).
        std::vector<float> denseSource = sourcePitchHz;
        if (!audioData.voicedMask.empty() &&
            audioData.voicedMask.size() == sourcePitchHz.size())
        {
            denseSource = interpolateWithUvMask(sourcePitchHz, audioData.voicedMask);
        }

        auto segments = collectNoteSegments(project.getNotes());
        if (!segments.empty())
        {
            audioData.basePitch = BasePitchCurve::generateForNotes(segments, totalFrames);
        }

        if (audioData.basePitch.size() != static_cast<size_t>(totalFrames))
        {
            // Fallback: derive base from source pitch directly
            audioData.basePitch.assign(static_cast<size_t>(totalFrames), 0.0f);
            for (int i = 0; i < totalFrames; ++i)
                audioData.basePitch[static_cast<size_t>(i)] = safeFreqToMidi(denseSource[i]);
        }

        // Dense delta: midi(source) - base
        audioData.deltaPitch.assign(static_cast<size_t>(totalFrames), 0.0f);
        for (int i = 0; i < totalFrames; ++i)
        {
            const float base = audioData.basePitch[static_cast<size_t>(i)];
            const float midi = safeFreqToMidi(denseSource[i]);
            audioData.deltaPitch[static_cast<size_t>(i)] = midi - base;
        }

        // Initialize originalDeltaPitch in each note from computed deltaPitch
        // This preserves the pristine pitch curve for non-destructive transformations
        for (auto& note : project.getNotes())
        {
            if (note.isRest()) continue;

            const int startFrame = note.getStartFrame();
            const int endFrame = note.getEndFrame();
            const int numFrames = endFrame - startFrame;

            if (numFrames <= 0) continue;

            std::vector<float> origDelta(static_cast<size_t>(numFrames));
            for (int i = 0; i < numFrames; ++i)
            {
                const int globalIdx = startFrame + i;
                if (globalIdx >= 0 && globalIdx < totalFrames)
                    origDelta[static_cast<size_t>(i)] = audioData.deltaPitch[static_cast<size_t>(globalIdx)];
            }
            note.setOriginalDeltaPitch(std::move(origDelta));
        }

        // A note the analysis just made remembers its pitch curve with its
        // analysed range ("Restore note separation", Pitch restore).
        for (auto& note : project.getNotes())
        {
            if (note.isRest() || note.getAnalysisSpans().size() != 1)
                continue;
            auto spans = note.getAnalysisSpans();
            auto& span = spans.front();
            if (span.first != note.getSrcStartFrame() ||
                span.second != note.getSrcEndFrame() || !span.originalDelta.empty())
                continue;
            span.midi = note.getMidiNote();
            span.originalDelta = note.getOriginalDeltaPitch();
            note.setAnalysisSpans(std::move(spans));
        }

        // Cache base F0 (Hz) for backwards compatibility
        audioData.baseF0.resize(static_cast<size_t>(totalFrames));
        for (int i = 0; i < totalFrames; ++i)
            audioData.baseF0[static_cast<size_t>(i)] = midiToFreq(audioData.basePitch[static_cast<size_t>(i)]);

        composeF0InPlace(project, /*applyUvMask=*/false);
    }

    std::vector<float> getDriftPreviewResponse(const Project& project, const Note& note)
    {
        const int frames = note.getEndFrame() - note.getStartFrame();
        if (frames <= 0 || sourceDeltaFor(note).empty()) return {};
        auto source = notetiming::curveOnNote(project.getAudioData(), note, sourceDeltaFor(note));
        const auto context = buildAdjacentContext(project.getNotes(), note);
        auto zero = PitchToolOperations::applyAllTransformations(source,
            note.getTiltLeft(), note.getTiltRight(), note.getVibrato(),
            note.getSmoothLeftFrames(), note.getSmoothRightFrames(), context, 0.0f,
            note.getTiltPivot(), &note.getFrozenTilt());
        auto one = PitchToolOperations::applyAllTransformations(source,
            note.getTiltLeft(), note.getTiltRight(), note.getVibrato(),
            note.getSmoothLeftFrames(), note.getSmoothRightFrames(), context, 1.0f,
            note.getTiltPivot(), &note.getFrozenTilt());
        for (size_t i = 0; i < one.size(); ++i)
            one[i] = (one[i] - zero[i]) * note.getDeltaScale();
        return one;
    }

    std::vector<int> refitUntouchedCurves(Project& project, int start, int end)
    {
        std::vector<int> changed;
        auto& audio = project.getAudioData();
        const auto& dense = audio.denseF0;
        if (dense.empty() || audio.basePitch.empty())
            return changed;
        for (auto& note : project.getNotes())
        {
            if (note.isRest() || note.getEndFrame() <= start || note.getStartFrame() >= end ||
                !note.isPitchUntouched())
                continue;
            if (note.getStartFrame() != note.getSrcStartFrame() ||
                note.getDurationFrames() != note.getSrcDurationFrames())
            {
                // Stretched: fit each frame on the timeline to the recorded
                // pitch the warp plays there, so the note sings the recording
                // exactly. The curve is per timeline frame, so it is refitted
                // whenever the warp changes (warp::apply).
                const int frames = note.getDurationFrames();
                if (frames <= 0 || note.getSrcDurationFrames() <= 0)
                    continue;
                const auto& stored = note.getOriginalDeltaPitch();
                auto fitted = notetiming::curveOnNote(audio, note, stored);
                fitted.resize(static_cast<size_t>(frames), 0.0f);
                // The track's own map, not the note's slice of it: the slice
                // runs between rounded note edges and drifts off it near them.
                const auto& map = audio.warpMap;
                // The recording is continuous: a frame reads it where the
                // warp plays it, past the note's own recorded frames at its
                // ends too, so where the note is cut changes nothing.
                const double srcFirst = 0.0;
                const double srcLast = static_cast<double>(dense.size() - 1);
                bool refit = static_cast<int>(stored.size()) != frames;
                for (int i = 0; i < frames; ++i)
                {
                    const int t = note.getStartFrame() + i;
                    if (t < 0 || static_cast<size_t>(t) >= audio.basePitch.size())
                        continue;
                    const double src = std::clamp(
                        map.sourceAt(static_cast<double>(t) + 0.5) - 0.5, srcFirst, srcLast);
                    const auto a = static_cast<size_t>(std::floor(src));
                    const auto b = std::min(a + 1, static_cast<size_t>(srcLast));
                    if (!(dense[a] > 0.0f) || !(dense[b] > 0.0f))
                        continue;
                    const float k = static_cast<float>(src - static_cast<double>(a));
                    const float value = freqToMidi(dense[a]) * (1.0f - k) +
                                        freqToMidi(dense[b]) * k -
                                        audio.basePitch[static_cast<size_t>(t)];
                    if (std::abs(value - fitted[static_cast<size_t>(i)]) > 0.0001f)
                    {
                        fitted[static_cast<size_t>(i)] = value;
                        refit = true;
                    }
                }
                if (!refit)
                    continue;
                note.setOriginalDeltaPitch(fitted);
                if (!note.getDeltaPitch().empty())
                    note.setDeltaPitch(fitted);
                changed.push_back(note.getStartFrame());
                continue;
            }
            auto curve = note.getOriginalDeltaPitch();
            if (static_cast<int>(curve.size()) != note.getDurationFrames())
                continue;
            bool touched = false;
            for (int i = 0; i < note.getDurationFrames(); ++i)
            {
                const auto f = static_cast<size_t>(note.getStartFrame() + i);
                if (f >= dense.size() || f >= audio.basePitch.size() || !(dense[f] > 0.0f))
                    continue;
                const float fitted = freqToMidi(dense[f]) - audio.basePitch[f];
                if (std::abs(fitted - curve[static_cast<size_t>(i)]) > 0.0001f)
                {
                    curve[static_cast<size_t>(i)] = fitted;
                    touched = true;
                }
            }
            if (!touched)
                continue;
            note.setOriginalDeltaPitch(curve);
            if (note.getDeltaPitch().size() == curve.size())
                note.setDeltaPitch(curve);
            changed.push_back(note.getStartFrame());
        }
        if (!changed.empty())
            rebuildBaseFromNotes(project);
        return changed;
    }

    void rebuildBaseFromNotes(Project& project)
    {
        auto& audioData = project.getAudioData();
        const int totalFrames = audioData.getNumFrames();
        ensureSizes(audioData, totalFrames);

        auto segments = collectNoteSegments(project.getNotes());
        if (!segments.empty())
        {
            audioData.basePitch = BasePitchCurve::generateForNotes(segments, totalFrames);
        }

        if (audioData.basePitch.size() != static_cast<size_t>(totalFrames))
        {
            audioData.basePitch.assign(static_cast<size_t>(totalFrames), 0.0f);
        }

        // CRITICAL: Rebuild deltaPitch from Note objects NON-DESTRUCTIVELY
        // Clear global deltaPitch first
        audioData.deltaPitch.assign(static_cast<size_t>(totalFrames), 0.0f);
        
        // Composite each note's deltaPitch curve (WITH transformations applied) into audioData.deltaPitch
        const auto& allNotes = project.getNotes();
        for (const auto& note : allNotes)
        {
            if (note.isRest())
                continue;
                
            // A committed baked contour overrides the immutable analysis
            // contour. Legacy projects may only have deltaPitch.
            const auto& rawSourceData = sourceDeltaFor(note);
            if (rawSourceData.empty())
                continue;

            const int startFrame = note.getStartFrame();
            const int endFrame = note.getEndFrame();
            const int numFrames = endFrame - startFrame;
            if (numFrames <= 0)
                continue;

            // Lay the stored contour over the note's current frames: a
            // stretched note holds its vowel, so the contour follows the
            // audio it was written over (NoteTiming.h).
            std::vector<float> resampledBuf;
            const std::vector<float>* sourceDataPtr = &rawSourceData;
            if (static_cast<int>(rawSourceData.size()) != numFrames)
            {
                resampledBuf = notetiming::curveOnNote(audioData, note, rawSourceData);
                sourceDataPtr = &resampledBuf;
            }
            const auto& sourceData = *sourceDataPtr;
            
            // Build adjacent note context
            auto adjacentContext = buildAdjacentContext(allNotes, note);
            
            // Apply all transformation parameters NON-DESTRUCTIVELY
            std::vector<float> transformedDelta = PitchToolOperations::applyAllTransformations(
                sourceData,
                note.getTiltLeft(),
                note.getTiltRight(),
                note.getVibrato(),
                note.getSmoothLeftFrames(),
                note.getSmoothRightFrames(),
                adjacentContext,
                note.getPitchDrift(),
                note.getTiltPivot(),
                &note.getFrozenTilt()
            );

            // Apply per-note delta scale/offset (from delta control handles)
            const float dScale = note.getDeltaScale();
            const float dOffset = note.getDeltaOffset();
            if (std::abs(dScale - 1.0f) > 0.0001f || std::abs(dOffset) > 0.0001f)
            {
                for (auto& v : transformedDelta)
                    v = v * dScale + dOffset;
            }
            
            for (int i = 0; i < numFrames && i < static_cast<int>(transformedDelta.size()); ++i)
            {
                const int globalIdx = startFrame + i;
                if (globalIdx >= 0 && globalIdx < totalFrames)
                    audioData.deltaPitch[static_cast<size_t>(globalIdx)] = transformedDelta[static_cast<size_t>(i)];
            }
        }

        // Update cached baseF0
        audioData.baseF0.resize(static_cast<size_t>(totalFrames));
        for (int i = 0; i < totalFrames; ++i)
            audioData.baseF0[static_cast<size_t>(i)] = midiToFreq(audioData.basePitch[static_cast<size_t>(i)]);

        composeF0InPlace(project, /*applyUvMask=*/false);
    }

    void rebuildBaseFromNotesForDrag(Project& project, const std::vector<Note*>& affectedNotes)
    {
        auto& audioData = project.getAudioData();
        const int totalFrames = audioData.getNumFrames();
        ensureSizes(audioData, totalFrames);

        // 1. Regenerate basePitch from ALL notes (needed for correct pitch display)
        auto segments = collectNoteSegments(project.getNotes());
        if (!segments.empty())
        {
            audioData.basePitch = BasePitchCurve::generateForNotes(segments, totalFrames);
        }

        if (audioData.basePitch.size() != static_cast<size_t>(totalFrames))
        {
            audioData.basePitch.assign(static_cast<size_t>(totalFrames), 0.0f);
        }

        // 2. Only rebuild deltaPitch for the affected notes (not all notes)
        //    First, zero the affected range in the global deltaPitch array
        int minFrame = totalFrames;
        int maxFrame = 0;
        for (auto* note : affectedNotes)
        {
            if (!note || note->isRest())
                continue;
            minFrame = std::min(minFrame, note->getStartFrame());
            maxFrame = std::max(maxFrame, note->getEndFrame());
        }
        minFrame = std::max(0, minFrame);
        maxFrame = std::min(totalFrames, maxFrame);
        for (int i = minFrame; i < maxFrame; ++i)
            audioData.deltaPitch[static_cast<size_t>(i)] = 0.0f;

        const auto& allNotes = project.getNotes();
        for (auto* note : affectedNotes)
        {
            if (!note || note->isRest())
                continue;

            const auto& rawSourceData = sourceDeltaFor(*note);
            if (rawSourceData.empty())
                continue;

            const int startFrame = note->getStartFrame();
            const int endFrame = note->getEndFrame();
            const int numFrames = endFrame - startFrame;
            if (numFrames <= 0)
                continue;

            std::vector<float> resampledBuf;
            const std::vector<float>* sourceDataPtr = &rawSourceData;
            if (static_cast<int>(rawSourceData.size()) != numFrames)
            {
                resampledBuf = notetiming::curveOnNote(audioData, *note, rawSourceData);
                sourceDataPtr = &resampledBuf;
            }
            const auto& sourceData = *sourceDataPtr;

            auto adjacentContext = buildAdjacentContext(allNotes, *note);

            std::vector<float> transformedDelta = PitchToolOperations::applyAllTransformations(
                sourceData,
                note->getTiltLeft(),
                note->getTiltRight(),
                note->getVibrato(),
                note->getSmoothLeftFrames(),
                note->getSmoothRightFrames(),
                adjacentContext,
                note->getPitchDrift(),
                note->getTiltPivot(),
                &note->getFrozenTilt()
            );

            const float dScale = note->getDeltaScale();
            const float dOffset = note->getDeltaOffset();
            if (std::abs(dScale - 1.0f) > 0.0001f || std::abs(dOffset) > 0.0001f)
            {
                for (auto& v : transformedDelta)
                    v = v * dScale + dOffset;
            }

            for (int i = 0; i < numFrames && i < static_cast<int>(transformedDelta.size()); ++i)
            {
                const int globalIdx = startFrame + i;
                if (globalIdx >= 0 && globalIdx < totalFrames)
                    audioData.deltaPitch[static_cast<size_t>(globalIdx)] = transformedDelta[static_cast<size_t>(i)];
            }
        }

        // 3. Update baseF0 only for the affected range
        audioData.baseF0.resize(static_cast<size_t>(totalFrames));
        for (int i = minFrame; i < maxFrame; ++i)
            audioData.baseF0[static_cast<size_t>(i)] = midiToFreq(audioData.basePitch[static_cast<size_t>(i)]);

        // 4. Recompose f0 only for the affected range (with padding for smoothing)
        const int f0Start = std::max(0, minFrame - 60);
        const int f0End = std::min(totalFrames, maxFrame + 60);
        for (int i = f0Start; i < f0End; ++i)
        {
            const float base = audioData.basePitch[static_cast<size_t>(i)];
            const float delta = audioData.deltaPitch[static_cast<size_t>(i)];
            audioData.f0[static_cast<size_t>(i)] = midiToFreq(base + delta);
        }
    }

    void rebuildDeltaForNotes(Project& project, const std::vector<Note*>& affectedNotes)
    {
        auto& audioData = project.getAudioData();
        const int totalFrames = audioData.getNumFrames();
        if (totalFrames <= 0 || affectedNotes.empty())
            return;

        const auto& allNotes = project.getNotes();
        int minAffectedFrame = totalFrames;
        int maxAffectedFrame = 0;

        for (auto* note : affectedNotes)
        {
            if (!note || note->isRest())
                continue;

            const auto& rawSourceData = sourceDeltaFor(*note);
            if (rawSourceData.empty())
                continue;

            const int startFrame = note->getStartFrame();
            const int endFrame = note->getEndFrame();
            const int numFrames = endFrame - startFrame;
            if (numFrames <= 0)
                continue;

            // Lay the stored contour over the note's current frames: a
            // stretched note holds its vowel, so the contour follows the
            // audio it was written over (NoteTiming.h).
            std::vector<float> resampledBuf;
            const std::vector<float>* sourceDataPtr = &rawSourceData;
            if (static_cast<int>(rawSourceData.size()) != numFrames)
            {
                resampledBuf = notetiming::curveOnNote(audioData, *note, rawSourceData);
                sourceDataPtr = &resampledBuf;
            }
            const auto& sourceData = *sourceDataPtr;

            // Build adjacent note context
            auto adjacentContext = buildAdjacentContext(allNotes, *note);

            std::vector<float> transformedDelta = PitchToolOperations::applyAllTransformations(
                sourceData,
                note->getTiltLeft(),
                note->getTiltRight(),
                note->getVibrato(),
                note->getSmoothLeftFrames(),
                note->getSmoothRightFrames(),
                adjacentContext,
                note->getPitchDrift(),
                note->getTiltPivot(),
                &note->getFrozenTilt()
            );

            // Apply per-note delta scale/offset
            const float dScale = note->getDeltaScale();
            const float dOffset = note->getDeltaOffset();
            if (std::abs(dScale - 1.0f) > 0.0001f || std::abs(dOffset) > 0.0001f)
            {
                for (auto& v : transformedDelta)
                    v = v * dScale + dOffset;
            }

            minAffectedFrame = std::min(minAffectedFrame, startFrame);
            maxAffectedFrame = std::max(maxAffectedFrame, endFrame);

            for (int i = 0; i < numFrames && i < static_cast<int>(transformedDelta.size()); ++i)
            {
                const int globalIdx = startFrame + i;
                if (globalIdx >= 0 && globalIdx < totalFrames)
                    audioData.deltaPitch[static_cast<size_t>(globalIdx)] = transformedDelta[static_cast<size_t>(i)];
            }
        }

        // Recompose f0 only for affected range
        if (minAffectedFrame < maxAffectedFrame)
        {
            const int rangeStart = std::max(0, minAffectedFrame);
            const int rangeEnd = std::min(totalFrames, maxAffectedFrame);
            for (int i = rangeStart; i < rangeEnd; ++i)
            {
                const float base = audioData.basePitch[static_cast<size_t>(i)];
                const float delta = audioData.deltaPitch[static_cast<size_t>(i)];
                audioData.f0[static_cast<size_t>(i)] = midiToFreq(base + delta);
            }
        }
    }

    std::vector<float> composeF0(const Project& project,
                                 bool applyUvMask,
                                 float globalPitchOffset)
    {
        const auto& audioData = project.getAudioData();
        const int totalFrames = static_cast<int>(audioData.basePitch.size());
        std::vector<float> result(static_cast<size_t>(totalFrames), 0.0f);

        for (int i = 0; i < totalFrames; ++i)
        {
            bool isVoiced = (i < static_cast<int>(audioData.voicedMask.size())) ? audioData.voicedMask[i] : true;
            if (applyUvMask && !isVoiced)
                continue;

            const float base = audioData.basePitch[static_cast<size_t>(i)];
            const float delta = (i < static_cast<int>(audioData.deltaPitch.size()))
                                    ? audioData.deltaPitch[static_cast<size_t>(i)]
                                    : 0.0f;
            const float midi = base + delta + globalPitchOffset;
            result[static_cast<size_t>(i)] = midiToFreq(midi);
        }

        return result;
    }

    void composeF0InPlace(Project& project,
                          bool applyUvMask,
                          float globalPitchOffset)
    {
        auto composed = composeF0(project, applyUvMask, globalPitchOffset);
        auto& audioData = project.getAudioData();
        audioData.f0 = std::move(composed);
    }
} // namespace PitchCurveProcessor
