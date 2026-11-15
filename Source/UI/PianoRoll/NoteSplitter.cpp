#include "NoteSplitter.h"
#include "../../Audio/NoteTiming.h"
#include "../../Utils/NoteAmplitude.h"
#include "../../Utils/PitchCurveProcessor.h"
#include "../../Utils/Constants.h"
#include <algorithm>
#include <cmath>

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

bool NoteSplitter::splitNoteAtFrame(Note* note, int splitFrame) {
    if (!note || !project)
        return false;

    int startFrame = note->getStartFrame();
    int endFrame = note->getEndFrame();
    const float previousPitchCenter = note->getAdjustedMidiNote();

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
    secondNote.setVibrato(note->getVibrato());
    secondNote.setPitchDrift(note->getPitchDrift());
    secondNote.setFormantShift(note->getFormantShift());
    secondNote.setSmoothLeftFrames(note->getSmoothLeftFrames());
    secondNote.setSmoothRightFrames(note->getSmoothRightFrames());
    secondNote.setDeltaScale(note->getDeltaScale());
    secondNote.setDeltaOffset(note->getDeltaOffset());
    secondNote.setPitchOffset(0.0f);

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
    const auto& audioData = project->getAudioData();
    const float leftPitchCenter = calculatePitchCenter(
        audioData, startFrame, splitFrame, previousPitchCenter);
    const float rightPitchCenter = calculatePitchCenter(
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
    // the audio already matches), keeping what an edited note sang.
    PitchCurveProcessor::rebuildBaseFromNotes(*project);
    keepSungPitch(*project, {startFrame, splitFrame}, startFrame, endFrame,
                  pitchBefore);

    // Create undo action - don't pass callback to avoid lifetime issues
    // UI refresh is handled by UndoManager's onUndoRedo callback
    if (undoManager) {
        const Note* left = noteStartingAt(*project, startFrame);
        const Note* right = noteStartingAt(*project, splitFrame);
        if (left && right)
            undoManager->addAction(std::make_unique<NoteSplitAction>(
                project, originalNote, *left, *right, nullptr));
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

    const auto &chunkRanges = project->getAudioData().segmentChunkRanges;
    const auto getRegionIndex = [&chunkRanges](const Note &note)
    {
        if (chunkRanges.empty())
            return 0;

        const int midpoint = note.getSrcStartFrame() +
                             std::max(0, note.getSrcDurationFrames()) / 2;
        for (size_t i = 0; i < chunkRanges.size(); ++i)
            if (midpoint >= chunkRanges[i].first && midpoint < chunkRanges[i].second)
                return static_cast<int>(i);
        return -1;
    };

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

            // Only expose merge on a true split: the output and source frame
            // ranges must remain continuous across the boundary.
            if (left.getEndFrame() != right.getStartFrame() ||
                left.getSrcEndFrame() != right.getSrcStartFrame() ||
                getRegionIndex(left) < 0 || getRegionIndex(left) != getRegionIndex(right))
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

    const Note firstNote = *first;
    const Note secondNote = *second;

    // Like gluing segments in Cubase's VariAudio, the merged note goes back
    // to the pitch the analysis gave it: the halves' pitch edits (drag,
    // drift, vibrato, tilt, smoothing, drawn curves) are dropped. Undo brings
    // them back.
    const bool reshaped =
        !(pitchUntouched(firstNote) && pitchUntouched(secondNote));

    Note mergedNote = firstNote;
    mergedNote.setEndFrame(secondNote.getEndFrame());
    mergedNote.setSrcEndFrame(secondNote.getSrcEndFrame());
    {
        auto spans = firstNote.getAnalysisSpans();
        const auto& more = secondNote.getAnalysisSpans();
        spans.insert(spans.end(), more.begin(), more.end());
        mergedNote.setAnalysisSpans(std::move(spans));
    }

    NoteAmplitude::update(mergedNote, project->getAudioData());

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
    const auto mergeDeviation = [mergedPitchCenter](
                                  const std::vector<float> &firstValues,
                                  float firstCenter,
                                  const std::vector<float> &secondValues,
                                  float secondCenter)
    {
        std::vector<float> values;
        values.reserve(firstValues.size() + secondValues.size());
        for (const auto value : firstValues)
            values.push_back(value + firstCenter - mergedPitchCenter);
        for (const auto value : secondValues)
            values.push_back(value + secondCenter - mergedPitchCenter);
        return values;
    };
    mergedNote.setOriginalDeltaPitch(mergeDeviation(
        firstNote.getOriginalDeltaPitch(), firstNote.getOriginalMidiNote(),
        secondNote.getOriginalDeltaPitch(), secondNote.getOriginalMidiNote()));
    mergedNote.setDeltaPitch(mergeDeviation(
        firstNote.getDeltaPitch(), firstNote.getAdjustedMidiNote(),
        secondNote.getDeltaPitch(), secondNote.getAdjustedMidiNote()));
    if (firstNote.hasBakedDeltaPitch() || secondNote.hasBakedDeltaPitch()) {
        const auto& firstActive = firstNote.hasBakedDeltaPitch()
                                      ? firstNote.getBakedDeltaPitch()
                                      : firstNote.getOriginalDeltaPitch();
        const auto& secondActive = secondNote.hasBakedDeltaPitch()
                                       ? secondNote.getBakedDeltaPitch()
                                       : secondNote.getOriginalDeltaPitch();
        mergedNote.setBakedDeltaPitch(mergeDeviation(
            firstActive, firstNote.getAdjustedMidiNote(),
            secondActive, secondNote.getAdjustedMidiNote()));
    }
    mergedNote.setMidiNote(mergedPitchCenter);
    mergedNote.setOriginalMidiNote(mergedPitchCenter);
    mergedNote.setPitchOffset(0.0f);
    const bool isDirty = firstNote.isDirty() || secondNote.isDirty();
    const bool isSynthDirty = firstNote.isSynthDirty() || secondNote.isSynthDirty();
    mergedNote.setDirty(isDirty);
    mergedNote.setRenderedEdit(firstNote.hasRenderedEdit() ||
                               secondNote.hasRenderedEdit());
    if (isSynthDirty)
        mergedNote.markSynthDirty();
    else
        mergedNote.setSynthDirty(false);

    // Back to the analysis: the detected note itself when the merge made it
    // whole again, otherwise the halves' analysed curves joined above.
    if (!mergedNote.restoreAnalysisPitch())
    {
        mergedNote.setTiltLeft(0.0f);
        mergedNote.setTiltRight(0.0f);
        mergedNote.setVibrato(1.0f);
        mergedNote.setPitchDrift(1.0f);
        mergedNote.setSmoothLeftFrames(0);
        mergedNote.setSmoothRightFrames(0);
        mergedNote.setDeltaScale(1.0f);
        mergedNote.setDeltaOffset(0.0f);
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

    // Rebuild the drawn pitch curve from the new notes (pitch-neutral edit:
    // the audio already matches).
    PitchCurveProcessor::rebuildBaseFromNotes(*project);


    if (undoManager)
        undoManager->addAction(std::make_unique<NoteMergeAction>(
            project, firstNote, secondNote, mergedNote, nullptr));
    if (onNoteSplit)
        onNoteSplit();

    return true;
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

    after.clear();
    for (const auto& n : project->getNotes())
        if (!n.isRest() && n.getSrcStartFrame() >= group.srcStart &&
            n.getSrcEndFrame() <= group.srcEnd)
            after.push_back(n);
    return true;
}
