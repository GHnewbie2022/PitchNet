#pragma once

#include "UndoableAction.h"
#include "../Models/Note.h"
#include "../Models/Project.h"
#include "../Models/NoteRef.h"
#include "../Utils/PitchCurveProcessor.h"
#include <algorithm>
#include <limits>
#include <vector>
#include <functional>

/**
 * Generic action for changing a single float property on a Note.
 * Uses a member function pointer to call the appropriate setter.
 */
class NoteFloatPropertyAction : public UndoableAction
{
public:
    using Setter = void (Note::*)(float);

    NoteFloatPropertyAction(Note *note, float oldVal, float newVal,
                            Setter setter, juce::String actionName,
                            std::function<void(Note *)> onNoteChanged = nullptr)
        : noteRef(note), oldVal(oldVal), newVal(newVal),
          setter(setter), actionName(std::move(actionName)),
          onNoteChanged(onNoteChanged) {}

    void undo() override
    {
        Note *note = noteRef.get();
        if (!note)
            return;
        (note->*setter)(oldVal);
        note->markDirty();
        if (onNoteChanged)
            onNoteChanged(note);
    }
    void redo() override
    {
        Note *note = noteRef.get();
        if (!note)
            return;
        (note->*setter)(newVal);
        note->markDirty();
        if (onNoteChanged)
            onNoteChanged(note);
    }
    juce::String getName() const override { return actionName; }

private:
    NoteRef noteRef;
    float oldVal;
    float newVal;
    Setter setter;
    juce::String actionName;
    std::function<void(Note *)> onNoteChanged;
};

/**
 * Generic action for changing a single float property on multiple Notes.
 * Uses a member function pointer to call the appropriate setter.
 */
class MultiNoteFloatPropertyAction : public UndoableAction
{
public:
    using Setter = void (Note::*)(float);

    MultiNoteFloatPropertyAction(const std::vector<Note *> &notes,
                                 const std::vector<float> &oldVals,
                                 const std::vector<float> &newVals,
                                 Setter setter, juce::String actionName,
                                 std::function<void()> onChanged = nullptr)
        : noteRefs(noteref::refs(notes)), oldVals(oldVals), newVals(newVals),
          setter(setter), actionName(std::move(actionName)),
          onChanged(onChanged) {}

    void undo() override
    {
        const auto notes = noteref::resolve(noteRefs);
        for (size_t i = 0; i < notes.size() && i < oldVals.size(); ++i)
        {
            if (notes[i])
            {
                (notes[i]->*setter)(oldVals[i]);
                notes[i]->markDirty();
            }
        }
        if (onChanged)
            onChanged();
    }

    void redo() override
    {
        const auto notes = noteref::resolve(noteRefs);
        for (size_t i = 0; i < notes.size() && i < newVals.size(); ++i)
        {
            if (notes[i])
            {
                (notes[i]->*setter)(newVals[i]);
                notes[i]->markDirty();
            }
        }
        if (onChanged)
            onChanged();
    }

    juce::String getName() const override { return actionName; }

private:
    std::vector<NoteRef> noteRefs;
    std::vector<float> oldVals;
    std::vector<float> newVals;
    Setter setter;
    juce::String actionName;
    std::function<void()> onChanged;
};

/**
 * Action for resetting tilt values on multiple notes.
 * Used for double-click on TiltLeft/TiltRight handles to reset to 0.
 */
class TiltResetAction : public UndoableAction
{
public:
    enum class TiltSide
    {
        Left,
        Right
    };

    TiltResetAction(const std::vector<Note *> &notes,
                    TiltSide side,
                    const std::vector<float> &oldTilts,
                    const std::vector<float> &oldMidiNotes,
                    std::function<void()> onChanged = nullptr)
        : noteRefs(noteref::refs(notes)), side(side), oldTilts(oldTilts),
          oldMidiNotes(oldMidiNotes), onChanged(onChanged) {}

    void undo() override
    {
        const auto notes = noteref::resolve(noteRefs);
        for (size_t i = 0; i < notes.size() && i < oldTilts.size(); ++i)
        {
            if (notes[i])
            {
                if (side == TiltSide::Left)
                    notes[i]->setTiltLeft(oldTilts[i]);
                else
                    notes[i]->setTiltRight(oldTilts[i]);

                if (i < oldMidiNotes.size())
                    notes[i]->setMidiNote(oldMidiNotes[i]);

                notes[i]->markDirty();
                notes[i]->markSynthDirty();
            }
        }
        if (onChanged)
            onChanged();
    }

    void redo() override
    {
        const auto notes = noteref::resolve(noteRefs);
        for (size_t i = 0; i < notes.size(); ++i)
        {
            if (notes[i])
            {
                if (side == TiltSide::Left)
                    notes[i]->setTiltLeft(0.0f);
                else
                    notes[i]->setTiltRight(0.0f);

                const float newTiltMean = notes[i]->getTiltMean();
                if (i < oldMidiNotes.size())
                {
                    const float oldTiltLeft = (side == TiltSide::Left) ? oldTilts[i] : notes[i]->getTiltLeft();
                    const float oldTiltRight = (side == TiltSide::Right) ? oldTilts[i] : notes[i]->getTiltRight();
                    juce::ignoreUnused(oldTiltLeft, oldTiltRight);
                    const float oldTiltMean = 0.0f; // see Note::getTiltMean
                    const float baseline = oldMidiNotes[i] - oldTiltMean;
                    notes[i]->setMidiNote(baseline + newTiltMean);
                }

                notes[i]->markDirty();
                notes[i]->markSynthDirty();
            }
        }
        if (onChanged)
            onChanged();
    }

    juce::String getName() const override
    {
        return side == TiltSide::Left ? "Reset Tilt Left" : "Reset Tilt Right";
    }

private:
    std::vector<NoteRef> noteRefs;
    TiltSide side;
    std::vector<float> oldTilts;
    std::vector<float> oldMidiNotes;
    std::function<void()> onChanged;
};

/**
 * Action for snapping a note to the nearest semitone (double-click).
 * Combines midiNote and pitchOffset into a rounded integer MIDI value.
 */
class NoteSnapToSemitoneAction : public UndoableAction
{
public:
    NoteSnapToSemitoneAction(Note *note,
                             float oldMidi, float oldOffset,
                             float newMidi,
                             std::function<void(Note *)> onNoteChanged = nullptr)
        : noteRef(note), oldMidi(oldMidi), oldOffset(oldOffset),
          newMidi(newMidi), onNoteChanged(onNoteChanged) {}

    void undo() override
    {
        Note *note = noteRef.get();
        if (note)
        {
            note->setMidiNote(oldMidi);
            note->setPitchOffset(oldOffset);
            note->markDirty();
            note->markSynthDirty();
        }
        if (onNoteChanged && note)
            onNoteChanged(note);
    }

    void redo() override
    {
        Note *note = noteRef.get();
        if (note)
        {
            note->setMidiNote(newMidi);
            note->setPitchOffset(0.0f);
            note->markDirty();
            note->markSynthDirty();
        }
        if (onNoteChanged && note)
            onNoteChanged(note);
    }

    juce::String getName() const override { return "Snap to Semitone"; }

private:
    NoteRef noteRef;
    float oldMidi;
    float oldOffset;
    float newMidi;
    std::function<void(Note *)> onNoteChanged;
};

/**
 * Action for snapping multiple notes to the nearest semitone.
 */
class MultiNoteSnapToSemitoneAction : public UndoableAction
{
public:
    MultiNoteSnapToSemitoneAction(const std::vector<Note *> &notes,
                                  std::vector<float> oldMidis,
                                  std::vector<float> oldOffsets,
                                  std::vector<float> newMidis,
                                  std::function<void(const std::vector<Note *> &)> onNotesChanged = nullptr)
        : noteRefs(noteref::refs(notes)),
          oldMidis(std::move(oldMidis)),
          oldOffsets(std::move(oldOffsets)),
          newMidis(std::move(newMidis)),
          onNotesChanged(onNotesChanged) {}

    void undo() override
    {
        const auto notes = noteref::resolve(noteRefs);
        for (size_t i = 0; i < notes.size(); ++i)
        {
            auto *note = notes[i];
            if (!note)
                continue;
            note->setMidiNote(oldMidis[i]);
            note->setPitchOffset(oldOffsets[i]);
            note->markDirty();
            note->markSynthDirty();
        }
        if (onNotesChanged)
            onNotesChanged(noteref::present(noteRefs));
    }

    void redo() override
    {
        const auto notes = noteref::resolve(noteRefs);
        for (size_t i = 0; i < notes.size(); ++i)
        {
            auto *note = notes[i];
            if (!note)
                continue;
            note->setMidiNote(newMidis[i]);
            note->setPitchOffset(0.0f);
            note->markDirty();
            note->markSynthDirty();
        }
        if (onNotesChanged)
            onNotesChanged(noteref::present(noteRefs));
    }

    juce::String getName() const override { return "Snap Notes to Semitone"; }

private:
    std::vector<NoteRef> noteRefs;
    std::vector<float> oldMidis;
    std::vector<float> oldOffsets;
    std::vector<float> newMidis;
    std::function<void(const std::vector<Note *> &)> onNotesChanged;
};

/**
 * Action for splitting a note into two.
 */
/**
 * A neighbouring note's curve changed by a split or merge, which bends the
 * base pitch around the boundary (NoteSplitter keeps what the notes sang).
 * Found again by start frame: the neighbours do not move.
 */
struct NoteCurveEdit
{
    int startFrame = 0;
    std::vector<float> originalBefore, originalAfter;
    std::vector<float> bakedBefore, bakedAfter;

    static void apply(Project &project, const std::vector<NoteCurveEdit> &edits,
                      bool after)
    {
        for (const auto &edit : edits)
            for (auto &note : project.getNotes())
                if (!note.isRest() && note.getStartFrame() == edit.startFrame)
                {
                    note.setOriginalDeltaPitch(after ? edit.originalAfter : edit.originalBefore);
                    note.setBakedDeltaPitch(after ? edit.bakedAfter : edit.bakedBefore);
                    break;
                }
    }
};

class NoteSplitAction : public UndoableAction
{
public:
    NoteSplitAction(Project *proj, const Note &original, const Note &firstPart, const Note &secondPart,
                    std::function<void()> onChanged = nullptr)
        : project(proj), originalNote(original), firstNote(firstPart), secondNote(secondPart),
          onChanged(onChanged) {}

    void undo() override
    {
        if (!project)
            return;
        project->removeNoteByStartFrame(secondNote.getStartFrame());
        for (auto &note : project->getNotes())
        {
            if (note.getStartFrame() == firstNote.getStartFrame())
            {
                note = originalNote;
                break;
            }
        }
        NoteCurveEdit::apply(*project, neighbourEdits, false);
        // The drawn pitch curve is built from the notes: rebuild it for the
        // notes now in place, or it keeps the other layout's curve.
        PitchCurveProcessor::rebuildBaseFromNotes(*project);
        if (onChanged)
            onChanged();
    }

    void redo() override
    {
        if (!project)
            return;
        for (auto &note : project->getNotes())
        {
            if (note.getStartFrame() == originalNote.getStartFrame())
            {
                note = firstNote;
                break;
            }
        }
        project->addNote(secondNote);
        NoteCurveEdit::apply(*project, neighbourEdits, true);
        PitchCurveProcessor::rebuildBaseFromNotes(*project);
        if (onChanged)
            onChanged();
    }

    juce::String getName() const override { return "Split Note"; }
    bool requiresAudioResynthesis() const override { return false; }

    void setNeighbourEdits(std::vector<NoteCurveEdit> edits) { neighbourEdits = std::move(edits); }

private:
    std::vector<NoteCurveEdit> neighbourEdits;
    Project *project;
    Note originalNote;
    Note firstNote;
    Note secondNote;
    std::function<void()> onChanged;
};

/**
 * Action for merging two adjacent note segments.
 */
class NoteMergeAction : public UndoableAction
{
public:
    NoteMergeAction(Project *proj, const Note &firstPart, const Note &secondPart,
                    const Note &merged, std::function<void()> onChanged = nullptr)
        : project(proj), firstNote(firstPart), secondNote(secondPart),
          mergedNote(merged), onChanged(onChanged) {}

    void undo() override
    {
        if (!project)
            return;
        for (auto &note : project->getNotes())
        {
            if (note.getStartFrame() == mergedNote.getStartFrame())
            {
                note = firstNote;
                break;
            }
        }
        project->addNote(secondNote);
        project->getDrawnCurves() = curvesBefore;
        NoteCurveEdit::apply(*project, neighbourEdits, false);
        markHalvesForRender();
        PitchCurveProcessor::rebuildBaseFromNotes(*project);
        if (onChanged)
            onChanged();
    }

    void redo() override
    {
        if (!project)
            return;
        for (auto &note : project->getNotes())
        {
            if (note.getStartFrame() == firstNote.getStartFrame())
            {
                note = mergedNote;
                break;
            }
        }
        project->removeNoteByStartFrame(secondNote.getStartFrame());
        project->getDrawnCurves() = curvesAfter;
        NoteCurveEdit::apply(*project, neighbourEdits, true);
        for (auto &note : project->getNotes())
            if (note.getStartFrame() == firstNote.getStartFrame())
            {
                note.markDirty();
                note.markSynthDirty();
            }
        PitchCurveProcessor::rebuildBaseFromNotes(*project);
        if (onChanged)
            onChanged();
    }

    juce::String getName() const override { return "Merge Notes"; }
    // A merge puts the pitch back to the analysis: the audio changes.
    bool requiresAudioResynthesis() const override { return true; }

    /** The drawn curves before the merge and after (it drops those on the
     *  notes it resets). */
    void setDrawnCurves(std::vector<DrawnCurve> before, std::vector<DrawnCurve> after)
    {
        curvesBefore = std::move(before);
        curvesAfter = std::move(after);
    }
    void setNeighbourEdits(std::vector<NoteCurveEdit> edits) { neighbourEdits = std::move(edits); }

private:
    void markHalvesForRender()
    {
        for (auto &note : project->getNotes())
            if (note.getStartFrame() == firstNote.getStartFrame() ||
                note.getStartFrame() == secondNote.getStartFrame())
            {
                note.markDirty();
                note.markSynthDirty();
            }
    }

    Project *project;
    std::vector<NoteCurveEdit> neighbourEdits;
    std::vector<DrawnCurve> curvesBefore;
    std::vector<DrawnCurve> curvesAfter;
    Note firstNote;
    Note secondNote;
    Note mergedNote;
    std::function<void()> onChanged;
};

/**
 * "Restore note separation": the notes of a span replaced by the notes the
 * analysis cut there (NoteSplitter::restoreSeparation).
 */
class NoteSeparationAction : public UndoableAction
{
public:
    NoteSeparationAction(Project *proj, std::vector<Note> beforeNotes,
                         std::vector<Note> afterNotes)
        : project(proj), before(std::move(beforeNotes)), after(std::move(afterNotes)) {}

    void undo() override { swap(after, before); }
    void redo() override { swap(before, after); }
    juce::String getName() const override { return "Restore Note Separation"; }
    // The restored notes get their analysed pitch back.
    bool requiresAudioResynthesis() const override { return true; }

private:
    void swap(const std::vector<Note> &out, const std::vector<Note> &in)
    {
        if (!project)
            return;
        for (const auto &note : out)
            project->removeNoteByStartFrame(note.getStartFrame());
        int first = std::numeric_limits<int>::max();
        int last = std::numeric_limits<int>::min();
        for (auto note : in)
        {
            note.markDirty();
            note.markSynthDirty();
            first = std::min(first, note.getStartFrame());
            last = std::max(last, note.getEndFrame());
            project->addNote(std::move(note));
        }
        project->setModified(true);
        PitchCurveProcessor::rebuildBaseFromNotes(*project);
        // The centres changed: untouched notes around sing the recording
        // over the new base (PitchCurveProcessor::refitUntouchedCurves).
        if (first <= last)
            PitchCurveProcessor::refitUntouchedCurves(*project, first - 12, last + 12);
        if (first <= last)
            project->markNoteEditDirtyRange(first, last);
    }

    Project *project;
    std::vector<Note> before;
    std::vector<Note> after;
};

/**
 * Action for resizing a note's source/timeline frame range in Select mode.
 */
class NoteResizeAction : public UndoableAction
{
public:
    NoteResizeAction(Project *proj, Note before, Note after,
                     std::vector<NoteCurveEdit> edits,
                     std::vector<DrawnCurve> cBefore,
                     std::vector<DrawnCurve> cAfter,
                     bool resynth,
                     std::function<void()> changed = nullptr)
        : project(proj),
          beforeNote(std::move(before)),
          afterNote(std::move(after)),
          neighbourEdits(std::move(edits)),
          curvesBefore(std::move(cBefore)),
          curvesAfter(std::move(cAfter)),
          needsResynthesis(resynth),
          onChanged(std::move(changed)) {}

    void undo() override
    {
        if (!project)
            return;
        Note *target = findNote(afterNote);
        if (target)
            *target = beforeNote;
        sortNotesByStartFrame();
        project->getDrawnCurves() = curvesBefore;
        NoteCurveEdit::apply(*project, neighbourEdits, false);
        PitchCurveProcessor::rebuildBaseFromNotes(*project);
        if (needsResynthesis)
        {
            project->getAudioData().warpLayoutPending = true;
            if (Note *restored = findNote(beforeNote))
            {
                restored->markDirty();
                restored->markSynthDirty();
            }
            project->markNoteEditDirtyRange(
                std::min(beforeNote.getStartFrame(), afterNote.getStartFrame()),
                std::max(beforeNote.getEndFrame(), afterNote.getEndFrame()));
        }
        project->setModified(true);
        if (onChanged)
            onChanged();
    }

    void redo() override
    {
        if (!project)
            return;
        Note *target = findNote(beforeNote);
        if (target)
            *target = afterNote;
        sortNotesByStartFrame();
        project->getDrawnCurves() = curvesAfter;
        NoteCurveEdit::apply(*project, neighbourEdits, true);
        PitchCurveProcessor::rebuildBaseFromNotes(*project);
        if (needsResynthesis)
        {
            project->getAudioData().warpLayoutPending = true;
            if (Note *resized = findNote(afterNote))
            {
                resized->markDirty();
                resized->markSynthDirty();
            }
            project->markNoteEditDirtyRange(
                std::min(beforeNote.getStartFrame(), afterNote.getStartFrame()),
                std::max(beforeNote.getEndFrame(), afterNote.getEndFrame()));
        }
        project->setModified(true);
        if (onChanged)
            onChanged();
    }

    juce::String getName() const override { return "Resize Note"; }
    bool requiresAudioResynthesis() const override { return needsResynthesis; }

private:
    Note *findNote(const Note &reference)
    {
        auto &notes = project->getNotes();
        if (reference.getUid() != 0)
        {
            for (auto &note : notes)
                if (note.getUid() == reference.getUid())
                    return &note;
        }
        for (auto &note : notes)
            if (!note.isRest() && note.getStartFrame() == reference.getStartFrame())
                return &note;
        return nullptr;
    }

    void sortNotesByStartFrame()
    {
        auto &notes = project->getNotes();
        if (!std::is_sorted(notes.begin(), notes.end(),
                            [](const Note &a, const Note &b)
                            { return a.getStartFrame() < b.getStartFrame(); }))
        {
            std::stable_sort(notes.begin(), notes.end(),
                             [](const Note &a, const Note &b)
                             { return a.getStartFrame() < b.getStartFrame(); });
        }
    }

    Project *project;
    Note beforeNote;
    Note afterNote;
    std::vector<NoteCurveEdit> neighbourEdits;
    std::vector<DrawnCurve> curvesBefore;
    std::vector<DrawnCurve> curvesAfter;
    bool needsResynthesis = false;
    std::function<void()> onChanged;
};

