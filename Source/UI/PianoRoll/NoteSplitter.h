#pragma once

#include "../../JuceHeader.h"
#include "../../Models/Project.h"
#include "../../Undo/UndoActions.h"
#include "CoordinateMapper.h"
#include <functional>
#include <optional>

/**
 * Handles note splitting operations.
 */
class NoteSplitter {
public:
    NoteSplitter() = default;
    ~NoteSplitter() = default;

    void setProject(Project* proj) { project = proj; }
    void setUndoManager(PitchUndoManager* manager) { undoManager = manager; }
    void setCoordinateMapper(CoordinateMapper* mapper) { coordMapper = mapper; }

    /**
     * Find note at the given world coordinates.
     */
    Note* findNoteAt(float x, float y);

    /**
     * Split a note at the given frame position.
     * Returns true if split was successful.
     */
    bool splitNoteAtFrame(Note* note, int splitFrame);

    /**
     * Split note at world X coordinate.
     * Returns true if split was successful.
     */
    bool splitNoteAtX(Note* note, float x);

    /** Returns the adjacent note pair represented by a split boundary at x/y. */
    std::pair<Note*, Note*> findMergeableNotesAt(float x, float y);

    /** Merges two adjacent segments that originated from the same note. */
    bool mergeNotes(Note* first, Note* second);

    /** Live preview for Select-mode boundary resizing (does not record undo). */
    bool previewResizeNote(Note* note, const Note& originalNote,
                           int newSrcStartFrame, int newSrcEndFrame);

    /** Commit Select-mode boundary resize and record undo action. */
    bool commitResizeNote(Note* note, const Note& originalNote,
                          int newSrcStartFrame, int newSrcEndFrame);

    /**
     * "Restore note separation": the notes that share analysed notes with
     * `note` (Note::getAnalysisSpans), cut back the way the analysis cut
     * them - merges undone, extra splits joined - and each note that is a
     * detected note again gets its analysed pitch back, as Cubase does.
     * Formant and level edits stay.
     */
    struct SeparationGroup
    {
        int srcStart = 0;
        int srcEnd = 0;
        std::vector<int> currentCuts; // source frames between the notes now
        std::vector<int> targetCuts;  // where the analysis cut
        std::vector<Note> notes;      // the notes now, in order
    };
    std::optional<SeparationGroup> separationGroupFor(const Note& note) const;
    bool canRestoreSeparation(const Note& note) const;
    /** Recuts the group. Records no undo; `after` receives the new notes. */
    bool restoreSeparation(const SeparationGroup& group, std::vector<Note>& after);

    // Callbacks
    std::function<void()> onNoteSplit;

private:
    Project* project = nullptr;
    PitchUndoManager* undoManager = nullptr;
    CoordinateMapper* coordMapper = nullptr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NoteSplitter)
};
