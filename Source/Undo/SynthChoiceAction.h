#pragma once

#include "UndoableAction.h"
#include "../Models/Note.h"
#include "../Models/NoteRef.h"
#include "../Models/Project.h"
#include "../Utils/FryDetector.h"
#include "../Utils/PitchCurveProcessor.h"

#include <algorithm>
#include <limits>
#include <vector>

/**
 * How a note is rendered, as the user chose it (spec Q12): the way of the
 * synthesis menu, and the stretches kept as recorded by hand (the two green
 * squares, Note::setKeptRange). The two go together - a range kept by hand
 * belongs to one way only - so they are remembered and put back together.
 */
struct SynthChoice
{
    Note::SynthEngine engine = Note::SynthEngine::Auto;
    bool keptByHand = false;
    int keptHeadUntil = Note::kKeepsNone;
    int keptTailFrom = Note::kKeepsNone;

    static SynthChoice of(const Note& note)
    {
        return {note.getSynthEngine(), note.hasKeptRange(), note.getKeptHeadUntil(),
                note.getKeptTailFrom()};
    }
    /** The way `engine` of the menu, chosen on `note`: any way but keeping
     *  the fry drops what was marked by hand; that one leaves it. */
    static SynthChoice ofMenu(const Note& note, Note::SynthEngine engine)
    {
        if (engine == Note::SynthEngine::VocoderRecordedHighsFry && note.hasKeptRange())
            return of(note);
        return {engine, false, Note::kKeepsNone, Note::kKeepsNone};
    }
    static SynthChoice byHand(int headUntil, int tailFrom)
    {
        return {Note::SynthEngine::VocoderRecordedHighsFry, true, headUntil, tailFrom};
    }
    void applyTo(Note& note) const
    {
        // The marks go first: "keep the fry" set again leaves them where
        // they are (Note::setSynthEngine), and this choice may be that way
        // with none.
        note.clearKeptRange();
        note.setSynthEngine(engine);
        if (keptByHand)
            note.setKeptRange(keptHeadUntil, keptTailFrom);
    }
    bool operator==(const SynthChoice& other) const
    {
        return engine == other.engine && keptByHand == other.keptByHand &&
               keptHeadUntil == other.keptHeadUntil && keptTailFrom == other.keptTailFrom;
    }
    bool operator!=(const SynthChoice& other) const { return !(*this == other); }
};

/**
 * The choice of the synthesis menu and a drag of a green square, as one
 * history step: both change a SynthChoice, and one undo of either puts back
 * the way and the stretches kept by hand as they were.
 *
 * Applying it (redo and undo alike) marks the edited notes for rendering.
 * A note kept whole as recorded sings its recorded pitch whatever it is
 * moved to (fry::wholeAsRecorded, PitchCurveProcessor), and a choice can
 * make it so or stop it being so: the pitch curve is rebuilt, the unedited
 * notes next to it refitted to the base as it then is, and the range marked
 * as a pitch edit marks it, so that the neighbour gliding into the note is
 * rendered to the pitch that plays. (The menu's action did none of that: a
 * note told to keep fry the detector misses kept its moved curve until some
 * other pitch edit rebuilt it.)
 */
class SynthChoiceAction final : public UndoableAction
{
public:
    SynthChoiceAction(Project& projectToChange, std::vector<Note*> notesToChange,
                      std::vector<SynthChoice> choicesBefore,
                      std::vector<SynthChoice> choicesAfter, juce::String actionName)
        : project(projectToChange), notes(noteref::refs(notesToChange)),
          before(std::move(choicesBefore)), after(std::move(choicesAfter)),
          name(std::move(actionName)) {}

    /** The same choice for every note, from what each is now. */
    static std::vector<SynthChoice> current(const std::vector<Note*>& notes)
    {
        std::vector<SynthChoice> choices;
        for (const auto* note : notes)
            choices.push_back(note != nullptr ? SynthChoice::of(*note) : SynthChoice{});
        return choices;
    }

    void undo() override { apply(before); }
    void redo() override { apply(after); }
    juce::String getName() const override { return name; }

private:
    void apply(const std::vector<SynthChoice>& choices)
    {
        const auto resolved = noteref::resolve(notes);
        const auto found = fry::find(project.getAudioData());
        int start = std::numeric_limits<int>::max();
        int end = std::numeric_limits<int>::min();
        bool curveMoves = false;
        for (size_t i = 0; i < resolved.size() && i < choices.size(); ++i)
        {
            auto* note = resolved[i];
            if (note == nullptr)
                continue;
            const bool wholeBefore = fry::wholeAsRecorded(*note, found);
            choices[i].applyTo(*note);
            if (note->isNeutralForOriginalWaveform())
                continue;
            note->markDirty();
            note->markSynthDirty();
            start = std::min(start, note->getStartFrame());
            end = std::max(end, note->getEndFrame());
            curveMoves = curveMoves || wholeBefore != fry::wholeAsRecorded(*note, found);
        }
        project.setModified(true);
        if (!curveMoves || start >= end)
            return;
        // As a restored note (ResetNoteEditsAction): the base under the
        // note moves to or from its recorded pitch, and with it the glide
        // of its neighbours.
        constexpr int kBaseReach = 12;
        PitchCurveProcessor::rebuildBaseFromNotes(project);
        PitchCurveProcessor::refitUntouchedCurves(project, start - kBaseReach,
                                                  end + kBaseReach);
        project.markNoteEditDirtyRange(start, end);
    }

    Project& project;
    std::vector<NoteRef> notes;
    std::vector<SynthChoice> before;
    std::vector<SynthChoice> after;
    juce::String name;
};
