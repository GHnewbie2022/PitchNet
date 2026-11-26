#pragma once

#include "UndoableAction.h"
#include "../Models/Note.h"
#include "../Models/NoteRef.h"

#include <vector>

/**
 * Notes locked or unlocked (Note::isLocked) as one history step: undo puts
 * each back as it was. It changes no audio.
 */
class LockAction final : public UndoableAction
{
public:
    /** `notesToChange` as they are before the change; they become `lock`. */
    LockAction(const std::vector<Note*>& notesToChange, bool lock, juce::String actionName)
        : notes(noteref::refs(notesToChange)), after(lock), name(std::move(actionName))
    {
        for (const auto* note : notesToChange)
            before.push_back(note != nullptr && note->isLocked());
    }

    /** What a press of the lock does to a selection: locks it, unless every
     *  note of it is locked already, then unlocks it. */
    static bool wanted(const std::vector<Note*>& notes)
    {
        for (const auto* note : notes)
            if (note != nullptr && !note->isLocked())
                return true;
        return false;
    }

    void undo() override
    {
        const auto resolved = noteref::resolve(notes);
        for (size_t i = 0; i < resolved.size() && i < before.size(); ++i)
            if (resolved[i] != nullptr)
                resolved[i]->setLocked(before[i]);
    }
    void redo() override
    {
        for (auto* note : noteref::present(notes))
            note->setLocked(after);
    }
    juce::String getName() const override { return name; }
    bool requiresAudioResynthesis() const override { return false; }

private:
    std::vector<NoteRef> notes;
    std::vector<bool> before;
    bool after = false;
    juce::String name;
};
