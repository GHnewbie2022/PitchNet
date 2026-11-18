// Undo actions find their notes again after the note list moved: a split or
// merge inserts into and erases from it (shifting notes, or reallocating the
// whole list), and undo used to write through the old Note*.
#include "../Source/Models/Project.h"
#include "../Source/Undo/DragActions.h"
#include "../Source/Undo/NoteActions.h"
#include "../Source/Undo/PitchShaperAction.h"
#include "../Source/Undo/PitchToolAction.h"
#include "TestAssert.h"
#include <cmath>
#include <iostream>

namespace
{
Note makeNote(int start, int end, float midi)
{
    Note note(start, end, midi);
    note.setOriginalDeltaPitch(std::vector<float>(static_cast<size_t>(end - start), 0.0f));
    note.setDeltaPitch(note.getOriginalDeltaPitch());
    return note;
}

Note* at(Project& project, int start)
{
    for (auto& note : project.getNotes())
        if (note.getStartFrame() == start)
            return &note;
    return nullptr;
}

void setUp(Project& project)
{
    for (int i = 0; i < 3; ++i)
        project.addNote(makeNote(i * 10, i * 10 + 10, 60.0f + i));
    auto& audio = project.getAudioData();
    audio.melSpectrogram.assign(400, std::vector<float>(1, 0));
    audio.f0.assign(400, 440);
    audio.basePitch.assign(400, 60);
    audio.deltaPitch.assign(400, 0);
    audio.voicedMask.assign(400, true);
}

// Shifts every note after the first and reallocates the list.
void reshuffle(Project& project)
{
    for (int i = 0; i < 50; ++i)
        project.addNote(makeNote(100 + i * 5, 105 + i * 5, 50.0f));
    project.removeNoteByStartFrame(0);
}
} // namespace

int main()
{
    {
        Note a = makeNote(0, 10, 60.0f);
        Note copy = a;
        Note fresh = makeNote(0, 10, 60.0f);
        CHECK(copy.getUid() == a.getUid());
        CHECK(fresh.getUid() != a.getUid());
    }

    // Pitch tool: undo after the list moved changes the right note.
    {
        Project project;
        setUp(project);
        Note* target = at(project, 20);
        const auto before = TransformParams::fromNote(*target);
        auto after = before;
        after.pitchDrift = -2.4f;
        PitchToolAction action(&project, {target}, {before}, {after});
        after.applyToNote(*target);
        reshuffle(project);
        CHECK(at(project, 20)->getPitchDrift() == -2.4f);
        action.undo();
        CHECK(at(project, 20)->getPitchDrift() == 1.0f);
        CHECK(at(project, 10)->getPitchDrift() == 1.0f);
        for (const auto& note : project.getNotes())
            if (note.getStartFrame() >= 100)
                CHECK(note.getPitchDrift() == 1.0f);
        action.redo();
        CHECK(at(project, 20)->getPitchDrift() == -2.4f);
    }

    // Pitch shaper (the action in the crash report).
    {
        Project project;
        setUp(project);
        Note* target = at(project, 10);
        const auto params = TransformParams::fromNote(*target);
        const std::vector<float> contour(10, 1.0f);
        PitchShaperAction action(&project, {{target, params, {}}},
                                  {{target, params, contour}});
        action.redo();
        reshuffle(project);
        action.undo();
        CHECK(!at(project, 10)->hasBakedDeltaPitch());
        action.redo();
        CHECK(at(project, 10)->getBakedDeltaPitch() == contour);
        CHECK(!at(project, 20)->hasBakedDeltaPitch());
    }

    // A note that no longer exists is skipped.
    {
        Project project;
        setUp(project);
        Note* target = at(project, 20);
        NotePitchDragAction drag(target, nullptr, 62.0f, 64.0f, {});
        MultiNoteFloatPropertyAction vibrato({at(project, 10), target}, {1.0f, 1.0f},
                                             {0.5f, 0.5f}, &Note::setVibrato, "v");
        reshuffle(project);
        project.removeNoteByStartFrame(20);
        drag.redo();
        vibrato.redo();
        CHECK(at(project, 10)->getVibrato() == 0.5f);
        for (const auto& note : project.getNotes())
            CHECK(note.getMidiNote() != 64.0f);
    }

    // The project itself gone: nothing to change, no crash.
    {
        std::unique_ptr<NoteSnapToSemitoneAction> snap;
        {
            Project project;
            setUp(project);
            snap = std::make_unique<NoteSnapToSemitoneAction>(at(project, 0), 60.2f,
                                                              0.0f, 60.0f);
        }
        snap->undo();
        snap->redo();
    }

    std::cout << "Undo note references survive list changes\n";
}
