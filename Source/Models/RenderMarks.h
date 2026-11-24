#pragma once

#include "Project.h"
#include "ProjectSerializer.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

/**
 * The marks a render answers, for a render made on a copy of the project.
 *
 * The standalone renders the project it shows, and the render clears that
 * project's marks as it lands (Project::clearAllDirty). The plug-in renders
 * a copy in the background and brings back the audio and two flags
 * (PluginProcessor.cpp, mergeRenderedState): the marks stayed on the project
 * shown, and every later render took all the notes edited before along with
 * the new one - unless a drag's own render, which works on the project
 * shown, had cleared them.
 *
 * of() is taken as the copy is made, and clearAnswered() as its render
 * lands: the marks of the notes still as they were rendered are cleared. A
 * note edited again while the render ran keeps its mark for the render that
 * edit asks for, and so does a range of pitch curve that has grown since.
 */
namespace rendermarks
{
struct Marks
{
    std::vector<std::uint64_t> notes; // uids of the notes marked to be rendered
    bool hasCurveRange = false;
    std::pair<int, int> curveRange{-1, -1};
};

inline Marks of(const Project& project)
{
    Marks marks;
    for (const auto& note : project.getNotes())
        if (!note.isRest() && note.isDirty())
            marks.notes.push_back(note.getUid());
    marks.hasCurveRange = project.hasF0DirtyRange();
    marks.curveRange = project.getF0DirtyRange();
    return marks;
}

inline void clearAnswered(Project& shown, const Project& rendered, const Marks& marks)
{
    for (auto& note : shown.getNotes())
    {
        if (!note.isDirty() ||
            std::find(marks.notes.begin(), marks.notes.end(), note.getUid()) == marks.notes.end())
            continue;
        for (const auto& other : rendered.getNotes())
            if (other.getUid() == note.getUid())
            {
                if (ProjectSerializer::writtenAlike(other, note))
                    note.clearDirty();
                break;
            }
    }
    if (marks.hasCurveRange && shown.hasF0DirtyRange() &&
        shown.getF0DirtyRange() == marks.curveRange)
        shown.clearF0DirtyRange();
}
} // namespace rendermarks
