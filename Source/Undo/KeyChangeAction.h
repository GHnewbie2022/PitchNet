#pragma once

#include "UndoableAction.h"
#include "../Models/Project.h"

#include <vector>

/**
 * The key of a project (spec section 15): its own key, on both of the levels
 * it is kept (the project's and the region's, Project::getScaleRootNote),
 * and the changes of key along the recording. What the wizard sets when the
 * user confirms a key, and what a flag on the time ruler edits.
 */
struct KeyState
{
    bool useProjectKey = true;
    int projectRoot = 0;
    ScaleMode projectMode = ScaleMode::Major;
    int regionRoot = 0;
    ScaleMode regionMode = ScaleMode::Major;
    std::vector<KeyChange> changes;

    static KeyState of(const Project& project)
    {
        return {project.getUseProjectKey(), project.getProjectKeyRoot(), project.getProjectKeyMode(),
                project.getRegionKeyRoot(), project.getRegionKeyMode(), project.getKeyChanges()};
    }

    /**
     * As `project` is, with the key the user confirmed: the major scale on
     * `root` up to the first change (on the level the project reads its key
     * from), then `keyChanges`. Always major: a minor song is kept as its
     * relative major (the user's decision), and a mode the analysis guessed
     * (ScaleUtils::detectAndApplyScale may say minor) goes.
     */
    static KeyState confirmed(const Project& project, int root, std::vector<KeyChange> keyChanges,
                              ScaleMode mode = ScaleMode::Major)
    {
        auto state = of(project);
        const int pitchClass = (root % 12 + 12) % 12;
        if (state.useProjectKey)
        {
            state.projectRoot = pitchClass;
            state.projectMode = mode;
        }
        else
        {
            state.regionRoot = pitchClass;
            state.regionMode = mode;
        }
        state.changes = std::move(keyChanges);
        return state;
    }

    void applyTo(Project& project) const
    {
        // The region's key first: setting it stops the project's being read.
        project.setScaleRootNote(regionRoot);
        project.setPreferredScaleMode(regionMode);
        project.setProjectKey(projectRoot, projectMode);
        project.setUseProjectKey(useProjectKey);
        project.setKeyChanges(changes);
    }

    bool operator==(const KeyState& other) const
    {
        return useProjectKey == other.useProjectKey && projectRoot == other.projectRoot &&
               projectMode == other.projectMode && regionRoot == other.regionRoot &&
               regionMode == other.regionMode && changes == other.changes;
    }
    bool operator!=(const KeyState& other) const { return !(*this == other); }
};

/** A change to the key as one history step. It changes no audio. */
class KeyChangeAction final : public UndoableAction
{
public:
    KeyChangeAction(Project& projectToChange, KeyState stateBefore, KeyState stateAfter,
                    juce::String actionName)
        : project(projectToChange), before(std::move(stateBefore)), after(std::move(stateAfter)),
          name(std::move(actionName)) {}

    void undo() override { before.applyTo(project); }
    void redo() override { after.applyTo(project); }
    juce::String getName() const override { return name; }
    bool requiresAudioResynthesis() const override { return false; }

private:
    Project& project;
    KeyState before, after;
    juce::String name;
};
