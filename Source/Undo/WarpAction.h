#pragma once

#include "UndoableAction.h"
#include "../Audio/WarpLayer.h"
#include "../Models/Project.h"

#include <utility>

/**
 * A warp-layer edit: the whole map before and after. Undo and redo set the
 * map back (warp::apply), which moves the notes with it and queues the
 * composite to be laid out again.
 */
class WarpAction final : public UndoableAction
{
public:
  WarpAction(Project &project, warp::WarpMap before, warp::WarpMap after,
             juce::String name = "Edit Timing")
      : project(project), before(std::move(before)), after(std::move(after)),
        name(std::move(name)) {}

  void undo() override { warp::apply(project, before); }
  void redo() override { warp::apply(project, after); }
  juce::String getName() const override { return name; }

private:
  Project &project;
  warp::WarpMap before;
  warp::WarpMap after;
  juce::String name;
};
