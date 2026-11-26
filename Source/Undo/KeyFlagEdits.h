#pragma once

#include "KeyChangeAction.h"
#include "../Audio/WarpLayer.h"
#include "../Utils/Constants.h"

#include <algorithm>
#include <cmath>
#include <vector>

/**
 * The changes of key as the piano roll shows and edits them (spec section
 * 15): the stretches between them on the timeline, and what a flag on the
 * time ruler does - moved, given another key, taken away, added. Each edit
 * gives the key as it would be (a KeyState); the piano roll applies it and
 * keeps it as one history step (KeyChangeAction).
 *
 * A change of key is kept on the recording (KeyChange::sourceFrame); a flag
 * stands where that frame plays on the timeline.
 */
namespace keyflags
{
/** No change of key nearer than this to another or to either end, in frames
 *  of the recording (about 0.1 s: the shortest note the key is found from). */
constexpr int kMinGapFrames = 9;

struct Stretch
{
    int sourceFrame = 0;       // where it starts on the recording; 0: the first
    double startSeconds = 0.0; // where that plays on the timeline
    int root = 0;              // -1: the project has no key
};

inline double secondsOfSourceFrame(const AudioData& audio, int sourceFrame)
{
    const double frame = audio.warpMap.isIdentity()
                             ? static_cast<double>(sourceFrame)
                             : audio.warpMap.destinationAt(static_cast<double>(sourceFrame));
    return frame * HOP_SIZE / SAMPLE_RATE;
}

inline int sourceFrameAtSeconds(const AudioData& audio, double seconds)
{
    return warp::sourceFrameAt(audio, static_cast<int>(std::floor(seconds * SAMPLE_RATE / HOP_SIZE)));
}

/** The frame boundary nearest to `seconds`: where a flag let go there, or a
 *  change of key added there, is put. (The pointer's place comes through
 *  single-precision pixels: a flag dropped on a boundary must not land a
 *  frame short of it.) */
inline int sourceFrameNearSeconds(const AudioData& audio, double seconds)
{
    return warp::sourceFrameAt(audio, static_cast<int>(std::lround(seconds * SAMPLE_RATE / HOP_SIZE)));
}

/** The frames of the recording. */
inline int sourceFrames(const AudioData& audio)
{
    return static_cast<int>(audio.f0.size());
}

/** In order: the first from the start in the project's own key, then one
 *  for each change of key. */
inline std::vector<Stretch> stretches(const Project& project)
{
    std::vector<Stretch> out;
    out.push_back({0, 0.0, project.getScaleRootNote()});
    for (const auto& change : project.getKeyChanges())
        out.push_back({change.sourceFrame,
                       secondsOfSourceFrame(project.getAudioData(), change.sourceFrame), change.root});
    return out;
}

/** The frames change `index` may be moved over: not past its neighbours. */
inline bool movableRange(const Project& project, int index, int& low, int& high)
{
    const auto& changes = project.getKeyChanges();
    const int count = static_cast<int>(changes.size());
    if (index < 0 || index >= count)
        return false;
    low = (index > 0 ? changes[static_cast<size_t>(index - 1)].sourceFrame : 0) + kMinGapFrames;
    high = (index + 1 < count ? changes[static_cast<size_t>(index + 1)].sourceFrame
                              : sourceFrames(project.getAudioData())) -
           kMinGapFrames;
    return high >= low;
}

/** Change `index` (0: the first change of key) at `sourceFrame`, as near to
 *  it as its neighbours and the ends of the recording let it be. */
inline KeyState moved(const Project& project, int index, int sourceFrame)
{
    auto state = KeyState::of(project);
    int low = 0, high = 0;
    if (movableRange(project, index, low, high))
        state.changes[static_cast<size_t>(index)].sourceFrame = std::clamp(sourceFrame, low, high);
    return state;
}

/** Stretch `stretch` in the key on `root`. Stretch 0 is the project's own
 *  key (the one the toolbar shows): its root changes, its mode stays. */
inline KeyState rekeyed(const Project& project, int stretch, int root)
{
    auto state = KeyState::of(project);
    const int pitchClass = (root % 12 + 12) % 12;
    if (stretch == 0)
        (state.useProjectKey ? state.projectRoot : state.regionRoot) = pitchClass;
    else if (stretch > 0 && stretch <= static_cast<int>(state.changes.size()))
        state.changes[static_cast<size_t>(stretch - 1)].root = pitchClass;
    return state;
}

/** Change `index` taken away: its stretch joins the one before it. */
inline KeyState removed(const Project& project, int index)
{
    auto state = KeyState::of(project);
    if (index >= 0 && index < static_cast<int>(state.changes.size()))
        state.changes.erase(state.changes.begin() + index);
    return state;
}

/** Whether a change of key can be put at `sourceFrame`. */
inline bool canAdd(const Project& project, int sourceFrame)
{
    if (sourceFrame < kMinGapFrames ||
        sourceFrame > sourceFrames(project.getAudioData()) - kMinGapFrames)
        return false;
    for (const auto& change : project.getKeyChanges())
        if (std::abs(change.sourceFrame - sourceFrame) < kMinGapFrames)
            return false;
    return true;
}

/** A change to the key on `root` at `sourceFrame`; nothing where one
 *  cannot be (canAdd). */
inline KeyState added(const Project& project, int sourceFrame, int root)
{
    auto state = KeyState::of(project);
    if (!canAdd(project, sourceFrame))
        return state;
    state.changes.push_back({sourceFrame, (root % 12 + 12) % 12});
    std::sort(state.changes.begin(), state.changes.end(),
              [](const KeyChange& a, const KeyChange& b) { return a.sourceFrame < b.sourceFrame; });
    return state;
}
} // namespace keyflags
