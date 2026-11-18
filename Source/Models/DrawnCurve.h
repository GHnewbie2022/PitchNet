#pragma once

#include <algorithm>
#include <vector>

/**
 * A pitch curve drawn with the Pitch Shaper tool, kept on the project so its
 * anchors stay on the pitch line and can be edited again later (like
 * RePitch's shape points).
 *
 * Anchors are frames only: an anchor sits on the sung pitch at its frame, so
 * it follows the note when the note is dragged. The pitch the curve replaced
 * is kept as its difference from the drawn pitch, frame by frame, so moving
 * or removing an anchor later can give those frames back - shifted with the
 * note too.
 */
struct DrawnCurve
{
    int id = 0;
    std::vector<int> anchorFrames;     // sorted
    int underStart = 0;                // first frame of underDiff
    std::vector<float> underDiff;      // pitch before the curve minus pitch with it

    int underEnd() const { return underStart + static_cast<int>(underDiff.size()); }
    bool spans(int frame) const
    {
        return !anchorFrames.empty() && frame >= anchorFrames.front() &&
               frame <= anchorFrames.back();
    }
    bool overlaps(int start, int end) const
    {
        return underStart < end && underEnd() > start;
    }
};

namespace drawncurve
{
/** The curves, without those that reach into [start, end). */
inline std::vector<DrawnCurve> without(const std::vector<DrawnCurve>& curves,
                                       int start, int end)
{
    std::vector<DrawnCurve> kept;
    for (const auto& curve : curves)
        if (!curve.overlaps(start, end))
            kept.push_back(curve);
    return kept;
}
} // namespace drawncurve
