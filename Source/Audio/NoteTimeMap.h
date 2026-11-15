#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

/**
 * Where a note's audio plays: source frames [sourceStart, sourceEnd) of the
 * recording over destination frames [destinationStart, destinationEnd) of
 * the timeline, piecewise even between the warp layer's anchors
 * (NoteTiming.h builds it from WarpMap.h).
 *
 * Positions are continuous frame coordinates in which frame k spans
 * [k, k + 1). Destination frame d reads the source at sourceAt(d + 0.5) - 0.5,
 * the frame-centre convention the synthesizer already used.
 *
 * JUCE-free.
 */
namespace notetiming
{
class TimeMap
{
public:
  struct Segment
  {
    double sourceStart = 0.0;
    double sourceEnd = 0.0;
    double destinationStart = 0.0;
    double destinationEnd = 0.0;
    bool vowel = false;
  };

  TimeMap() = default;

  /** The source range spread evenly over the destination range. */
  static TimeMap uniform(double sourceStart, double sourceEnd,
                         double destinationStart, double destinationEnd)
  {
    TimeMap map;
    map.segments_.push_back(
        {sourceStart, sourceEnd, destinationStart, destinationEnd, false});
    return map;
  }

  /** Piecewise even: source points[i].first plays at destination
   *  points[i].second, linearly in between (the warp layer). */
  static TimeMap fromPoints(const std::vector<std::pair<double, double>> &points)
  {
    TimeMap map;
    for (size_t i = 0; i + 1 < points.size(); ++i)
      if (points[i + 1].first > points[i].first)
        map.segments_.push_back({points[i].first, points[i + 1].first,
                                 points[i].second, points[i + 1].second,
                                 false});
    return map;
  }

  /** Source position played at destination position `destination`. */
  double sourceAt(double destination) const
  {
    if (segments_.empty())
      return destination;
    destination = std::clamp(destination, segments_.front().destinationStart,
                             segments_.back().destinationEnd);
    for (const auto &segment : segments_)
    {
      const double length = segment.destinationEnd - segment.destinationStart;
      if (length <= 0.0 || (destination >= segment.destinationEnd &&
                            &segment != &segments_.back()))
        continue;
      return segment.sourceStart + (destination - segment.destinationStart) *
                                       (segment.sourceEnd -
                                        segment.sourceStart) /
                                       length;
    }
    return segments_.back().sourceEnd;
  }

  /** Destination position at which source position `source` plays. */
  double destinationAt(double source) const
  {
    if (segments_.empty())
      return source;
    source = std::clamp(source, segments_.front().sourceStart,
                        segments_.back().sourceEnd);
    for (const auto &segment : segments_)
    {
      const double length = segment.sourceEnd - segment.sourceStart;
      if (length <= 0.0 ||
          (source >= segment.sourceEnd && &segment != &segments_.back()))
        continue;
      return segment.destinationStart +
             (source - segment.sourceStart) *
                 (segment.destinationEnd - segment.destinationStart) / length;
    }
    return segments_.back().destinationEnd;
  }

  /** Fractional source frame read for destination frame `frame`. */
  double sourceFrameAt(int frame) const
  {
    return sourceAt(static_cast<double>(frame) + 0.5) - 0.5;
  }

  const std::vector<Segment> &segments() const { return segments_; }

private:
  std::vector<Segment> segments_;
};
} // namespace notetiming
