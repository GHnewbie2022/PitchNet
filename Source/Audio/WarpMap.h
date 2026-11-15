#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

/**
 * The warp layer: one time map for the whole track, AudioWarp style.
 *
 * A list of anchors, each pinning a source position (in the recording) to a
 * destination position (on the timeline). Between two anchors the recording
 * is stretched evenly - consonants, vowels and gaps alike. The two ends of
 * the track are implicit anchors that never move, so the track keeps its
 * length and dragging an anchor stretches the two segments next to it in
 * opposite directions while everything else stays put.
 *
 * Positions are continuous frame coordinates (frame k spans [k, k + 1)), at
 * the analysis hop. Notes store source frames; their timeline frames are
 * derived from this map (WarpNotes.h).
 *
 * JUCE-free so it can be unit tested on its own.
 */
namespace warp
{
// Hard limits per segment (spec Q2): at least one frame long on the
// timeline, and 12.5% .. 800% of its recorded length.
constexpr double kMinSegmentFrames = 1.0;
constexpr double kMinRatio = 0.125;
constexpr double kMaxRatio = 8.0;
// Outside this range signalsmith-stretch's quality is not assured; the UI
// shows the ratio in a warning colour but still allows it.
constexpr double kCleanMinRatio = 0.5;
constexpr double kCleanMaxRatio = 2.0;

struct Anchor
{
  double source = 0.0;
  double destination = 0.0;

  bool operator==(const Anchor &other) const
  {
    return source == other.source && destination == other.destination;
  }
  bool operator!=(const Anchor &other) const { return !(*this == other); }
};

class WarpMap
{
public:
  WarpMap() = default;
  explicit WarpMap(double lengthFrames) : length_(std::max(0.0, lengthFrames)) {}

  double length() const { return length_; }
  void setLength(double lengthFrames)
  {
    length_ = std::max(0.0, lengthFrames);
    anchors_.erase(std::remove_if(anchors_.begin(), anchors_.end(),
                                  [this](const Anchor &a)
                                  {
                                    return a.source <= 0.0 ||
                                           a.source >= length_ ||
                                           a.destination <= 0.0 ||
                                           a.destination >= length_;
                                  }),
                   anchors_.end());
  }

  /** User anchors, sorted, without the two implicit end anchors. */
  const std::vector<Anchor> &anchors() const { return anchors_; }
  void setAnchors(std::vector<Anchor> anchors)
  {
    std::sort(anchors.begin(), anchors.end(),
              [](const Anchor &a, const Anchor &b)
              { return a.source < b.source; });
    anchors_ = std::move(anchors);
  }
  void clear() { anchors_.clear(); }

  bool operator==(const WarpMap &other) const
  {
    return length_ == other.length_ && anchors_ == other.anchors_;
  }
  bool operator!=(const WarpMap &other) const { return !(*this == other); }

  /** Every anchor, including the fixed ends (0,0) and (length,length). */
  std::vector<Anchor> points() const
  {
    std::vector<Anchor> result;
    result.reserve(anchors_.size() + 2);
    result.push_back({0.0, 0.0});
    for (const auto &a : anchors_)
      result.push_back(a);
    result.push_back({length_, length_});
    return result;
  }

  /** True when no anchor has moved: every segment plays at its own place. */
  bool isIdentity() const
  {
    for (const auto &a : anchors_)
      if (a.source != a.destination)
        return false;
    return true;
  }

  double destinationAt(double source) const
  {
    return interpolate(source, &Anchor::source, &Anchor::destination);
  }

  double sourceAt(double destination) const
  {
    return interpolate(destination, &Anchor::destination, &Anchor::source);
  }

  /** Fractional source frame read for timeline frame `frame`, using the
   *  frame-centre convention the synthesizer already used. */
  double sourceFrameAt(int frame) const
  {
    return sourceAt(static_cast<double>(frame) + 0.5) - 0.5;
  }

  /** Stretch ratio (timeline length over recorded length) of the segment that
   *  contains timeline position `destination`. 1 = unchanged. */
  double ratioAtDestination(double destination) const
  {
    const auto p = points();
    for (size_t i = 0; i + 1 < p.size(); ++i)
      if (destination < p[i + 1].destination || i + 2 == p.size())
        return segmentRatio(p[i], p[i + 1]);
    return 1.0;
  }

  static double segmentRatio(const Anchor &left, const Anchor &right)
  {
    const double sourceLength = right.source - left.source;
    const double destinationLength = right.destination - left.destination;
    return sourceLength > 0.0 ? destinationLength / sourceLength : 1.0;
  }

  /** Index of the anchor at source position `source` (within `tolerance`),
   *  or -1. */
  int findAnchorAtSource(double source, double tolerance = 1.0e-6) const
  {
    for (size_t i = 0; i < anchors_.size(); ++i)
      if (std::abs(anchors_[i].source - source) <= tolerance)
        return static_cast<int>(i);
    return -1;
  }

  /** Index of the anchor at timeline position `destination`, or -1. */
  int findAnchorAtDestination(double destination,
                              double tolerance = 1.0e-6) const
  {
    for (size_t i = 0; i < anchors_.size(); ++i)
      if (std::abs(anchors_[i].destination - destination) <= tolerance)
        return static_cast<int>(i);
    return -1;
  }

  /**
   * Pins the recording where it currently plays at timeline position
   * `destination`. Does not change the sound. Returns the anchor's index,
   * the existing one if there already is an anchor there, or -1 if the
   * position is an end of the track or closer than a frame to another anchor.
   */
  int addAnchorAtDestination(double destination)
  {
    if (!(destination > 0.0 && destination < length_))
      return -1;
    const int existing = findAnchorAtDestination(destination);
    if (existing >= 0)
      return existing;
    const Anchor anchor{sourceAt(destination), destination};
    for (const auto &p : points())
      if (std::abs(p.destination - anchor.destination) < kMinSegmentFrames ||
          std::abs(p.source - anchor.source) < kMinSegmentFrames)
        return -1;
    const auto it = std::lower_bound(anchors_.begin(), anchors_.end(), anchor,
                                     [](const Anchor &a, const Anchor &b)
                                     { return a.source < b.source; });
    return static_cast<int>(anchors_.insert(it, anchor) - anchors_.begin());
  }

  /** Same, for a source position. */
  int addAnchorAtSource(double source)
  {
    const int existing = findAnchorAtSource(source);
    if (existing >= 0)
      return existing;
    if (!(source > 0.0 && source < length_))
      return -1;
    return addAnchorAtDestination(destinationAt(source));
  }

  /** Deletes an anchor; its two neighbours' segments merge and are
   *  re-stretched evenly between the anchors that remain. */
  bool removeAnchor(int index)
  {
    if (index < 0 || index >= static_cast<int>(anchors_.size()))
      return false;
    anchors_.erase(anchors_.begin() + index);
    return true;
  }

  /**
   * The range of timeline positions anchor `index` may be dragged to: each
   * neighbouring segment at least kMinSegmentFrames long and within
   * kMinRatio .. kMaxRatio of its recorded length.
   */
  std::pair<double, double> destinationRange(int index) const
  {
    if (index < 0 || index >= static_cast<int>(anchors_.size()))
      return {0.0, 0.0};
    const auto p = points();
    const auto &left = p[static_cast<size_t>(index)];
    const auto &self = p[static_cast<size_t>(index) + 1];
    const auto &right = p[static_cast<size_t>(index) + 2];
    const double leftSource = self.source - left.source;
    const double rightSource = right.source - self.source;
    const double low = std::max(
        left.destination +
            std::max(kMinSegmentFrames, kMinRatio * leftSource),
        right.destination - kMaxRatio * rightSource);
    const double high = std::min(
        left.destination + kMaxRatio * leftSource,
        right.destination -
            std::max(kMinSegmentFrames, kMinRatio * rightSource));
    if (low > high)
      return {self.destination, self.destination};
    return {low, high};
  }

  /** Drags anchor `index` to `destination`, clamped to destinationRange().
   *  Returns the position it ended up at. */
  double moveAnchor(int index, double destination)
  {
    if (index < 0 || index >= static_cast<int>(anchors_.size()))
      return destination;
    const auto [low, high] = destinationRange(index);
    anchors_[static_cast<size_t>(index)].destination =
        std::clamp(destination, low, high);
    return anchors_[static_cast<size_t>(index)].destination;
  }

  /** True when both maps place every source position at the same timeline
   *  position: they differ at most by anchors that pin the recording where
   *  it already plays. Such a change does not alter the sound. */
  bool sameMappingAs(const WarpMap &other, double tolerance = 1.0e-6) const
  {
    if (length_ != other.length_)
      return false;
    for (const auto &a : anchors_)
      if (std::abs(other.destinationAt(a.source) - a.destination) > tolerance)
        return false;
    for (const auto &a : other.anchors_)
      if (std::abs(destinationAt(a.source) - a.destination) > tolerance)
        return false;
    return true;
  }

  /**
   * Where the timeline of map `from` lands on the timeline of map `to`: a
   * piecewise linear function given by its breakpoints, each an Anchor with
   * `source` on `from`'s timeline and `destination` on `to`'s. Sorted, from
   * (0, 0) to (length, length). Used to draw audio laid out by one map where
   * another map puts it.
   */
  static std::vector<Anchor> relayout(const WarpMap &from, const WarpMap &to)
  {
    std::vector<Anchor> result;
    for (const auto &p : to.points())
      result.push_back({from.destinationAt(p.source), p.destination});
    for (const auto &p : from.points())
      result.push_back({p.destination, to.destinationAt(p.source)});
    std::sort(result.begin(), result.end(),
              [](const Anchor &a, const Anchor &b)
              { return a.destination < b.destination; });
    std::vector<Anchor> unique;
    for (const auto &a : result)
      if (unique.empty() || a.destination - unique.back().destination > 1.0e-9)
        unique.push_back(a);
    return unique;
  }

  /** True when every segment satisfies the hard limits. */
  bool isValid() const
  {
    const auto p = points();
    for (size_t i = 0; i + 1 < p.size(); ++i)
    {
      const double s = p[i + 1].source - p[i].source;
      const double d = p[i + 1].destination - p[i].destination;
      if (!(s > 0.0) || d < kMinSegmentFrames - 1.0e-9 ||
          d < kMinRatio * s - 1.0e-9 || d > kMaxRatio * s + 1.0e-9)
        return false;
    }
    return true;
  }

private:
  double interpolate(double x, double Anchor::*from, double Anchor::*to) const
  {
    if (anchors_.empty() || !(length_ > 0.0))
      return x;
    // Outside the track the map continues 1:1, so positions just past the
    // ends (padding, the frame-centre half frame) stay meaningful.
    if (x <= 0.0 || x >= length_)
      return x;
    const auto p = points();
    size_t hi = 1;
    while (hi + 1 < p.size() && x >= p[hi].*from)
      ++hi;
    const auto &a = p[hi - 1];
    const auto &b = p[hi];
    const double span = b.*from - a.*from;
    if (!(span > 0.0))
      return a.*to;
    return a.*to + (x - a.*from) * (b.*to - a.*to) / span;
  }

  double length_ = 0.0;
  std::vector<Anchor> anchors_;
};
} // namespace warp
