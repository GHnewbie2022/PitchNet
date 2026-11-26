#pragma once

#include "../../Audio/KeyFinder/KeyFinder.h"
#include "../../Audio/WarpLayer.h"
#include "../../Models/Project.h"
#include "../../Undo/KeyChangeAction.h"
#include "../../Utils/Constants.h"

#include <algorithm>
#include <array>
#include <tuple>
#include <utility>
#include <vector>

/**
 * The wizard (spec section 15), without its window: which files it has been
 * given and whether it may go on, the stretches of the key check with their
 * two candidates and the user's choice, and the key that is written to the
 * project when the user confirms. The overlay shows this and the window
 * acts on it; the tests drive it as they do.
 *
 * Entered two ways. Fresh (the welcome screen's guided mode): a recording
 * and its accompaniment are chosen, and the user ticks that the two start at
 * the same place - asked once a file, on the page the files are chosen on,
 * before anything is analysed (a length says nothing: 晚餐歌's two files are
 * 0.6 s apart in length and aligned). From a project that is open: the
 * recording is the project's; the accompaniment is the one it links, asked
 * about only if it was never confirmed, or none - then the key is found from
 * the voice alone.
 */
namespace wizard
{
enum class Page { files, analysing, keyCheck, nextStep };

/** A stretch of the song in one key: the two likeliest keys, and the one chosen. */
struct Stretch
{
    double startSeconds = 0.0, endSeconds = 0.0;
    int first = 0, second = 0; // roots, 0 = C
    int chosen = 0;
};

/** Where a mode's tonic lies above the root of the major scale of the same
 *  notes: A minor is C major's notes from 9 up. The modes that are not a
 *  major scale's notes (harmonic and melodic minor, blues, phrygian
 *  dominant) are taken by the major scale nearest to them; whole tone has
 *  none and stays on the root. */
inline int tonicOffset(ScaleMode mode)
{
    switch (mode)
    {
    case ScaleMode::Minor:
    case ScaleMode::HarmonicMinor:
    case ScaleMode::MelodicMinor:
    case ScaleMode::MinorPentatonic:
    case ScaleMode::Blues: return 9;
    case ScaleMode::Dorian: return 2;
    case ScaleMode::Phrygian:
    case ScaleMode::PhrygianDominant: return 4;
    case ScaleMode::Lydian: return 5;
    case ScaleMode::Mixolydian: return 7;
    case ScaleMode::Locrian: return 11;
    default: return 0;
    }
}

class Model
{
public:
    /** The mode of the whole song (a change of key keeps it): major or
     *  minor as the analysis finds, any other by the user's choice. The
     *  stretches' keys below are the major roots of their seven notes; what
     *  is shown, heard and written to the project is the mode's tonic. */
    ScaleMode mode = ScaleMode::Major;
    int tonicOf(int majorRoot) const { return ((majorRoot + tonicOffset(mode)) % 12 + 12) % 12; }
    int majorRootOf(int tonic) const { return ((tonic - tonicOffset(mode)) % 12 + 12) % 12; }

    /** No change of key nearer than this to another, or to either end. */
    static constexpr double kMinStretchSeconds = 1.0;

    bool existingProject = false; // the voice is the open project's recording
    juce::File voice;             // a new project's recording
    juce::File backing;           // the accompaniment; none: the voice alone
    bool aligned = false;         // the user's tick for these two files
    Page page = Page::files;

    static Model fresh() { return {}; }

    /** For the project that is open: its accompaniment and what was said of it. */
    static Model forProject(const Project& project)
    {
        Model model;
        model.existingProject = true;
        model.backing = project.getBacking().file;
        model.aligned = project.getBacking().isSet() && project.getBacking().aligned;
        return model;
    }

    void setVoice(const juce::File& file) { voice = file; }

    /** Another accompaniment has not been said to be aligned. */
    void setBacking(const juce::File& file)
    {
        if (file == backing)
            return;
        backing = file;
        aligned = false;
    }

    bool hasBacking() const { return backing != juce::File{}; }

    /** The files page may go on. A new project needs both files and the
     *  tick; an open one needs the tick only if it has an accompaniment. */
    bool canAnalyse() const
    {
        if (existingProject)
            return !hasBacking() || aligned;
        return voice != juce::File{} && hasBacking() && aligned;
    }

    /** An open project whose accompaniment was confirmed before: there is
     *  nothing to ask, and the wizard goes straight to the analysis. */
    bool nothingToAsk() const { return existingProject && hasBacking() && aligned; }

    // ---- the key check

    void setResult(const keyfinder::Result& found)
    {
        result = found;
        mode = found.minor ? ScaleMode::Minor : ScaleMode::Major;
        stretches.clear();
        for (const auto& s : found.stretches)
            stretches.push_back({s.startSeconds, s.endSeconds, s.key, s.second, s.key});
    }

    const keyfinder::Result& getResult() const { return result; }
    const std::vector<Stretch>& getStretches() const { return stretches; }
    double duration() const { return stretches.empty() ? 0.0 : stretches.back().endSeconds; }

    int stretchAt(double seconds) const
    {
        for (size_t i = 0; i < stretches.size(); ++i)
            if (seconds < stretches[i].endSeconds)
                return static_cast<int>(i);
        return static_cast<int>(stretches.size()) - 1;
    }

    /**
     * The key the project was confirmed in before, in place of the
     * analysis's own stretches: `spans` are where each stretch starts
     * (seconds; the first from 0) and its root, in order. Each keeps the two
     * candidates the evidence gives for its span; its own key is the one
     * chosen. A stretch too short to stand is left out.
     */
    void adopt(const std::vector<std::pair<double, int>>& spans, ScaleMode projectMode = ScaleMode::Major)
    {
        if (spans.empty() || stretches.empty())
            return;
        if (projectMode != ScaleMode::None && projectMode != ScaleMode::Chromatic)
            mode = projectMode; // the roots given are that mode's tonics
        const double end = duration();
        std::vector<Stretch> adopted;
        for (size_t i = 0; i < spans.size(); ++i)
        {
            const double start = i == 0 ? 0.0 : spans[i].first;
            const double stop = i + 1 < spans.size() ? spans[i + 1].first : end;
            if (stop - start < kMinStretchSeconds)
                continue;
            const auto like = stretches[static_cast<size_t>(stretchAt(start))];
            const auto two = candidates(start, stop, like);
            adopted.push_back({start, stop, two.first, two.second, majorRootOf(spans[i].second)});
        }
        if (!adopted.empty())
        {
            adopted.front().startSeconds = 0.0;
            adopted.back().endSeconds = end;
            for (size_t i = 0; i + 1 < adopted.size(); ++i)
                adopted[i].endSeconds = adopted[i + 1].startSeconds;
            stretches = std::move(adopted);
        }
    }

    void choose(int index, int root)
    {
        if (index >= 0 && index < static_cast<int>(stretches.size()))
            stretches[static_cast<size_t>(index)].chosen = (root % 12 + 12) % 12;
    }

    /**
     * The two keys the evidence likes best between two times: the windows
     * whose middle lies there, their costs added (KeyFinder.h). With no
     * window there, the stretch `like` is of.
     */
    std::pair<int, int> candidates(double startSeconds, double endSeconds, const Stretch& like) const
    {
        std::array<double, 12> cost{};
        int windows = 0;
        for (size_t i = 0; i < result.windows.size(); ++i)
        {
            const double start = result.windows[i].startSeconds;
            const double end = i + 1 < result.windows.size() ? result.windows[i + 1].startSeconds
                                                             : start + keyfinder::kWindowSeconds;
            const double middle = 0.5 * (start + end);
            if (middle < startSeconds || middle >= endSeconds)
                continue;
            for (size_t r = 0; r < 12; ++r)
                cost[r] += result.windows[i].cost[r];
            ++windows;
        }
        if (windows == 0)
            return {like.first, like.second};
        int first = 0;
        for (int r = 1; r < 12; ++r)
            if (cost[static_cast<size_t>(r)] < cost[static_cast<size_t>(first)])
                first = r;
        int second = first == 0 ? 1 : 0;
        for (int r = 0; r < 12; ++r)
            if (r != first && cost[static_cast<size_t>(r)] < cost[static_cast<size_t>(second)])
                second = r;
        return {first, second};
    }

    /**
     * A change of key at `seconds`: the stretch there becomes two, each with
     * its own two candidates. The earlier keeps the key chosen; the later
     * starts on its own likeliest. False when it is too near another change
     * or an end.
     */
    bool addChangeAt(double seconds)
    {
        const int index = stretchAt(seconds);
        if (index < 0)
            return false;
        const Stretch whole = stretches[static_cast<size_t>(index)];
        if (seconds - whole.startSeconds < kMinStretchSeconds ||
            whole.endSeconds - seconds < kMinStretchSeconds)
            return false;
        Stretch earlier = whole, later = whole;
        earlier.endSeconds = seconds;
        later.startSeconds = seconds;
        std::tie(earlier.first, earlier.second) = candidates(earlier.startSeconds, earlier.endSeconds, whole);
        std::tie(later.first, later.second) = candidates(later.startSeconds, later.endSeconds, whole);
        later.chosen = later.first;
        stretches[static_cast<size_t>(index)] = earlier;
        stretches.insert(stretches.begin() + index + 1, later);
        return true;
    }

    /** The change of key stretch `index` starts with is taken away: it joins
     *  the stretch before, in that one's key. */
    bool removeChange(int index)
    {
        if (index < 1 || index >= static_cast<int>(stretches.size()))
            return false;
        Stretch joined = stretches[static_cast<size_t>(index - 1)];
        joined.endSeconds = stretches[static_cast<size_t>(index)].endSeconds;
        std::tie(joined.first, joined.second) = candidates(joined.startSeconds, joined.endSeconds, joined);
        stretches[static_cast<size_t>(index - 1)] = joined;
        stretches.erase(stretches.begin() + index);
        return true;
    }

    /** The change stretch `index` starts with, moved to `seconds`: held
     *  kMinStretchSeconds inside its two neighbours. Candidates follow. */
    bool moveChange(int index, double seconds)
    {
        if (index < 1 || index >= static_cast<int>(stretches.size()))
            return false;
        auto& before = stretches[static_cast<size_t>(index - 1)];
        auto& after = stretches[static_cast<size_t>(index)];
        const double low = before.startSeconds + kMinStretchSeconds;
        const double high = after.endSeconds - kMinStretchSeconds;
        if (high < low)
            return false;
        seconds = std::clamp(seconds, low, high);
        before.endSeconds = seconds;
        after.startSeconds = seconds;
        std::tie(before.first, before.second) = candidates(before.startSeconds, before.endSeconds, before);
        std::tie(after.first, after.second) = candidates(after.startSeconds, after.endSeconds, after);
        return true;
    }

    // ---- confirming

    /**
     * The key as the user left it, for `project`: the first stretch's key,
     * always major (KeyState::confirmed), and a change of key wherever a
     * stretch is in another key than the one before it - at the frame of the
     * recording that plays where the stretch starts on the timeline (the
     * accompaniment is on the timeline, a change of key is kept on the
     * recording: KeyChange).
     */
    KeyState confirmed(const Project& project) const
    {
        std::vector<KeyChange> changes;
        for (size_t i = 1; i < stretches.size(); ++i)
        {
            if (stretches[i].chosen == stretches[i - 1].chosen)
                continue;
            const int timelineFrame = secondsToFrames(static_cast<float>(stretches[i].startSeconds));
            changes.push_back({warp::sourceFrameAt(project.getAudioData(), timelineFrame),
                               tonicOf(stretches[i].chosen)});
        }
        return KeyState::confirmed(project, stretches.empty() ? project.getScaleRootNote()
                                                              : tonicOf(stretches.front().chosen),
                                   std::move(changes), mode);
    }

private:
    keyfinder::Result result;
    std::vector<Stretch> stretches;
};
} // namespace wizard
