#include "KeyFinder.h"

#include "../../Utils/Constants.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace keyfinder
{
// What a change of key costs, in points (see find()). Set between what the
// songs at hand need (2026-11-26):
//   above 12  晚餐歌 (accompaniment and voice): under it the one window that
//             straddles its change of key, which reads as a third key, is
//             cut out as a stretch of its own;
//   above 9   the 180 recordings of CSD that stay in one key (voice alone):
//             under it one of them is cut; 六等星, whose accompaniment leans
//             to B in three places, is left whole from 3 up;
//   up to 20  the 8 recordings of CSD that do change key (3 songs up a
//             semitone twice, 1 away for 20 s and back, each in two keys):
//             all cut where the annotation changes; at 22 one is missed, at
//             26 two.
// 晚餐歌's own change holds past 300.
const double kChangeCost = 16.0;

namespace
{
// How far `pitch` is from the nearest note of the major scale on `root`.
double distanceToScale(double pitch, int root)
{
    int below = static_cast<int>(std::floor(pitch));
    int above = below + 1;
    while (!inMajorScale(below, root))
        --below;
    while (!inMajorScale(above, root))
        ++above;
    return std::min(std::abs(pitch - below), std::abs(pitch - above));
}

double secondsOf(int frame) { return static_cast<double>(frame) * HOP_SIZE / SAMPLE_RATE; }

// How the pitch classes `share` (from the tonic up) go with a key profile.
double correlation(const std::array<double, 12> &share, const std::array<double, 12> &profile)
{
    double meanA = 0.0, meanB = 0.0;
    for (size_t i = 0; i < 12; ++i)
    {
        meanA += share[i] / 12.0;
        meanB += profile[i] / 12.0;
    }
    double ab = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 0; i < 12; ++i)
    {
        ab += (share[i] - meanA) * (profile[i] - meanB);
        aa += (share[i] - meanA) * (share[i] - meanA);
        bb += (profile[i] - meanB) * (profile[i] - meanB);
    }
    return aa > 0.0 && bb > 0.0 ? ab / std::sqrt(aa * bb) : 0.0;
}

// Krumhansl and Kessler's probe-tone profiles (1982), from the tonic up.
constexpr std::array<double, 12> kMajorProfile{6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88};
constexpr std::array<double, 12> kMinorProfile{6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17};
} // namespace

std::vector<VocalNote> vocalNotes(const Project& project)
{
    std::vector<VocalNote> notes;
    constexpr float moved = 1.0e-3f;
    for (const auto& note : project.getNotes())
    {
        const int frames = note.getSrcEndFrame() - note.getSrcStartFrame();
        if (note.isRest() || frames < kMinNoteFrames)
            continue;
        VocalNote out;
        out.startSeconds = secondsOf(note.getStartFrame());
        out.weight = frames;
        // Put by hand: the centre set (a drag, a snap; the Pitch Center macro
        // moves midiNote and leaves lastNonMacroMidiNote), or offset. A tilt,
        // vibrato or drift alone leaves the centre where it was sung.
        out.placed = std::abs(note.getLastNonMacroMidiNote() - note.getOriginalMidiNote()) > moved ||
                     std::abs(note.getPitchOffset()) > moved;
        out.pitch = out.placed ? note.getLastNonMacroMidiNote() + note.getPitchOffset()
                               : note.getOriginalMidiNote();
        notes.push_back(out);
    }
    return notes;
}

double vocalOffset(const std::vector<VocalNote>& notes)
{
    std::vector<double> offsets;
    for (const auto& note : notes)
        if (!note.placed)
            // A note exactly between two semitones goes to the even one (as
            // numpy rounds): which way is arbitrary, and this one makes the
            // result the Python's that the measurements were made with.
            offsets.push_back(note.pitch - std::nearbyint(note.pitch));
    if (offsets.empty())
        return 0.0;
    std::sort(offsets.begin(), offsets.end());
    const size_t middle = offsets.size() / 2;
    return offsets.size() % 2 == 1 ? offsets[middle]
                                   : 0.5 * (offsets[middle - 1] + offsets[middle]);
}

bool vocalFit(const std::vector<VocalNote>& notes, double offset, std::array<double, 12>& fit)
{
    fit.fill(0.0);
    double weight = 0.0;
    for (const auto& note : notes)
        weight += note.weight;
    if (weight <= 0.0)
        return false;
    for (int root = 0; root < 12; ++root)
    {
        double distance = 0.0;
        for (const auto& note : notes)
            distance += note.weight *
                        distanceToScale(note.placed ? note.pitch : note.pitch - offset, root);
        fit[static_cast<size_t>(root)] = distance / weight;
    }
    return true;
}

Result find(const std::vector<VocalNote>& notes, const Chroma* backing, double changeCost)
{
    Result result;
    result.hasBacking = backing != nullptr && !backing->empty();
    result.tuneSemitones = result.hasBacking ? backing->tuneSemitones : 0.0;
    result.vocalOffsetSemitones = vocalOffset(notes);

    // Windows of 10 s, a whole number of the accompaniment's frames each.
    const int windowFrames =
        result.hasBacking ? std::max(1, static_cast<int>(std::lround(kWindowSeconds / backing->frameSeconds)))
                          : 0;
    const double windowSeconds =
        result.hasBacking ? windowFrames * backing->frameSeconds : kWindowSeconds;
    const int backingFrames = result.hasBacking ? static_cast<int>(backing->frames.size()) : 0;
    double lastNote = -1.0;
    for (const auto& note : notes)
        lastNote = std::max(lastNote, note.startSeconds);
    int count = result.hasBacking ? (backingFrames + windowFrames - 1) / windowFrames : 0;
    if (lastNote >= 0.0)
        count = std::max(count, static_cast<int>(std::floor(lastNote / windowSeconds)) + 1);
    if (count <= 0)
        return result;

    result.windows.assign(static_cast<size_t>(count), {});
    std::vector<double> energy(static_cast<size_t>(count), 0.0), sung(static_cast<size_t>(count), 0.0);
    std::vector<std::vector<VocalNote>> byWindow(static_cast<size_t>(count));
    for (const auto& note : notes)
    {
        const int k = std::clamp(static_cast<int>(std::floor(note.startSeconds / windowSeconds)), 0,
                                 count - 1);
        byWindow[static_cast<size_t>(k)].push_back(note);
        sung[static_cast<size_t>(k)] += note.weight;
    }
    double energyTotal = 0.0, sungTotal = 0.0;
    int voiced = 0;
    for (int k = 0; k < count; ++k)
    {
        auto& window = result.windows[static_cast<size_t>(k)];
        window.startSeconds = k * windowSeconds;
        window.notes = static_cast<int>(byWindow[static_cast<size_t>(k)].size());
        if (result.hasBacking)
        {
            const auto classes = sum(*backing, k * windowFrames, (k + 1) * windowFrames);
            for (const double value : classes)
                energy[static_cast<size_t>(k)] += value;
            window.hasBacking = energy[static_cast<size_t>(k)] > 0.0;
            window.backingShare = scaleShares(classes);
            energyTotal += energy[static_cast<size_t>(k)];
        }
        window.hasVoice = vocalFit(byWindow[static_cast<size_t>(k)], result.vocalOffsetSemitones,
                                   window.voiceFit);
        if (window.hasVoice)
        {
            sungTotal += sung[static_cast<size_t>(k)];
            ++voiced;
        }
    }

    // Each window's evidence in points above its best scale. A window weighs
    // as much as it holds - the accompaniment's energy, the length sung -
    // against the song's average window, so that with no change of key the
    // sums are the whole song's: its share of energy and its notes' distance,
    // a point of the one against a cent of the other.
    const double energyMean = energyTotal > 0.0 ? energyTotal / count : 0.0;
    const double sungMean = voiced > 0 ? sungTotal / voiced : 0.0;
    for (int k = 0; k < count; ++k)
    {
        auto& window = result.windows[static_cast<size_t>(k)];
        if (window.hasBacking && energyMean > 0.0)
        {
            const double best = *std::max_element(window.backingShare.begin(), window.backingShare.end());
            const double weight = energy[static_cast<size_t>(k)] / energyMean;
            for (size_t r = 0; r < 12; ++r)
                window.cost[r] += weight * 100.0 * (best - window.backingShare[r]);
        }
        if (window.hasVoice && sungMean > 0.0)
        {
            const double best = *std::min_element(window.voiceFit.begin(), window.voiceFit.end());
            const double weight = sung[static_cast<size_t>(k)] / sungMean;
            for (size_t r = 0; r < 12; ++r)
                window.cost[r] += kPointsPerCent * weight * 100.0 * (window.voiceFit[r] - best);
        }
    }

    // The cheapest path through the windows, a change of key costing
    // changeCost. Staying wins a tie.
    std::vector<std::array<double, 12>> total(static_cast<size_t>(count));
    std::vector<std::array<int, 12>> from(static_cast<size_t>(count));
    total[0] = result.windows[0].cost;
    from[0].fill(-1);
    for (int k = 1; k < count; ++k)
    {
        const auto& before = total[static_cast<size_t>(k - 1)];
        int cheapest = 0;
        for (int r = 1; r < 12; ++r)
            if (before[static_cast<size_t>(r)] < before[static_cast<size_t>(cheapest)])
                cheapest = r;
        for (int r = 0; r < 12; ++r)
        {
            const double stay = before[static_cast<size_t>(r)];
            const double change = before[static_cast<size_t>(cheapest)] + changeCost;
            const bool stays = stay <= change;
            total[static_cast<size_t>(k)][static_cast<size_t>(r)] =
                result.windows[static_cast<size_t>(k)].cost[static_cast<size_t>(r)] + (stays ? stay : change);
            from[static_cast<size_t>(k)][static_cast<size_t>(r)] = stays ? r : cheapest;
        }
    }
    int key = 0;
    for (int r = 1; r < 12; ++r)
        if (total[static_cast<size_t>(count - 1)][static_cast<size_t>(r)] <
            total[static_cast<size_t>(count - 1)][static_cast<size_t>(key)])
            key = r;
    for (int k = count - 1; k >= 0; --k)
    {
        result.windows[static_cast<size_t>(k)].key = key;
        if (k > 0)
            key = from[static_cast<size_t>(k)][static_cast<size_t>(key)];
    }

    // Major or minor: the pitch classes of the whole song, each window's
    // counted from its own key (so a song that changes key adds up), the
    // accompaniment and the voice each as a share.
    {
        std::array<double, 12> played{}, sungClasses{};
        for (int k = 0; k < count; ++k)
        {
            const int root = result.windows[static_cast<size_t>(k)].key;
            if (result.hasBacking)
            {
                const auto classes = sum(*backing, k * windowFrames, (k + 1) * windowFrames);
                for (int pc = 0; pc < 12; ++pc)
                    played[static_cast<size_t>((pc - root + 12) % 12)] += classes[static_cast<size_t>(pc)];
            }
            for (const auto &note : byWindow[static_cast<size_t>(k)])
            {
                const double pitch = note.placed ? note.pitch : note.pitch - result.vocalOffsetSemitones;
                const int pc = ((static_cast<int>(std::lround(pitch)) - root) % 12 + 12) % 12;
                sungClasses[static_cast<size_t>(pc)] += note.weight;
            }
        }
        std::array<double, 12> share{};
        for (const auto *part : {&played, &sungClasses})
        {
            double total = 0.0;
            for (const double value : *part)
                total += value;
            if (total > 0.0)
                for (size_t pc = 0; pc < 12; ++pc)
                    share[pc] += (*part)[pc] / total;
        }
        std::array<double, 12> fromMinorTonic{};
        for (size_t pc = 0; pc < 12; ++pc)
            fromMinorTonic[pc] = share[(pc + 9) % 12];
        result.majorFit = correlation(share, kMajorProfile);
        result.minorFit = correlation(fromMinorTonic, kMinorProfile);
        result.minor = result.minorFit > result.majorFit;
    }

    // The stretches between changes, each with its two cheapest keys.
    const double end = std::max(result.hasBacking ? backing->duration() : 0.0, count * windowSeconds);
    for (int k = 0; k < count;)
    {
        int last = k;
        while (last + 1 < count &&
               result.windows[static_cast<size_t>(last + 1)].key == result.windows[static_cast<size_t>(k)].key)
            ++last;
        std::array<double, 12> cost{};
        for (int w = k; w <= last; ++w)
            for (size_t r = 0; r < 12; ++r)
                cost[r] += result.windows[static_cast<size_t>(w)].cost[r];
        Stretch stretch;
        stretch.startSeconds = result.windows[static_cast<size_t>(k)].startSeconds;
        stretch.endSeconds = last + 1 < count ? result.windows[static_cast<size_t>(last + 1)].startSeconds : end;
        stretch.key = result.windows[static_cast<size_t>(k)].key;
        stretch.second = stretch.key == 0 ? 1 : 0;
        for (int r = 0; r < 12; ++r)
            if (r != stretch.key && cost[static_cast<size_t>(r)] < cost[static_cast<size_t>(stretch.second)])
                stretch.second = r;
        stretch.lead = (cost[static_cast<size_t>(stretch.second)] - cost[static_cast<size_t>(stretch.key)]) /
                       static_cast<double>(last - k + 1);
        result.stretches.push_back(stretch);
        k = last + 1;
    }
    return result;
}

Result find(const std::vector<VocalNote>& notes, const Chroma* backing)
{
    return find(notes, backing, kChangeCost);
}
} // namespace keyfinder
