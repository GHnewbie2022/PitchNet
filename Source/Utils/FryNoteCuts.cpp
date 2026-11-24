#include "FryNoteCuts.h"

#include "../UI/PianoRoll/NoteSplitter.h"
#include "FryDetector.h"
#include "PitchCurveProcessor.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>

namespace frycut
{
namespace
{
std::atomic<bool> cutting{true};

Note* noteStartingAtSource(Project& project, int sourceStart)
{
    for (auto& note : project.getNotes())
        if (!note.isRest() && note.getSrcStartFrame() == sourceStart)
            return &note;
    return nullptr;
}

// The note is one the analysis made, as it made it.
bool analysedAsItIs(const Note& note)
{
    const auto& spans = note.getAnalysisSpans();
    return spans.size() == 1 && spans.front().first == note.getSrcStartFrame() &&
           spans.front().second == note.getSrcEndFrame();
}

// ...and from here on this piece is: its unedited centre and curve are what
// "Restore note separation" and a pitch restore go back to.
void analysedFromNowOn(Note& note)
{
    note.setAnalysisSpans({{note.getSrcStartFrame(), note.getSrcEndFrame(),
                            note.getOriginalMidiNote(), note.getOriginalDeltaPitch()}});
}

bool cut(Project& project, const Cut& wantedCut)
{
    Note* note = noteStartingAtSource(project, wantedCut.noteStart);
    if (note == nullptr)
        return false;
    const bool analysed = analysedAsItIs(*note);
    // The split works on the timeline: the frame goes through the warp map,
    // and the source edges are then set to it exactly (as
    // NoteSplitter::restoreSeparation cuts where the analysis did).
    const int at = static_cast<int>(
        std::lround(project.getAudioData().warpMap.destinationAt(wantedCut.at)));
    NoteSplitter splitter;
    splitter.setProject(&project);
    // Both pieces stay on the note's centre: the few low frames of the fry
    // would put the piece that starts with it well under the voice it is.
    if (!splitter.splitNoteAtFrame(note, at, true))
        return false;
    Note* left = noteStartingAtSource(project, wantedCut.noteStart);
    Note* right = nullptr;
    for (auto& candidate : project.getNotes())
        if (!candidate.isRest() && candidate.getStartFrame() == at && &candidate != left)
            right = &candidate;
    if (left == nullptr || right == nullptr)
        return true;
    left->setSrcEndFrame(wantedCut.at);
    right->setSrcStartFrame(wantedCut.at);
    if (analysed)
    {
        analysedFromNowOn(*left);
        analysedFromNowOn(*right);
    }
    return true;
}
} // namespace

std::vector<Cut> wanted(const Project& project)
{
    std::vector<Cut> cuts;
    const auto& notes = project.getNotes();
    for (const auto& [first, last] : fry::onsets(project.getAudioData().rawF0))
        for (const auto& note : notes)
            if (!note.isRest() && note.getSrcStartFrame() < first && last < note.getSrcEndFrame())
                cuts.push_back({note.getSrcStartFrame(), first});
    return cuts;
}

void setEnabled(bool enabled) { cutting.store(enabled); }

int apply(Project& project)
{
    if (!cutting.load())
        return 0;
    // The split rebuilds the pitch curves over the project's frames
    // (PitchCurveProcessor::rebuildBaseFromNotes): with none known yet - a
    // linked project before its recording is read - it would empty them.
    auto& audio = project.getAudioData();
    const auto frames = static_cast<size_t>(std::max(0, audio.getNumFrames()));
    if (frames == 0 || audio.basePitch.size() != frames || audio.deltaPitch.size() != frames)
        return 0;
    if (wanted(project).empty())
        return 0;

    // The curves and the notes as they are: put back below.
    struct Curves
    {
        std::vector<float> base, delta, f0, baseF0;
    };
    const auto curves = [&audio] {
        return Curves{audio.basePitch, audio.deltaPitch, audio.f0, audio.baseF0};
    };
    const Curves loaded = curves();
    const auto notesBefore = project.getNotes();
    // ...and the curves as these notes lay them out, which is what the cuts
    // are held against: a file's own can be far from that between its notes
    // (a format 6 file of 煙幕 keeps a deviation there, which a rebuild
    // leaves at none: 5088 frames, up to 7 semitones), and held against
    // those, every cut looked like a change of pitch.
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    const Curves laidOut = curves();
    std::vector<std::uint64_t> cutNotes; // the left piece keeps the note's uid

    int made = 0;
    // Each cut moves the notes, so the list is taken again after it. A place
    // is tried once: a cut that cannot be made (a stretched note with no
    // frame of its own there on the timeline) is left.
    std::vector<int> tried;
    for (;;)
    {
        const auto cuts = wanted(project);
        const auto next = std::find_if(cuts.begin(), cuts.end(), [&](const Cut& c) {
            return std::find(tried.begin(), tried.end(), c.at) == tried.end();
        });
        if (next == cuts.end())
            break;
        tried.push_back(next->at);
        const Note* note = noteStartingAtSource(project, next->noteStart);
        const auto uid = note != nullptr ? note->getUid() : 0;
        if (cut(project, *next))
        {
            ++made;
            cutNotes.push_back(uid);
        }
    }
    const auto asLoaded = [&] {
        audio.basePitch = loaded.base;
        audio.deltaPitch = loaded.delta;
        audio.f0 = loaded.f0;
        audio.baseF0 = loaded.baseF0;
    };
    if (made == 0)
    {
        asLoaded();
        return 0;
    }

    // Both pieces keep the note's centre and its curve, so the rebuilds
    // inside the split give the pitch curves back as these notes laid them
    // out - to the last digits (a tilt folded into a piece's curve came out
    // 1e-4 of a semitone off at one frame of 煙幕). Then the cuts changed
    // nothing that is sung, and all but the notes cut is put back as it was
    // read: the curves, value for value, and every other note - the split
    // had written 600 of the 620 notes' analysed curves of that song again, a
    // digit off in the fourth place, and the next rebuild read them. So
    // nothing rendered or judged from the curves moves by that rounding, here
    // or in the rest of the song: a project opens as it did before notes were
    // cut, but for the notes cut. Further off than that would be a change of
    // pitch, which a cut is not to make: then all stays as the split left it.
    constexpr float kSemitones = 1.0e-3f;
    constexpr float kHertz = 0.05f;
    const auto within = [](const std::vector<float>& now, const std::vector<float>& before,
                           float tolerance) {
        if (now.size() != before.size())
            return false;
        for (size_t i = 0; i < now.size(); ++i)
            if (std::abs(now[i] - before[i]) > tolerance)
                return false;
        return true;
    };
    if (within(audio.basePitch, laidOut.base, kSemitones) &&
        within(audio.deltaPitch, laidOut.delta, kSemitones) && within(audio.f0, laidOut.f0, kHertz))
    {
        asLoaded();
        for (auto& note : project.getNotes())
        {
            if (std::find(cutNotes.begin(), cutNotes.end(), note.getUid()) != cutNotes.end())
                continue;
            const auto was = std::find_if(notesBefore.begin(), notesBefore.end(),
                                          [&](const Note& n) { return n.getUid() == note.getUid(); });
            if (was != notesBefore.end())
                note = *was;
        }
    }

    // Not an edit of the user's: the debug journal goes on from the notes
    // as they are cut.
    project.getDebugJournal().resync(project.getNotes());
    return made;
}
} // namespace frycut
