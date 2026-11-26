// The wizard without its window (spec section 15): when it may go on from
// the files, what the key check holds, and the key it writes - with the
// calls the App's window makes (wizard::Model, keyfinder::find, KeyState).
#include "TestAssert.h"

#include "../Source/UI/Wizard/WizardModel.h"
#include "../Source/Undo/PitchUndoManager.h"

#include <cmath>
#include <cstdio>

namespace
{
const juce::File voiceFile("/tmp/voice.wav"), backingFile("/tmp/backing.wav"),
    otherBacking("/tmp/other.wav");

// A chroma made by hand: 10 s windows, each with the major scale on keys[k].
keyfinder::Chroma chromaOf(const std::vector<int>& keys)
{
    keyfinder::Chroma chroma;
    chroma.frameSeconds = 1024.0 / 12000.0;
    const int perWindow = static_cast<int>(std::lround(keyfinder::kWindowSeconds / chroma.frameSeconds));
    for (const int key : keys)
        for (int f = 0; f < perWindow; ++f)
        {
            std::array<float, 12> frame{};
            for (int c = 0; c < 12; ++c)
                frame[static_cast<size_t>(c)] = keyfinder::inMajorScale(c, key) ? 1.0f : 0.05f;
            chroma.frames.push_back(frame);
        }
    return chroma;
}

std::vector<keyfinder::VocalNote> sing(const std::vector<double>& pitches, int count, double start)
{
    std::vector<keyfinder::VocalNote> notes;
    for (int i = 0; i < count; ++i)
        notes.push_back({start + 0.5 * i, 26.0, pitches[static_cast<size_t>(i) % pitches.size()], false});
    return notes;
}

// G# for 90 s, then A for 50 s: the accompaniment and the voice alike.
keyfinder::Result modulating()
{
    std::vector<int> keys(9, 8);
    keys.insert(keys.end(), 5, 9);
    const auto chroma = chromaOf(keys);
    auto notes = sing({68.0, 70.0, 72.0, 73.0, 75.0, 77.0, 79.0}, 170, 0.0);
    const auto later = sing({69.0, 71.0, 73.0, 74.0, 76.0, 78.0, 80.0}, 90, 90.0);
    notes.insert(notes.end(), later.begin(), later.end());
    return keyfinder::find(notes, &chroma);
}

keyfinder::Result steady(int key, int windows)
{
    const auto chroma = chromaOf(std::vector<int>(static_cast<size_t>(windows), key));
    return keyfinder::find(sing({60.0 + key, 62.0 + key, 64.0 + key, 67.0 + key, 69.0 + key}, 20 * windows, 0.0),
                           &chroma);
}

void fill(Project& project, int frames)
{
    auto& audio = project.getAudioData();
    audio.sampleRate = SAMPLE_RATE;
    audio.f0.assign(static_cast<size_t>(frames), 220.0f);
    audio.voicedMask.assign(static_cast<size_t>(frames), true);
    audio.warpMap.setLength(static_cast<double>(frames));
}

// 29. A new project: both files and the tick, before it may go on.
void aNewProjectNeedsBothFilesAndTheTick()
{
    auto model = wizard::Model::fresh();
    CHECK(model.page == wizard::Page::files && !model.existingProject);
    CHECK(!model.canAnalyse());
    model.setVoice(voiceFile);
    CHECK(!model.canAnalyse());
    model.aligned = true; // a tick with no accompaniment to tick for
    CHECK(!model.canAnalyse());
    model.setBacking(backingFile);
    CHECK(!model.aligned && !model.canAnalyse()); // the tick was not for this file
    model.aligned = true;
    CHECK(model.canAnalyse());
    CHECK(!model.nothingToAsk());
    model.setVoice(juce::File{});
    CHECK(!model.canAnalyse());
}

// 30. An open project whose accompaniment was confirmed: nothing to ask.
void anOpenProjectIsAskedOnce()
{
    Project project;
    BackingTrack backing;
    backing.file = backingFile;
    backing.aligned = true;
    project.setBacking(backing);
    auto model = wizard::Model::forProject(project);
    CHECK(model.existingProject && model.backing == backingFile && model.aligned);
    CHECK(model.nothingToAsk() && model.canAnalyse());

    // Linked but never confirmed (chosen in the left panel): asked.
    backing.aligned = false;
    project.setBacking(backing);
    model = wizard::Model::forProject(project);
    CHECK(!model.nothingToAsk() && !model.canAnalyse());
    model.aligned = true;
    CHECK(model.canAnalyse());

    // No accompaniment: it may go on, on the voice alone.
    Project bare;
    model = wizard::Model::forProject(bare);
    CHECK(!model.hasBacking() && !model.nothingToAsk() && model.canAnalyse());
    model.aligned = false;
    CHECK(model.canAnalyse());
}

// 31. Another accompaniment: the tick is for the file, and goes with it.
void anotherAccompanimentIsAskedAgain()
{
    Project project;
    BackingTrack backing;
    backing.file = backingFile;
    backing.aligned = true;
    project.setBacking(backing);
    auto model = wizard::Model::forProject(project);
    model.setBacking(backingFile); // the same file: still confirmed
    CHECK(model.aligned && model.nothingToAsk());
    model.setBacking(otherBacking);
    CHECK(!model.aligned && !model.canAnalyse() && !model.nothingToAsk());
    model.aligned = true;
    CHECK(model.canAnalyse());
    model.setBacking(juce::File{}); // taken away: the voice alone
    CHECK(!model.aligned && model.canAnalyse());
}

// 34. The key check: one stretch when no change is found; a change added
// makes two, each with its two candidates; taken away, one again.
void theStretchesOfTheKeyCheck()
{
    wizard::Model model;
    model.setResult(steady(0, 12)); // C for 2 minutes
    CHECK(model.getStretches().size() == 1);
    auto whole = model.getStretches().front();
    CHECK(whole.first == 0 && whole.chosen == 0 && whole.second != 0);
    CHECK(std::abs(whole.startSeconds) < 1.0e-9 && whole.endSeconds > 119.0);
    CHECK(model.stretchAt(0.0) == 0 && model.stretchAt(500.0) == 0);

    CHECK(model.addChangeAt(60.0));
    CHECK(model.getStretches().size() == 2);
    const auto a = model.getStretches()[0], b = model.getStretches()[1];
    CHECK(std::abs(a.endSeconds - 60.0) < 1.0e-9 && std::abs(b.startSeconds - 60.0) < 1.0e-9);
    CHECK(a.first == 0 && b.first == 0 && a.second != a.first && b.second != b.first);
    CHECK(a.chosen == 0 && b.chosen == 0);
    CHECK(model.stretchAt(59.9) == 0 && model.stretchAt(60.0) == 1);
    // Not within a second of another change or an end.
    CHECK(!model.addChangeAt(60.5) && !model.addChangeAt(59.5) && !model.addChangeAt(0.5));
    CHECK(model.getStretches().size() == 2);

    // Any of the twelve keys can be chosen for a stretch.
    model.choose(1, 14);
    CHECK(model.getStretches()[1].chosen == 2);
    CHECK(model.moveChange(1, 75.0));
    CHECK(std::abs(model.getStretches()[1].startSeconds - 75.0) < 1.0e-9 &&
          model.getStretches()[1].chosen == 2);
    CHECK(model.moveChange(1, 500.0)); // held a second inside the end
    CHECK(model.getStretches()[1].startSeconds < model.getStretches()[1].endSeconds - 0.99);
    CHECK(!model.removeChange(0) && model.removeChange(1));
    CHECK(model.getStretches().size() == 1 && model.getStretches()[0].chosen == 0);
    CHECK(std::abs(model.getStretches()[0].endSeconds - whole.endSeconds) < 1.0e-9);

    // A song that changes key comes as two stretches, each on its own key.
    model.setResult(modulating());
    CHECK(model.getStretches().size() == 2);
    CHECK(model.getStretches()[0].chosen == 8 && model.getStretches()[1].chosen == 9);
    CHECK(model.getStretches()[0].second != 8 && model.getStretches()[1].second != 9);
    // A change added inside the second: its two halves are both on A.
    CHECK(model.addChangeAt(120.0));
    CHECK(model.getStretches().size() == 3);
    CHECK(model.getStretches()[1].first == 9 && model.getStretches()[2].first == 9);
}

// 32, 33. Confirmed: the first stretch's key, always major, and the changes
// of key, as one history step - over whatever the analysis had guessed.
void whatIsWrittenWhenTheKeyIsConfirmed()
{
    Project project;
    fill(project, 12100); // 140 s
    // As the analysis may leave it (ScaleUtils::detectAndApplyScale): minor.
    project.setProjectKey(5, ScaleMode::Minor);
    wizard::Model model;
    model.setResult(modulating());
    PitchUndoManager history;

    const auto before = KeyState::of(project);
    const auto after = model.confirmed(project);
    after.applyTo(project);
    history.addAction(std::make_unique<KeyChangeAction>(project, before, after, "Confirm Key"));
    CHECK(project.getScaleRootNote() == 8 && project.getPreferredScaleMode() == ScaleMode::Major);
    CHECK(project.getKeyChanges().size() == 1);
    const auto change = project.getKeyChanges().front();
    CHECK(change.root == 9);
    const double changeSeconds = static_cast<double>(change.sourceFrame) * HOP_SIZE / SAMPLE_RATE;
    CHECK(std::abs(changeSeconds - model.getStretches()[1].startSeconds) < 0.02);
    CHECK(project.keyRootAt(100) == 8 && project.keyRootAt(change.sourceFrame + 1) == 9);

    history.undo(); // the whole of it
    CHECK(project.getKeyChanges().empty());
    CHECK(project.getScaleRootNote() == 5 && project.getPreferredScaleMode() == ScaleMode::Minor);
    history.redo();
    CHECK(project.getScaleRootNote() == 8 && project.getKeyChanges().size() == 1);

    // The user's choice, not the first candidate, is what is written; two
    // stretches left in one key are one stretch.
    model.choose(0, 1);
    model.choose(1, 1);
    const auto same = model.confirmed(project);
    CHECK(same.projectRoot == 1 && same.changes.empty());
    model.choose(1, 3);
    CHECK(model.confirmed(project).changes.size() == 1 && model.confirmed(project).changes[0].root == 3);

    // A stretched timeline: the change is kept where it was recorded. The
    // second half of the recording plays 10 s late: the stretch that starts
    // at 90 s on the timeline starts at 80 s of the recording.
    Project stretched;
    fill(stretched, 12100);
    auto& map = stretched.getAudioData().warpMap;
    const double from = 80.0 * SAMPLE_RATE / HOP_SIZE, to = 90.0 * SAMPLE_RATE / HOP_SIZE;
    const int anchor = map.addAnchorAtDestination(from);
    CHECK(anchor >= 0);
    map.moveAnchor(anchor, to);
    wizard::Model again;
    again.setResult(modulating());
    const auto moved = again.confirmed(stretched);
    CHECK(moved.changes.size() == 1);
    const double recorded = static_cast<double>(moved.changes[0].sourceFrame) * HOP_SIZE / SAMPLE_RATE;
    std::printf("  a change at %.2f s of the timeline is at %.2f s of the recording\n",
                again.getStretches()[1].startSeconds, recorded);
    CHECK(recorded < again.getStretches()[1].startSeconds - 1.0);
}
} // namespace

// A project confirmed before (decision 乙): its own stretches and keys are
// what the wizard shows chosen, with this analysis's candidates.
void aConfirmedProjectIsAdopted()
{
    wizard::Model model;
    keyfinder::Result found;
    found.stretches.push_back({0.0, 170.0, 8, 1, 3.0});
    found.stretches.push_back({170.0, 230.0, 9, 2, 3.0});
    model.setResult(found);
    // The user had moved the change to 175 s and called the first part C#.
    model.adopt({{0.0, 1}, {175.0, 9}});
    const auto& stretches = model.getStretches();
    CHECK(stretches.size() == 2);
    CHECK(stretches[0].startSeconds == 0.0 && stretches[0].endSeconds == 175.0);
    CHECK(stretches[1].startSeconds == 175.0 && stretches[1].endSeconds == 230.0);
    CHECK(stretches[0].chosen == 1 && stretches[1].chosen == 9);
    CHECK(stretches[0].first == 8 && stretches[0].second == 1); // the candidates are the analysis's
    // One key throughout, where the analysis found two.
    model.setResult(found);
    model.adopt({{0.0, 4}});
    CHECK(model.getStretches().size() == 1 && model.getStretches()[0].chosen == 4);
    CHECK(model.getStretches()[0].endSeconds == 230.0);
    // A change too near the end is left out; nothing given changes nothing.
    model.setResult(found);
    model.adopt({{0.0, 8}, {229.5, 3}});
    CHECK(model.getStretches().size() == 1 && model.getStretches()[0].endSeconds == 230.0);
    model.setResult(found);
    model.adopt({});
    CHECK(model.getStretches().size() == 2 && model.getStretches()[0].chosen == 8);
}

// The mode (2026-11-26): the analysis says major or minor; what is written
// to the project is the mode's tonic, on each stretch.
void theModeNamesTheTonic()
{
    wizard::Model model;
    keyfinder::Result found;
    found.minor = true;
    found.stretches.push_back({0.0, 100.0, 0, 5, 3.0});   // C major's notes
    found.stretches.push_back({100.0, 200.0, 2, 7, 3.0}); // D major's
    model.setResult(found);
    CHECK(model.mode == ScaleMode::Minor && model.tonicOf(0) == 9 && model.majorRootOf(9) == 0);
    Project project;
    auto state = model.confirmed(project);
    CHECK(state.projectRoot == 9 && state.projectMode == ScaleMode::Minor); // A minor
    CHECK(state.changes.size() == 1 && state.changes[0].root == 11);        // then B minor
    model.mode = ScaleMode::Dorian;
    state = model.confirmed(project);
    CHECK(state.projectRoot == 2 && state.projectMode == ScaleMode::Dorian && state.changes[0].root == 4);
    // A project confirmed in A minor then B minor comes back as it was.
    found.minor = false;
    model.setResult(found);
    model.adopt({{0.0, 9}, {100.0, 11}}, ScaleMode::Minor);
    CHECK(model.mode == ScaleMode::Minor);
    CHECK(model.getStretches()[0].chosen == 0 && model.getStretches()[1].chosen == 2);
    for (const auto mode : {ScaleMode::Major, ScaleMode::MajorPentatonic, ScaleMode::WholeTone})
        CHECK(wizard::tonicOffset(mode) == 0);
}

int main()
{
    juce::ScopedJuceInitialiser_GUI juce;
    aConfirmedProjectIsAdopted();
    theModeNamesTheTonic();
    aNewProjectNeedsBothFilesAndTheTick();
    anOpenProjectIsAskedOnce();
    anotherAccompanimentIsAskedAgain();
    theStretchesOfTheKeyCheck();
    whatIsWrittenWhenTheKeyIsConfirmed();
    std::printf("WizardModelTests: all passed\n");
    return 0;
}
