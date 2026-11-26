// The welcome screen and the wizard's pages (spec section 15): they show the
// model they are given and say what the user did - through their own
// buttons, wired to the model as the App's window wires them
// (MainComponent_Wizard.cpp).
#include "../Source/UI/MixerPanel.h"
#include "../Source/UI/Wizard/WelcomeOverlay.h"
#include "../Source/UI/Wizard/WizardOverlay.h"
#include "TestAssert.h"

#include <cstdio>
#include <vector>

namespace
{
void pump() { juce::MessageManager::getInstance()->runDispatchLoopUntil(50); }

juce::File tempFile(const juce::String& name)
{
    const auto file = juce::File::getSpecialLocation(juce::File::tempDirectory)
                          .getChildFile("pitchnet_wizard_tests")
                          .getChildFile(name);
    file.create();
    return file;
}

bool chosen(juce::Button* button)
{
    auto* key = dynamic_cast<wizardui::KeyButton*>(button);
    return key != nullptr && key->isChosen();
}

void theWelcomeScreen()
{
    WelcomeOverlay welcome;
    welcome.setSize(1280, 800);
    int freeWay = 0, guided = 0, open = 0;
    std::vector<juce::File> recent;
    welcome.onFreeMode = [&] { ++freeWay; };
    welcome.onWizard = [&] { ++guided; };
    welcome.onOpenFile = [&] { ++open; };
    welcome.onOpenRecent = [&](const juce::File& file) { recent.push_back(file); };

    welcome.freeCard().triggerClick();
    welcome.wizardCard().triggerClick();
    welcome.openButton().triggerClick();
    pump();
    CHECK(freeWay == 1 && guided == 1 && open == 1);
    // Escape is the free way.
    CHECK(welcome.keyPressed(juce::KeyPress(juce::KeyPress::escapeKey)));
    CHECK(freeWay == 2);
    CHECK(!welcome.keyPressed(juce::KeyPress('z', juce::ModifierKeys::commandModifier, 0)));

    // Recent files: those that are still there, five at most, newest first.
    juce::StringArray paths;
    paths.add(tempFile("gone.pitchnet").getFullPathName());
    tempFile("gone.pitchnet").deleteFile();
    for (int i = 0; i < 7; ++i)
        paths.add(tempFile("song" + juce::String(i) + ".pitchnet").getFullPathName());
    welcome.setRecentFiles(paths);
    CHECK(welcome.recentCount() == WelcomeOverlay::kMaxRecent);
    CHECK(welcome.cardBounds().getBottom() <= 800 && welcome.cardBounds().getY() >= 0);
    welcome.setRecentFiles({});
    CHECK(welcome.recentCount() == 0);
}

// The files page of a new project: both files and the tick, then on.
void theFilesOfANewProject()
{
    auto model = wizard::Model::fresh();
    WizardOverlay overlay(model);
    overlay.setSize(1280, 800);
    overlay.setVisible(true);
    int started = 0, cancelled = 0, voiceAsked = 0, backingAsked = 0;
    overlay.onStart = [&] { ++started; };
    overlay.onCancel = [&] { ++cancelled; };
    overlay.onChooseVoice = [&] { ++voiceAsked; };
    overlay.onChooseBacking = [&] { ++backingAsked; };
    // As the window does.
    overlay.onFileDropped = [&](const juce::File& file, bool forVoice)
    {
        if (forVoice)
            model.setVoice(file);
        else
            model.setBacking(file);
        overlay.refresh();
    };
    overlay.onAligned = [&](bool on)
    {
        model.aligned = on && model.hasBacking();
        overlay.refresh();
    };
    overlay.refresh();

    CHECK(overlay.chooseVoiceButton().isVisible() && overlay.chooseBackingButton().isVisible());
    CHECK(!overlay.startButton().isEnabled());
    CHECK(!overlay.alignedSwitch().isEnabled()); // nothing to be aligned yet
    overlay.chooseVoiceButton().triggerClick();
    overlay.chooseBackingButton().triggerClick();
    pump();
    CHECK(voiceAsked == 1 && backingAsked == 1);

    // Dropped on the accompaniment's row: the accompaniment, though the
    // voice's row is the first one empty.
    const auto voice = tempFile("voice.wav"), backing = tempFile("backing.wav"),
               other = tempFile("other.wav");
    CHECK(overlay.takeDroppedFile(backing, overlay.rowBounds(false).getCentre()));
    CHECK(model.backing == backing && model.voice == juce::File{});
    // Dropped beside the rows: the first one empty.
    CHECK(overlay.takeDroppedFile(voice, overlay.cardBounds().getBottomLeft().translated(8, -8)));
    CHECK(model.voice == voice);
    CHECK(!overlay.startButton().isEnabled()); // the tick is missing
    CHECK(overlay.alignedSwitch().isEnabled());

    overlay.alignedSwitch().triggerClick();
    pump();
    CHECK(model.aligned && overlay.startButton().isEnabled());
    overlay.startButton().triggerClick();
    pump();
    CHECK(started == 1);

    // Another accompaniment has not been said to be aligned.
    CHECK(overlay.takeDroppedFile(other, overlay.rowBounds(false).getCentre()));
    CHECK(model.backing == other && !model.aligned);
    CHECK(!overlay.startButton().isEnabled() && !overlay.alignedSwitch().getToggleState());

    // The keyboard: escape cancels; the menu's shortcuts are let through;
    // nothing else reaches what is under the overlay.
    CHECK(overlay.keyPressed(juce::KeyPress(juce::KeyPress::escapeKey)) && cancelled == 1);
    CHECK(!overlay.keyPressed(juce::KeyPress('q', juce::ModifierKeys::commandModifier, 0)));
    CHECK(overlay.keyPressed(juce::KeyPress('a')));
    overlay.cancelButton().triggerClick();
    pump();
    CHECK(cancelled == 2);

    // Not on this page: a drop is the window's (it opens the file).
    model.page = wizard::Page::analysing;
    overlay.refresh();
    CHECK(!overlay.takeDroppedFile(voice, overlay.cardBounds().getCentre()));
    CHECK(!overlay.startButton().isVisible() && !overlay.keyPage().isVisible());
    CHECK(overlay.keyPressed(juce::KeyPress(juce::KeyPress::escapeKey)) && cancelled == 2); // it waits
}

// An open project: its recording is the voice, an accompaniment is optional.
void theFilesOfAnOpenProject()
{
    wizard::Model model;
    model.existingProject = true;
    WizardOverlay overlay(model);
    overlay.setSize(1280, 800);
    overlay.setVisible(true);
    overlay.onFileDropped = [&](const juce::File& file, bool forVoice)
    {
        CHECK(!forVoice); // there is no voice to choose
        model.setBacking(file);
        overlay.refresh();
    };
    overlay.refresh();
    CHECK(!overlay.chooseVoiceButton().isVisible());
    CHECK(overlay.startButton().isEnabled()); // by the voice alone
    CHECK(overlay.takeDroppedFile(tempFile("backing.wav"), overlay.rowBounds(true).getCentre()));
    CHECK(model.hasBacking() && !overlay.startButton().isEnabled()); // now the tick is asked
}

// The key check: a key pressed is said, with its stretch; the answer shown
// is the model's.
void theKeyCheck()
{
    wizard::Model model;
    model.existingProject = true;
    keyfinder::Result found;
    found.hasBacking = true;
    found.stretches.push_back({0.0, 100.0, 8, 1, 3.0});
    found.stretches.push_back({100.0, 200.0, 9, 2, 3.0});
    model.setResult(found);
    model.page = wizard::Page::keyCheck;

    WizardOverlay overlay(model);
    overlay.setSize(1280, 800);
    overlay.setVisible(true);
    auto& keys = overlay.keyPage();
    std::vector<std::pair<int, int>> pressed;
    std::vector<int> removed;
    int confirmed = 0, back = 0, playStop = 0, added = 0, cancelled = 0;
    // As the window does.
    keys.onKey = [&](int stretch, int root)
    {
        pressed.push_back({stretch, root});
        model.choose(stretch, root);
        overlay.refresh();
    };
    std::vector<ScaleMode> modes;
    keys.onMode = [&](ScaleMode mode)
    {
        modes.push_back(mode);
        model.mode = mode;
        overlay.refresh();
    };
    keys.onRemoveChange = [&](int stretch)
    {
        removed.push_back(stretch);
        if (model.removeChange(stretch))
            overlay.refresh();
    };
    keys.onConfirm = [&] { ++confirmed; };
    keys.onBack = [&] { ++back; };
    keys.onPlayStop = [&] { ++playStop; };
    keys.onAddChange = [&] { ++added; };
    overlay.onCancel = [&] { ++cancelled; };
    overlay.refresh();

    CHECK(keys.isVisible() && keys.rowCount() == 2);
    CHECK(overlay.cardBounds().getHeight() <= 800);
    // The likeliest key of each stretch is the answer until another is pressed.
    CHECK(chosen(keys.keyButton(0, 8, false)) && !chosen(keys.keyButton(0, 1, false)));
    CHECK(chosen(keys.keyButton(1, 9, false)) && keys.tonicList(1)->getSelectedId() == 9 + 1);
    // The two are named by tonic and mode; the lists hold every other key.
    CHECK(keys.keyButton(0, 8, false)->getTitle() == juce::String("G# ") + wizardui::modeName(ScaleMode::Major) && keys.keyButton(0, 8, true) == nullptr);
    CHECK(keys.tonicList(0)->getNumItems() == 12 && keys.modeList(0)->getNumItems() == 14);
    CHECK(keys.modeList(0)->getText() == wizardui::modeName(ScaleMode::Major));
    // Only a change of key can be taken away: not the first stretch.
    CHECK(!keys.removeButton(0)->isVisible() && keys.removeButton(1)->isVisible());

    // The second of the two.
    keys.keyButton(1, 2, false)->triggerClick();
    pump();
    CHECK(pressed.size() == 1 && pressed[0] == std::make_pair(1, 2));
    CHECK(chosen(keys.keyButton(1, 2, false)) && keys.tonicList(1)->getSelectedId() == 2 + 1);
    CHECK(!chosen(keys.keyButton(1, 9, false)));
    CHECK(chosen(keys.keyButton(0, 8, false))); // the other stretch keeps its own
    // One of the twelve that is neither.
    keys.tonicList(0)->setSelectedId(5 + 1, juce::sendNotificationSync); // F, from the list
    pump();
    CHECK(pressed.size() == 2 && pressed[1] == std::make_pair(0, 5));
    CHECK(model.getStretches()[0].chosen == 5 && keys.tonicList(0)->getSelectedId() == 5 + 1);

    // The mode from the list: the same seven notes from another tonic. F
    // major's are D minor's; the other stretch's D major's are B minor's.
    modes.clear();
    keys.modeList(1)->setSelectedId(2, juce::sendNotificationSync); // Minor
    pump();
    CHECK(modes.size() == 1 && modes[0] == ScaleMode::Minor && model.mode == ScaleMode::Minor);
    CHECK(model.getStretches()[0].chosen == 5 && model.tonicOf(5) == 2);
    CHECK(keys.tonicList(0)->getSelectedId() == 2 + 1 && keys.tonicList(1)->getSelectedId() == 11 + 1);
    CHECK(keys.keyButton(1, 2, false)->getTitle() == juce::String("B ") + wizardui::modeName(ScaleMode::Minor));
    CHECK(keys.modeList(0)->getText() == wizardui::modeName(ScaleMode::Minor));
    // A tonic chosen in a minor song is that minor key: A minor is C major's notes.
    keys.tonicList(0)->setSelectedId(9 + 1, juce::sendNotificationSync);
    pump();
    CHECK(model.getStretches()[0].chosen == 0 && model.tonicOf(0) == 9);
    keys.modeList(0)->setSelectedId(1, juce::sendNotificationSync); // Major again
    keys.tonicList(0)->setSelectedId(5 + 1, juce::sendNotificationSync);
    pump();
    CHECK(model.mode == ScaleMode::Major && model.getStretches()[0].chosen == 5);
    CHECK(!chosen(keys.keyButton(0, 8, false)) && !chosen(keys.keyButton(0, 1, false)));

    // The transport, the footer, the keyboard.
    keys.playButton().triggerClick();
    keys.addChangeButton().triggerClick();
    pump();
    CHECK(playStop == 1 && added == 1);
    CHECK(overlay.keyPressed(juce::KeyPress(juce::KeyPress::spaceKey)) && playStop == 2);
    CHECK(overlay.keyPressed(juce::KeyPress(juce::KeyPress::escapeKey)) && cancelled == 1);

    // The change of key taken away: one stretch, in the first one's key. The
    // row goes after the click has returned.
    keys.removeButton(1)->triggerClick();
    pump();
    CHECK(removed.size() == 1 && removed[0] == 1);
    CHECK(keys.rowCount() == 1 && model.getStretches().size() == 1);
    CHECK(model.getStretches()[0].endSeconds == 200.0 && model.getStretches()[0].chosen == 5);

    keys.backButton().triggerClick();
    keys.confirmButton().triggerClick();
    pump();
    CHECK(back == 1 && confirmed == 1);
}

// After the key: by hand, or - not on offer yet - by the wizard.
void theNextStep()
{
    wizard::Model model;
    model.existingProject = true;
    model.page = wizard::Page::nextStep;
    WizardOverlay overlay(model);
    overlay.setSize(1280, 800);
    overlay.setVisible(true);
    int byHand = 0;
    overlay.onManual = [&] { ++byHand; };
    overlay.refresh();
    CHECK(overlay.manualCard().isVisible() && overlay.manualCard().isEnabled());
    CHECK(overlay.autoCard().isVisible() && !overlay.autoCard().isEnabled());
    CHECK(!overlay.startButton().isVisible() && !overlay.keyPage().isVisible());
    overlay.manualCard().triggerClick();
    pump();
    CHECK(byHand == 1);
}

// The wizard's door on the left panel: open with or without a project.
void theDoorOnTheLeftPanel()
{
    MixerPanel panel;
    panel.setSize(141, panel.getPreferredHeight());
    int opened = 0;
    panel.onOpenWizard = [&] { ++opened; };
    CHECK(panel.wizardEntry().isVisible() && panel.wizardEntry().isEnabled());
    CHECK(panel.wizardEntry().getBottom() <= panel.getPreferredHeight());
    CHECK(panel.wizardEntry().getY() >= panel.chooseButton().getBottom());
    panel.wizardEntry().triggerClick();
    pump();
    CHECK(opened == 1);
}
} // namespace

int main()
{
    juce::ScopedJuceInitialiser_GUI juce;
    theWelcomeScreen();
    theFilesOfANewProject();
    theFilesOfAnOpenProject();
    theKeyCheck();
    theNextStep();
    theDoorOnTheLeftPanel();
    juce::File::getSpecialLocation(juce::File::tempDirectory)
        .getChildFile("pitchnet_wizard_tests")
        .deleteRecursively();
    std::printf("WizardOverlayTests: all passed\n");
    return 0;
}
