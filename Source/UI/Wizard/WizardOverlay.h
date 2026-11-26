#pragma once

#include "KeyCheckPage.h"
#include "WizardModel.h"
#include "WizardWidgets.h"
#include "../Components/StyledWidgets.h"
#include "../../Utils/Localization.h"

#include <functional>

/**
 * The wizard's window (spec section 15): an overlay in the main window, a
 * page for each step of wizard::Model.
 *
 *   files      the recording and its accompaniment, each with a button to
 *              choose it (or dropped on its row), and the tick that the two
 *              start at the same place;
 *   analysing  a line to wait by, while the accompaniment is read;
 *   keyCheck   KeyCheckPage;
 *   nextStep   tune by hand, or let the wizard tune (not on offer yet).
 *
 * It shows the model and says what was pressed; the window does the rest.
 */
class WizardOverlay : public juce::Component, private juce::Timer
{
public:
    explicit WizardOverlay(wizard::Model& modelToShow);
    ~WizardOverlay() override;

    /** After the model changed. */
    void refresh();
    /** An open project's recording, named on the files page. */
    void setVoiceName(const juce::String& name) { voiceName = name; repaint(); }
    /** A line under the files, when something went wrong. */
    void setMessage(const juce::String& text) { message = text; repaint(); }

    KeyCheckPage& keyPage() { return keyCheck; }

    // The files page.
    std::function<void()> onChooseVoice, onChooseBacking, onStart, onCancel;
    std::function<void(bool)> onAligned;
    // The last page.
    std::function<void()> onManual;
    // Asked 20 times a second while the key check shows.
    std::function<void()> onTick;

    /** A file dropped on the files page: true when it was taken (the row
     *  under `position`, or the first one empty). */
    bool takeDroppedFile(const juce::File& file, juce::Point<int> position);
    std::function<void(const juce::File&, bool forVoice)> onFileDropped;

    void paint(juce::Graphics& g) override;
    void resized() override;
    bool keyPressed(const juce::KeyPress& key) override;
    void visibilityChanged() override;

    // For tests and screenshots.
    juce::Button& chooseVoiceButton() { return chooseVoice; }
    juce::Button& chooseBackingButton() { return chooseBacking; }
    juce::Button& alignedSwitch() { return alignedToggle; }
    juce::Button& startButton() { return start; }
    juce::Button& cancelButton() { return cancel; }
    juce::Button& manualCard() { return manual; }
    juce::Button& autoCard() { return automatic; }
    juce::Rectangle<int> cardBounds() const { return card; }
    juce::Rectangle<int> rowBounds(bool ofTheVoice) const { return ofTheVoice ? voiceRow() : backingRow(); }

private:
    void timerCallback() override;
    void refreshLocalisedText();
    int cardHeight() const;
    juce::Rectangle<int> voiceRow() const;
    juce::Rectangle<int> backingRow() const;

    wizard::Model& model;
    juce::String voiceName, message;
    juce::Rectangle<int> card;

    wizardui::PillButton chooseVoice, chooseBacking, cancel, start{true};
    StyledToggleButton alignedToggle;
    KeyCheckPage keyCheck;
    wizardui::ChoiceCard manual, automatic;

    LocalisationWatcher languageWatcher{[this] { refreshLocalisedText(); }};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WizardOverlay)
};
