#pragma once

#include "WizardWidgets.h"
#include "../../Utils/Localization.h"

#include <functional>
#include <memory>
#include <vector>

/**
 * What the standalone shows when it is opened by itself, with no file
 * (spec section 15): the two ways to begin - free, an empty project the user
 * drops a recording into; guided, the wizard, which asks for the recording
 * and its accompaniment and finds the key - and, under them, a project to
 * open instead. A file opened by any other road (Finder, a drop, the menu)
 * takes the screen away.
 */
class WelcomeOverlay : public juce::Component
{
public:
    static constexpr int kMaxRecent = 5;

    WelcomeOverlay();

    /** The newest first; those that are still there are listed, kMaxRecent at most. */
    void setRecentFiles(const juce::StringArray& paths);

    std::function<void()> onFreeMode;
    std::function<void()> onWizard;
    std::function<void()> onOpenFile;
    std::function<void(const juce::File&)> onOpenRecent;

    void paint(juce::Graphics& g) override;
    void resized() override;
    bool keyPressed(const juce::KeyPress& key) override;

    // For tests and screenshots.
    juce::Button& freeCard() { return freeModeCard; }
    juce::Button& wizardCard() { return guidedCard; }
    juce::Button& openButton() { return openFileButton; }
    int recentCount() const { return static_cast<int>(recentButtons.size()); }
    juce::Rectangle<int> cardBounds() const { return card; }

private:
    void refreshLocalisedText();
    int cardHeight() const;

    wizardui::ChoiceCard freeModeCard, guidedCard;
    wizardui::LinkButton openFileButton;
    std::vector<std::unique_ptr<wizardui::LinkButton>> recentButtons;
    std::vector<juce::File> recentFiles;
    juce::Rectangle<int> card;

    LocalisationWatcher languageWatcher{[this] { refreshLocalisedText(); }};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WelcomeOverlay)
};
