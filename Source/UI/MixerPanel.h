#pragma once

#include "../JuceHeader.h"
#include "Components/StyledWidgets.h"
#include "Workspace/PanelContent.h"
#include "../Utils/Localization.h"

#include <functional>

/**
 * Content of the left side panel in the standalone (spec section 15): the
 * voice and the accompaniment as two tracks, each with its level; the
 * accompaniment with the switch that plays it along, the file it is, and
 * the buttons to choose another or take it away.
 *
 * (In the plug-in the left panel lists the host's tracks, TrackListPanel:
 * the host plays the accompaniment there.)
 *
 * The panel holds no project: it shows the State it is given and says what
 * the user did. The window puts that on the project and the playback engine.
 */
class MixerPanel : public juce::Component, public PanelContent
{
public:
    struct State
    {
        bool hasProject = false; // nothing can be set without one
        float voiceDb = 0.0f;
        bool hasBacking = false; // the project links one
        juce::String backingName;
        bool backingMissing = false; // linked, but the file could not be read
        bool backingLoading = false;
        bool backingAudible = false;
        float backingDb = -6.0f;
    };

    static constexpr double kMinDb = -60.0, kMaxDb = 12.0, kStepDb = 0.5;
    static constexpr float kVoiceDefaultDb = 0.0f, kBackingDefaultDb = -6.0f;

    MixerPanel();
    ~MixerPanel() override;

    void setState(const State& newState);
    const State& getState() const { return state; }

    int getPreferredHeight() const override;
    void paint(juce::Graphics& g) override;
    void resized() override;

    std::function<void(float)> onVoiceGainChanged;   // dB, while dragging too
    std::function<void(float)> onBackingGainChanged; // dB
    std::function<void(bool)> onBackingAudibleChanged;
    std::function<void()> onChooseBacking;
    std::function<void()> onRemoveBacking;
    std::function<void()> onOpenWizard; // "Key / auto-tune…", under the two tracks

    /** "-6.0 dB", or "-∞ dB" at the fader's bottom. */
    static juce::String levelText(float dB);

    // For tests and screenshots: the parts by name.
    juce::Slider& voiceFader() { return voiceSlider; }
    juce::Slider& backingFader() { return backingSlider; }
    juce::Button& playAlongSwitch() { return audibleToggle; }
    juce::Button& chooseButton() { return chooseBackingButton; }
    juce::Button& removeButton() { return removeBackingButton; }
    juce::Button& wizardEntry() { return wizardButton; }

private:
    juce::Rectangle<int> voiceCard() const;
    juce::Rectangle<int> backingCard() const;
    juce::Rectangle<int> switchLabelBounds() const;
    void mouseUp(const juce::MouseEvent& e) override;
    void refreshLocalisedText();
    void applyState();

    State state;
    MacroSlider voiceSlider, backingSlider;
    StyledToggleButton audibleToggle;
    CompactSelectionButton chooseBackingButton, removeBackingButton, wizardButton;
    bool applying = false;

    LocalisationWatcher languageWatcher{[this] { refreshLocalisedText(); }};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MixerPanel)
};
