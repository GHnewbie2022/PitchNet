#include "MixerPanel.h"
#include "Components/AppFont.h"
#include "../Utils/UI/Theme.h"

#include <cmath>

namespace
{
constexpr int kPad = 9;          // inside a card
constexpr int kTitleHeight = 20; // the track's name and its level
constexpr int kSliderHeight = 22;
constexpr int kSwitchHeight = 24;
constexpr int kNameHeight = 18;  // the accompaniment's file
constexpr int kButtonHeight = 24;
constexpr int kGap = 4;
constexpr int kCardGap = 10;
constexpr int kSwitchWidth = 40;
constexpr int kSwitchLabelX = 35;
constexpr int kLevelWidth = 54;   // "-60.0 dB"
constexpr float kCorner = 6.0f;
constexpr float kBorder = 1.5f;

const juce::Colour cardBorder{0x4DE6E6E6u};
const juce::Colour titleColour{0xFFE6E6E6u};

int voiceCardHeight() { return kPad + kTitleHeight + kGap + kSliderHeight + kPad; }
int backingCardHeight()
{
    return kPad + kTitleHeight + kGap + kSliderHeight + kGap + kSwitchHeight + kGap + kNameHeight +
           kGap + kButtonHeight + kPad;
}

void setUpFader(juce::Slider& slider, float defaultDb)
{
    slider.setRange(MixerPanel::kMinDb, MixerPanel::kMaxDb, MixerPanel::kStepDb);
    slider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    slider.setDoubleClickReturnValue(true, defaultDb);
    slider.setValue(defaultDb, juce::dontSendNotification);
    slider.setMouseCursor(juce::MouseCursor::PointingHandCursor);
}
} // namespace

MixerPanel::MixerPanel()
{
    setOpaque(false);
    setUpFader(voiceSlider, kVoiceDefaultDb);
    setUpFader(backingSlider, kBackingDefaultDb);
    voiceSlider.onValueChange = [this]
    {
        if (applying)
            return;
        state.voiceDb = static_cast<float>(voiceSlider.getValue());
        repaint();
        if (onVoiceGainChanged)
            onVoiceGainChanged(state.voiceDb);
    };
    backingSlider.onValueChange = [this]
    {
        if (applying)
            return;
        state.backingDb = static_cast<float>(backingSlider.getValue());
        repaint();
        if (onBackingGainChanged)
            onBackingGainChanged(state.backingDb);
    };
    // The label is painted by the panel (it may need the panel's width); the
    // toggle draws the switch alone, as in the track list.
    audibleToggle.setClickingTogglesState(true);
    audibleToggle.onClick = [this]
    {
        if (applying)
            return;
        state.backingAudible = audibleToggle.getToggleState();
        if (onBackingAudibleChanged)
            onBackingAudibleChanged(state.backingAudible);
    };
    chooseBackingButton.onClick = [this]
    {
        if (onChooseBacking)
            onChooseBacking();
    };
    removeBackingButton.onClick = [this]
    {
        if (onRemoveBacking)
            onRemoveBacking();
    };
    addAndMakeVisible(voiceSlider);
    addAndMakeVisible(backingSlider);
    addAndMakeVisible(audibleToggle);
    addAndMakeVisible(chooseBackingButton);
    addAndMakeVisible(removeBackingButton);
    wizardButton.onClick = [this]
    {
        if (onOpenWizard)
            onOpenWizard();
    };
    addAndMakeVisible(wizardButton);
    refreshLocalisedText();
    applyState();
}

MixerPanel::~MixerPanel() = default;

juce::String MixerPanel::levelText(float dB)
{
    if (dB <= static_cast<float>(kMinDb))
        return juce::String::fromUTF8("-\xe2\x88\x9e dB");
    return juce::String(dB, 1) + " dB";
}

void MixerPanel::refreshLocalisedText()
{
    audibleToggle.setTitle(TR("mixer.play_along"));
    removeBackingButton.setButtonText(TR("mixer.remove"));
    wizardButton.setButtonText(TR("mixer.wizard"));
    chooseBackingButton.setButtonText(state.hasBacking ? TR("mixer.change") : TR("mixer.choose_backing"));
    repaint();
}

void MixerPanel::setState(const State& newState)
{
    state = newState;
    applyState();
}

void MixerPanel::applyState()
{
    applying = true;
    voiceSlider.setValue(state.voiceDb, juce::dontSendNotification);
    backingSlider.setValue(state.backingDb, juce::dontSendNotification);
    audibleToggle.setToggleState(state.backingAudible, juce::dontSendNotification);
    applying = false;
    const bool playable = state.hasProject && state.hasBacking && !state.backingMissing &&
                          !state.backingLoading;
    voiceSlider.setEnabled(state.hasProject);
    backingSlider.setEnabled(playable);
    audibleToggle.setEnabled(playable);
    chooseBackingButton.setEnabled(state.hasProject && !state.backingLoading);
    removeBackingButton.setVisible(state.hasBacking);
    removeBackingButton.setEnabled(state.hasProject && !state.backingLoading);
    chooseBackingButton.setButtonText(state.hasBacking ? TR("mixer.change") : TR("mixer.choose_backing"));
    resized();
    repaint();
}

juce::Rectangle<int> MixerPanel::voiceCard() const { return {0, 0, getWidth(), voiceCardHeight()}; }

juce::Rectangle<int> MixerPanel::backingCard() const
{
    return {0, voiceCardHeight() + kCardGap, getWidth(), backingCardHeight()};
}

int MixerPanel::getPreferredHeight() const
{
    return voiceCardHeight() + kCardGap + backingCardHeight() + kCardGap + kButtonHeight;
}

juce::Rectangle<int> MixerPanel::switchLabelBounds() const
{
    auto card = backingCard().reduced(kPad);
    card.removeFromTop(kTitleHeight + kGap + kSliderHeight + kGap);
    return card.removeFromTop(kSwitchHeight).withTrimmedLeft(kSwitchLabelX);
}

void MixerPanel::resized()
{
    auto voice = voiceCard().reduced(kPad);
    voice.removeFromTop(kTitleHeight + kGap);
    voiceSlider.setBounds(voice.removeFromTop(kSliderHeight));

    auto backing = backingCard().reduced(kPad);
    backing.removeFromTop(kTitleHeight + kGap);
    backingSlider.setBounds(backing.removeFromTop(kSliderHeight));
    backing.removeFromTop(kGap);
    // Nudged 3px left: the switch is drawn 4px inside the toggle's bounds.
    audibleToggle.setBounds(backing.removeFromTop(kSwitchHeight)
                                .withWidth(kSwitchWidth)
                                .translated(-3, 0));
    backing.removeFromTop(kGap + kNameHeight + kGap);
    auto buttons = backing.removeFromTop(kButtonHeight);
    if (state.hasBacking)
    {
        const int each = (buttons.getWidth() - kGap) / 2;
        chooseBackingButton.setBounds(buttons.removeFromLeft(each));
        buttons.removeFromLeft(kGap);
        removeBackingButton.setBounds(buttons);
    }
    else
    {
        chooseBackingButton.setBounds(buttons);
    }
    wizardButton.setBounds(0, backingCard().getBottom() + kCardGap, getWidth(), kButtonHeight);
}

void MixerPanel::mouseUp(const juce::MouseEvent& e)
{
    // The label flips the switch too.
    if (!e.mouseWasDraggedSinceMouseDown() && audibleToggle.isEnabled() &&
        switchLabelBounds().contains(e.getPosition()))
        audibleToggle.triggerClick();
}

void MixerPanel::paint(juce::Graphics& g)
{
    const auto drawCard = [&](juce::Rectangle<int> card, const juce::String& title, float dB, bool live)
    {
        g.setColour(cardBorder);
        g.drawRoundedRectangle(card.toFloat().reduced(kBorder * 0.5f), kCorner, kBorder);
        auto row = card.reduced(kPad).removeFromTop(kTitleHeight);
        g.setFont(AppFont::getFont(13.0f));
        // The level has its own room on the right; the name takes the rest
        // (the panel is narrow, and "Accompaniment" is long in some languages).
        const auto level = row.removeFromRight(kLevelWidth);
        g.setColour(APP_COLOR_TEXT_MUTED);
        g.drawText(levelText(dB), level, juce::Justification::centredRight, false);
        g.setColour(live ? titleColour : APP_COLOR_TEXT_MUTED);
        g.drawFittedText(title, row, juce::Justification::centredLeft, 1, 0.75f);
    };
    const bool playable = state.hasProject && state.hasBacking && !state.backingMissing &&
                          !state.backingLoading;
    drawCard(voiceCard(), TR("mixer.voice"), state.voiceDb, state.hasProject);
    drawCard(backingCard(), TR("mixer.backing"), state.backingDb, playable);

    g.setFont(AppFont::getFont(13.0f));
    g.setColour(playable ? APP_COLOR_TEXT_PRIMARY : APP_COLOR_TEXT_MUTED);
    g.drawFittedText(TR("mixer.play_along"), switchLabelBounds(), juce::Justification::centredLeft, 1,
                     1.0f);

    // The file: its name, or what is the matter with it.
    auto nameRow = backingCard().reduced(kPad);
    nameRow.removeFromTop(kTitleHeight + kGap + kSliderHeight + kGap + kSwitchHeight + kGap);
    nameRow = nameRow.removeFromTop(kNameHeight);
    juce::String text = state.backingName;
    juce::Colour colour = APP_COLOR_TEXT_MUTED;
    if (!state.hasBacking)
        text = TR("mixer.no_backing");
    else if (state.backingLoading)
        text = TR("mixer.loading");
    else if (state.backingMissing)
    {
        text = TR("mixer.missing") + juce::String::fromUTF8("\xef\xbc\x9a") + state.backingName;
        colour = APP_COLOR_ALERT_WARNING;
    }
    g.setFont(AppFont::getFont(12.0f));
    g.setColour(colour);
    g.drawText(text, nameRow, juce::Justification::centredLeft, true);
}
