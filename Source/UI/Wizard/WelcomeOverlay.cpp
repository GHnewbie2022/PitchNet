#include "WelcomeOverlay.h"

namespace
{
constexpr int kWidth = 620;
constexpr int kPad = 24;
constexpr int kTitleHeight = 30;
constexpr int kCardsHeight = 104;
constexpr int kLinkHeight = 24;
constexpr int kGap = 12;
} // namespace

WelcomeOverlay::WelcomeOverlay()
{
    setWantsKeyboardFocus(true);
    freeModeCard.onClick = [this]
    {
        if (onFreeMode)
            onFreeMode();
    };
    guidedCard.onClick = [this]
    {
        if (onWizard)
            onWizard();
    };
    openFileButton.onClick = [this]
    {
        if (onOpenFile)
            onOpenFile();
    };
    addAndMakeVisible(freeModeCard);
    addAndMakeVisible(guidedCard);
    addAndMakeVisible(openFileButton);
    refreshLocalisedText();
}

void WelcomeOverlay::refreshLocalisedText()
{
    freeModeCard.setTexts(TR("welcome.free"), TR("welcome.free_detail"));
    guidedCard.setTexts(TR("welcome.wizard"), TR("welcome.wizard_detail"));
    openFileButton.setButtonText(TR("welcome.open"));
    repaint();
}

void WelcomeOverlay::setRecentFiles(const juce::StringArray& paths)
{
    for (auto& button : recentButtons)
        removeChildComponent(button.get());
    recentButtons.clear();
    recentFiles.clear();
    for (const auto& path : paths)
    {
        const juce::File file(path);
        if (!file.existsAsFile())
            continue;
        recentFiles.push_back(file);
        auto button = std::make_unique<wizardui::LinkButton>();
        button->setButtonText(file.getFileName());
        button->setTooltip(file.getFullPathName());
        button->onClick = [this, file]
        {
            if (onOpenRecent)
                onOpenRecent(file);
        };
        addAndMakeVisible(*button);
        recentButtons.push_back(std::move(button));
        if (static_cast<int>(recentButtons.size()) >= kMaxRecent)
            break;
    }
    resized();
    repaint();
}

int WelcomeOverlay::cardHeight() const
{
    int height = kPad + kTitleHeight + kGap + kCardsHeight + kGap + kLinkHeight + kPad;
    if (!recentButtons.empty())
        height += kGap + kLinkHeight * (1 + static_cast<int>(recentButtons.size()));
    return height;
}

void WelcomeOverlay::resized()
{
    card = getLocalBounds().withSizeKeepingCentre(juce::jmin(kWidth, getWidth() - 40), cardHeight());
    auto area = card.reduced(kPad);
    area.removeFromTop(kTitleHeight + kGap);
    auto cards = area.removeFromTop(kCardsHeight);
    const int each = (cards.getWidth() - kGap) / 2;
    freeModeCard.setBounds(cards.removeFromLeft(each));
    cards.removeFromLeft(kGap);
    guidedCard.setBounds(cards);
    area.removeFromTop(kGap);
    openFileButton.setBounds(area.removeFromTop(kLinkHeight));
    if (!recentButtons.empty())
    {
        area.removeFromTop(kGap + kLinkHeight); // the "Recent" heading, painted
        for (auto& button : recentButtons)
            button->setBounds(area.removeFromTop(kLinkHeight).withTrimmedLeft(12));
    }
}

void WelcomeOverlay::paint(juce::Graphics& g)
{
    wizardui::paintCard(g, getLocalBounds(), card);
    auto area = card.reduced(kPad);
    g.setColour(wizardui::kText);
    g.setFont(AppFont::getBoldFont(19.0f));
    g.drawText(TR("welcome.title"), area.removeFromTop(kTitleHeight), juce::Justification::centredLeft, true);
    if (!recentButtons.empty())
    {
        area.removeFromTop(kGap + kCardsHeight + kGap + kLinkHeight + kGap);
        g.setColour(APP_COLOR_TEXT_MUTED);
        g.setFont(AppFont::getFont(12.0f));
        g.drawText(TR("menu.recent_files"), area.removeFromTop(kLinkHeight), juce::Justification::centredLeft,
                   true);
    }
}

bool WelcomeOverlay::keyPressed(const juce::KeyPress& key)
{
    // Escape is the free way: the screen goes, the empty project stays.
    if (key == juce::KeyPress::escapeKey)
    {
        if (onFreeMode)
            onFreeMode();
        return true;
    }
    return false;
}
