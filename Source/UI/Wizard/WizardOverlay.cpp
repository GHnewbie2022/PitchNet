#include "WizardOverlay.h"

namespace
{
constexpr int kWidth = 660;
constexpr int kPad = 24;
constexpr int kTitleHeight = 28;
constexpr int kHintHeight = 22;
constexpr int kRowHeight = 38;
constexpr int kTickHeight = 28;
constexpr int kMessageHeight = 22;
constexpr int kFooterHeight = 34;
constexpr int kGap = 12;
constexpr int kRowLabelWidth = 124; // "Accompaniment"
constexpr int kChooseWidth = 92;
constexpr int kCardsHeight = 104;
constexpr int kWaitHeight = 90;
} // namespace

WizardOverlay::WizardOverlay(wizard::Model& modelToShow) : model(modelToShow)
{
    setWantsKeyboardFocus(true);
    chooseVoice.onClick = [this]
    {
        if (onChooseVoice)
            onChooseVoice();
    };
    chooseBacking.onClick = [this]
    {
        if (onChooseBacking)
            onChooseBacking();
    };
    cancel.onClick = [this]
    {
        if (onCancel)
            onCancel();
    };
    start.onClick = [this]
    {
        if (onStart)
            onStart();
    };
    alignedToggle.setClickingTogglesState(true);
    alignedToggle.onClick = [this]
    {
        if (onAligned)
            onAligned(alignedToggle.getToggleState());
    };
    manual.onClick = [this]
    {
        if (onManual)
            onManual();
    };
    automatic.setEnabled(false); // the rules are not written yet
    keyCheck.setModel(&model);
    for (auto* child : std::initializer_list<juce::Component*>{&chooseVoice, &chooseBacking, &cancel, &start,
                                                               &alignedToggle, &keyCheck, &manual, &automatic})
        addChildComponent(child);
    refreshLocalisedText();
}

WizardOverlay::~WizardOverlay() { stopTimer(); }

void WizardOverlay::refreshLocalisedText()
{
    chooseVoice.setButtonText(TR("wizard.choose"));
    chooseBacking.setButtonText(TR("wizard.choose"));
    cancel.setButtonText(TR("dialog.cancel"));
    start.setButtonText(TR("wizard.start"));
    alignedToggle.setTitle(TR("wizard.aligned"));
    manual.setTexts(TR("wizard.manual"), TR("wizard.manual_detail"));
    automatic.setTexts(TR("wizard.auto"), TR("wizard.auto_detail"));
    refresh();
}

int WizardOverlay::cardHeight() const
{
    switch (model.page)
    {
    case wizard::Page::files:
        return kPad + kTitleHeight + kHintHeight + kGap + kRowHeight * 2 + kGap + kTickHeight +
               kMessageHeight + kGap + kFooterHeight + kPad;
    case wizard::Page::analysing:
        return kPad + kWaitHeight + kPad;
    case wizard::Page::keyCheck:
        return kPad + keyCheck.getPreferredHeight() + kPad;
    case wizard::Page::nextStep:
        return kPad + kTitleHeight + kGap + kCardsHeight + kPad;
    }
    return 200;
}

juce::Rectangle<int> WizardOverlay::voiceRow() const
{
    auto area = card.reduced(kPad);
    area.removeFromTop(kTitleHeight + kHintHeight + kGap);
    return area.removeFromTop(kRowHeight);
}

juce::Rectangle<int> WizardOverlay::backingRow() const
{
    return voiceRow().translated(0, kRowHeight);
}

void WizardOverlay::refresh()
{
    const auto page = model.page;
    const bool files = page == wizard::Page::files;
    // An open project's recording is the project's: nothing to choose.
    chooseVoice.setVisible(files && !model.existingProject);
    chooseBacking.setVisible(files);
    alignedToggle.setVisible(files);
    alignedToggle.setEnabled(model.hasBacking());
    alignedToggle.setToggleState(model.aligned, juce::dontSendNotification);
    cancel.setVisible(files);
    start.setVisible(files);
    start.setEnabled(model.canAnalyse());
    keyCheck.setVisible(page == wizard::Page::keyCheck);
    manual.setVisible(page == wizard::Page::nextStep);
    automatic.setVisible(page == wizard::Page::nextStep);
    if (page == wizard::Page::keyCheck)
        keyCheck.refresh();
    visibilityChanged();
    resized();
    repaint();
}

void WizardOverlay::visibilityChanged()
{
    if (isShowing() && model.page == wizard::Page::keyCheck)
        startTimerHz(20);
    else
        stopTimer();
}

void WizardOverlay::timerCallback()
{
    if (onTick)
        onTick();
}

void WizardOverlay::resized()
{
    card = getLocalBounds().withSizeKeepingCentre(juce::jmin(kWidth, getWidth() - 40),
                                                  juce::jmin(cardHeight(), getHeight() - 24));
    auto area = card.reduced(kPad);
    switch (model.page)
    {
    case wizard::Page::files:
    {
        chooseVoice.setBounds(voiceRow().removeFromRight(kChooseWidth).reduced(0, 4));
        chooseBacking.setBounds(backingRow().removeFromRight(kChooseWidth).reduced(0, 4));
        area.removeFromTop(kTitleHeight + kHintHeight + kGap + kRowHeight * 2 + kGap);
        alignedToggle.setBounds(area.removeFromTop(kTickHeight).removeFromLeft(40));
        auto footer = card.reduced(kPad).removeFromBottom(kFooterHeight);
        start.setBounds(footer.removeFromRight(130));
        footer.removeFromRight(10);
        cancel.setBounds(footer.removeFromRight(96));
        break;
    }
    case wizard::Page::keyCheck:
        keyCheck.setBounds(area);
        break;
    case wizard::Page::nextStep:
    {
        area.removeFromTop(kTitleHeight + kGap);
        auto cards = area.removeFromTop(kCardsHeight);
        const int each = (cards.getWidth() - kGap) / 2;
        manual.setBounds(cards.removeFromLeft(each));
        cards.removeFromLeft(kGap);
        automatic.setBounds(cards);
        break;
    }
    case wizard::Page::analysing:
        break;
    }
}

void WizardOverlay::paint(juce::Graphics& g)
{
    wizardui::paintCard(g, getLocalBounds(), card);
    auto area = card.reduced(kPad);
    const auto title = [&](const juce::String& text)
    {
        g.setColour(wizardui::kText);
        g.setFont(AppFont::getBoldFont(17.0f));
        g.drawText(text, area.removeFromTop(kTitleHeight), juce::Justification::centredLeft, true);
    };
    switch (model.page)
    {
    case wizard::Page::files:
    {
        title(TR("wizard.files_title"));
        g.setColour(APP_COLOR_TEXT_MUTED);
        g.setFont(AppFont::getFont(12.5f));
        g.drawFittedText(model.existingProject ? TR("wizard.files_hint_project") : TR("wizard.files_hint"),
                         area.removeFromTop(kHintHeight), juce::Justification::centredLeft, 1, 0.85f);
        const auto row = [&](juce::Rectangle<int> bounds, const juce::String& label, const juce::String& name,
                             const juce::String& empty, bool chooser)
        {
            g.setColour(wizardui::kText);
            g.setFont(AppFont::getFont(14.0f));
            g.drawFittedText(label, bounds.removeFromLeft(kRowLabelWidth).withTrimmedRight(8),
                             juce::Justification::centredLeft, 1, 0.8f);
            if (chooser)
                bounds.removeFromRight(kChooseWidth + 10);
            const auto well = bounds.reduced(0, 4);
            g.setColour(wizardui::kInset);
            g.fillRoundedRectangle(well.toFloat(), 6.0f);
            g.setColour(name.isEmpty() ? APP_COLOR_TEXT_MUTED : wizardui::kText);
            g.setFont(AppFont::getFont(13.0f));
            g.drawText(name.isEmpty() ? empty : name, well.reduced(10, 0), juce::Justification::centredLeft, true);
        };
        row(voiceRow(), TR("wizard.voice"),
            model.existingProject ? voiceName : model.voice.getFileName(), TR("wizard.drop_here"),
            !model.existingProject);
        row(backingRow(), TR("wizard.backing"), model.backing.getFileName(),
            model.existingProject ? TR("wizard.no_backing") : TR("wizard.drop_here"), true);
        area.removeFromTop(kGap + kRowHeight * 2 + kGap);
        auto tick = area.removeFromTop(kTickHeight);
        tick.removeFromLeft(44);
        g.setColour(model.hasBacking() ? wizardui::kText : APP_COLOR_TEXT_MUTED);
        g.setFont(AppFont::getFont(13.5f));
        g.drawFittedText(TR("wizard.aligned"), tick, juce::Justification::centredLeft, 1, 0.85f);
        g.setColour(APP_COLOR_ALERT_WARNING);
        g.setFont(AppFont::getFont(12.5f));
        g.drawFittedText(message, area.removeFromTop(kMessageHeight), juce::Justification::centredLeft, 1, 0.85f);
        break;
    }
    case wizard::Page::analysing:
        g.setColour(wizardui::kText);
        g.setFont(AppFont::getFont(15.0f));
        g.drawText(TR("wizard.analysing"), area, juce::Justification::centred, true);
        break;
    case wizard::Page::nextStep:
        title(TR("wizard.next_title"));
        break;
    case wizard::Page::keyCheck:
        break;
    }
}

bool WizardOverlay::keyPressed(const juce::KeyPress& key)
{
    // The menu's shortcuts (quit, open, save) stay the menu's; the window
    // lets through those that do not touch the project under the wizard.
    if (key.getModifiers().isCommandDown())
        return false;
    if (key == juce::KeyPress::escapeKey && model.page != wizard::Page::analysing)
    {
        if (onCancel)
            onCancel();
        return true;
    }
    // The space bar is the transport's, on the key check too.
    if (key == juce::KeyPress::spaceKey && model.page == wizard::Page::keyCheck)
    {
        if (keyCheck.onPlayStop)
            keyCheck.onPlayStop();
        return true;
    }
    return true; // nothing under the overlay takes the keyboard
}

bool WizardOverlay::takeDroppedFile(const juce::File& file, juce::Point<int> position)
{
    if (model.page != wizard::Page::files || !onFileDropped)
        return false;
    bool forVoice = false;
    if (!model.existingProject)
    {
        if (voiceRow().contains(position))
            forVoice = true;
        else if (backingRow().contains(position))
            forVoice = false;
        else
            forVoice = model.voice == juce::File{}; // the first row that is empty
    }
    onFileDropped(file, forVoice);
    return true;
}
