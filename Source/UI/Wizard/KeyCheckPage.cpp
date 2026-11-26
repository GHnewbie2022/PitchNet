#include "KeyCheckPage.h"

namespace
{
constexpr int kTitleHeight = 28;
constexpr int kHintHeight = 38;
constexpr int kControlsHeight = 30;
constexpr int kStripHeight = 34;
constexpr int kFooterHeight = 34;
constexpr int kGap = 10;
constexpr int kLabelWidth = 150;  // "Stretch 1  0:00-2:44"
constexpr int kBigKeyWidth = 150, kBigKeyHeight = 40;
constexpr int kSmallKeyWidth = 34, kSmallKeyHeight = 26, kSmallKeyGap = 3;
constexpr int kOtherLabelWidth = 150;
} // namespace

// ---------------------------------------------------------------------------

class KeyCheckPage::PlayButton : public juce::Button
{
public:
    PlayButton() : juce::Button("play") { setMouseCursor(juce::MouseCursor::PointingHandCursor); }
    bool playing = false;
    void paintButton(juce::Graphics& g, bool over, bool down) override
    {
        const auto bounds = getLocalBounds().toFloat().reduced(1.0f);
        g.setColour(down ? wizardui::kRaised.darker(0.15f)
                         : over ? wizardui::kRaised.brighter(0.1f) : wizardui::kRaised);
        g.fillRoundedRectangle(bounds, 6.0f);
        g.setColour(wizardui::kText);
        const auto icon = bounds.withSizeKeepingCentre(11.0f, 12.0f);
        if (playing)
        {
            g.fillRect(icon.withWidth(4.0f));
            g.fillRect(icon.withLeft(icon.getRight() - 4.0f));
        }
        else
        {
            juce::Path triangle;
            triangle.addTriangle(icon.getX(), icon.getY(), icon.getRight(), icon.getCentreY(),
                                 icon.getX(), icon.getBottom());
            g.fillPath(triangle);
        }
    }
};

// The song from its start to its end: the stretches, the changes of key
// between them, and the playhead. A press seeks.
class KeyCheckPage::Strip : public juce::Component
{
public:
    Strip() { setMouseCursor(juce::MouseCursor::PointingHandCursor); }
    const wizard::Model* model = nullptr;
    double playhead = 0.0;
    std::function<void(double)> onSeek;

    juce::Rectangle<float> bar() const
    {
        return getLocalBounds().toFloat().withTrimmedTop(14.0f).reduced(0.5f, 3.0f);
    }
    double duration() const { return model != nullptr ? model->duration() : 0.0; }
    float xOf(double seconds) const
    {
        const auto b = bar();
        const double total = duration();
        return total > 0.0 ? b.getX() + static_cast<float>(seconds / total) * b.getWidth() : b.getX();
    }
    double secondsAt(float x) const
    {
        const auto b = bar();
        return duration() * juce::jlimit(0.0, 1.0, static_cast<double>((x - b.getX()) / b.getWidth()));
    }
    void paint(juce::Graphics& g) override
    {
        const auto b = bar();
        g.setColour(wizardui::kInset);
        g.fillRoundedRectangle(b, 4.0f);
        if (model == nullptr || duration() <= 0.0)
            return;
        const auto& stretches = model->getStretches();
        g.setFont(AppFont::getFont(11.0f));
        for (size_t i = 0; i < stretches.size(); ++i)
        {
            const float x0 = xOf(stretches[i].startSeconds), x1 = xOf(stretches[i].endSeconds);
            // Every other stretch a shade lighter, its key in the middle.
            g.setColour(juce::Colour(i % 2 == 0 ? 0xFF2F3A47u : 0xFF3B4757u));
            g.fillRoundedRectangle(juce::Rectangle<float>(x0, b.getY(), x1 - x0, b.getHeight()).reduced(1.0f), 3.0f);
            g.setColour(wizardui::kText);
            g.drawText(wizardui::noteName(model->tonicOf(stretches[i].chosen)),
                       juce::Rectangle<float>(x0, b.getY(), x1 - x0, b.getHeight()),
                       juce::Justification::centred, false);
            if (i > 0)
            {
                g.setColour(wizardui::kChosen);
                g.fillRect(x0 - 0.75f, 12.0f, 1.5f, static_cast<float>(getHeight()) - 12.0f);
                g.setColour(APP_COLOR_TEXT_MUTED);
                g.drawText(wizardui::clock(stretches[i].startSeconds),
                           juce::Rectangle<float>(x0 - 30.0f, 0.0f, 60.0f, 12.0f), juce::Justification::centred,
                           false);
            }
        }
        g.setColour(juce::Colours::white);
        const float px = xOf(juce::jlimit(0.0, duration(), playhead));
        g.fillRect(px - 0.75f, b.getY() - 2.0f, 1.5f, b.getHeight() + 4.0f);
    }
    void mouseDown(const juce::MouseEvent& e) override
    {
        if (onSeek && duration() > 0.0)
            onSeek(secondsAt(static_cast<float>(e.x)));
    }
    void mouseDrag(const juce::MouseEvent& e) override { mouseDown(e); }
};

// One stretch: its name and times, its two candidates, the twelve keys, and
// (from the second on) the change of key it starts with, to take away.
class KeyCheckPage::Row : public juce::Component
{
public:
    Row()
    {
        addAndMakeVisible(first);
        addAndMakeVisible(second);
        addChildComponent(remove);
        remove.justification = juce::Justification::centredRight;
        // Any other key: its tonic and the mode, each from a list.
        for (int tonic = 0; tonic < 12; ++tonic)
            roots.addItem(wizardui::noteName(tonic), tonic + 1);
        int id = 1;
        for (const auto& item : wizardui::modes())
            modes.addItem(TR(item.second), id++);
        roots.onChange = [this]
        {
            if (onTonic && roots.getSelectedId() > 0)
                onTonic(roots.getSelectedId() - 1);
        };
        modes.onChange = [this]
        {
            const int index = modes.getSelectedId() - 1;
            if (onMode && index >= 0 && index < static_cast<int>(wizardui::modes().size()))
                onMode(wizardui::modes()[static_cast<size_t>(index)].first);
        };
        addAndMakeVisible(roots);
        addAndMakeVisible(modes);
    }
    wizardui::KeyButton first{0, true}, second{0, true};
    StyledComboBox roots, modes;
    std::function<void(int tonic)> onTonic;
    std::function<void(ScaleMode)> onMode;
    wizardui::LinkButton remove;
    juce::String name, times;

    void show(const wizard::Stretch& stretch, int index, int soundingRoot, const wizard::Model& model)
    {
        name = TR("wizard.stretch").replace("{n}", juce::String(index + 1));
        times = wizardui::clock(stretch.startSeconds) + juce::String::fromUTF8("\xe2\x80\x93") +
                wizardui::clock(stretch.endSeconds);
        first.setRoot(stretch.first);
        second.setRoot(stretch.second);
        first.setChosen(stretch.chosen == stretch.first);
        second.setChosen(stretch.chosen == stretch.second);
        first.setSounding(soundingRoot == stretch.first);
        second.setSounding(soundingRoot == stretch.second);
        // The two likeliest by name: the mode's tonic on those seven notes.
        const auto named = [&](int majorRoot)
        {
            return juce::String(wizardui::noteName(model.tonicOf(majorRoot))) + " " + wizardui::modeName(model.mode);
        };
        first.setText(named(stretch.first));
        second.setText(named(stretch.second));
        roots.setSelectedId(model.tonicOf(stretch.chosen) + 1, juce::dontSendNotification);
        int id = 1;
        for (const auto& item : wizardui::modes())
        {
            if (item.first == model.mode)
                modes.setSelectedId(id, juce::dontSendNotification);
            ++id;
        }
        remove.setVisible(index > 0);
        remove.setButtonText(TR("wizard.remove_change"));
        repaint();
    }
    void resized() override
    {
        auto area = getLocalBounds();
        auto top = area.removeFromTop(kBigKeyHeight + 4);
        top.removeFromLeft(kLabelWidth);
        first.setBounds(top.removeFromLeft(kBigKeyWidth).withSizeKeepingCentre(kBigKeyWidth, kBigKeyHeight));
        top.removeFromLeft(8);
        second.setBounds(top.removeFromLeft(kBigKeyWidth).withSizeKeepingCentre(kBigKeyWidth, kBigKeyHeight));
        remove.setBounds(top.removeFromRight(juce::jmin(190, top.getWidth() - 8)));
        area.removeFromTop(6);
        auto bottom = area.removeFromTop(kSmallKeyHeight);
        bottom.removeFromLeft(kOtherLabelWidth);
        roots.setBounds(bottom.removeFromLeft(96));
        bottom.removeFromLeft(8);
        modes.setBounds(bottom.removeFromLeft(210));
    }
    void paint(juce::Graphics& g) override
    {
        auto area = getLocalBounds();
        auto top = area.removeFromTop(kBigKeyHeight + 4).removeFromLeft(kLabelWidth - 8);
        g.setColour(wizardui::kText);
        g.setFont(AppFont::getFont(14.0f));
        g.drawText(name, top.removeFromTop(top.getHeight() / 2 + 2), juce::Justification::bottomLeft, true);
        g.setColour(APP_COLOR_TEXT_MUTED);
        g.setFont(AppFont::getFont(12.0f));
        g.drawText(times, top, juce::Justification::topLeft, true);
        area.removeFromTop(6);
        g.drawFittedText(TR("wizard.other_keys"), area.removeFromTop(kSmallKeyHeight).removeFromLeft(kOtherLabelWidth - 8),
                         juce::Justification::centredLeft, 1, 0.8f);
    }
};

// ---------------------------------------------------------------------------

KeyCheckPage::KeyCheckPage()
{
    play = std::make_unique<PlayButton>();
    timeline = std::make_unique<Strip>();
    play->onClick = [this]
    {
        if (onPlayStop)
            onPlayStop();
    };
    timeline->onSeek = [this](double seconds)
    {
        if (onSeek)
            onSeek(seconds);
    };
    scaleSlider.setRange(kScaleMinDb, kScaleMaxDb, 0.5);
    scaleSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    scaleSlider.setDoubleClickReturnValue(true, 0.0);
    scaleSlider.setValue(0.0, juce::dontSendNotification);
    scaleSlider.onValueChange = [this]
    {
        if (onScaleLevel)
            onScaleLevel(static_cast<float>(scaleSlider.getValue()));
    };
    voiceToggle.setClickingTogglesState(true);
    voiceToggle.onClick = [this]
    {
        if (onVoiceToo)
            onVoiceToo(voiceToggle.getToggleState());
    };
    addButton.onClick = [this]
    {
        if (onAddChange)
            onAddChange();
    };
    back.onClick = [this]
    {
        if (onBack)
            onBack();
    };
    confirm.onClick = [this]
    {
        if (onConfirm)
            onConfirm();
    };
    viewport.setViewedComponent(&rowHolder, false);
    viewport.setScrollBarsShown(true, false);
    addAndMakeVisible(*play);
    addAndMakeVisible(*timeline);
    addAndMakeVisible(scaleSlider);
    addAndMakeVisible(voiceToggle);
    addAndMakeVisible(viewport);
    addAndMakeVisible(addButton);
    addAndMakeVisible(back);
    addAndMakeVisible(confirm);
    refreshLocalisedText();
}

KeyCheckPage::~KeyCheckPage() = default;

juce::Button& KeyCheckPage::playButton() { return *play; }
juce::Component& KeyCheckPage::strip() { return *timeline; }

juce::Button* KeyCheckPage::keyButton(int stretch, int root, bool amongTheTwelve)
{
    if (stretch < 0 || stretch >= static_cast<int>(rows.size()))
        return nullptr;
    auto& row = *rows[static_cast<size_t>(stretch)];
    if (amongTheTwelve)
        return nullptr; // the other keys are the two lists now (tonicList, modeList)
    if (row.first.getRoot() == root)
        return &row.first;
    return row.second.getRoot() == root ? &row.second : nullptr;
}

juce::Button* KeyCheckPage::removeButton(int stretch)
{
    return stretch >= 0 && stretch < static_cast<int>(rows.size()) ? &rows[static_cast<size_t>(stretch)]->remove
                                                                    : nullptr;
}

void KeyCheckPage::refreshLocalisedText()
{
    addButton.setButtonText(TR("wizard.add_change"));
    back.setButtonText(TR("wizard.back"));
    confirm.setButtonText(TR("wizard.confirm_key"));
    voiceToggle.setTitle(TR("wizard.voice_too"));
    refresh();
}

void KeyCheckPage::setModel(const wizard::Model* modelToShow)
{
    model = modelToShow;
    timeline->model = model;
    refresh();
}

void KeyCheckPage::refresh()
{
    const int count = model != nullptr ? static_cast<int>(model->getStretches().size()) : 0;
    while (static_cast<int>(rows.size()) > count)
    {
        rowHolder.removeChildComponent(rows.back().get());
        rows.pop_back();
    }
    while (static_cast<int>(rows.size()) < count)
    {
        auto row = std::make_unique<Row>();
        Row* raw = row.get();
        const auto press = [this, raw](wizardui::KeyButton& key)
        {
            key.onClick = [this, raw, &key]
            {
                for (size_t i = 0; i < rows.size(); ++i)
                    if (rows[i].get() == raw && onKey)
                        onKey(static_cast<int>(i), key.getRoot());
            };
        };
        press(row->first);
        press(row->second);
        row->onTonic = [this, raw](int tonic)
        {
            for (size_t i = 0; i < rows.size(); ++i)
                if (rows[i].get() == raw && onKey && model != nullptr)
                    onKey(static_cast<int>(i), model->majorRootOf(tonic));
        };
        row->onMode = [this](ScaleMode mode)
        {
            if (onMode)
                onMode(mode);
        };
        row->remove.onClick = [this, raw]
        {
            // Taking a change away takes a row away, perhaps this one: not
            // from inside its own button's click.
            for (size_t i = 0; i < rows.size(); ++i)
                if (rows[i].get() == raw)
                {
                    juce::Component::SafePointer<KeyCheckPage> safeThis(this);
                    const int index = static_cast<int>(i);
                    juce::MessageManager::callAsync([safeThis, index]
                    {
                        if (safeThis != nullptr && safeThis->onRemoveChange)
                            safeThis->onRemoveChange(index);
                    });
                    break;
                }
        };
        rowHolder.addAndMakeVisible(*row);
        rows.push_back(std::move(row));
    }
    for (int i = 0; i < count; ++i)
        rows[static_cast<size_t>(i)]->show(model->getStretches()[static_cast<size_t>(i)], i,
                                           i == soundingStretch ? soundingRoot : -1, *model);
    resized();
    repaint();
}

void KeyCheckPage::setPlayback(double seconds, bool playing)
{
    if (std::abs(seconds - playhead) < 1.0e-3 && playing == isPlaying)
        return;
    playhead = seconds;
    isPlaying = playing;
    play->playing = playing;
    play->repaint();
    timeline->playhead = seconds;
    timeline->repaint();
    repaint(getLocalBounds().removeFromTop(kTitleHeight + kHintHeight + kGap + kControlsHeight));
}

void KeyCheckPage::setSounding(int stretch, int root)
{
    if (stretch == soundingStretch && (stretch < 0 || root == soundingRoot))
        return;
    soundingStretch = stretch;
    soundingRoot = root;
    refresh();
}

void KeyCheckPage::setVoiceToo(bool on, bool offered)
{
    voiceOffered = offered;
    voiceToggle.setToggleState(on, juce::dontSendNotification);
    voiceToggle.setVisible(offered);
    repaint();
}

void KeyCheckPage::setScaleLevelDb(float dB) { scaleSlider.setValue(dB, juce::dontSendNotification); }

int KeyCheckPage::getPreferredHeight() const
{
    const int shown = juce::jlimit(1, kMaxRowsShown, static_cast<int>(rows.size()));
    return kTitleHeight + kHintHeight + kGap + kControlsHeight + kGap + kStripHeight + kGap +
           shown * kRowHeight + kGap + kFooterHeight;
}

void KeyCheckPage::resized()
{
    auto area = getLocalBounds();
    area.removeFromTop(kTitleHeight + kHintHeight + kGap);
    auto controls = area.removeFromTop(kControlsHeight);
    play->setBounds(controls.removeFromLeft(36));
    // From the right: the voice's switch (when there is an accompaniment to
    // hear without it), then the scale's level.
    if (voiceOffered)
    {
        auto voice = controls.removeFromRight(150);
        voiceToggle.setBounds(voice.removeFromLeft(40).withSizeKeepingCentre(40, 24));
        controls.removeFromRight(12);
    }
    scaleSlider.setBounds(controls.removeFromRight(110));
    area.removeFromTop(kGap);
    timeline->setBounds(area.removeFromTop(kStripHeight));
    area.removeFromTop(kGap);
    auto footer = area.removeFromBottom(kFooterHeight);
    area.removeFromBottom(kGap);
    viewport.setBounds(area);
    const int rowWidth = area.getWidth() - (static_cast<int>(rows.size()) > kMaxRowsShown ? 12 : 0);
    rowHolder.setSize(rowWidth, static_cast<int>(rows.size()) * kRowHeight);
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i]->setBounds(0, static_cast<int>(i) * kRowHeight, rowWidth, kRowHeight - 12);
    addButton.setBounds(footer.removeFromLeft(juce::jmin(240, footer.getWidth() / 2)));
    confirm.setBounds(footer.removeFromRight(120));
    footer.removeFromRight(10);
    back.setBounds(footer.removeFromRight(96));
}

void KeyCheckPage::paint(juce::Graphics& g)
{
    auto area = getLocalBounds();
    g.setColour(wizardui::kText);
    g.setFont(AppFont::getBoldFont(17.0f));
    g.drawText(TR("wizard.key_title"), area.removeFromTop(kTitleHeight), juce::Justification::centredLeft, true);
    g.setColour(APP_COLOR_TEXT_MUTED);
    g.setFont(AppFont::getFont(12.5f));
    g.drawFittedText(TR("wizard.key_hint"), area.removeFromTop(kHintHeight), juce::Justification::topLeft, 2, 0.9f);
    area.removeFromTop(kGap);
    auto controls = area.removeFromTop(kControlsHeight);
    controls.removeFromLeft(36 + 10);
    g.setFont(AppFont::getFont(13.0f));
    g.drawText(wizardui::clock(playhead) + " / " + wizardui::clock(model != nullptr ? model->duration() : 0.0),
               controls.removeFromLeft(110), juce::Justification::centredLeft, false);
    if (voiceOffered)
    {
        auto voice = controls.removeFromRight(150);
        voice.removeFromLeft(40);
        g.drawFittedText(TR("wizard.voice_too"), voice, juce::Justification::centredLeft, 1, 0.8f);
        controls.removeFromRight(12);
    }
    controls.removeFromRight(110 + 8);
    g.drawFittedText(TR("wizard.scale_level"), controls.removeFromRight(110), juce::Justification::centredRight, 1,
                     0.8f);
}

juce::ComboBox* KeyCheckPage::tonicList(int stretch)
{
    return stretch >= 0 && stretch < static_cast<int>(rows.size()) ? &rows[static_cast<size_t>(stretch)]->roots
                                                                    : nullptr;
}

juce::ComboBox* KeyCheckPage::modeList(int stretch)
{
    return stretch >= 0 && stretch < static_cast<int>(rows.size()) ? &rows[static_cast<size_t>(stretch)]->modes
                                                                    : nullptr;
}
