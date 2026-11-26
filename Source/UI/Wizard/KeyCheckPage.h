#pragma once

#include "WizardModel.h"
#include "WizardWidgets.h"
#include "../Components/StyledWidgets.h"
#include "../../Utils/Localization.h"

#include <array>
#include <functional>
#include <memory>
#include <vector>

/**
 * The wizard's key check (spec section 15): for each stretch of the song,
 * the two likeliest keys as two large buttons and all twelve under them.
 * A key pressed is heard - the accompaniment from that stretch, the key's
 * scale in sine over it - and chosen: the ringed one is the answer. Above
 * the stretches, the song as a strip with its changes of key and the
 * playhead; a change can be added where the playhead is, or taken away.
 *
 * The page shows the wizard::Model it is given and says what was pressed;
 * the window plays the sound and changes the model.
 */
class KeyCheckPage : public juce::Component
{
public:
    static constexpr int kRowHeight = 92;
    static constexpr int kMaxRowsShown = 3; // more scroll
    static constexpr double kScaleMinDb = -24.0, kScaleMaxDb = 12.0;

    KeyCheckPage();
    ~KeyCheckPage() override;

    void setModel(const wizard::Model* modelToShow);
    /** After the model changed: stretches, candidates, choices. */
    void refresh();
    /** The transport, for the strip, the clock and the play button. */
    void setPlayback(double seconds, bool playing);
    /** The key whose scale is sounding, or stretch -1 for none. */
    void setSounding(int stretch, int root);
    void setVoiceToo(bool on, bool offered);
    void setScaleLevelDb(float dB);

    int getPreferredHeight() const;
    void paint(juce::Graphics& g) override;
    void resized() override;

    /** `root`: the major root of the key's seven notes (wizard::Model). */
    std::function<void(int stretch, int root)> onKey;
    /** The mode of the song, chosen from a stretch's list. */
    std::function<void(ScaleMode)> onMode;
    std::function<void()> onPlayStop;
    std::function<void(double seconds)> onSeek;
    std::function<void()> onAddChange;
    std::function<void(int stretch)> onRemoveChange;
    std::function<void(bool)> onVoiceToo;
    std::function<void(float dB)> onScaleLevel;
    std::function<void()> onBack;
    std::function<void()> onConfirm;

    // For tests and screenshots.
    int rowCount() const { return static_cast<int>(rows.size()); }
    juce::Button* keyButton(int stretch, int root, bool amongTheTwelve);
    juce::Button* removeButton(int stretch);
    juce::ComboBox* tonicList(int stretch);
    juce::ComboBox* modeList(int stretch);
    juce::Button& addChangeButton() { return addButton; }
    juce::Button& confirmButton() { return confirm; }
    juce::Button& backButton() { return back; }
    juce::Button& playButton();
    juce::Component& strip();

private:
    class Row;
    class Strip;
    class PlayButton;

    void refreshLocalisedText();

    const wizard::Model* model = nullptr;
    double playhead = 0.0;
    bool isPlaying = false;
    bool voiceOffered = true;
    int soundingStretch = -1, soundingRoot = 0;

    std::unique_ptr<PlayButton> play;
    std::unique_ptr<Strip> timeline;
    MacroSlider scaleSlider;
    StyledToggleButton voiceToggle;
    juce::Viewport viewport;
    juce::Component rowHolder;
    std::vector<std::unique_ptr<Row>> rows;
    wizardui::PillButton addButton, back, confirm{true};

    LocalisationWatcher languageWatcher{[this] { refreshLocalisedText(); }};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(KeyCheckPage)
};
