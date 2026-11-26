// The standalone's left panel (spec section 15): it shows the state it is
// given and says what the user did - through its own sliders and buttons,
// as the App's window wires them.
#include "../Source/UI/MixerPanel.h"
#include "../Source/UI/Main/ExportHelper.h"
#include "TestAssert.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
MixerPanel::State withBacking()
{
    MixerPanel::State state;
    state.hasProject = true;
    state.hasBacking = true;
    state.backingName = "backing.wav";
    return state;
}

void whatItShows()
{
    MixerPanel panel;
    panel.setSize(141, panel.getPreferredHeight());
    // No project: nothing can be set.
    CHECK(!panel.voiceFader().isEnabled() && !panel.backingFader().isEnabled());
    CHECK(!panel.playAlongSwitch().isEnabled() && !panel.chooseButton().isEnabled());
    CHECK(!panel.removeButton().isVisible());

    // A project without an accompaniment: the voice's level, and the choice.
    MixerPanel::State state;
    state.hasProject = true;
    state.voiceDb = -4.5f;
    panel.setState(state);
    CHECK(panel.voiceFader().isEnabled() && panel.voiceFader().getValue() == -4.5);
    CHECK(panel.chooseButton().isEnabled() && !panel.removeButton().isVisible());
    CHECK(!panel.backingFader().isEnabled() && !panel.playAlongSwitch().isEnabled());
    CHECK(panel.chooseButton().getWidth() > 100); // the whole row

    // With one: its level, the switch, and both buttons side by side.
    state = withBacking();
    state.backingAudible = true;
    state.backingDb = -9.5f;
    panel.setState(state);
    CHECK(panel.backingFader().isEnabled() && panel.backingFader().getValue() == -9.5);
    CHECK(panel.playAlongSwitch().isEnabled() && panel.playAlongSwitch().getToggleState());
    CHECK(panel.removeButton().isVisible() && panel.removeButton().isEnabled());
    CHECK(panel.chooseButton().getRight() < panel.removeButton().getX());

    // Missing or still being read: it cannot play, but another can be chosen
    // once the read is over.
    state.backingMissing = true;
    panel.setState(state);
    CHECK(!panel.backingFader().isEnabled() && !panel.playAlongSwitch().isEnabled());
    CHECK(panel.chooseButton().isEnabled() && panel.removeButton().isEnabled());
    state.backingMissing = false;
    state.backingLoading = true;
    panel.setState(state);
    CHECK(!panel.backingFader().isEnabled() && !panel.chooseButton().isEnabled());

    CHECK(MixerPanel::levelText(-6.0f) == "-6.0 dB");
    CHECK(MixerPanel::levelText(0.0f) == "0.0 dB");
    CHECK(MixerPanel::levelText(-60.0f).startsWith("-") && !MixerPanel::levelText(-60.0f).contains("60"));
}

void whatItSays()
{
    MixerPanel panel;
    panel.setSize(141, panel.getPreferredHeight());
    std::vector<float> voice, backing;
    std::vector<bool> audible;
    int chosen = 0, removed = 0;
    panel.onVoiceGainChanged = [&](float dB) { voice.push_back(dB); };
    panel.onBackingGainChanged = [&](float dB) { backing.push_back(dB); };
    panel.onBackingAudibleChanged = [&](bool on) { audible.push_back(on); };
    panel.onChooseBacking = [&] { ++chosen; };
    panel.onRemoveBacking = [&] { ++removed; };

    // Being given a state says nothing back.
    auto state = withBacking();
    state.voiceDb = -3.0f;
    state.backingDb = -12.0f;
    state.backingAudible = true;
    panel.setState(state);
    CHECK(voice.empty() && backing.empty() && audible.empty());

    // The faders, in half dB steps, between -60 and +12.
    panel.voiceFader().setValue(-7.26, juce::sendNotificationSync);
    panel.backingFader().setValue(-100.0, juce::sendNotificationSync);
    panel.backingFader().setValue(40.0, juce::sendNotificationSync);
    CHECK(voice.size() == 1 && voice[0] == -7.5f);
    CHECK(backing.size() == 2 && backing[0] == -60.0f && backing[1] == 12.0f);
    CHECK(panel.getState().voiceDb == -7.5f && panel.getState().backingDb == 12.0f);
    // A double click puts a fader back: 0 dB for the voice, -6 for the other.
    CHECK(panel.voiceFader().getDoubleClickReturnValue() == 0.0);
    CHECK(panel.backingFader().getDoubleClickReturnValue() == -6.0);

    // The switch and the buttons.
    panel.playAlongSwitch().triggerClick();
    juce::MessageManager::getInstance()->runDispatchLoopUntil(50);
    CHECK(audible.size() == 1 && audible[0] == false);
    CHECK(!panel.getState().backingAudible);
    panel.chooseButton().triggerClick();
    panel.removeButton().triggerClick();
    juce::MessageManager::getInstance()->runDispatchLoopUntil(50);
    CHECK(chosen == 1 && removed == 1);
}
// The audio an export writes, by the track asked for: the call the App's
// export makes (ExportHelper::buildExportAudio).
void whatAnExportWrites()
{
    const double pi = 3.14159265358979323846;
    const int rate = 44100, length = 44100;
    juce::AudioBuffer<float> voice(1, length), backingAudio(2, 48000);
    for (int i = 0; i < length; ++i)
        voice.setSample(0, i, static_cast<float>(0.5 * std::sin(2 * pi * 220.0 * i / rate)));
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 48000; ++i)
            backingAudio.setSample(ch, i, static_cast<float>((ch == 0 ? 0.4 : 0.2) *
                                                             std::sin(2 * pi * 330.0 * i / 48000.0)));
    const auto file = juce::File::getSpecialLocation(juce::File::tempDirectory)
                          .getChildFile("pitchnet_export_test_backing.wav");
    file.deleteFile();
    {
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::OutputStream> stream = std::make_unique<juce::FileOutputStream>(file);
        auto writer = wav.createWriterFor(stream, juce::AudioFormatWriterOptions{}
                                                      .withSampleRate(48000)
                                                      .withNumChannels(2)
                                                      .withBitsPerSample(32));
        CHECK(writer != nullptr);
        CHECK(writer->writeFromAudioSampleBuffer(backingAudio, 0, backingAudio.getNumSamples()));
    }
    BackingTrack backing;
    backing.file = file;
    backing.gainDb = -6.0f;

    ExportHelper::ExportSettings settings;
    settings.sampleRate = 48000;
    settings.channels = 2;

    // The voice: the channels, then the rate - what the export always did,
    // whatever the faders are at.
    settings.track = ExportHelper::ExportTrack::voice;
    auto built = ExportHelper::buildExportAudio(settings, voice, rate, backing, -12.0f);
    auto wanted = ExportHelper::resampleAudio(ExportHelper::convertChannels(voice, 2), rate, 48000);
    CHECK(built.error.isEmpty() && built.audio.getNumSamples() == wanted.getNumSamples());
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < wanted.getNumSamples(); ++i)
            CHECK(built.audio.getSample(ch, i) == wanted.getSample(ch, i));

    // The accompaniment: its file as it is (the same rate: nothing converted).
    settings.track = ExportHelper::ExportTrack::backing;
    built = ExportHelper::buildExportAudio(settings, {}, 0, backing, 0.0f);
    CHECK(built.error.isEmpty() && built.audio.getNumSamples() == 48000);
    for (int i = 0; i < 48000; ++i)
        CHECK(std::abs(built.audio.getSample(1, i) - backingAudio.getSample(1, i)) < 1.0e-6f);

    // The whole: each at its fader.
    settings.track = ExportHelper::ExportTrack::mix;
    built = ExportHelper::buildExportAudio(settings, voice, rate, backing, -3.0f);
    CHECK(built.error.isEmpty());
    const float gv = juce::Decibels::decibelsToGain(-3.0f), gb = juce::Decibels::decibelsToGain(-6.0f);
    for (int i = 0; i < 47000; ++i)
        CHECK(std::abs(built.audio.getSample(0, i) -
                       (gv * wanted.getSample(0, i) + gb * backingAudio.getSample(0, i))) < 1.0e-5f);
    CHECK(built.mixPeak > 0.3f && built.mixPeak < 1.0f);

    // No file: said, nothing written.
    file.deleteFile();
    built = ExportHelper::buildExportAudio(settings, voice, rate, backing, 0.0f);
    CHECK(built.error.isNotEmpty() && built.audio.getNumSamples() == 0);
    // ...and the voice needs none.
    settings.track = ExportHelper::ExportTrack::voice;
    CHECK(ExportHelper::buildExportAudio(settings, voice, rate, backing, 0.0f).error.isEmpty());
}
} // namespace

int main()
{
    juce::ScopedJuceInitialiser_GUI juce;
    whatItShows();
    whatItSays();
    whatAnExportWrites();
    std::printf("MixerPanelTests: all passed\n");
    return 0;
}
