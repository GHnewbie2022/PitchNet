#pragma once

#include "../../JuceHeader.h"
#include "../../Utils/Constants.h"
#include "../../Utils/UI/Theme.h"
#include "../../Utils/Localization.h"
#include "../../Models/Project.h"
#include <functional>
#include <optional>

// =============================================================================
// Audio export utilities — extracted from the anonymous namespace in
// MainComponent.cpp to reduce file size and improve discoverability.
// =============================================================================

namespace ExportHelper {

enum class ExportFormat { wav, flac, aiff, ogg };

// What is exported (spec section 15): the voice as ever, the accompaniment
// the project links, or the two as they play together (Mixdown.h). Chosen
// each time - it is not kept with the other settings, so an export is the
// voice unless the user says otherwise.
enum class ExportTrack { voice, backing, mix };

struct ExportSettings {
  ExportTrack track = ExportTrack::voice;
  ExportFormat format = ExportFormat::wav;
  int sampleRate = SAMPLE_RATE;
  int channels = 1;
  int bitsPerSample = 16;
};

juce::String getFormatDisplayName(ExportFormat format);
juce::String getFormatExtension(ExportFormat format);
juce::String getFormatWildcard(ExportFormat format);

juce::AudioBuffer<float> convertChannels(const juce::AudioBuffer<float> &input,
                                          int outChannels);
juce::AudioBuffer<float> resampleAudio(const juce::AudioBuffer<float> &input,
                                        int sourceRate, int targetRate);

juce::AudioFormat *findFormatForExtension(juce::AudioFormatManager &manager,
                                           const juce::String &extension);

/** What an export writes, or why it cannot. */
struct ExportAudio {
  juce::AudioBuffer<float> audio; // at settings.sampleRate, settings.channels
  float mixPeak = 0.0f;           // of the whole: over 1 clips in a fixed-point file
  juce::String error;             // empty when the audio is there
};

/**
 * The audio of the track `settings` asks for, at its channels and rate.
 *   voice    `voice` as it was always exported: the channels, then the rate,
 *            no fader.
 *   backing  the accompaniment's file, the same way.
 *   mix      the two as they play together, each at its fader (Mixdown.h).
 * `voice` is the project's audio as playback builds it (sourceaudio::build);
 * it is not read for the accompaniment alone. Reads the accompaniment's
 * file: not for the message thread on a long song.
 */
ExportAudio buildExportAudio(const ExportSettings &settings,
                             const juce::AudioBuffer<float> &voice, int voiceRate,
                             const BackingTrack &backing, float voiceDb);

/** `offerTracks`: the project links an accompaniment, so the dialog asks
 *  which track. The recording's own rate is named under the rate chosen. */
void showExportSettingsDialogAsync(
    juce::Component *parent, int inputSampleRate,
    const std::optional<ExportSettings> &savedSettings,
    std::function<void(std::optional<ExportSettings>)> onDone,
    bool offerTracks = false);

/** The dialog's content alone, for a screenshot. */
std::unique_ptr<juce::Component> makeExportSettingsContent(int inputSampleRate, bool offerTracks);

} // namespace ExportHelper
