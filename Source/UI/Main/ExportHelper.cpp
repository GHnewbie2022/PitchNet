#include "ExportHelper.h"
#include "../../Utils/AudioResampler.h"
#include "../../Audio/Mixdown.h"
#include "../../Utils/Localization.h"
#include "../StyledComponents.h"
#include <cmath>

namespace ExportHelper {

juce::String getFormatDisplayName(ExportFormat format) {
  switch (format) {
  case ExportFormat::wav:
    return "WAV";
  case ExportFormat::flac:
    return "FLAC";
  case ExportFormat::aiff:
    return "AIFF";
  case ExportFormat::ogg:
    return "OGG";
  }
  return "WAV";
}

juce::String getFormatExtension(ExportFormat format) {
  switch (format) {
  case ExportFormat::wav:
    return "wav";
  case ExportFormat::flac:
    return "flac";
  case ExportFormat::aiff:
    return "aiff";
  case ExportFormat::ogg:
    return "ogg";
  }
  return "wav";
}

juce::String getFormatWildcard(ExportFormat format) {
  return "*." + getFormatExtension(format);
}

juce::AudioBuffer<float> convertChannels(const juce::AudioBuffer<float> &input,
                                          int outChannels) {
  // One rule for a single track and for the whole (Mixdown.h).
  return mixdown::toChannels(input, outChannels);
}

juce::AudioBuffer<float> resampleAudio(const juce::AudioBuffer<float> &input,
                                        int sourceRate, int targetRate) {
  return AudioResampler::resample(input, sourceRate, targetRate);
}

ExportAudio buildExportAudio(const ExportSettings &settings,
                             const juce::AudioBuffer<float> &voice, int voiceRate,
                             const BackingTrack &backing, float voiceDb) {
  ExportAudio result;
  if (settings.track == ExportTrack::voice) {
    // The voice, as it was always exported: no fader on it.
    result.audio = convertChannels(voice, settings.channels);
    result.audio = resampleAudio(result.audio, voiceRate, settings.sampleRate);
    return result;
  }

  juce::AudioBuffer<float> backingAudio;
  int backingRate = 0;
  {
    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(backing.file));
    if (reader != nullptr && reader->lengthInSamples > 0) {
      backingRate = static_cast<int>(std::lround(reader->sampleRate));
      backingAudio.setSize(static_cast<int>(reader->numChannels),
                           static_cast<int>(reader->lengthInSamples));
      reader->read(&backingAudio, 0, backingAudio.getNumSamples(), 0, true, true);
    }
  }
  if (backingAudio.getNumSamples() <= 0) {
    result.error = TR("dialog.backing_unreadable") + "\n" + backing.file.getFullPathName();
    return result;
  }
  if (settings.track == ExportTrack::backing) {
    result.audio = convertChannels(backingAudio, settings.channels);
    result.audio = resampleAudio(result.audio, backingRate, settings.sampleRate);
    return result;
  }
  // The two as they play together: each at its fader.
  auto mixed = mixdown::mix({&voice, voiceRate, voiceDb},
                            {&backingAudio, backingRate, backing.gainDb},
                            settings.sampleRate, settings.channels);
  result.audio = std::move(mixed.audio);
  result.mixPeak = mixed.peak;
  return result;
}

juce::AudioFormat *findFormatForExtension(juce::AudioFormatManager &manager,
                                           const juce::String &extension) {
  const auto normalizedExtension = extension.trimCharactersAtStart(".");
  for (int i = 0; i < manager.getNumKnownFormats(); ++i) {
    auto *fmt = manager.getKnownFormat(i);
    if (!fmt)
      continue;
    auto exts = fmt->getFileExtensions();
    for (const auto &ext : exts) {
      if (ext.trimCharactersAtStart(".").equalsIgnoreCase(normalizedExtension))
        return fmt;
    }
  }
  return nullptr;
}

// =============================================================================
// ExportSettingsContent — dialog component for choosing export parameters
// =============================================================================

class ExportSettingsContent final : public juce::Component, private juce::Button::Listener {
public:
  ExportSettingsContent(int inputSampleRate,
                        const std::optional<ExportSettings> &savedSettings,
                        std::function<void(std::optional<ExportSettings>)> done,
                        bool offerTracks)
      : onDone(std::move(done)), tracksOffered(offerTracks) {
    setOpaque(false);

    // Which track, when there is an accompaniment to choose.
    if (tracksOffered)
      setupCombo(trackBox, trackLabel, TR("export.track"),
                 {TR("export.track_voice"), TR("export.track_backing"), TR("export.track_mix")}, 1);
    // The recording's own rate, under the rate chosen: the same exports
    // without a conversion.
    addAndMakeVisible(sourceRateNote);
    sourceRateNote.setText(TR("export.source_rate").replace("{rate}", juce::String(inputSampleRate)),
                           juce::dontSendNotification);
    sourceRateNote.setJustificationType(juce::Justification::centredRight);
    sourceRateNote.setColour(juce::Label::textColourId, APP_COLOR_TEXT_MUTED);
    sourceRateNote.setFont(AppFont::getFont(12.0f));
    sourceRateNote.setMinimumHorizontalScale(0.7f);

    addAndMakeVisible(title);
    title.setText(TR("export.title"), juce::dontSendNotification);
    title.setJustificationType(juce::Justification::centredLeft);
    title.setColour(juce::Label::textColourId, APP_COLOR_TEXT_PRIMARY);
    title.setFont(AppFont::getBoldFont(17.0f));

    setupCombo(formatBox, formatLabel, TR("export.format"),
               {"WAV", "FLAC", "AIFF", "OGG"}, 1);

    int srId = 3;
    if (inputSampleRate >= 47000)
      srId = 4;
    else if (inputSampleRate >= 43000)
      srId = 3;
    else if (inputSampleRate >= 30000)
      srId = 2;
    else
      srId = 1;
    setupCombo(sampleRateBox, sampleRateLabel, TR("settings.sample_rate"),
               {"22050", "32000", "44100", "48000"}, srId);
    setupCombo(bitDepthBox, bitDepthLabel, TR("export.bit_depth"),
               {"16", "24", "32"}, 1);
    const juce::StringArray channelItems{TR("settings.mono"),
                                        TR("settings.stereo")};
    setupCombo(channelsBox, channelsLabel, TR("settings.output_channels"),
               channelItems, 1);

    if (savedSettings)
    {
      formatBox.setSelectedId(static_cast<int>(savedSettings->format) + 1,
                              juce::dontSendNotification);
      const juce::StringArray sampleRates{"22050", "32000", "44100", "48000"};
      sampleRateBox.setSelectedId(sampleRates.indexOf(juce::String(savedSettings->sampleRate)) + 1,
                                  juce::dontSendNotification);
      const juce::StringArray bitDepths{"16", "24", "32"};
      bitDepthBox.setSelectedId(bitDepths.indexOf(juce::String(savedSettings->bitsPerSample)) + 1,
                                juce::dontSendNotification);
      channelsBox.setSelectedId(savedSettings->channels, juce::dontSendNotification);
    }

    addAndMakeVisible(cancelButton);
    addAndMakeVisible(exportButton);
    cancelButton.setButtonText(TR("dialog.cancel"));
    exportButton.setButtonText(TR("export.export"));
    configureButton(cancelButton);
    configureButton(exportButton);
    cancelButton.addListener(this);
    exportButton.addListener(this);
  }

  ~ExportSettingsContent() override {
    cancelButton.removeListener(this);
    exportButton.removeListener(this);
    cancelButton.setLookAndFeel(nullptr);
    exportButton.setLookAndFeel(nullptr);
  }

  void paint(juce::Graphics &g) override {
    auto bounds = getLocalBounds().toFloat();
    g.setColour(juce::Colour(0xFF232323u));
    g.fillRoundedRectangle(bounds, 10.0f);

    auto card = cardBounds.toFloat();
    g.setColour(juce::Colour(0xFF151515u));
    g.fillRoundedRectangle(card, 8.0f);
    g.setColour(APP_COLOR_BORDER.withAlpha(0.55f));
    g.drawRoundedRectangle(card.reduced(0.5f), 8.0f, 0.75f);
  }

  void resized() override {
    auto area = getLocalBounds().reduced(16);
    title.setBounds(area.removeFromTop(28));
    area.removeFromTop(6);

    // Keep the export options grouped in the card, with the dialog actions
    // deliberately placed below it.
    const int rowHeight = 32;
    const int rowGap = 6;
    constexpr int cardVerticalPadding = 12;
    const int rows = tracksOffered ? 5 : 4;
    const int cardHeight =
        rows * rowHeight + (rows - 1) * rowGap + kSourceRateNoteHeight + 2 * cardVerticalPadding;
    cardBounds = area.removeFromTop(cardHeight);
    area.removeFromTop(16);
    auto buttonRow = area.removeFromTop(32);
    auto content = cardBounds.reduced(16, cardVerticalPadding);
    const int controlWidth = juce::jlimit(150, 190, content.getWidth() / 2);
    const int labelWidth = content.getWidth() - controlWidth - 24;

    auto layoutRow = [&](juce::Label &label, juce::Component &control) {
      auto row = content.removeFromTop(rowHeight);
      label.setBounds(row.removeFromLeft(labelWidth));
      control.setBounds(row.removeFromRight(controlWidth).reduced(0, 2));
      if (&label != &channelsLabel)
        content.removeFromTop(rowGap);
    };

    if (tracksOffered)
      layoutRow(trackLabel, trackBox);
    layoutRow(formatLabel, formatBox);
    layoutRow(sampleRateLabel, sampleRateBox);
    // The note sits in the gap under the rate, and takes its own line.
    content.setTop(content.getY() - rowGap);
    sourceRateNote.setBounds(content.removeFromTop(kSourceRateNoteHeight));
    content.removeFromTop(rowGap);
    layoutRow(bitDepthLabel, bitDepthBox);
    layoutRow(channelsLabel, channelsBox);

    auto right = buttonRow.removeFromRight(190);
    cancelButton.setBounds(right.removeFromLeft(90));
    right.removeFromLeft(10);
    exportButton.setBounds(right);
  }

private:
  void setupCombo(juce::ComboBox &box, juce::Label &label, const juce::String &labelText,
                  const juce::StringArray &items, int selectedId) {
    addAndMakeVisible(label);
    addAndMakeVisible(box);
    label.setText(labelText, juce::dontSendNotification);
    label.setColour(juce::Label::textColourId, APP_COLOR_TEXT_PRIMARY);
    label.setFont(AppFont::getFont(15.0f));
    label.setJustificationType(juce::Justification::centredLeft);
    box.addItemList(items, 1);
    box.setSelectedId(selectedId, juce::dontSendNotification);
  }

  static void configureButton(juce::TextButton &button) {
    button.setLookAndFeel(&DarkLookAndFeel::getInstance());
    button.setColour(juce::TextButton::buttonColourId, juce::Colour(0xFF5B5B5Bu));
    button.setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xFF3E3E3Eu));
    button.setColour(juce::TextButton::textColourOffId, juce::Colour(0xFFEFEFEFu));
    button.setColour(juce::TextButton::textColourOnId, juce::Colour(0xFFEFEFEFu));
    button.setMouseCursor(juce::MouseCursor::PointingHandCursor);
  }

  ExportSettings getSettings() const {
    ExportSettings settings;
    if (tracksOffered)
      settings.track = trackBox.getSelectedId() == 2   ? ExportTrack::backing
                       : trackBox.getSelectedId() == 3 ? ExportTrack::mix
                                                       : ExportTrack::voice;
    switch (formatBox.getSelectedId()) {
    case 2:
      settings.format = ExportFormat::flac;
      break;
    case 3:
      settings.format = ExportFormat::aiff;
      break;
    case 4:
      settings.format = ExportFormat::ogg;
      break;
    default:
      settings.format = ExportFormat::wav;
      break;
    }
    settings.sampleRate = sampleRateBox.getText().getIntValue();
    settings.bitsPerSample = bitDepthBox.getText().getIntValue();
    settings.channels = channelsBox.getSelectedId() == 2 ? 2 : 1;
    return settings;
  }

  void buttonClicked(juce::Button *button) override {
    auto closeWith = [this](std::optional<ExportSettings> result) {
      if (onDone)
        onDone(std::move(result));
      if (auto *dw = findParentComponentOfClass<juce::DialogWindow>())
        dw->exitModalState(0);
    };

    if (button == &exportButton)
      closeWith(getSettings());
    else if (button == &cancelButton)
      closeWith(std::nullopt);
  }

public:
  static constexpr int kSourceRateNoteHeight = 16;
  /** The dialog's height for its rows. */
  static int heightFor(bool offerTracks) { return 284 + kSourceRateNoteHeight + (offerTracks ? 38 : 0); }

private:
  std::function<void(std::optional<ExportSettings>)> onDone;
  bool tracksOffered = false;
  juce::Rectangle<int> cardBounds;
  juce::Label title;
  juce::Label trackLabel, formatLabel, sampleRateLabel, bitDepthLabel, channelsLabel;
  juce::Label sourceRateNote;
  StyledComboBox trackBox, formatBox, sampleRateBox, bitDepthBox, channelsBox;
  juce::TextButton cancelButton, exportButton;
};

std::unique_ptr<juce::Component> makeExportSettingsContent(int inputSampleRate, bool offerTracks) {
  auto content = std::make_unique<ExportSettingsContent>(inputSampleRate, std::nullopt,
                                                         [](std::optional<ExportSettings>) {},
                                                         offerTracks);
  content->setSize(420, ExportSettingsContent::heightFor(offerTracks));
  return content;
}

void showExportSettingsDialogAsync(
    juce::Component *parent, int inputSampleRate,
    const std::optional<ExportSettings> &savedSettings,
    std::function<void(std::optional<ExportSettings>)> onDone, bool offerTracks) {
  constexpr int dialogWidth = 420;
  const int dialogHeight = ExportSettingsContent::heightFor(offerTracks);
  auto *content =
      new ExportSettingsContent(inputSampleRate, savedSettings, std::move(onDone), offerTracks);
  content->setSize(dialogWidth, dialogHeight);

  juce::DialogWindow::LaunchOptions opts;
  opts.content.setOwned(content);
  opts.dialogTitle = TR("export.title");
  opts.componentToCentreAround = parent;
  opts.dialogBackgroundColour = APP_COLOR_SURFACE;
  opts.escapeKeyTriggersCloseButton = true;
  opts.useNativeTitleBar = false;
  opts.resizable = false;
  opts.useBottomRightCornerResizer = false;
  if (auto *window = opts.launchAsync()) {
    window->setTitleBarHeight(0);
    if (parent != nullptr)
      window->centreAroundComponent(parent, dialogWidth, dialogHeight);
    else
      window->centreWithSize(dialogWidth, dialogHeight);
  }
}

} // namespace ExportHelper
