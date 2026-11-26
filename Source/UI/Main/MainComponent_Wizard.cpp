// The welcome screen and the wizard (spec section 15, standalone only): what
// the window does when their buttons are pressed. The overlays themselves are
// in UI/Wizard; the wizard's state is wizard::Model.
//
//   welcome   free way: the screen goes. Guided way: the wizard, for a new
//             project. A project opened instead: the screen goes.
//   files     a new project's recording is loaded and analysed as any other
//             (loadAudioFile), its accompaniment linked (loadBackingAudio)
//             and marked aligned - the user's tick;
//   analysing the accompaniment is read again, off the message thread, for
//             its pitch classes (keyfinder::analyse); with the project's
//             notes they give the stretches and their two likeliest keys;
//   keyCheck  a key pressed is chosen and heard: the transport plays the
//             accompaniment from that stretch, the key's scale laid over it;
//   confirm   the project's key and its changes of key, one step of the
//             history (KeyChangeAction).

#include "../MainComponent.h"

#include "../../Audio/KeyFinder/AccompanimentChroma.h"
#include "../../Audio/KeyFinder/KeyFinder.h"
#include "../../Audio/KeyFinder/ScaleTone.h"
#include "../../Undo/KeyChangeAction.h"
#include "../../Undo/KeyFlagEdits.h"
#include "../../Utils/Localization.h"
#include "../../Utils/ScaleUtils.h"

#include <cstdlib>
#include <memory>

namespace {
// The scale of a key that is heard: 0.5 s of the accompaniment alone first,
// when the transport starts for it.
constexpr double kAuditionLeadSeconds = 0.5;
constexpr double kAuditionSeconds = kAuditionLeadSeconds + 8 * 0.4 + 0.3;
// A stretch is heard from a second before its first sung note.
constexpr double kAuditionBeforeVoice = 1.0;
constexpr int kToneRate = 48000;
} // namespace

// ---------------------------------------------------------------------------
// The welcome screen

void MainComponent::showWelcome() {
  if (isPluginMode() || hasAnalyzedProject() || isLoadingAudio.load())
    return;
  if (welcomeOverlay == nullptr) {
    welcomeOverlay = std::make_unique<WelcomeOverlay>();
    addChildComponent(*welcomeOverlay);
    welcomeOverlay->onFreeMode = [this] { hideWelcome(); };
    welcomeOverlay->onWizard = [this] { showWizard(true); };
    welcomeOverlay->onOpenFile = [this] { openFile(); }; // the screen goes when a file is chosen
    welcomeOverlay->onOpenRecent = [this](const juce::File &file) { openRecentFile(file); };
  }
  welcomeOverlay->setRecentFiles(recentFiles);
  layOutStartScreen(*welcomeOverlay);
  welcomeOverlay->setVisible(true);
  welcomeOverlay->toFront(true);
  welcomeOverlay->grabKeyboardFocus();
}

void MainComponent::layOutStartScreen(juce::Component &overlay) {
  // 1 at the smallest window the cards fit in; 1.6 from 1760 x 1152 up.
  const float scale = juce::jlimit(1.0f, 1.6f,
                                   std::min(static_cast<float>(getWidth()) / 1100.0f,
                                            static_cast<float>(getHeight()) / 720.0f));
  overlay.setTransform(juce::AffineTransform::scale(scale));
  overlay.setBounds(0, 0, static_cast<int>(std::ceil(static_cast<float>(getWidth()) / scale)),
                    static_cast<int>(std::ceil(static_cast<float>(getHeight()) / scale)));
}

void MainComponent::hideWelcome() {
  if (welcomeOverlay == nullptr || !welcomeOverlay->isVisible())
    return;
  welcomeOverlay->setVisible(false);
  grabKeyboardFocus();
}

void MainComponent::dismissStartScreens() {
  hideWelcome();
  if (wizardIsShowing() || afterAudioLoaded != nullptr)
    closeWizard(false);
}

// ---------------------------------------------------------------------------
// The wizard: opening, closing, its files

void MainComponent::showWizard(bool fromWelcome) {
  if (isPluginMode() || isLoadingAudio.load() || wizardIsShowing())
    return;
  hideWelcome();
  const auto *project = getProject();
  const bool existing = project != nullptr && hasAnalyzedProject();
  wizardModel = existing ? wizard::Model::forProject(*project) : wizard::Model::fresh();
  wizardFromWelcome = fromWelcome;

  if (wizardOverlay == nullptr) {
    wizardOverlay = std::make_unique<WizardOverlay>(wizardModel);
    addChildComponent(*wizardOverlay);
    wizardOverlay->onChooseVoice = [this] { wizardChooseFile(true); };
    wizardOverlay->onChooseBacking = [this] { wizardChooseFile(false); };
    wizardOverlay->onFileDropped = [this](const juce::File &file, bool forVoice) {
      wizardTakeFile(file, forVoice);
    };
    wizardOverlay->onAligned = [this](bool on) {
      wizardModel.aligned = on && wizardModel.hasBacking();
      wizardOverlay->refresh();
    };
    wizardOverlay->onStart = [this] { wizardStart(); };
    wizardOverlay->onCancel = [this] { closeWizard(true); };
    wizardOverlay->onManual = [this] { closeWizard(false); };
    wizardOverlay->onTick = [this] { wizardTick(); };

    auto &keys = wizardOverlay->keyPage();
    keys.onKey = [this](int stretch, int root) {
      wizardModel.choose(stretch, root);
      wizardAudition(stretch, root);
      wizardOverlay->refresh();
    };
    keys.onMode = [this](ScaleMode mode) {
      // The same seven notes read from another tonic: the names change, the
      // stretches and what is chosen in them stay.
      wizardModel.mode = mode;
      wizardOverlay->refresh();
    };
    keys.onPlayStop = [this] { wizardPlayStop(); };
    keys.onSeek = [this](double seconds) {
      if (auto *engine = editorController ? editorController->getAudioEngine() : nullptr)
        engine->stopOverlay();
      setPlayhead(seconds, false);
    };
    keys.onAddChange = [this] {
      auto *engine = editorController ? editorController->getAudioEngine() : nullptr;
      const double now = isPlaying && engine != nullptr ? engine->getPosition()
                                                        : pianoRoll.getPlayheadTime();
      if (wizardModel.addChangeAt(now))
        wizardOverlay->refresh();
    };
    keys.onRemoveChange = [this](int stretch) {
      if (wizardModel.removeChange(stretch))
        wizardOverlay->refresh();
    };
    keys.onVoiceToo = [this](bool on) {
      wizardVoiceToo = on;
      wizardApplyMonitor();
    };
    keys.onScaleLevel = [this](float dB) {
      wizardScaleDb = dB;
      if (auto *engine = editorController ? editorController->getAudioEngine() : nullptr)
        engine->setOverlayGainDb(dB);
    };
    keys.onBack = [this] {
      wizardLeaveKeyCheck();
      showWizardPage(wizard::Page::files);
    };
    keys.onConfirm = [this] { wizardConfirm(); };
  }

  wizardOverlay->setVoiceName(existing ? project->getFilePath().getFileName()
                                       : juce::String());
  // Confirmed before, and the engine has it: nothing to ask.
  auto *engine = editorController ? editorController->getAudioEngine() : nullptr;
  if (wizardModel.nothingToAsk() && engine != nullptr && engine->hasBacking() &&
      loadedBackingFile == wizardModel.backing && !backingLoading && !backingMissing) {
    wizardAnalyse();
    return;
  }
  showWizardPage(wizard::Page::files);
}

void MainComponent::showWizardPage(wizard::Page page, const juce::String &message) {
  if (wizardOverlay == nullptr)
    return;
  wizardModel.page = page;
  wizardOverlay->setMessage(message);
  layOutStartScreen(*wizardOverlay);
  wizardOverlay->setVisible(true);
  wizardOverlay->toFront(true);
  wizardOverlay->refresh();
  wizardOverlay->grabKeyboardFocus();
}

void MainComponent::closeWizard(bool cancelled) {
  ++wizardGeneration; // what is on its way is for a wizard that is gone
  afterAudioLoaded = nullptr;
  wizardLeaveKeyCheck();
  if (wizardOverlay != nullptr)
    wizardOverlay->setVisible(false);
  // Cancelled before anything was opened: back to where it was started.
  if (cancelled && wizardFromWelcome && !hasAnalyzedProject() && !isLoadingAudio.load())
    showWelcome();
  else
    grabKeyboardFocus();
}

void MainComponent::wizardChooseFile(bool forVoice) {
  if (fileChooser != nullptr)
    return;
  // UI tests only: the files are named, so the choice runs without the
  // system's Open panel (which cannot be driven in the background).
  // Everything after the choice is the same.
  if (const char *named = std::getenv(forVoice ? "PITCHNET_WIZARD_VOICE_FILE"
                                               : "PITCHNET_WIZARD_BACKING_FILE")) {
    wizardTakeFile(juce::File(juce::String::fromUTF8(named)), forVoice);
    return;
  }
  // Beside the other file, most likely.
  juce::File start = forVoice ? wizardModel.backing : wizardModel.voice;
  if (start == juce::File{} && getProject() != nullptr)
    start = getProject()->getFilePath();
  fileChooser = std::make_unique<juce::FileChooser>(
      forVoice ? TR("dialog.select_audio") : TR("dialog.select_backing"),
      start == juce::File{} ? juce::File{} : start.getParentDirectory(),
      "*.wav;*.aif;*.aiff;*.flac;*.mp3;*.m4a;*.ogg");
  juce::Component::SafePointer<MainComponent> safeThis(this);
  fileChooser->launchAsync(
      juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
      [safeThis, forVoice](const juce::FileChooser &chooser) {
        if (safeThis == nullptr)
          return;
        const auto selected = chooser.getResult();
        safeThis->fileChooser.reset();
        safeThis->wizardTakeFile(selected, forVoice);
      });
}

void MainComponent::wizardTakeFile(const juce::File &file, bool forVoice) {
  if (!file.existsAsFile() || wizardOverlay == nullptr ||
      wizardModel.page != wizard::Page::files)
    return;
  if (forVoice && !wizardModel.existingProject)
    wizardModel.setVoice(file);
  else
    wizardModel.setBacking(file);
  wizardOverlay->setMessage({});
  wizardOverlay->refresh();
}

// ---------------------------------------------------------------------------
// From the files to the keys

void MainComponent::wizardStart() {
  if (!wizardModel.canAnalyse() || isLoadingAudio.load())
    return;
  if (wizardModel.existingProject) {
    wizardLinkBacking();
    return;
  }
  if (wizardModel.voice == wizardModel.backing) {
    showWizardPage(wizard::Page::files, TR("wizard.same_file"));
    return;
  }
  // The recording is loaded and analysed as one opened from the menu; the
  // wizard waits behind the progress, and goes on when the project shows.
  wizardOverlay->setVisible(false);
  const int generation = ++wizardGeneration;
  afterAudioLoaded = [this, generation](bool shown) {
    if (generation != wizardGeneration)
      return;
    if (!shown || !hasAnalyzedProject()) {
      showWizardPage(wizard::Page::files, TR("wizard.voice_failed"));
      return;
    }
    wizardModel.existingProject = true;
    wizardOverlay->setVoiceName(wizardModel.voice.getFileName());
    wizardLinkBacking();
  };
  loadAudioFile(wizardModel.voice);
}

void MainComponent::wizardLinkBacking() {
  auto *project = getProject();
  auto *engine = editorController ? editorController->getAudioEngine() : nullptr;
  if (project == nullptr || engine == nullptr) {
    closeWizard(false);
    return;
  }
  if (!wizardModel.hasBacking()) {
    wizardAnalyse(); // an open project, by its voice alone
    return;
  }
  showWizardPage(wizard::Page::analysing);
  // The tick on the files page is the user's word that the two start at the
  // same place; the project keeps it with the file.
  const auto markAligned = [this] {
    auto *current = getProject();
    if (current == nullptr)
      return;
    auto backing = current->getBacking();
    if (!backing.aligned) {
      backing.aligned = true;
      backing.audible = true; // given to the wizard to be heard with the voice
      current->setBacking(backing);
      current->setModified(true);
    }
    refreshMixerPanel();
  };
  if (project->getBacking().file == wizardModel.backing && loadedBackingFile == wizardModel.backing &&
      engine->hasBacking() && !backingLoading) {
    markAligned();
    wizardAnalyse();
    return;
  }
  const int generation = ++wizardGeneration;
  loadBackingAudio(wizardModel.backing, true, [this, generation, markAligned](bool loaded) {
    if (generation != wizardGeneration)
      return;
    if (!loaded) {
      showWizardPage(wizard::Page::files, TR("dialog.backing_unreadable"));
      return;
    }
    markAligned();
    wizardAnalyse();
  });
}

void MainComponent::wizardAnalyse() {
  showWizardPage(wizard::Page::analysing);
  const int generation = ++wizardGeneration;
  const juce::File file = wizardModel.backing;
  juce::Component::SafePointer<MainComponent> safeThis(this);
  juce::Thread::launch([safeThis, file, generation]() {
    // The accompaniment as it is on disk, for its pitch classes: the engine
    // has it at the device's rate, and a song takes a second or two here.
    auto chroma = std::make_shared<keyfinder::Chroma>();
    if (file != juce::File{}) {
      juce::AudioFormatManager formats;
      formats.registerBasicFormats();
      std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
      if (reader != nullptr && reader->lengthInSamples > 0 && reader->sampleRate > 0.0) {
        juce::AudioBuffer<float> audio(juce::jlimit(1, 2, static_cast<int>(reader->numChannels)),
                                       static_cast<int>(reader->lengthInSamples));
        reader->read(&audio, 0, audio.getNumSamples(), 0, true, true);
        *chroma = keyfinder::analyse(audio, reader->sampleRate);
      }
    }
    juce::MessageManager::callAsync([safeThis, generation, chroma]() {
      if (safeThis != nullptr && generation == safeThis->wizardGeneration)
        safeThis->wizardShowKeys(*chroma);
    });
  });
}

void MainComponent::wizardShowKeys(const keyfinder::Chroma &chroma) {
  const auto *project = getProject();
  if (project == nullptr || wizardOverlay == nullptr) {
    closeWizard(false);
    return;
  }
  const auto notes = keyfinder::vocalNotes(*project);
  const auto found = keyfinder::find(notes, chroma.empty() ? nullptr : &chroma);
  if (found.stretches.empty()) {
    showWizardPage(wizard::Page::files, TR("wizard.nothing_found"));
    return;
  }
  wizardModel.setResult(found);
  // Confirmed before: its own key and changes of key are what is chosen to
  // begin with (the candidates are still this analysis's).
  if (project->isKeyConfirmed()) {
    std::vector<std::pair<double, int>> spans;
    for (const auto &stretch : keyflags::stretches(*project))
      if (stretch.root >= 0)
        spans.push_back({stretch.startSeconds, stretch.root});
    wizardModel.adopt(spans, project->getPreferredScaleMode());
  }

  // The scale is played in the octave the voice sings in.
  double weight = 0.0, sum = 0.0;
  for (const auto &note : notes) {
    weight += note.weight;
    sum += note.weight * note.pitch;
  }
  wizardAroundMidi = weight > 0.0 ? sum / weight : 64.0;

  auto &keys = wizardOverlay->keyPage();
  // With no accompaniment the voice is all there is to hear the scale over.
  keys.setVoiceToo(wizardVoiceToo || !found.hasBacking, found.hasBacking);
  keys.setScaleLevelDb(wizardScaleDb);
  keys.setSounding(-1, 0);
  wizardSounding = false;
  wizardEnterKeyCheck();
  showWizardPage(wizard::Page::keyCheck);
  wizardTick();
}

// ---------------------------------------------------------------------------
// The key check: what is heard

void MainComponent::wizardEnterKeyCheck() {
  auto *engine = editorController ? editorController->getAudioEngine() : nullptr;
  if (engine == nullptr || wizardKeyCheckActive)
    return;
  if (isPlaying)
    pause();
  wizardKeyCheckActive = true;
  // A stretch is heard through, wherever the project's loop is.
  engine->setLoopEnabled(false);
  wizardApplyMonitor();
}

void MainComponent::wizardLeaveKeyCheck() {
  if (!wizardKeyCheckActive)
    return;
  wizardKeyCheckActive = false;
  wizardSounding = false;
  auto *engine = editorController ? editorController->getAudioEngine() : nullptr;
  if (engine == nullptr)
    return;
  if (isPlaying)
    pause();
  engine->stopOverlay();
  // Back to what the project says: both faders, the accompaniment's switch,
  // the loop.
  engine->setVoiceAudible(true);
  if (const auto *project = getProject())
    engine->setLoopEnabled(project->getLoopRange().enabled);
  syncMixerWithProject();
}

void MainComponent::wizardApplyMonitor() {
  auto *engine = editorController ? editorController->getAudioEngine() : nullptr;
  if (engine == nullptr || !wizardKeyCheckActive)
    return;
  const bool hasBacking = wizardModel.getResult().hasBacking && engine->hasBacking();
  engine->setBackingAudible(hasBacking);
  engine->setVoiceAudible(!hasBacking || wizardVoiceToo);
}

void MainComponent::wizardPlayFrom(double seconds) {
  auto *engine = editorController ? editorController->getAudioEngine() : nullptr;
  if (engine == nullptr || getProject() == nullptr)
    return;
  wizardApplyMonitor();
  setPlayhead(seconds, false);
  isPlaying = true;
  toolbar.setPlaying(true);
  pianoRoll.setPlaybackActive(true);
  engine->play();
}

void MainComponent::wizardPlayStop() {
  if (!wizardKeyCheckActive)
    return;
  if (isPlaying) {
    pause(); // the scale ends with the transport
    return;
  }
  wizardPlayFrom(pianoRoll.getPlayheadTime());
}

void MainComponent::wizardAudition(int stretch, int root) {
  auto *engine = editorController ? editorController->getAudioEngine() : nullptr;
  const auto *project = getProject();
  const auto &stretches = wizardModel.getStretches();
  if (engine == nullptr || project == nullptr || !wizardKeyCheckActive || stretch < 0 ||
      stretch >= static_cast<int>(stretches.size()))
    return;
  const auto &span = stretches[static_cast<size_t>(stretch)];

  // Where the transport is, if the scale fits into this stretch from there;
  // otherwise a second before the stretch's first sung note.
  const double now = isPlaying ? engine->getPosition() : pianoRoll.getPlayheadTime();
  const bool carryOn = isPlaying && now >= span.startSeconds &&
                       now + kAuditionSeconds - kAuditionLeadSeconds <= span.endSeconds;
  double start = now;
  if (!carryOn && !(now >= span.startSeconds && now + kAuditionSeconds <= span.endSeconds)) {
    start = span.startSeconds;
    for (const auto &note : keyfinder::vocalNotes(*project))
      if (note.startSeconds >= span.startSeconds && note.startSeconds < span.endSeconds) {
        start = std::max(span.startSeconds, note.startSeconds - kAuditionBeforeVoice);
        break;
      }
    start = std::max(span.startSeconds, std::min(start, span.endSeconds - kAuditionSeconds));
  }

  const auto &found = wizardModel.getResult();
  const double tune = found.hasBacking ? found.tuneSemitones : found.vocalOffsetSemitones;
  // The chosen mode's own scale, from its tonic up to the octave.
  std::vector<int> degrees;
  for (int semitone = 0; semitone < 12; ++semitone)
    if (ScaleUtils::isPitchClassInScale(wizardModel.mode, semitone, 0))
      degrees.push_back(semitone);
  degrees.push_back(12);
  const auto tone = keyfinder::scaleTone(
      keyfinder::scaleRootNear(wizardModel.tonicOf(root), wizardAroundMidi), tune, kToneRate,
      keyfinder::kScalePeak, degrees);
  // Started for this: the song alone for a moment, then the scale.
  const int lead = carryOn ? 0 : static_cast<int>(kAuditionLeadSeconds * kToneRate);
  juce::AudioBuffer<float> sound(1, lead + tone.getNumSamples());
  sound.clear();
  sound.copyFrom(0, lead, tone, 0, 0, tone.getNumSamples());

  if (!carryOn)
    wizardPlayFrom(start);
  engine->setOverlayGainDb(wizardScaleDb);
  engine->playOverlay(sound, kToneRate);
  wizardSounding = true;
  wizardOverlay->keyPage().setSounding(stretch, root);
}

void MainComponent::wizardTick() {
  auto *engine = editorController ? editorController->getAudioEngine() : nullptr;
  if (engine == nullptr || wizardOverlay == nullptr || wizardModel.page != wizard::Page::keyCheck)
    return;
  auto &keys = wizardOverlay->keyPage();
  keys.setPlayback(isPlaying ? engine->getPosition() : pianoRoll.getPlayheadTime(), isPlaying);
  if (wizardSounding && !engine->isOverlayPlaying()) {
    wizardSounding = false;
    keys.setSounding(-1, 0);
  }
}

// ---------------------------------------------------------------------------
// The answer

void MainComponent::wizardConfirm() {
  auto *project = getProject();
  if (project == nullptr || wizardModel.getStretches().empty()) {
    closeWizard(false);
    return;
  }
  wizardLeaveKeyCheck();
  const auto before = KeyState::of(*project);
  const auto after = wizardModel.confirmed(*project);
  project->setKeyConfirmed(true);
  // The rows of the grid shaded by the scale (the panel's Grid switch), so
  // the key just confirmed - and each stretch's own - is seen at once.
  project->setScaleMode(after.useProjectKey ? after.projectMode : after.regionMode);
  parameterPanel.setProject(project);
  if (!(before == after)) {
    after.applyTo(*project);
    project->setModified(true);
    if (undoManager)
      undoManager->addAction(
          std::make_unique<KeyChangeAction>(*project, before, after, TR("wizard.confirm_key")));
    refreshKeyDisplays();
    parameterPanel.updateGlobalSliders();
    pianoRoll.repaint();
    if (commandManager)
      commandManager->commandStatusChanged();
  }
  refreshKeyDisplays();
  pianoRoll.repaint();
  showWizardPage(wizard::Page::nextStep);
}

void MainComponent::refreshKeyDisplays() {
  auto *project = getProject();
  if (project == nullptr)
    return;
  pianoRoll.refreshScaleFromProject();
  toolbar.setProject(project); // its key label is read from the project
  refreshToolbarKey();
}

void MainComponent::refreshToolbarKey() {
  const auto *project = getProject();
  if (project == nullptr || project->getKeyChanges().empty()) {
    toolbar.setKeyStretch(0, 0);
    return;
  }
  // The change of key last passed where the playhead is on the recording.
  const int frame = keyflags::sourceFrameAtSeconds(project->getAudioData(),
                                                   std::max(0.0, pianoRoll.getCursorTime()));
  const auto &changes = project->getKeyChanges();
  int stretch = 0;
  while (stretch < static_cast<int>(changes.size()) &&
         changes[static_cast<size_t>(stretch)].sourceFrame <= frame)
    ++stretch;
  toolbar.setKeyStretch(stretch, stretch > 0 ? changes[static_cast<size_t>(stretch - 1)].root : 0);
}
