// =============================================================================
// MainComponent member functions for project/audio file I/O.
// Split from MainComponent.cpp to reduce file size.
// =============================================================================

#include "../../Audio/SourceAudio.h"
#include "../../Audio/RecordingChange.h"
#include "../MainComponent.h"
#include "../../Audio/BreathSibilanceAnalysis.h"
#include "../../Models/ProjectSerializer.h"
#include "../../Utils/AppLogger.h"
#include "../../Utils/Constants.h"
#include "../../Utils/Localization.h"
#include "../../Utils/MelSpectrogram.h"
#include "../../Utils/PitchCurveProcessor.h"
#include "../../Utils/PlatformPaths.h"
#include "../../Utils/SHA256Utils.h"

void MainComponent::openProjectFile(const juce::File &file) {
  if (isLoadingAudio.load())
    return;
  addRecentFile(file);

  // A .pitchnet embeds the whole rendered waveform (hundreds of MB of JSON for
  // a full song), so read it off the message thread and show progress; the
  // window froze with no sign of life before.
  isLoadingAudio = true;
  loadingProgress = 0.0;
  shownLoadingProgress = 0.0;
  {
    const juce::ScopedLock sl(loadingMessageLock);
    loadingMessage = TR("progress.loading_project_title");
  }
  analysisProgressPopup.setTitle(TR("progress.loading_project_title"));
  showAnalysisProgress(0.0);

  juce::Component::SafePointer<MainComponent> safeThis(this);
  juce::Thread::launch([safeThis, file]() {
    auto loadedProject = std::make_shared<Project>();
    const bool ok = ProjectSerializer::loadFromFile(
        *loadedProject, file, [safeThis](double p) {
          juce::MessageManager::callAsync([safeThis, p]() {
            if (safeThis != nullptr)
              safeThis->loadingProgress = p;
          });
        });
    // thread_local in the serializer: read it on this thread.
    const bool legacy = !ok && ProjectSerializer::lastLoadWasLegacy();
    juce::MessageManager::callAsync([safeThis, file, loadedProject, ok, legacy]() {
      if (safeThis == nullptr)
        return;
      safeThis->isLoadingAudio = false;
      safeThis->analysisProgressPopup.setTitle({});
      safeThis->hideAnalysisProgress();
      if (!ok) {
        StyledMessageBox::show(safeThis.getComponent(), TR("dialog.open_failed"),
                               (legacy ? TR("dialog.project_legacy")
                                       : TR("dialog.project_load_failed")) +
                                   "\n" + file.getFullPathName(),
                               StyledMessageBox::WarningIcon);
        return;
      }
      safeThis->openLoadedProject(file, loadedProject);
    });
  });
}

void MainComponent::openLoadedProject(const juce::File &file,
                                      std::shared_ptr<Project> loadedProject) {
  loadedProject->setProjectFilePath(file);

  // A project that links its recording (spec Q11) is rebuilt from it.
  if (loadedProject->linksRecording()) {
    openLinkedProject(file, loadedProject);
    return;
  }

  // .pitchnet files that embed the rendered waveform and analysis data open
  // as saved, even if the original audio file has moved or changed.
  if (loadedProject->getAudioData().waveform.getNumSamples() > 0) {
    showOpenedProject(file, *loadedProject, true);
    return;
  }

  auto continueOpenWithAudio = [this, loadedProject](const juce::File &audioFile) {
    if (!audioFile.existsAsFile()) {
      StyledMessageBox::show(this, TR("dialog.open_failed"),
                             TR("dialog.project_audio_missing") + "\n" +
                                 audioFile.getFullPathName(),
                             StyledMessageBox::WarningIcon);
      return;
    }

    loadedProject->setFilePath(audioFile);

    const juce::String currentAudioSha = SHA256Utils::fileSHA256(audioFile);
    const juce::String savedAudioSha = loadedProject->getAudioSha256();
    const bool shaMatched = savedAudioSha.isNotEmpty() &&
                            savedAudioSha.equalsIgnoreCase(currentAudioSha);

    auto proceedWithProject =
        [this, loadedProject, audioFile, currentAudioSha](bool reanalyze) {
          if (reanalyze) {
            loadAudioFile(audioFile, loadedProject->getMidiReference());
            return;
          }

          isLoadingAudio = true;
          loadingProgress = 0.0;
          {
            const juce::ScopedLock sl(loadingMessageLock);
            loadingMessage = TR("progress.loading_audio");
          }
          toolbar.hideProgress();
          showAnalysisProgress(0.0);

          juce::Component::SafePointer<MainComponent> safeThis(this);
          fileManager->loadAudioFileAsync(
              audioFile,
              [safeThis](double p, const juce::String &msg) {
                if (safeThis == nullptr)
                  return;
                safeThis->loadingProgress = juce::jlimit(0.0, 1.0, p);
                const juce::ScopedLock sl(safeThis->loadingMessageLock);
                safeThis->loadingMessage = msg;
              },
              [safeThis, loadedProject, currentAudioSha](
                  juce::AudioBuffer<float> &&buffer, int sampleRate,
                  const juce::File &loadedFile) mutable {
                if (safeThis == nullptr)
                  return;

                auto projectToUse = std::make_unique<Project>(*loadedProject);
                projectToUse->setFilePath(loadedFile);
                projectToUse->setAudioSha256(currentAudioSha);

                auto &audioData = projectToUse->getAudioData();
                audioData.waveform = std::move(buffer);
                audioData.sampleRate = sampleRate;
                audioData.originalWaveform.makeCopyOf(audioData.waveform);
                // Playback and export use the recording itself (spec Q5).
                sourceaudio::load(audioData, loadedFile);

                // Recompute mel only; keep pitch/note edits from project file.
                if (audioData.waveform.getNumSamples() > 0) {
                  const float *samples = audioData.waveform.getReadPointer(0);
                  const int numSamples = audioData.waveform.getNumSamples();
                  MelSpectrogram melComputer(audioData.sampleRate, N_FFT,
                                             HOP_SIZE, NUM_MELS, FMIN, FMAX);
                  audioData.melSpectrogram =
                      melComputer.compute(samples, numSamples);
                }

                if (audioData.voicedMask.empty() && !audioData.f0.empty()) {
                  audioData.voicedMask.resize(audioData.f0.size(), false);
                  for (size_t i = 0; i < audioData.f0.size(); ++i)
                    audioData.voicedMask[i] = audioData.f0[i] > 0.0f;
                }
                if (audioData.rawF0.empty()) {
                  audioData.rawF0 = audioData.f0;
                  if (audioData.voicedMask.size() == audioData.rawF0.size()) {
                    for (size_t i = 0; i < audioData.rawF0.size(); ++i)
                      if (!audioData.voicedMask[i])
                        audioData.rawF0[i] = 0.0f;
                  }
                }
                if (audioData.cleanedF0.empty())
                  audioData.cleanedF0 = audioData.rawF0;
                if (audioData.denseF0.empty())
                  audioData.denseF0 = audioData.f0;

                // Not saved with the project: relabel from the reloaded audio.
                breathsibilance::ensure(audioData);

                if (audioData.basePitch.empty() || audioData.deltaPitch.empty()) {
                  if (!audioData.f0.empty()) {
                    PitchCurveProcessor::rebuildCurvesFromSource(*projectToUse,
                                                                 audioData.f0);
                  } else if (!audioData.melSpectrogram.empty()) {
                    // Legacy project fallback: rebuild base from notes and use
                    // zero delta so reopening can still synthesize edited notes.
                    PitchCurveProcessor::rebuildBaseFromNotes(*projectToUse);
                  }
                } else {
                  PitchCurveProcessor::composeF0InPlace(*projectToUse,
                                                        /*applyUvMask=*/false);
                }

                if (safeThis->undoManager)
                  safeThis->undoManager->clear();

                if (safeThis->editorController)
                  safeThis->editorController->setProject(std::move(projectToUse));

                auto *project = safeThis->getProject();
                if (!project) {
                  safeThis->isLoadingAudio = false;
                  return;
                }

                safeThis->pianoRoll.setProject(project);
                safeThis->pianoRollView.setProject(project);
                safeThis->restoreProjectPresentation();
                safeThis->savedProjectSnapshot = safeThis->projectSaveSnapshot();
                safeThis->parameterPanel.setProject(project);
                safeThis->toolbar.setTotalTime(project->getAudioData().getDuration());
                safeThis->toolbar.setLoopEnabled(project->getLoopRange().enabled);
                safeThis->toolbar.setTransportEnabled(true);

                auto &activeAudioData = project->getAudioData();
                if (safeThis->isPluginMode()) {
                  // plugin mode: no audio engine
                } else if (auto *engine = safeThis->editorController
                                               ? safeThis->editorController->getAudioEngine()
                                               : nullptr) {
                  try {
                    engine->loadProjectAudio(activeAudioData);
                    const auto &loopRange = project->getLoopRange();
                    if (loopRange.enabled)
                      engine->setLoopRange(loopRange.startSeconds,
                                           loopRange.endSeconds);
                    engine->setLoopEnabled(loopRange.enabled);
                    engine->setVolumeDb(project->getVolume());
                  } catch (...) {
                  }
                }

                if (auto *vocoder = safeThis->editorController
                                        ? safeThis->editorController->getVocoder()
                                        : nullptr;
                    vocoder && !vocoder->isLoaded()) {
                  auto modelPath = PlatformPaths::getVocoderModelFile();
                  if (modelPath.exists()) {
                    if (!vocoder->loadModel(modelPath)) {
                      juce::AlertWindow::showMessageBoxAsync(
                          juce::AlertWindow::WarningIcon, TR("error.inference_failed"),
                          TR("error.vocoder_load_failed") + "\n" +
                              modelPath.getFullPathName() + "\n\n" +
                              TR("error.check_model_install"));
                      safeThis->isLoadingAudio = false;
                      return;
                    }
                  } else {
                    juce::AlertWindow::showMessageBoxAsync(
                        juce::AlertWindow::WarningIcon, TR("error.missing_model"),
                        TR("error.vocoder_not_found") + "\n" +
                            modelPath.getFullPathName() + "\n\n" +
                            TR("error.install_models"));
                    safeThis->isLoadingAudio = false;
                    return;
                  }
                }

                // Skip full re-analysis; run vocoder from loaded edits.
                const int totalFrames = std::max(
                    static_cast<int>(activeAudioData.melSpectrogram.size()),
                    std::max(static_cast<int>(activeAudioData.f0.size()),
                             static_cast<int>(activeAudioData.basePitch.size())));
                if (totalFrames > 0) {
                  project->setF0DirtyRange(0, totalFrames);

                  safeThis->toolbar.showProgress(TR("progress.synthesizing"));
                  safeThis->toolbar.setProgress(-1.0f);
                  safeThis->setToolGroupEnabled(false);

                  safeThis->editorController->resynthesizeIncrementalAsync(
                      *project,
                      [safeThis](const juce::String &message) {
                        if (safeThis == nullptr)
                          return;
                        safeThis->toolbar.showProgress(message);
                      },
                      [safeThis](bool success) {
                        if (safeThis == nullptr)
                          return;

                        safeThis->setToolGroupEnabled(true);
                        safeThis->toolbar.hideProgress();
                        safeThis->isLoadingAudio = false;
                        safeThis->repaint();

                        if (!success) {
                          StyledMessageBox::show(
                              safeThis.getComponent(), TR("dialog.open_warning"),
                              TR("dialog.pitch_edits_failed"),
                              StyledMessageBox::WarningIcon);
                          return;
                        }

                        if (safeThis->isPluginMode())
                          safeThis->notifyProjectDataChanged();
                      },
                      safeThis->pendingIncrementalResynth,
                      safeThis->isPluginMode());
                  return;
                }

                safeThis->repaint();
                safeThis->isLoadingAudio = false;
                if (safeThis->isPluginMode())
                  safeThis->notifyProjectDataChanged();
              });
        };

    if (!shaMatched) {
      juce::AlertWindow::showOkCancelBox(
          juce::AlertWindow::WarningIcon, TR("dialog.audio_changed"),
          TR("dialog.audio_hash_mismatch") + "\n" +
              audioFile.getFullPathName() + "\n\n" +
              TR("dialog.reanalyze_question"),
          TR("dialog.reanalyze"), TR("dialog.use_saved_edits"), this,
          juce::ModalCallbackFunction::create([proceedWithProject](int result) {
            proceedWithProject(result != 0);
          }));
      return;
    }

    proceedWithProject(false);
  };

  const juce::File audioFile = loadedProject->getFilePath();
  if (!audioFile.existsAsFile()) {
    juce::AlertWindow::showOkCancelBox(
        juce::AlertWindow::WarningIcon, TR("dialog.audio_missing"),
        TR("dialog.project_audio_missing") + "\n" +
            audioFile.getFullPathName() + "\n\n" +
            TR("dialog.locate_question"),
        TR("dialog.locate_audio"), TR("dialog.cancel"), this,
        juce::ModalCallbackFunction::create(
            [this, continueOpenWithAudio](int result) {
              if (result == 0)
                return;
              if (fileChooser != nullptr)
                return;
              fileChooser = std::make_unique<juce::FileChooser>(
                  TR("dialog.select_audio"), juce::File{},
                  "*.wav;*.mp3;*.flac;*.aiff;*.ogg;*.m4a");
              auto chooserFlags = juce::FileBrowserComponent::openMode |
                                  juce::FileBrowserComponent::canSelectFiles;
              juce::Component::SafePointer<MainComponent> safeThis(this);
              fileChooser->launchAsync(
                  chooserFlags, [safeThis, continueOpenWithAudio](
                                    const juce::FileChooser &fc) {
                    if (safeThis == nullptr)
                      return;
                    auto selected = fc.getResult();
                    safeThis->fileChooser.reset();
                    if (selected.existsAsFile())
                      continueOpenWithAudio(selected);
                  });
            }));
    return;
  }

  continueOpenWithAudio(audioFile);
}

void MainComponent::showOpenedProject(const juce::File &file,
                                      const Project &openedProject,
                                      bool findRecording) {
  // The history belongs to the project being replaced: its actions point
  // into that project's curves (NotePitchDragAction keeps &audioData.f0),
  // so an undo past the open would write into freed memory. The branch
  // that resynthesizes below clears it for the same reason.
  if (undoManager)
    undoManager->clear();
  if (editorController)
    editorController->setProject(std::make_unique<Project>(openedProject));

  auto *project = getProject();
  if (!project)
    return;

  pianoRoll.setProject(project);
  pianoRollView.setProject(project);
  restoreProjectPresentation();
  parameterPanel.setProject(project);
  parameterPanel.setSelectedNote(nullptr);
  toolbar.setTotalTime(project->getAudioData().getDuration());
  toolbar.setLoopEnabled(project->getLoopRange().enabled);
  toolbar.setTransportEnabled(true);

  // Playback plays the recording itself (spec Q5), so find it: the saved
  // path, else a recording beside the project with the project's name. A
  // linked project has read it already.
  const bool recordingAttached =
      !findRecording || (!isPluginMode() && attachProjectRecording(*project, file));
  // As saved, after the recording is attached: an embedded project then
  // learns its recording's fingerprint, which is not an edit.
  savedProjectSnapshot = projectSaveSnapshot();

  if (!isPluginMode()) {
    if (auto *engine = editorController ? editorController->getAudioEngine()
                                         : nullptr) {
      try {
        engine->loadProjectAudio(project->getAudioData());
        const auto &loopRange = project->getLoopRange();
        if (loopRange.enabled)
          engine->setLoopRange(loopRange.startSeconds, loopRange.endSeconds);
        engine->setLoopEnabled(loopRange.enabled);
        engine->setVolumeDb(project->getVolume());
      } catch (...) {
      }
    }
  }

  if (hasAnalyzedProject())
    fitAnalyzedPitchRangeToView(*project);
  // Back where the project was last saved (overrides the fit above).
  restoreViewFromProject();
  repaint();
  if (isPluginMode())
    notifyProjectDataChanged();
  if (!isPluginMode() && !recordingAttached)
    askForProjectRecording();
}

void MainComponent::openLinkedProject(const juce::File &file,
                                      std::shared_ptr<Project> loadedProject,
                                      const juce::File &chosen) {
  // The saved path, else a recording beside the project with its name.
  juce::File audioFile = chosen;
  if (audioFile == juce::File{}) {
    juce::Array<juce::File> candidates{loadedProject->getFilePath()};
    for (const auto *extension : {".wav", ".aif", ".aiff", ".flac"})
      candidates.add(file.withFileExtension(extension));
    for (const auto &candidate : candidates)
      if (candidate.existsAsFile()) {
        audioFile = candidate;
        break;
      }
  }

  if (!audioFile.existsAsFile()) {
    juce::Component::SafePointer<MainComponent> safeThis(this);
    juce::AlertWindow::showOkCancelBox(
        juce::AlertWindow::WarningIcon, TR("dialog.audio_missing"),
        TR("dialog.project_audio_missing") + "\n" +
            loadedProject->getFilePath().getFullPathName() + "\n\n" +
            TR("dialog.locate_question"),
        TR("dialog.locate_audio"), TR("dialog.cancel"), this,
        juce::ModalCallbackFunction::create([safeThis, file, loadedProject](int result) {
          if (safeThis == nullptr || result == 0 || safeThis->fileChooser != nullptr)
            return;
          safeThis->fileChooser = std::make_unique<juce::FileChooser>(
              TR("dialog.select_audio"), juce::File{}, "*.wav;*.aif;*.aiff;*.flac");
          safeThis->fileChooser->launchAsync(
              juce::FileBrowserComponent::openMode |
                  juce::FileBrowserComponent::canSelectFiles,
              [safeThis, file, loadedProject](const juce::FileChooser &fc) {
                if (safeThis == nullptr)
                  return;
                const auto selected = fc.getResult();
                safeThis->fileChooser.reset();
                if (selected.existsAsFile())
                  safeThis->openLinkedProject(file, loadedProject, selected);
              });
        }));
    return;
  }

  // Every edit is placed by frame: a recording of another rate, channel
  // count or length cannot carry them (spec Q11).
  const auto &saved = loadedProject->getRecordingFingerprint();
  const auto found = recordingchange::shapeOf(audioFile);
  if (!saved.sameShape(found)) {
    StyledMessageBox::show(this, TR("dialog.open_failed"),
                           TR("dialog.recording_other_shape") + "\n" +
                               audioFile.getFullPathName() + "\n\n" +
                               TR("dialog.recording_shape_project") + " " +
                               recordingchange::describe(saved) + "\n" +
                               TR("dialog.recording_shape_file") + " " +
                               recordingchange::describe(found),
                           StyledMessageBox::WarningIcon);
    return;
  }

  isLoadingAudio = true;
  loadingProgress = 0.0;
  {
    const juce::ScopedLock sl(loadingMessageLock);
    loadingMessage = TR("progress.reading_recording");
  }
  analysisProgressPopup.setTitle(TR("progress.loading_project_title"));
  showAnalysisProgress(0.0);

  juce::Component::SafePointer<MainComponent> safeThis(this);
  auto *controller = editorController;
  juce::Thread::launch([safeThis, controller, file, audioFile, loadedProject]() {
    // The pitch detector is only needed where the recording changed; wait
    // for its models then (they load in the background at startup).
    const recordingchange::Detect detect =
        [controller](const float *samples, int numSamples, int frames,
                     std::vector<float> &f0, std::vector<bool> &voiced) {
          if (controller == nullptr)
            return false;
          for (int waited = 0; !controller->isSelectedPitchDetectorLoaded() && waited < 600;
               ++waited)
            juce::Thread::sleep(100);
          return controller->detectPitch(samples, numSamples, frames, f0, voiced);
        };
    const auto opened = recordingchange::openLinked(*loadedProject, audioFile, detect);
    juce::MessageManager::callAsync([safeThis, file, loadedProject, opened]() {
      if (safeThis == nullptr)
        return;
      safeThis->analysisProgressPopup.setTitle({});
      safeThis->hideAnalysisProgress();
      if (opened.result != recordingchange::Opened::Result::opened) {
        safeThis->isLoadingAudio = false;
        StyledMessageBox::show(safeThis.getComponent(), TR("dialog.open_failed"),
                               TR("dialog.project_load_failed") + "\n" +
                                   loadedProject->getFilePath().getFullPathName(),
                               StyledMessageBox::WarningIcon);
        return;
      }
      safeThis->showOpenedProject(file, *loadedProject, false);
      if (!opened.changed) {
        safeThis->isLoadingAudio = false;
        return;
      }
      // Changed outside: the project now differs from the file (new pitch
      // where it changed, the new fingerprint), so closing asks to save.
      safeThis->savedProjectSnapshot = {};
      const auto report = TR("dialog.recording_changed_detail")
                              .replace("%1", juce::String(static_cast<int>(
                                                 opened.update.audioFrames.size())))
                              .replace("%2", juce::String(opened.update.notesToRender)) +
                          (opened.update.detected ? juce::String()
                                                  : "\n" + TR("dialog.recording_changed_no_pitch"));
      auto *project = safeThis->getProject();
      if (project == nullptr || opened.update.notesToRender == 0) {
        safeThis->isLoadingAudio = false;
        StyledMessageBox::show(safeThis.getComponent(), TR("dialog.recording_changed"),
                               report, StyledMessageBox::InfoIcon);
        return;
      }
      safeThis->toolbar.showProgress(TR("progress.synthesizing"));
      safeThis->toolbar.setProgress(-1.0f);
      safeThis->setToolGroupEnabled(false);
      safeThis->editorController->resynthesizeIncrementalAsync(
          *project,
          [safeThis](const juce::String &message) {
            if (safeThis != nullptr)
              safeThis->toolbar.showProgress(message);
          },
          [safeThis, report](bool success) {
            if (safeThis == nullptr)
              return;
            safeThis->setToolGroupEnabled(true);
            safeThis->toolbar.hideProgress();
            safeThis->isLoadingAudio = false;
            safeThis->repaint();
            StyledMessageBox::show(safeThis.getComponent(), TR("dialog.recording_changed"),
                                   success ? report : TR("dialog.pitch_edits_failed"),
                                   success ? StyledMessageBox::InfoIcon
                                           : StyledMessageBox::WarningIcon);
          },
          safeThis->pendingIncrementalResynth, safeThis->isPluginMode());
    });
  });
}

bool MainComponent::attachProjectRecording(Project &project,
                                           const juce::File &projectFile,
                                           const juce::File &chosen) {
  return sourceaudio::attachRecording(project, projectFile, chosen);
}

void MainComponent::askForProjectRecording() {
  juce::AlertWindow::showOkCancelBox(
      juce::AlertWindow::WarningIcon, TR("dialog.audio_missing"),
      TR("dialog.recording_missing_playback"), TR("dialog.locate_audio"),
      TR("dialog.cancel"), this,
      juce::ModalCallbackFunction::create([this](int result) {
        if (result == 0 || fileChooser != nullptr)
          return;
        fileChooser = std::make_unique<juce::FileChooser>(
            TR("dialog.select_audio"), juce::File{},
            "*.wav;*.aif;*.aiff;*.flac");
        juce::Component::SafePointer<MainComponent> safeThis(this);
        fileChooser->launchAsync(
            juce::FileBrowserComponent::openMode |
                juce::FileBrowserComponent::canSelectFiles,
            [safeThis](const juce::FileChooser &fc) {
              if (safeThis == nullptr)
                return;
              const auto selected = fc.getResult();
              safeThis->fileChooser.reset();
              auto *project = safeThis->getProject();
              if (project == nullptr || !selected.existsAsFile())
                return;
              if (!safeThis->attachProjectRecording(
                      *project, project->getProjectFilePath(), selected)) {
                StyledMessageBox::show(safeThis.getComponent(),
                                       TR("dialog.open_failed"),
                                       TR("dialog.project_audio_missing") +
                                           "\n" + selected.getFullPathName(),
                                       StyledMessageBox::WarningIcon);
                return;
              }
              if (auto *engine = safeThis->editorController
                                     ? safeThis->editorController->getAudioEngine()
                                     : nullptr)
                engine->loadProjectAudio(project->getAudioData());
            });
      }));
}

void MainComponent::loadAudioFile(const juce::File &file,
                                  std::optional<MidiReference::State> reference) {
  if (isLoadingAudio.load())
    return;
  addRecentFile(file);

  isLoadingAudio = true;
  loadingProgress = 0.0;
  {
    const juce::ScopedLock sl(loadingMessageLock);
    loadingMessage = TR("progress.loading_audio");
  }
  toolbar.hideProgress();
  clearProjectForNewLoad(std::move(reference));
  showAnalysisProgress(0.0);

  juce::Component::SafePointer<MainComponent> safeThis(this);
  if (!editorController) {
    isLoadingAudio = false;
    return;
  }

  const auto *loadingProject = getProject();
  const auto loadingParameters = loadingProject ? loadingProject->getMacroParameters() : nullptr;
  editorController->loadAudioFileAsync(
      file,
      [safeThis](double p, const juce::String &msg) {
        if (safeThis == nullptr)
          return;
        safeThis->loadingProgress = juce::jlimit(0.0, 1.0, p);
        const juce::ScopedLock sl(safeThis->loadingMessageLock);
        safeThis->loadingMessage = msg;
      },
      [safeThis, loadingParameters](const juce::AudioBuffer<float> &original) {
        if (safeThis == nullptr)
          return;

        // Clear undo history before replacing project to avoid dangling pointers
        if (safeThis->undoManager)
          safeThis->undoManager->clear();

        auto *project = safeThis->getProject();
        if (!project)
          return;
        if (loadingParameters)
          project->setMidiReference(loadingParameters->midiReference);
        if (!safeThis->isPluginMode())
          project->setTimelineDisplayMode(TimelineDisplayMode::Time);

        // Update UI
        safeThis->pianoRoll.setProject(project);
        safeThis->pianoRollView.setProject(project);
        safeThis->parameterPanel.setProject(project);
        safeThis->toolbar.setTotalTime(project->getAudioData().getDuration());
        safeThis->toolbar.setLoopEnabled(project->getLoopRange().enabled);
        safeThis->toolbar.setTransportEnabled(true);

        auto &audioData = project->getAudioData();

        if (safeThis->isPluginMode()) {
          // plugin mode: no audio engine
        } else if (auto *engine = safeThis->editorController
                                       ? safeThis->editorController->getAudioEngine()
                                       : nullptr) {
          try {
            engine->loadProjectAudio(audioData);
            const auto &loopRange = project->getLoopRange();
            if (loopRange.enabled)
              engine->setLoopRange(loopRange.startSeconds,
                                   loopRange.endSeconds);
            engine->setLoopEnabled(loopRange.enabled);
            engine->setVolumeDb(project->getVolume());
          } catch (...) {
          }
        }

        safeThis->fitAnalyzedPitchRangeToView(*project);
        safeThis->restoreViewFromProject();

        if (auto *vocoder = safeThis->editorController
                                ? safeThis->editorController->getVocoder()
                                : nullptr;
            vocoder && !vocoder->isLoaded()) {
          auto modelPath = PlatformPaths::getVocoderModelFile();
          if (modelPath.exists()) {
            if (!vocoder->loadModel(modelPath)) {
              juce::AlertWindow::showMessageBoxAsync(
                  juce::AlertWindow::WarningIcon, TR("error.inference_failed"),
                  TR("error.vocoder_load_failed") + "\n" +
                      modelPath.getFullPathName() + "\n\n" +
                      TR("error.check_model_install"));
              return;
            }
          } else {
            juce::AlertWindow::showMessageBoxAsync(
                juce::AlertWindow::WarningIcon, TR("error.missing_model"),
                TR("error.vocoder_not_found") + "\n" +
                    modelPath.getFullPathName() + "\n\n" +
                    TR("error.install_models"));
            return;
          }
        }

        safeThis->repaint();
        safeThis->isLoadingAudio = false;

        if (safeThis->isPluginMode())
          safeThis->notifyProjectDataChanged();
      },
      [safeThis]() {
        if (safeThis == nullptr)
          return;
        safeThis->isLoadingAudio = false;
      });
}

void MainComponent::clearProjectForNewLoad(std::optional<MidiReference::State> reference) {
  isPlaying = false;
  pendingCursorTime.store(0.0);
  hasPendingCursorUpdate.store(false);

  if (undoManager)
    undoManager->clear();

  if (auto *audioEngine = editorController ? editorController->getAudioEngine() : nullptr) {
    audioEngine->stop();
    audioEngine->seek(0.0);
    audioEngine->clearLoopRange();
  }

  if (editorController) {
    auto emptyProject = std::make_unique<Project>();
    if (!reference && getProject())
      reference = getProject()->getMidiReference();
    if (reference)
      emptyProject->setMidiReference(std::move(*reference));
    if (!isPluginMode())
      emptyProject->setTimelineDisplayMode(TimelineDisplayMode::Time);
    editorController->setProject(std::move(emptyProject));
  }

  auto *project = getProject();
  pianoRoll.setProject(project);
  pianoRollView.setProject(project);
  parameterPanel.setProject(project);
  parameterPanel.setSelectedNote(nullptr);
  toolbar.setPlaying(false);
  pianoRoll.setPlaybackActive(false);
  toolbar.setCurrentTime(0.0);
  toolbar.setTotalTime(0.0);
  toolbar.setLoopEnabled(false);
  toolbar.setTransportEnabled(isPluginMode());
  pianoRoll.setCursorTime(0.0);
}
