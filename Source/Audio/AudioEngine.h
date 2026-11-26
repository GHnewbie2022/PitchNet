#pragma once
#include <array>

#include "../JuceHeader.h"
#include "../Models/Project.h"
#include <atomic>
#include <functional>
#include <memory>

/**
 * Audio engine for playback and synthesis.
 */
class AudioEngine : public juce::AudioSource,
                    public juce::ChangeListener,
                    private juce::AsyncUpdater {
public:
  AudioEngine();
  ~AudioEngine() override;

  // AudioSource interface
  void prepareToPlay(int samplesPerBlockExpected, double sampleRate) override;
  void releaseResources() override;
  void
  getNextAudioBlock(const juce::AudioSourceChannelInfo &bufferToFill) override;

  // ChangeListener
  void changeListenerCallback(juce::ChangeBroadcaster *source) override;

  // Playback control
  void setProject(Project *proj) { project = proj; }
  void loadWaveform(const juce::AudioBuffer<float> &buffer, int sampleRate,
                    bool preservePosition = false);
  // The project's audio as it should be heard: the recording at its own rate
  // with edited ranges replaced (SourceAudio.h, spec Q5).
  void loadProjectAudio(const AudioData &audioData,
                        bool preservePosition = false);

  void play();
  void pause();
  void stop();
  void seek(double timeSeconds);

  // Loop control (seconds)
  void setLoopRange(double startSeconds, double endSeconds);
  void setLoopEnabled(bool enabled);
  void clearLoopRange();
  bool isLoopEnabled() const { return loopEnabled.load(); }

  // Temporary audition source. It loops independently of the project
  // transport, so previewing a note never moves the timeline playhead.
  // keepPosition: a new take of the same audio (the dragged note rendered
  // at a new pitch) carries on where the old one was, crossfaded, instead
  // of starting over. loopOverlapSamples: how much of the tail is
  // crossfaded into the head at the loop join.
  void beginAuditionLoop(const juce::AudioBuffer<float> &buffer,
                         int sampleRate, bool keepPosition = false,
                         int loopOverlapSamples = 8192);
  void endAuditionLoop();
  bool isAuditioning() const { return auditionActive.load(); }

  // The accompaniment (spec section 15): heard beside the project's audio at
  // the same place on the transport - through a seek, a loop, to the end.
  // Shorter than the project's audio, it is silent from where it ends; the
  // transport's length stays the project's. Converted to the device rate
  // once, like the project's audio. Not for the audio thread.
  void setBacking(const juce::AudioBuffer<float> &buffer, int sampleRate);
  void clearBacking();
  bool hasBacking() const { return backingPresent.load(); }
  void setBackingAudible(bool audible) { backingAudible.store(audible); }
  bool isBackingAudible() const { return backingAudible.load(); }
  // dB, -60 (silent) to +12.
  void setBackingGainDb(float dB);
  float getBackingGainDb() const;

  // The project's own audio beside it: its level, and whether it is heard
  // at all (a key is checked against the accompaniment alone).
  void setVoiceGainDb(float dB);
  float getVoiceGainDb() const;
  void setVoiceAudible(bool audible) { voiceAudible.store(audible); }
  bool isVoiceAudible() const { return voiceAudible.load(); }

  // A sound played once over the transport, from its start to its end, while
  // the transport plays: the scale a key is confirmed with (ScaleTone.h). It
  // moves no playhead, and is gone when it ends or the transport stops.
  // (The audition loop above takes the transport's place; this is laid over.)
  void playOverlay(const juce::AudioBuffer<float> &buffer, int sampleRate);
  void stopOverlay();
  bool isOverlayPlaying() const { return overlayActive.load(); }
  void setOverlayGainDb(float dB);
  float getOverlayGainDb() const;

  // The device changed rate: what was loaded is converted to it again (the
  // project's audio, the accompaniment). Called by the engine itself on the
  // message thread; public for tests, which have no message loop.
  void convertForDeviceRate();

  bool isPlaying() const { return playing; }
  double getPosition() const; // Returns position in seconds
  double getDuration() const;

  // Callbacks
  using PositionCallback = std::function<void(double)>;
  using FinishCallback = std::function<void()>;

  void setPositionCallback(PositionCallback callback) {
    std::atomic_store(&positionCallback,
                      std::make_shared<PositionCallback>(std::move(callback)));
  }
  void setFinishCallback(FinishCallback callback) {
    std::atomic_store(&finishCallback,
                      std::make_shared<FinishCallback>(std::move(callback)));
  }
  void clearCallbacks() {
    std::atomic_store(&positionCallback, std::shared_ptr<PositionCallback>{});
    std::atomic_store(&finishCallback, std::shared_ptr<FinishCallback>{});
  }

  // Audio device management
  juce::AudioDeviceManager &getDeviceManager() { return deviceManager; }
  /// Opens the output device. @p savedStateXml is the state previously
  /// captured with juce::AudioDeviceManager::createStateXml() (empty on first
  /// run); when @p followSystemDefault is set the restored state only supplies
  /// the device type and the output falls back to that type's current default
  /// device.
  void initializeAudio(const juce::String &savedStateXml = {},
                       bool followSystemDefault = true);
  void shutdownAudio();

  // Volume control (dB, -60 to +12)
  void setVolumeDb(float dB);
  float getVolumeDb() const;

private:
  struct PositionUpdateState {
    std::atomic<double> latestSeconds{0.0};
    std::atomic<bool> callbackPending{false};
  };

  juce::AudioDeviceManager deviceManager;
  juce::AudioSourcePlayer audioSourcePlayer;

  /// Publish a freshly rendered waveform for the audio thread to pick up.
  /// Called on the message thread; the swap itself happens in
  /// getNextAudioBlock() so playback is never gated on the copy.
  void publishWaveform(const juce::AudioBuffer<float> &buffer, int sampleRate);
  // Audio at another rate than the device is converted once with r8brain
  // (spec Q5 B) instead of interpolated while it plays. Message thread.
  juce::AudioBuffer<float> toDeviceRate(const juce::AudioBuffer<float> &buffer,
                                        int &sampleRate) const;
  // The device changed rate: convert the last waveform again.
  void handleAsyncUpdate() override;
  juce::AudioBuffer<float> deviceSourceWaveform; // last loadWaveform, as given
  int deviceSourceSampleRate = 0;

  /// Audio-thread side of the handoff: installs pendingWaveform and blends the
  /// outgoing samples into it across kWaveformSwapFadeSamples at the playhead.
  void applyPendingWaveformSwap();

  Project *project = nullptr;
  juce::AudioBuffer<float> currentWaveform;
  int waveformSampleRate = 44100;
  juce::AudioBuffer<float> auditionWaveform;
  int auditionSampleRate = 44100;
  juce::AudioBuffer<float> previousAuditionWaveform;
  int previousAuditionSampleRate = 44100;
  int auditionLoopOverlap = 8192;
  int previousAuditionLoopOverlap = 8192;

  std::atomic<int64_t> currentPosition{0}; // Position in waveform samples
  std::atomic<int64_t> auditionPosition{0};
  std::atomic<double> auditionReadPosition{0.0};
  std::atomic<double> previousAuditionReadPosition{0.0};
  std::atomic<int> auditionTransitionSamplesRemaining{0};
  std::atomic<int> auditionTransitionSamplesTotal{0};
  std::atomic<bool> playing{false};
  std::atomic<bool> auditionActive{false};
  std::atomic<bool> shouldStop{false};

  std::shared_ptr<PositionCallback> positionCallback;
  std::shared_ptr<FinishCallback> finishCallback;

  std::shared_ptr<PositionUpdateState> positionUpdateState =
      std::make_shared<PositionUpdateState>();

  double currentSampleRate = 44100.0;

  // For sample rate conversion
  juce::LagrangeInterpolator interpolator;
  double playbackRatio = 1.0;      // waveformSampleRate / deviceSampleRate
  double fractionalPosition = 0.0; // Sub-sample position for interpolation

  // Thread safety for waveform updates
  juce::SpinLock waveformLock;

  // Deferred waveform handoff. An incremental resynthesis commit publishes the
  // new composite here instead of stopping playback and deep-copying under
  // waveformLock; the audio thread installs it between blocks and crossfades
  // the outgoing samples into it, so a render that lands mid-playback is
  // inaudible rather than a hard silence gap on both edges.
  static constexpr int kWaveformSwapFadeSamples = 512;

  juce::AudioBuffer<float> pendingWaveform;
  int pendingWaveformSampleRate = 44100;
  // Holds the outgoing buffer so its deallocation happens on the message
  // thread (in publishWaveform) rather than inside the audio callback.
  juce::AudioBuffer<float> retiredWaveform;
  juce::SpinLock pendingWaveformLock;
  std::atomic<bool> waveformSwapPending{false};

  // Where the pending buffer differs from the one playing: a crossfade at the
  // playhead is only clean where the two are the same audio. Inside the
  // re-rendered part they are two renders at unrelated phases and the
  // crossfade cancels (a click once, never there again on replay), so the
  // swap waits for the playhead to leave that part (or a loop to wrap).
  // Block hashes of each buffer find the part; the hashes of the playing
  // and pending buffers trade places with the buffers. All under
  // pendingWaveformLock.
  static constexpr int kMaxSwapDeferBlocks = 200; // audio blocks (~2 s): then crossfade anyway
  std::vector<uint64_t> playingHashes, pendingHashes;
  // The outgoing audio at the swap point, crossfaded from as it is output
  // (audio thread only; fixed size, nothing allocated there).
  std::array<std::array<float, kWaveformSwapFadeSamples>, 2> swapTail{};
  int swapTailChannels = 0, swapTailLength = 0, swapTailRemaining = 0;
  int64_t pendingDiffStart = -1, pendingDiffEnd = -1; // samples, [start, end)
  int pendingDeferredBlocks = 0;                       // audio thread
  static std::vector<uint64_t> blockHashes(const juce::AudioBuffer<float> &buffer);

public:
  // Swaps made while playing, for the debug recording (DebugJournal.h): how
  // many, and the last one's playhead, wait and whether it had to crossfade
  // inside the re-rendered part after all.
  struct SwapReport
  {
    int count = 0;
    double positionSeconds = 0.0, waitedSeconds = 0.0;
    bool forcedInside = false;
  };
  SwapReport getSwapReport() const
  {
    SwapReport r;
    r.count = swapCount.load();
    r.positionSeconds = lastSwapSeconds.load();
    r.waitedSeconds = lastSwapWaitedSeconds.load();
    r.forcedInside = lastSwapForced.load();
    return r;
  }

private:
  std::atomic<int> swapCount{0};
  std::atomic<double> lastSwapSeconds{0.0}, lastSwapWaitedSeconds{0.0};
  std::atomic<bool> lastSwapForced{false};

  // Volume control (linear gain, lock-free for audio thread)
  std::atomic<float> volumeGain{1.0f};

  // The accompaniment: as given (converted again when the device changes
  // rate), and at the device rate when that differs. Swapped in under
  // waveformLock, read by the audio thread under it.
  juce::AudioBuffer<float> backingSource;
  int backingSourceRate = 0;
  juce::AudioBuffer<float> backingConverted; // empty while the rates agree
  int backingRate = 0;                       // of what the audio thread reads
  std::atomic<bool> backingPresent{false};
  std::atomic<bool> backingAudible{false};
  std::atomic<float> backingGain{0.5f};
  std::atomic<float> voiceGain{1.0f};
  std::atomic<bool> voiceAudible{true};

  // The sound laid over the transport once (playOverlay).
  juce::AudioBuffer<float> overlayWaveform; // at the device rate
  int overlayRate = 0;
  int overlayPosition = 0;                  // audio thread, under waveformLock
  std::atomic<bool> overlayActive{false};
  std::atomic<float> overlayGain{1.0f};

  // Loop range (in waveform samples)
  std::atomic<bool> loopEnabled{false};
  std::atomic<int64_t> loopStartSample{0};
  std::atomic<int64_t> loopEndSample{0};

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioEngine)
};
