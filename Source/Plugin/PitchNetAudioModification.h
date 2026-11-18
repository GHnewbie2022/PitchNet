#pragma once

#include "../JuceHeader.h"
#include "../Audio/SourceAudio.h"
#include "../Utils/AudioResampler.h"
#include <limits>

#if JucePlugin_Enable_ARA

#include <atomic>
#include <memory>

// PROCESSED audio and project archive owned by the ARA audio modification, so
// the analysed/processed result lives independently of the editor: the timeline
// draws each clip from here and the playback renderer mixes from here, both of
// which work with the UI closed.
//
// State is held in single slots, not keyed containers. ARA gives persistent
// identity to the audio modification, and every playback region referencing one
// is a window onto the same edit layer, so a modification owns exactly one
// project archive and one processed-audio buffer. A key would carry no
// information.
//
// The modification's host-assigned persistent ID is deliberately NOT used as an
// identity here. ARA treats it as an archive-reconnection token and a mutable
// model property - hosts may adjust it on restore or import, and REAPER changes
// it from an absolute to a project-relative path on a project's first save. It
// belongs in the archive stream and in diagnostics, nowhere else. Live identity
// is getLiveKey(), a process-local serial minted at construction.
class PitchNetAudioModification final : public juce::ARAAudioModification {
public:
  struct ProcessedRegionData {
    juce::AudioBuffer<float> audio;
    double sampleRate = 0.0;
    juce::int64 startSampleInModification = 0;

    juce::AudioFormatManager thumbnailFormatManager;
    juce::AudioThumbnailCache thumbnailCache{8};
    juce::AudioThumbnail thumbnail;

    ProcessedRegionData() : thumbnail(128, thumbnailFormatManager, thumbnailCache) {
      thumbnailFormatManager.registerBasicFormats();
    }

    void setAudio(const juce::AudioBuffer<float> &buffer, double sampleRateIn,
                  juce::int64 startSampleInModificationIn) {
      audio.makeCopyOf(buffer);
      sampleRate = sampleRateIn;
      startSampleInModification = startSampleInModificationIn;
      thumbnail.reset(audio.getNumChannels(), sampleRate, audio.getNumSamples());
      thumbnail.addBlock(0, audio, 0, audio.getNumSamples());
    }

    void clear() {
      audio.setSize(0, 0);
      sampleRate = 0.0;
      startSampleInModification = 0;
      thumbnail.clear();
    }

    bool hasAudio() const {
      return audio.getNumSamples() > 0 && sampleRate > 0.0;
    }

    // Spec Q5: what playback actually lays over the host's own audio. `audio`
    // above stays the 44.1 kHz analysis composite (the editor restores from
    // it); these pieces are the same edit at the recording's rate and channel
    // count, only where it differs from the recording. Empty for archives
    // written before them: playback then falls back to `audio`.
    std::vector<sourceaudio::Piece> pieces;
    double piecesSampleRate = 0.0;

    bool hasPieces() const { return piecesSampleRate > 0.0; }
  };

  PitchNetAudioModification(
      juce::ARAAudioSource *audioSource,
      ARA::ARAAudioModificationHostRef hostRef,
      const juce::ARAAudioModification *optionalModificationToClone)
      : juce::ARAAudioModification(audioSource, hostRef,
                                   optionalModificationToClone),
        liveKey("mod:" + juce::String(nextLiveKeySerial().fetch_add(1))) {
    // ARA defines the source passed here as the object whose internal state the
    // clone must copy, and the host assigns the clone's own properties straight
    // afterwards. Copying synchronously here is therefore complete: nothing can
    // arrive on the source in between, and later edits to the source correctly
    // do not bleed into an existing clone.
    if (const auto *sourceModification =
            dynamic_cast<const PitchNetAudioModification *>(
                optionalModificationToClone)) {
      const juce::SpinLock::ScopedLockType lock(
          sourceModification->processedAudioLock);
      if (sourceModification->processedAudio != nullptr &&
          sourceModification->processedAudio->hasAudio()) {
        processedAudio = std::make_unique<ProcessedRegionData>();
        processedAudio->setAudio(
            sourceModification->processedAudio->audio,
            sourceModification->processedAudio->sampleRate,
            sourceModification->processedAudio->startSampleInModification);
        copyPieces(*sourceModification->processedAudio, *processedAudio);
      }
      projectArchive = sourceModification->projectArchive;
    }
  }

  //============================================================================
  // Live identity
  //
  // Stable for this object's lifetime and unique within the process. A raw
  // address would also be lifetime-stable, but destruction lets the allocator
  // reuse it, so a stale key could come to equal a live object's key. A serial
  // cannot collide.
  //
  // MUST NEVER be written to any archive. It is meaningless across runs.
  const juce::String &getLiveKey() const noexcept { return liveKey; }

  //============================================================================
  // Mutators
  void setProcessedAudio(const juce::AudioBuffer<float> &buffer,
                         double sampleRateIn,
                         juce::int64 startSampleInModificationIn) {
    const juce::SpinLock::ScopedLockType lock(processedAudioLock);
    if (processedAudio == nullptr)
      processedAudio = std::make_unique<ProcessedRegionData>();
    processedAudio->setAudio(buffer, sampleRateIn, startSampleInModificationIn);
    processedAudio->pieces.clear();
    processedAudio->piecesSampleRate = 0.0;
    contentChangedLocked();
  }

  // Sets the composite and its playback pieces together, so the renderer
  // never sees one without the other.
  void setProcessedAudio(const juce::AudioBuffer<float> &buffer,
                         double sampleRateIn,
                         juce::int64 startSampleInModificationIn,
                         std::vector<sourceaudio::Piece> pieces,
                         double piecesSampleRate) {
    const juce::SpinLock::ScopedLockType lock(processedAudioLock);
    if (processedAudio == nullptr)
      processedAudio = std::make_unique<ProcessedRegionData>();
    processedAudio->setAudio(buffer, sampleRateIn, startSampleInModificationIn);
    processedAudio->pieces = std::move(pieces);
    processedAudio->piecesSampleRate = piecesSampleRate;
    contentChangedLocked();
  }

  void clearProcessedAudio() {
    const juce::SpinLock::ScopedLockType lock(processedAudioLock);
    processedAudio.reset();
    contentChangedLocked();
  }

  //============================================================================
  // Spec Q5 at a host rate that differs from the source: the playback audio
  // (the host's audio with the edit pieces laid over it) converted once with
  // r8brain, as the standalone converts for its device, instead of Lagrange
  // per block. Built on a worker thread; until it is ready the renderer keeps
  // its per-block conversion.
  struct HostRateAudio {
    juce::AudioBuffer<float> audio;
    double rate = 0.0;
  };

  // Message/model thread. `now`: build on this thread (offline rendering).
  void requestHostRate(double hostRate, bool now = false) {
    auto *source = getAudioSource();
    if (source == nullptr || hostRate <= 0.0 ||
        juce::approximatelyEqual(hostRate, source->getSampleRate()))
      return;
    {
      const juce::SpinLock::ScopedLockType lock(processedAudioLock);
      if (juce::approximatelyEqual(hostRate, requestedHostRate) && !now &&
          hostRateAudio != nullptr)
        return;
      requestedHostRate = hostRate;
    }
    if (now) {
      buildHostRateAudio();
      return;
    }
    if (hostRateBuilder == nullptr)
      hostRateBuilder = std::make_unique<HostRateBuilder>(*this);
    hostRateBuilder->notify();
  }

  // Audio thread, while holding tryLockProcessedAudio(): the converted audio
  // if it is current for `rate`, else nullptr.
  const HostRateAudio *getHostRateAudio(double rate) const noexcept {
    return hostRateAudio != nullptr &&
                   juce::approximatelyEqual(hostRateAudio->rate, rate)
               ? hostRateAudio.get()
               : nullptr;
  }

  void setProjectArchive(const void *data, size_t sizeInBytes) const {
    if (data == nullptr || sizeInBytes == 0)
      return;

    const juce::SpinLock::ScopedLockType lock(processedAudioLock);
    projectArchive = juce::MemoryBlock(data, sizeInBytes);
  }

  //============================================================================
  // Queries
  bool copyProjectArchive(juce::MemoryBlock &dest) const {
    const juce::SpinLock::ScopedLockType lock(processedAudioLock);
    if (projectArchive.getSize() == 0)
      return false;

    dest = projectArchive;
    return true;
  }

  bool hasProjectArchive() const {
    const juce::SpinLock::ScopedLockType lock(processedAudioLock);
    return projectArchive.getSize() > 0;
  }

  bool hasProcessedAudio() const {
    const juce::SpinLock::ScopedLockType lock(processedAudioLock);
    return processedAudio != nullptr && processedAudio->hasAudio();
  }

  bool copyProcessedAudio(juce::AudioBuffer<float> &buffer,
                          double &sampleRateOut,
                          juce::int64 &startSampleOut) const {
    const juce::SpinLock::ScopedLockType lock(processedAudioLock);
    if (processedAudio == nullptr || !processedAudio->hasAudio())
      return false;

    buffer.makeCopyOf(processedAudio->audio);
    sampleRateOut = processedAudio->sampleRate;
    startSampleOut = processedAudio->startSampleInModification;
    return true;
  }

  // Read access must hold the lock while the returned pointer is used on the
  // audio thread. tryLockProcessedAudio() gives a try-lock for the renderer so
  // it never blocks; getProcessedData() returns the raw pointer.
  juce::SpinLock::ScopedTryLockType tryLockProcessedAudio() const {
    return juce::SpinLock::ScopedTryLockType(processedAudioLock);
  }

  juce::SpinLock &getProcessedAudioLock() const noexcept {
    return processedAudioLock;
  }

  const ProcessedRegionData *getProcessedData() const noexcept {
    return processedAudio.get();
  }

  ProcessedRegionData *getProcessedData() noexcept {
    return processedAudio.get();
  }

  // Every playback region referencing this modification shares its edit layer,
  // so they all serve the same processed audio. This reverses the earlier
  // per-region rule, which existed when state was owned by regions rather than
  // by the modification.

  //============================================================================
  // Persistence. Stream this modification's processed audio.
  bool writeProcessedAudioToStream(juce::OutputStream &output) const {
    const juce::SpinLock::ScopedLockType lock(processedAudioLock);
    if (processedAudio == nullptr || !processedAudio->hasAudio())
      return false;

    const auto &data = *processedAudio;
    output.writeDouble(data.sampleRate);
    output.writeInt64(data.startSampleInModification);
    output.writeInt(data.audio.getNumChannels());
    output.writeInt(data.audio.getNumSamples());
    for (int ch = 0; ch < data.audio.getNumChannels(); ++ch)
      if (!output.write(data.audio.getReadPointer(ch),
                        static_cast<size_t>(data.audio.getNumSamples()) *
                            sizeof(float)))
        return false;
    return writePieces(output, data);
  }

  // `withPieces`: archive version 6 on. Older archives end after the audio.
  bool readProcessedAudioFromStream(juce::InputStream &input,
                                    bool withPieces) {
    const auto sampleRateIn = input.readDouble();
    const auto startSampleIn = input.readInt64();
    const auto numChannels = input.readInt();
    const auto numSamples = input.readInt();
    const int bytesPerSample = sizeof(float);
    if (!std::isfinite(sampleRateIn) || sampleRateIn <= 0.0 ||
        numChannels < 0 ||
        numSamples < 0 || numSamples > std::numeric_limits<int>::max() / bytesPerSample)
      return false;

    juce::AudioBuffer<float> restored(numChannels, numSamples);
    const int channelBytes = numSamples * bytesPerSample;
    for (int ch = 0; ch < numChannels; ++ch)
      if (input.read(restored.getWritePointer(ch), channelBytes) != channelBytes)
        return false;

    std::vector<sourceaudio::Piece> pieces;
    double piecesRate = 0.0;
    if (withPieces && !readPieces(input, pieces, piecesRate))
      return false;
    setProcessedAudio(restored, sampleRateIn, startSampleIn, std::move(pieces),
                      piecesRate);
    return true;
  }

  static bool skipProcessedAudioFromStream(juce::InputStream &input,
                                           bool withPieces) {
    input.readDouble();
    input.readInt64();
    const auto numChannels = input.readInt();
    const auto numSamples = input.readInt();
    const int bytesPerSample = sizeof(float);
    if (numChannels < 0 || numSamples < 0 ||
        numSamples > std::numeric_limits<int>::max() / bytesPerSample)
      return false;
    juce::HeapBlock<char> skip(static_cast<size_t>(numSamples) * sizeof(float));
    const int channelBytes = numSamples * bytesPerSample;
    for (int ch = 0; ch < numChannels; ++ch)
      if (input.read(skip.get(), channelBytes) != channelBytes)
        return false;
    std::vector<sourceaudio::Piece> pieces;
    double piecesRate = 0.0;
    return !withPieces || readPieces(input, pieces, piecesRate);
  }

private:
  class HostRateBuilder final : public juce::Thread {
  public:
    explicit HostRateBuilder(PitchNetAudioModification &ownerIn)
        : juce::Thread("PitchNet host rate"), owner(ownerIn) {
      startThread(juce::Thread::Priority::background);
    }
    ~HostRateBuilder() override { stopThread(30000); }
    void run() override {
      while (!threadShouldExit()) {
        owner.buildHostRateAudio();
        wait(-1);
      }
    }

  private:
    PitchNetAudioModification &owner;
  };

  // Called with processedAudioLock held: the edit changed, so any converted
  // audio is stale. It is dropped at once (the renderer falls back) and
  // rebuilt if a host rate was asked for.
  void contentChangedLocked() {
    ++contentVersion;
    staleHostRateAudio = std::move(hostRateAudio);
    if (requestedHostRate > 0.0 && hostRateBuilder != nullptr)
      hostRateBuilder->notify();
  }

  void buildHostRateAudio() {
    auto *source = getAudioSource();
    if (source == nullptr || source->getSampleRate() <= 0.0 ||
        source->getChannelCount() <= 0 || source->getSampleCount() <= 0 ||
        source->getSampleCount() > std::numeric_limits<int>::max())
      return;

    double rate = 0.0;
    juce::uint64 version = 0;
    std::vector<sourceaudio::Piece> pieces;
    {
      const juce::SpinLock::ScopedLockType lock(processedAudioLock);
      rate = requestedHostRate;
      version = contentVersion;
      staleHostRateAudio.reset();
      if (processedAudio != nullptr && processedAudio->hasAudio()) {
        // An archive from before the pieces holds only the 44.1 kHz render;
        // the source alone would drop its edit, so leave it to the renderer.
        if (!processedAudio->hasPieces() ||
            !juce::approximatelyEqual(processedAudio->piecesSampleRate,
                                      source->getSampleRate()))
          return;
        ProcessedRegionData copy;
        copyPieces(*processedAudio, copy);
        pieces = std::move(copy.pieces);
      }
    }
    if (rate <= 0.0)
      return;

    const int channels = source->getChannelCount();
    const int samples = static_cast<int>(source->getSampleCount());
    juce::AudioBuffer<float> recording(channels, samples);
    juce::ARAAudioSourceReader reader(source);
    if (!reader.read(&recording, 0, samples, 0, true, channels > 1))
      return;
    sourceaudio::overlay(pieces, 0, recording, 0, samples);

    auto built = std::make_unique<HostRateAudio>();
    built->audio =
        AudioResampler::resample(recording, source->getSampleRate(), rate);
    built->rate = rate;

    std::unique_ptr<HostRateAudio> replaced;
    {
      const juce::SpinLock::ScopedLockType lock(processedAudioLock);
      if (version != contentVersion ||
          !juce::approximatelyEqual(rate, requestedHostRate))
        return; // changed meanwhile; the builder has been notified again
      replaced = std::move(hostRateAudio);
      hostRateAudio = std::move(built);
    }
  }

  static void copyPieces(const ProcessedRegionData &from,
                         ProcessedRegionData &to) {
    to.pieces.clear();
    for (const auto &piece : from.pieces) {
      sourceaudio::Piece copy;
      copy.start = piece.start;
      copy.audio.makeCopyOf(piece.audio);
      to.pieces.push_back(std::move(copy));
    }
    to.piecesSampleRate = from.piecesSampleRate;
  }

  static bool writePieces(juce::OutputStream &output,
                          const ProcessedRegionData &data) {
    if (!output.writeDouble(data.piecesSampleRate) ||
        !output.writeInt(static_cast<int>(data.pieces.size())))
      return false;
    for (const auto &piece : data.pieces) {
      if (!output.writeInt64(piece.start) ||
          !output.writeInt(piece.audio.getNumChannels()) ||
          !output.writeInt(piece.audio.getNumSamples()))
        return false;
      for (int ch = 0; ch < piece.audio.getNumChannels(); ++ch)
        if (!output.write(piece.audio.getReadPointer(ch),
                          static_cast<size_t>(piece.audio.getNumSamples()) *
                              sizeof(float)))
          return false;
    }
    return true;
  }

  static bool readPieces(juce::InputStream &input,
                         std::vector<sourceaudio::Piece> &pieces,
                         double &piecesRate) {
    piecesRate = input.readDouble();
    const int count = input.readInt();
    if (!std::isfinite(piecesRate) || piecesRate < 0.0 || count < 0)
      return false;
    for (int i = 0; i < count; ++i) {
      sourceaudio::Piece piece;
      piece.start = input.readInt64();
      const int channels = input.readInt();
      const int samples = input.readInt();
      if (piece.start < 0 || channels <= 0 || channels > 64 || samples <= 0 ||
          samples > std::numeric_limits<int>::max() /
                        static_cast<int>(sizeof(float)))
        return false;
      piece.audio.setSize(channels, samples);
      const int bytes = samples * static_cast<int>(sizeof(float));
      for (int ch = 0; ch < channels; ++ch)
        if (input.read(piece.audio.getWritePointer(ch), bytes) != bytes)
          return false;
      pieces.push_back(std::move(piece));
    }
    return true;
  }

  static std::atomic<juce::uint64> &nextLiveKeySerial() {
    static std::atomic<juce::uint64> serial{1};
    return serial;
  }

  const juce::String liveKey;
  mutable juce::SpinLock processedAudioLock;
  std::unique_ptr<ProcessedRegionData> processedAudio;
  mutable juce::MemoryBlock projectArchive;
  double requestedHostRate = 0.0;
  juce::uint64 contentVersion = 0;
  std::unique_ptr<HostRateAudio> hostRateAudio;
  // Held until the builder's next run so it is not freed under the lock on
  // a thread the renderer might be waiting on.
  std::unique_ptr<HostRateAudio> staleHostRateAudio;
  // Last: destroyed first, so the worker stops before anything it uses.
  std::unique_ptr<HostRateBuilder> hostRateBuilder;
};

#endif // JucePlugin_Enable_ARA
