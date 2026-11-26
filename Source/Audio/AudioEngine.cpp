#include "AudioEngine.h"
#include "../Utils/AppLogger.h"
#include "../Utils/AudioResampler.h"
#include "SourceAudio.h"
#include "Mixdown.h"
#include "BufferHashes.h"
#include <tuple>
#include <algorithm>
#include <cstring>
#include <cstdlib>

AudioEngine::AudioEngine() {}

AudioEngine::~AudioEngine() { shutdownAudio(); }

namespace {
// Name of the current device type's default output, or empty if it has none.
juce::String defaultOutputDeviceName(juce::AudioDeviceManager &manager) {
  auto *type = manager.getCurrentDeviceTypeObject();
  if (type == nullptr)
    return {};

  auto devices = type->getDeviceNames(false);
  const int defaultIndex = type->getDefaultDeviceIndex(false);
  if (juce::isPositiveAndBelow(defaultIndex, devices.size()))
    return devices[defaultIndex];

  return devices.size() > 0 ? devices[0] : juce::String();
}
} // namespace

void AudioEngine::initializeAudio(const juce::String &savedStateXml,
                                  bool followSystemDefault) {
  // Initialize audio device, restoring the device type / device / sample rate
  // / buffer size the user last chose. Passing the state to initialise() (0 in,
  // stereo out) falls back to the default device if the saved one is gone.
  std::unique_ptr<juce::XmlElement> savedState;
  if (savedStateXml.isNotEmpty())
    savedState = juce::parseXML(savedStateXml);

  for (auto *type : deviceManager.getAvailableDeviceTypes())
    LOG("AudioEngine: driver '" + type->getTypeName() + "' outputs: " +
        type->getDeviceNames(false).joinIntoString(", "));

  if (savedState != nullptr)
    deviceManager.initialise(0, 2, savedState.get(), true);
  else
    deviceManager.initialiseWithDefaultDevices(0, 2);

  // "System Default" stores the driver but no device name, and JUCE opens
  // nothing for an empty name. Resolve it against the restored driver so the
  // output tracks that driver's current default device. This has to run before
  // the fallback below, or a perfectly usable driver looks dead.
  if (followSystemDefault && savedState != nullptr) {
    const auto targetName = defaultOutputDeviceName(deviceManager);
    auto *current = deviceManager.getCurrentAudioDevice();
    if (targetName.isNotEmpty() &&
        (current == nullptr || current->getName() != targetName)) {
      auto setup = deviceManager.getAudioDeviceSetup();
      setup.outputDeviceName = targetName;
      setup.sampleRate = 0.0;
      setup.bufferSize = 0;
      setup.useDefaultOutputChannels = true;
      setup.outputChannels.clear();
      deviceManager.setAudioDeviceSetup(setup, true);
    }
  }

  // Nothing opened, so the saved driver really is unusable (e.g. JACK with no
  // jackd running). JUCE's own fallback retries the default device of that
  // same dead driver, so drop to the first available one explicitly.
  if (deviceManager.getCurrentAudioDevice() == nullptr) {
    if (auto *firstType = deviceManager.getAvailableDeviceTypes().getFirst())
      deviceManager.setCurrentAudioDeviceType(firstType->getTypeName(), false);
    deviceManager.initialiseWithDefaultDevices(0, 2);
  }

  if (auto *device = deviceManager.getCurrentAudioDevice())
    LOG("AudioEngine: output device '" + device->getName() + "' (driver " +
        device->getTypeName() + ", " +
        juce::String(device->getCurrentSampleRate(), 0) + " Hz, " +
        juce::String(device->getCurrentBufferSizeSamples()) + " samples)");
  else
    LOG("AudioEngine: no output device could be opened");

  // UI tests only: with PITCHNET_SILENT_OUTPUT set, everything plays at no
  // level - the transport runs as it does, nothing is heard. For running a
  // flow that plays (the wizard's key check) with nobody listening.
  if (std::getenv("PITCHNET_SILENT_OUTPUT") != nullptr)
    audioSourcePlayer.setGain(0.0f);
  deviceManager.addAudioCallback(&audioSourcePlayer);
  audioSourcePlayer.setSource(this);
}

void AudioEngine::shutdownAudio() {
  audioSourcePlayer.setSource(nullptr);
  deviceManager.removeAudioCallback(&audioSourcePlayer);
  deviceManager.closeAudioDevice();
}

void AudioEngine::prepareToPlay(int samplesPerBlockExpected,
                                double sampleRate) {
  currentSampleRate = sampleRate;
  playbackRatio = static_cast<double>(waveformSampleRate) / sampleRate;
  interpolator.reset();
  fractionalPosition = 0.0;
  if (deviceSourceSampleRate > 0 &&
      std::abs(static_cast<double>(waveformSampleRate) - sampleRate) > 0.5)
    triggerAsyncUpdate();

}

void AudioEngine::releaseResources() {}

void AudioEngine::getNextAudioBlock(
    const juce::AudioSourceChannelInfo &bufferToFill) {
  // Install any waveform published since the last block. This runs before the
  // !playing early-out so a swap published while paused still lands promptly.
  if (waveformSwapPending.load(std::memory_order_acquire))
    applyPendingWaveformSwap();

  const bool auditioning = auditionActive.load();
  if (!playing ||
      (auditioning ? auditionWaveform.getNumSamples()
                   : currentWaveform.getNumSamples()) == 0) {
    bufferToFill.clearActiveBufferRegion();
    return;
  }

  const juce::SpinLock::ScopedTryLockType lock(waveformLock);
  if (!lock.isLocked()) {
    // Waveform is being updated, output silence to avoid glitches
    bufferToFill.clearActiveBufferRegion();
    return;
  }

  // Audition is a self-contained loop: crossfade its tail into its head and
  // then continue after the already-crossfaded head section. This eliminates
  // the click/gap of a hard wrap without affecting the project transport.
  if (auditioning) {
    auto *outputBuffer = bufferToFill.buffer;
    const int numOutputSamples = bufferToFill.numSamples;
    const int startSample = bufferToFill.startSample;
    auto renderLoopSample = [this](const juce::AudioBuffer<float> &waveform,
                                   int sampleRate, int loopOverlap,
                                   double &position) {
      const int sampleCount = waveform.getNumSamples();
      const int overlap =
          std::min(std::max(1, loopOverlap), std::max(1, sampleCount / 2));
      const double overlapStart = static_cast<double>(sampleCount - overlap);
      const double sampleRatio = currentSampleRate > 0.0
                                     ? static_cast<double>(sampleRate) / currentSampleRate
                                     : 1.0;
      auto sampleAt = [&waveform, sampleCount](double samplePosition) {
        samplePosition = juce::jlimit(0.0,
                                      static_cast<double>(sampleCount - 1),
                                      samplePosition);
        const int left = static_cast<int>(samplePosition);
        const int right = std::min(sampleCount - 1, left + 1);
        const float fraction = static_cast<float>(samplePosition - left);
        const auto *data = waveform.getReadPointer(0);
        return data[left] + fraction * (data[right] - data[left]);
      };

      float value = sampleAt(position);
      if (position >= overlapStart) {
        const float t = static_cast<float>((position - overlapStart) / overlap);
        value = value * std::cos(t * juce::MathConstants<float>::halfPi) +
                sampleAt(position - overlapStart) *
                    std::sin(t * juce::MathConstants<float>::halfPi);
      }
      position += sampleRatio;
      while (position >= sampleCount)
        position -= static_cast<double>(sampleCount - overlap);
      return value;
    };

    outputBuffer->clear(startSample, numOutputSamples);
    auto *output = outputBuffer->getWritePointer(0, startSample);
    double position = auditionReadPosition.load();
    double previousPosition = previousAuditionReadPosition.load();
    int transitionRemaining = auditionTransitionSamplesRemaining.load();
    const int transitionTotal = auditionTransitionSamplesTotal.load();
    for (int i = 0; i < numOutputSamples; ++i) {
      float value = renderLoopSample(auditionWaveform, auditionSampleRate,
                                     auditionLoopOverlap, position);
      if (transitionRemaining > 0 && transitionTotal > 0 &&
          previousAuditionWaveform.getNumSamples() > 0) {
        const float t = 1.0f - static_cast<float>(transitionRemaining) /
                                    static_cast<float>(transitionTotal);
        const float oldValue = renderLoopSample(previousAuditionWaveform,
                                                previousAuditionSampleRate,
                                                previousAuditionLoopOverlap,
                                                previousPosition);
        value = oldValue * std::cos(t * juce::MathConstants<float>::halfPi) +
                value * std::sin(t * juce::MathConstants<float>::halfPi);
        --transitionRemaining;
      }
      output[i] = value;
    }

    const float gain = volumeGain.load();
    if (std::abs(gain - 1.0f) > 0.0001f)
      juce::FloatVectorOperations::multiply(output, gain, numOutputSamples);
    for (int ch = 1; ch < outputBuffer->getNumChannels(); ++ch)
      outputBuffer->copyFrom(ch, startSample, output, numOutputSamples);
    auditionReadPosition.store(position);
    previousAuditionReadPosition.store(previousPosition);
    auditionTransitionSamplesRemaining.store(transitionRemaining);
    return;
  }

  auto *outputBuffer = bufferToFill.buffer;
  auto numOutputSamples = bufferToFill.numSamples;
  auto startSample = bufferToFill.startSample;

  int64_t pos = auditioning ? auditionPosition.load() : currentPosition.load();
  const auto &activeWaveform = auditioning ? auditionWaveform : currentWaveform;
  const int activeSampleRate = auditioning ? auditionSampleRate : waveformSampleRate;
  const double activePlaybackRatio =
      currentSampleRate > 0.0 ? static_cast<double>(activeSampleRate) / currentSampleRate
                              : 1.0;
  int64_t waveformLength = activeWaveform.getNumSamples();

  bool loopActive = auditioning || loopEnabled.load();
  int64_t loopStart = auditioning ? 0 : loopStartSample.load();
  int64_t loopEnd = auditioning ? waveformLength : loopEndSample.load();

  if (loopActive) {
    loopStart = juce::jlimit<int64_t>(0, waveformLength, loopStart);
    loopEnd = juce::jlimit<int64_t>(0, waveformLength, loopEnd);
    if (loopEnd <= loopStart)
      loopActive = false;
  }

  if (!loopActive && pos >= waveformLength) {
    bufferToFill.clearActiveBufferRegion();
    playing = false;
    overlayActive.store(false);
    currentPosition.store(0);
    interpolator.reset();

    // Schedule callback on message thread
    if (auto cb = std::atomic_load(&finishCallback))
      juce::MessageManager::callAsync([cb]() { (*cb)(); });
    return;
  }

  if (loopActive && pos >= loopEnd) {
    pos = loopStart;
    interpolator.reset();
  }

  // Audio is converted to the device rate when it is loaded (toDeviceRate),
  // so it normally plays sample for sample. The interpolator only covers the
  // moment between a device rate change and that conversion.
  const bool unity = std::abs(activePlaybackRatio - 1.0) < 1.0e-9;
  const int64_t blockStart = pos; // where this block starts reading
  const int sourceChannels = activeWaveform.getNumChannels();
  const float *inputData = activeWaveform.getReadPointer(0);
  float *outputData = outputBuffer->getWritePointer(0, startSample);

  outputBuffer->clear(startSample, numOutputSamples);

  int samplesRemaining = numOutputSamples;
  int writeOffset = 0;

  while (samplesRemaining > 0) {
    int64_t segmentEnd = loopActive ? loopEnd : waveformLength;
    int64_t inputAvailable = segmentEnd - pos;

    if (inputAvailable <= 0) {
      if (loopActive) {
        pos = loopStart;
        interpolator.reset();
        continue;
      }
      break;
    }

    int maxOutput = static_cast<int>(inputAvailable / activePlaybackRatio);
    if (maxOutput <= 0) {
      if (loopActive) {
        pos = loopStart;
        interpolator.reset();
        continue;
      }
      break;
    }

    int outCount = std::min(samplesRemaining, maxOutput);
    int samplesUsed = outCount;
    if (unity) {
      for (int ch = 0; ch < outputBuffer->getNumChannels(); ++ch)
        outputBuffer->copyFrom(ch, startSample + writeOffset, activeWaveform,
                               std::min(ch, sourceChannels - 1),
                               static_cast<int>(pos), outCount);
    } else {
      samplesUsed = interpolator.process(activePlaybackRatio, inputData + pos,
                                         outputData + writeOffset, outCount,
                                         static_cast<int>(inputAvailable),
                                         0 // No wrap
      );
      for (int ch = 1; ch < outputBuffer->getNumChannels(); ++ch)
        outputBuffer->copyFrom(ch, startSample + writeOffset,
                               outputData + writeOffset, outCount);
    }

    pos += samplesUsed;
    samplesRemaining -= outCount;
    writeOffset += outCount;

    if (loopActive && pos >= loopEnd) {
      pos = loopStart;
      interpolator.reset();
    }
  }

  // Just after a swap: crossfade from the outgoing audio kept at the swap
  // into what was output, sample for sample (unity rate only: otherwise the
  // tail and the output are not on the same grid, and the swap is plain).
  if (swapTailRemaining > 0 && swapTailChannels > 0) {
    if (unity) {
      const int count = std::min(numOutputSamples, swapTailRemaining);
      const int first = swapTailLength - swapTailRemaining;
      for (int ch = 0; ch < outputBuffer->getNumChannels(); ++ch) {
        const auto &tail = swapTail[static_cast<size_t>(std::min(ch, swapTailChannels - 1))];
        float *out = outputBuffer->getWritePointer(ch, startSample);
        for (int i = 0; i < count; ++i) {
          const float t = static_cast<float>(first + i) / static_cast<float>(swapTailLength);
          const float weight = t * t * (3.0f - 2.0f * t);
          const float old = tail[static_cast<size_t>(first + i)];
          out[i] = old + weight * (out[i] - old);
        }
      }
      swapTailRemaining -= count;
    } else {
      swapTailRemaining = 0;
    }
  }

  // Apply volume gain (lock-free read): the project's volume, and the
  // voice's own level beside the accompaniment (silent when not heard).
  float gain = volumeGain.load() * (voiceAudible.load() ? voiceGain.load() : 0.0f);
  if (std::abs(gain - 1.0f) > 0.0001f) // Only apply if not unity gain
    outputBuffer->applyGain(startSample, numOutputSamples, gain);

  // The accompaniment, read at the places the block just read the project's
  // audio at (the same walk through a loop). Only sample for sample: in the
  // moment after a device rate change the two are not on one grid.
  const auto &backing =
      backingConverted.getNumSamples() > 0 ? backingConverted : backingSource;
  if (unity && backingAudible.load() && backing.getNumSamples() > 0 &&
      backingRate == activeSampleRate) {
    const float backingLevel = backingGain.load();
    const int64_t backingLength = backing.getNumSamples();
    const int backingChannels = backing.getNumChannels();
    int64_t at = blockStart;
    int remaining = numOutputSamples;
    int offset = 0;
    while (remaining > 0 && backingLevel > 0.0f) {
      const int64_t segmentEnd = loopActive ? loopEnd : waveformLength;
      const int64_t available = segmentEnd - at;
      if (available <= 0) {
        if (!loopActive)
          break;
        at = loopStart;
        continue;
      }
      const int count = static_cast<int>(std::min<int64_t>(remaining, available));
      const int readable = static_cast<int>(
          std::max<int64_t>(0, std::min<int64_t>(count, backingLength - at)));
      if (readable > 0)
        for (int ch = 0; ch < outputBuffer->getNumChannels(); ++ch)
          outputBuffer->addFrom(ch, startSample + offset, backing,
                                std::min(ch, backingChannels - 1),
                                static_cast<int>(at), readable, backingLevel);
      at += count;
      remaining -= count;
      offset += count;
    }
  }

  // The sound laid over once (the scale a key is confirmed with): from where
  // it had got to, whatever the transport did.
  if (overlayActive.load()) {
    const int overlayLength = overlayWaveform.getNumSamples();
    const int count = std::min(numOutputSamples, overlayLength - overlayPosition);
    if (count > 0 && overlayRate == static_cast<int>(std::lround(currentSampleRate))) {
      const float overlayLevel = overlayGain.load();
      const int overlayChannels = overlayWaveform.getNumChannels();
      for (int ch = 0; ch < outputBuffer->getNumChannels(); ++ch)
        outputBuffer->addFrom(ch, startSample, overlayWaveform,
                              std::min(ch, overlayChannels - 1), overlayPosition,
                              count, overlayLevel);
    }
    overlayPosition += std::max(0, count);
    if (count <= 0 || overlayPosition >= overlayLength)
      overlayActive.store(false);
  }

  if (auditioning)
    auditionPosition.store(pos);
  else
    currentPosition.store(pos);


  if (!loopActive && samplesRemaining > 0) {
    playing = false;
    overlayActive.store(false);
    currentPosition.store(0);
    interpolator.reset();
    if (auto cb = std::atomic_load(&finishCallback))
      juce::MessageManager::callAsync([cb]() { (*cb)(); });
  }

  // Update position callback
  if (!auditioning)
    if (auto cb = std::atomic_load(&positionCallback)) {
    auto state = positionUpdateState;
    state->latestSeconds.store(static_cast<double>(currentPosition.load()) /
                               waveformSampleRate);

    // Schedule at most one pending callback to avoid flooding the message
    // thread
    if (!state->callbackPending.exchange(true)) {
      juce::MessageManager::callAsync([cb, state]() {
        state->callbackPending.store(false);
        (*cb)(state->latestSeconds.load());
      });
    }
    }
}

void AudioEngine::changeListenerCallback(juce::ChangeBroadcaster *source) {}

std::vector<uint64_t> AudioEngine::blockHashes(const juce::AudioBuffer<float> &buffer) {
  return bufferhashes::of(buffer);
}

void AudioEngine::publishWaveform(const juce::AudioBuffer<float> &buffer,
                                  int sampleRate) {
  auto hashes = blockHashes(buffer);
  {
    const juce::SpinLock::ScopedLockType lock(pendingWaveformLock);
    // The part that differs from what is playing (a different length: all).
    std::tie(pendingDiffStart, pendingDiffEnd) =
        bufferhashes::differing(hashes, playingHashes, buffer.getNumSamples());
    pendingHashes = std::move(hashes);
    pendingDeferredBlocks = 0;
    // Release the previously retired buffer here, on the message thread, so
    // the audio thread's move-assignment in applyPendingWaveformSwap() never
    // has to free anything.
    retiredWaveform.setSize(0, 0);
    pendingWaveform = buffer;
    pendingWaveformSampleRate = sampleRate;
  }
  waveformSwapPending.store(true, std::memory_order_release);
}

void AudioEngine::applyPendingWaveformSwap() {
  const juce::SpinLock::ScopedTryLockType pendingLock(pendingWaveformLock);
  if (!pendingLock.isLocked())
    return; // A publish is in flight; pick it up on the next block.

  const juce::SpinLock::ScopedTryLockType lock(waveformLock);
  if (!lock.isLocked())
    return;

  if (!waveformSwapPending.load(std::memory_order_acquire))
    return;

  const int newLength = pendingWaveform.getNumSamples();

  // Playing inside the part the two buffers differ in: keep the outgoing
  // audio until the playhead is out of it (crossfading two renders there
  // cancels), unless that takes too long (a loop inside the part).
  const int64_t playhead = currentPosition.load();
  const bool inside = pendingDiffStart >= 0 &&
                      playhead + kWaveformSwapFadeSamples > pendingDiffStart &&
                      playhead < pendingDiffEnd;
  if (playing.load() && inside && pendingDeferredBlocks < kMaxSwapDeferBlocks) {
    ++pendingDeferredBlocks;
    return;
  }
  if (playing.load()) {
    const double rate = std::max(1, pendingWaveformSampleRate);
    lastSwapSeconds.store(static_cast<double>(playhead) / rate);
    lastSwapWaitedSeconds.store(currentSampleRate > 0.0
                                    ? pendingDeferredBlocks * 512.0 / currentSampleRate
                                    : 0.0);
    lastSwapForced.store(inside);
    swapCount.fetch_add(1);
  }

  // Keep the outgoing audio's next samples at the playhead; the block
  // that plays them crossfades from them into the incoming audio as it is
  // output (getNextAudioBlock). The crossfade used to be written into the
  // incoming buffer itself, so a swap made inside the re-rendered part (two
  // renders at unrelated phases, cancelling) stayed in the playing buffer as
  // a click heard on every pass until the buffer was rebuilt - never in the
  // saved or exported audio. The ramp is linear (smoothstep-weighted) rather
  // than equal-power: cos/sin sums to sqrt(2) on correlated content and would
  // trade the click for a level bump.
  swapTailRemaining = 0;
  const int64_t position = currentPosition.load();
  if (playing.load() && newLength > 0 && position >= 0 &&
      position < static_cast<int64_t>(newLength) &&
      currentWaveform.getNumSamples() > position) {
    const int available = static_cast<int>(
        std::min<int64_t>(newLength - position,
                          currentWaveform.getNumSamples() - position));
    const int fadeSamples = std::min(kWaveformSwapFadeSamples, available);
    swapTailChannels = std::min(static_cast<int>(swapTail.size()),
                                currentWaveform.getNumChannels());
    for (int channel = 0; channel < swapTailChannels; ++channel) {
      const float *outgoing =
          currentWaveform.getReadPointer(channel, static_cast<int>(position));
      std::copy(outgoing, outgoing + fadeSamples,
                swapTail[static_cast<size_t>(channel)].begin());
    }
    swapTailLength = swapTailRemaining = fadeSamples;
  }

  retiredWaveform = std::move(currentWaveform);
  currentWaveform = std::move(pendingWaveform);
  waveformSampleRate = pendingWaveformSampleRate;
  std::swap(playingHashes, pendingHashes); // no allocation on the audio thread
  pendingDiffStart = pendingDiffEnd = -1;

  if (currentSampleRate > 0)
    playbackRatio = static_cast<double>(waveformSampleRate) / currentSampleRate;
  else
    playbackRatio = 1.0;

  waveformSwapPending.store(false, std::memory_order_release);
}

juce::AudioBuffer<float>
AudioEngine::toDeviceRate(const juce::AudioBuffer<float> &buffer,
                          int &sampleRate) const {
  const int deviceRate = static_cast<int>(std::lround(currentSampleRate));
  if (deviceRate <= 0 || sampleRate <= 0 || deviceRate == sampleRate ||
      buffer.getNumSamples() <= 0)
    return buffer;
  auto converted = AudioResampler::resample(buffer, sampleRate, deviceRate);
  sampleRate = deviceRate;
  return converted;
}

void AudioEngine::loadProjectAudio(const AudioData &audioData,
                                   bool preservePosition) {
  const auto playback = sourceaudio::build(audioData);
  loadWaveform(playback.audio, playback.sampleRate, preservePosition);
}

void AudioEngine::handleAsyncUpdate() { convertForDeviceRate(); }

void AudioEngine::convertForDeviceRate() {
  if (deviceSourceWaveform.getNumSamples() > 0) {
    const auto source = deviceSourceWaveform; // loadWaveform replaces it
    loadWaveform(source, deviceSourceSampleRate, true);
  }
  if (backingSource.getNumSamples() > 0) {
    const auto source = backingSource; // setBacking replaces it
    setBacking(source, backingSourceRate);
  }
  // A scale half played is not worth converting: it is played again.
  stopOverlay();
}

void AudioEngine::setBacking(const juce::AudioBuffer<float> &buffer, int sampleRate) {
  if (buffer.getNumSamples() <= 0 || sampleRate <= 0) {
    clearBacking();
    return;
  }
  int rate = sampleRate;
  auto converted = toDeviceRate(buffer, rate); // before the lock: it takes a while
  const bool differs = rate != sampleRate;
  juce::AudioBuffer<float> source;
  source.makeCopyOf(buffer);
  if (!differs)
    converted.setSize(0, 0);
  {
    const juce::SpinLock::ScopedLockType lock(waveformLock);
    std::swap(backingSource, source);
    std::swap(backingConverted, converted);
    backingSourceRate = sampleRate;
    backingRate = rate;
    backingPresent.store(true);
  }
  // `source` and `converted` now hold what was there: freed here, not on
  // the audio thread.
}

void AudioEngine::clearBacking() {
  juce::AudioBuffer<float> source, converted;
  {
    const juce::SpinLock::ScopedLockType lock(waveformLock);
    std::swap(backingSource, source);
    std::swap(backingConverted, converted);
    backingSourceRate = 0;
    backingRate = 0;
    backingPresent.store(false);
  }
}

namespace {
// The faders: -60 dB and under is silence, +12 at most (mixdown::faderGain,
// shared with the export of the whole).
float faderGain(float dB) { return mixdown::faderGain(dB); }
float faderDb(float gain) { return juce::Decibels::gainToDecibels(gain, -60.0f); }
} // namespace

void AudioEngine::setBackingGainDb(float dB) { backingGain.store(faderGain(dB)); }
float AudioEngine::getBackingGainDb() const { return faderDb(backingGain.load()); }
void AudioEngine::setVoiceGainDb(float dB) { voiceGain.store(faderGain(dB)); }
float AudioEngine::getVoiceGainDb() const { return faderDb(voiceGain.load()); }
void AudioEngine::setOverlayGainDb(float dB) { overlayGain.store(faderGain(dB)); }
float AudioEngine::getOverlayGainDb() const { return faderDb(overlayGain.load()); }

void AudioEngine::playOverlay(const juce::AudioBuffer<float> &buffer, int sampleRate) {
  if (buffer.getNumSamples() <= 0 || sampleRate <= 0) {
    stopOverlay();
    return;
  }
  int rate = sampleRate;
  auto converted = toDeviceRate(buffer, rate);
  const juce::SpinLock::ScopedLockType lock(waveformLock);
  std::swap(overlayWaveform, converted);
  overlayRate = rate;
  overlayPosition = 0;
  overlayActive.store(true);
}

void AudioEngine::stopOverlay() {
  const juce::SpinLock::ScopedLockType lock(waveformLock);
  overlayActive.store(false);
  overlayPosition = 0;
}

void AudioEngine::loadWaveform(const juce::AudioBuffer<float> &sourceBuffer,
                               int sourceSampleRate, bool preservePosition) {
  // Positions are kept in seconds across a rate change.
  const double positionSeconds = getPosition();
  const int previousRate = waveformSampleRate;
  deviceSourceWaveform.makeCopyOf(sourceBuffer);
  deviceSourceSampleRate = sourceSampleRate;
  int sampleRate = sourceSampleRate;
  const auto buffer = toDeviceRate(sourceBuffer, sampleRate);
  const int64_t newLength = buffer.getNumSamples();
  // A handoff crossfades sample for sample, so it needs both buffers at the
  // same rate; a rate change swaps under the lock and keeps the time instead.
  const bool rateChanged = previousRate != sampleRate;
  if (rateChanged && previousRate > 0) {
    const double scale = static_cast<double>(sampleRate) / previousRate;
    loopStartSample.store(
        static_cast<int64_t>(std::llround(loopStartSample.load() * scale)));
    loopEndSample.store(
        static_cast<int64_t>(std::llround(loopEndSample.load() * scale)));
  }

  // An incremental resynthesis commit that lands during playback goes through
  // the deferred handoff: stopping the transport and deep-copying under
  // waveformLock would emit hard silence on both edges of the swap, which is
  // audible as a click for every render the user triggers while playing.
  if (preservePosition && playing.load() && !rateChanged) {
    publishWaveform(buffer, sampleRate);
  } else {
    auto hashes = blockHashes(buffer); // before the locks: the audio thread waits on them
    const juce::SpinLock::ScopedLockType lock(waveformLock);

    // Drop any handoff that has not been picked up yet; this buffer supersedes
    // it, and leaving the flag set would overwrite us on the next block.
    waveformSwapPending.store(false, std::memory_order_release);
    {
      const juce::SpinLock::ScopedLockType pending(pendingWaveformLock);
      pendingWaveform.setSize(0, 0);
      retiredWaveform.setSize(0, 0);
      playingHashes = std::move(hashes);
      pendingDiffStart = pendingDiffEnd = -1;
    }

    currentWaveform = buffer;
    waveformSampleRate = sampleRate;

    if (!preservePosition) {
      currentPosition.store(0);
      fractionalPosition = 0.0;
    } else if (rateChanged) {
      currentPosition.store(juce::jlimit<int64_t>(
          0, newLength,
          static_cast<int64_t>(positionSeconds * waveformSampleRate)));
    }

    // Update playback ratio for sample rate conversion
    if (currentSampleRate > 0)
      playbackRatio =
          static_cast<double>(waveformSampleRate) / currentSampleRate;
    else
      playbackRatio = 1.0;

    interpolator.reset();
  }

  if (loopEnabled.load()) {
    auto loopStart = loopStartSample.load();
    auto loopEnd = loopEndSample.load();
    loopStart = juce::jlimit<int64_t>(0, newLength, loopStart);
    loopEnd = juce::jlimit<int64_t>(0, newLength, loopEnd);
    loopStartSample.store(loopStart);
    loopEndSample.store(loopEnd);
    if (loopEnd <= loopStart)
      loopEnabled.store(false);
  }
}

void AudioEngine::play() {
  if (currentWaveform.getNumSamples() == 0) {
    return;
  }

  playing = true;
}

void AudioEngine::pause() {
  playing = false;
  overlayActive.store(false); // not carried on when the transport plays again
}

void AudioEngine::stop() {
  playing = false;

  const juce::SpinLock::ScopedLockType lock(waveformLock);
  overlayActive.store(false); // what was laid over the transport goes with it
  overlayPosition = 0;
  currentPosition.store(0);
  interpolator.reset();
  fractionalPosition = 0.0;
}

void AudioEngine::seek(double timeSeconds) {
  const juce::SpinLock::ScopedLockType lock(waveformLock);
  int64_t newPos = static_cast<int64_t>(timeSeconds * waveformSampleRate);
  newPos = juce::jlimit<int64_t>(0, currentWaveform.getNumSamples(), newPos);
  currentPosition.store(newPos);
  interpolator.reset();
  fractionalPosition = 0.0;
}

void AudioEngine::setLoopRange(double startSeconds, double endSeconds) {
  if (startSeconds > endSeconds)
    std::swap(startSeconds, endSeconds);

  const juce::SpinLock::ScopedLockType lock(waveformLock);
  const int64_t waveformLength = currentWaveform.getNumSamples();
  int64_t startSample =
      static_cast<int64_t>(startSeconds * waveformSampleRate);
  int64_t endSample = static_cast<int64_t>(endSeconds * waveformSampleRate);

  startSample = juce::jlimit<int64_t>(0, waveformLength, startSample);
  endSample = juce::jlimit<int64_t>(0, waveformLength, endSample);

  loopStartSample.store(startSample);
  loopEndSample.store(endSample);
  loopEnabled.store(endSample > startSample);
}

void AudioEngine::setLoopEnabled(bool enabled) {
  if (enabled && loopEndSample.load() <= loopStartSample.load())
    loopEnabled.store(false);
  else
    loopEnabled.store(enabled);
}

void AudioEngine::clearLoopRange() {
  loopEnabled.store(false);
  loopStartSample.store(0);
  loopEndSample.store(0);
}

void AudioEngine::beginAuditionLoop(const juce::AudioBuffer<float> &sourceBuffer,
                                    int sourceSampleRate, bool keepPosition,
                                    int loopOverlapSamples) {
  if (sourceBuffer.getNumSamples() <= 0)
    return;
  int sampleRate = sourceSampleRate;
  const auto buffer = toDeviceRate(sourceBuffer, sampleRate);

  const juce::SpinLock::ScopedLockType lock(waveformLock);
  constexpr int kTransitionSamples = 4096;
  if (auditionActive.load() && auditionWaveform.getNumSamples() > 0) {
    previousAuditionWaveform.makeCopyOf(auditionWaveform);
    previousAuditionSampleRate = auditionSampleRate;
    previousAuditionLoopOverlap = auditionLoopOverlap;
    previousAuditionReadPosition.store(auditionReadPosition.load());
    auditionTransitionSamplesTotal.store(kTransitionSamples);
    auditionTransitionSamplesRemaining.store(kTransitionSamples);
  } else {
    previousAuditionWaveform.setSize(0, 0);
    auditionTransitionSamplesTotal.store(0);
    auditionTransitionSamplesRemaining.store(0);
  }
  const bool continuing = keepPosition && auditionActive.load() &&
                          auditionWaveform.getNumSamples() > 0;
  const double carriedPosition =
      continuing ? juce::jlimit(0.0,
                                static_cast<double>(buffer.getNumSamples() - 1),
                                auditionReadPosition.load())
                 : 0.0;
  auditionWaveform.makeCopyOf(buffer);
  auditionSampleRate = sampleRate > 0 ? sampleRate : waveformSampleRate;
  auditionLoopOverlap = std::max(1, loopOverlapSamples);
  auditionPosition.store(0);
  auditionReadPosition.store(carriedPosition);
  auditionActive.store(true);
  interpolator.reset();
  playing.store(true);
}

void AudioEngine::endAuditionLoop() {
  const juce::SpinLock::ScopedLockType lock(waveformLock);
  auditionActive.store(false);
  auditionPosition.store(0);
  auditionReadPosition.store(0.0);
  previousAuditionWaveform.setSize(0, 0);
  previousAuditionReadPosition.store(0.0);
  auditionTransitionSamplesTotal.store(0);
  auditionTransitionSamplesRemaining.store(0);
  interpolator.reset();
}

double AudioEngine::getPosition() const {
  return static_cast<double>(currentPosition.load()) / waveformSampleRate;
}

double AudioEngine::getDuration() const {
  if (currentWaveform.getNumSamples() == 0)
    return 0.0;
  return static_cast<double>(currentWaveform.getNumSamples()) /
         waveformSampleRate;
}

void AudioEngine::setVolumeDb(float dB) {
  // Clamp dB to range: -12 dB to +12 dB (symmetric around 0)
  dB = juce::jlimit(-12.0f, 12.0f, dB);
  // Convert dB to linear gain: gain = 10^(dB/20)
  float gain = juce::Decibels::decibelsToGain(dB, -60.0f);
  volumeGain.store(gain);
}

float AudioEngine::getVolumeDb() const {
  float gain = volumeGain.load();
  return juce::Decibels::gainToDecibels(gain, -60.0f);
}
