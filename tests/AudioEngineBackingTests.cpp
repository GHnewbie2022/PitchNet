// The accompaniment beside the project's audio, and a sound laid over the
// transport once (spec section 15): what the playback engine puts out, block
// by block, as the App's device callback asks for it.
#include "TestAssert.h"

#include "../Source/Audio/AudioEngine.h"
#include "../Source/Audio/KeyFinder/ScaleTone.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
constexpr double kRate = 48000.0;
constexpr double kPi = 3.14159265358979323846;
constexpr int kBlock = 512;

juce::AudioBuffer<float> tone(int channels, int samples, double hz, double level, double rate = kRate)
{
    juce::AudioBuffer<float> b(channels, samples);
    for (int ch = 0; ch < channels; ++ch)
        for (int i = 0; i < samples; ++i)
            b.setSample(ch, i, static_cast<float>(level * (ch == 0 ? 1.0 : 0.5) *
                                                  std::sin(2 * kPi * hz * i / rate)));
    return b;
}

// Plays `blocks` blocks of 512 and returns what came out, both channels.
juce::AudioBuffer<float> playBlocks(AudioEngine &engine, int blocks)
{
    juce::AudioBuffer<float> out(2, blocks * kBlock);
    juce::AudioBuffer<float> block(2, kBlock);
    for (int k = 0; k < blocks; ++k)
    {
        juce::AudioSourceChannelInfo info(&block, 0, kBlock);
        engine.getNextAudioBlock(info);
        for (int ch = 0; ch < 2; ++ch)
            out.copyFrom(ch, k * kBlock, block, ch, 0, kBlock);
    }
    return out;
}

double largestDifference(const juce::AudioBuffer<float> &out, int channel, int from, int count,
                         const std::function<double(int)> &wanted)
{
    double worst = 0.0;
    for (int i = 0; i < count; ++i)
        worst = std::max(worst, std::abs(static_cast<double>(out.getSample(channel, from + i)) - wanted(i)));
    return worst;
}

float gainOf(float dB) { return juce::Decibels::decibelsToGain(dB); }

// 18. With the accompaniment off, the engine puts out the voice as it did.
void offItChangesNothing()
{
    const int length = static_cast<int>(kRate * 2);
    const auto voice = tone(1, length, 220.0, 0.5);
    const auto backing = tone(2, length, 330.0, 0.4);

    AudioEngine plain;
    plain.prepareToPlay(kBlock, kRate);
    plain.loadWaveform(voice, static_cast<int>(kRate), false);
    plain.seek(0.25);
    plain.play();
    const auto before = playBlocks(plain, 40);

    AudioEngine engine;
    engine.prepareToPlay(kBlock, kRate);
    engine.loadWaveform(voice, static_cast<int>(kRate), false);
    engine.setBacking(backing, static_cast<int>(kRate));
    CHECK(engine.hasBacking() && !engine.isBackingAudible());
    engine.setBackingGainDb(-3.0f);
    engine.seek(0.25);
    engine.play();
    const auto out = playBlocks(engine, 40);
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < out.getNumSamples(); ++i)
            CHECK(out.getSample(ch, i) == before.getSample(ch, i));

    // Taken away again: still the voice alone.
    engine.setBackingAudible(true);
    engine.clearBacking();
    CHECK(!engine.hasBacking());
    engine.seek(0.25);
    const auto cleared = playBlocks(engine, 40);
    for (int i = 0; i < cleared.getNumSamples(); ++i)
        CHECK(cleared.getSample(0, i) == before.getSample(0, i));
}

// 19, 22. On: the voice and the accompaniment, each at its own level, added;
// the voice not heard: the accompaniment alone.
void theTwoAreAdded()
{
    const int length = static_cast<int>(kRate * 2);
    const auto voice = tone(1, length, 220.0, 0.5);
    const auto backing = tone(2, length, 330.0, 0.4);
    AudioEngine engine;
    engine.prepareToPlay(kBlock, kRate);
    engine.loadWaveform(voice, static_cast<int>(kRate), false);
    engine.setBacking(backing, static_cast<int>(kRate));
    engine.setBackingAudible(true);
    engine.setBackingGainDb(-6.0f);
    engine.setVoiceGainDb(-3.0f);
    CHECK(std::abs(engine.getBackingGainDb() + 6.0f) < 1.0e-4f);
    CHECK(std::abs(engine.getVoiceGainDb() + 3.0f) < 1.0e-4f);
    const int start = static_cast<int>(0.5 * kRate);
    engine.seek(0.5);
    engine.play();
    auto out = playBlocks(engine, 20);
    for (int ch = 0; ch < 2; ++ch)
    {
        const double worst = largestDifference(out, ch, 0, out.getNumSamples(), [&](int i) {
            return static_cast<double>(gainOf(-3.0f)) * voice.getSample(0, start + i) +
                   static_cast<double>(gainOf(-6.0f)) * backing.getSample(ch, start + i);
        });
        CHECK(worst < 1.0e-6);
    }
    CHECK(std::abs(engine.getPosition() - (0.5 + 20 * kBlock / kRate)) < 1.0e-9);

    // The voice not heard (a key is checked against the accompaniment alone).
    engine.setVoiceAudible(false);
    engine.seek(0.5);
    out = playBlocks(engine, 20);
    for (int ch = 0; ch < 2; ++ch)
        CHECK(largestDifference(out, ch, 0, out.getNumSamples(), [&](int i) {
                  return static_cast<double>(gainOf(-6.0f)) * backing.getSample(ch, start + i);
              }) < 1.0e-6);
    engine.setVoiceAudible(true);

    // The fader at its bottom is silence; a mono accompaniment goes to both sides.
    engine.setBackingGainDb(-60.0f);
    engine.seek(0.5);
    out = playBlocks(engine, 4);
    CHECK(largestDifference(out, 1, 0, out.getNumSamples(), [&](int i) {
              return static_cast<double>(gainOf(-3.0f)) * voice.getSample(0, start + i);
          }) < 1.0e-6);
    engine.setBackingGainDb(0.0f);
    engine.setVoiceGainDb(0.0f);
    const auto mono = tone(1, length, 330.0, 0.4);
    engine.setBacking(mono, static_cast<int>(kRate));
    engine.seek(0.5);
    out = playBlocks(engine, 4);
    for (int ch = 0; ch < 2; ++ch)
        CHECK(largestDifference(out, ch, 0, out.getNumSamples(), [&](int i) {
                  return static_cast<double>(voice.getSample(0, start + i)) + mono.getSample(0, start + i);
              }) < 1.0e-6);
}

// 20. A seek and a loop: the accompaniment is where the voice is.
void itFollowsTheTransport()
{
    const int length = static_cast<int>(kRate * 3);
    // Each is its own position: the sample at i is i / length.
    juce::AudioBuffer<float> voice(1, length), backing(1, length);
    for (int i = 0; i < length; ++i)
    {
        voice.setSample(0, i, static_cast<float>(i) / length);
        backing.setSample(0, i, -static_cast<float>(i) / length);
    }
    AudioEngine engine;
    engine.prepareToPlay(kBlock, kRate);
    engine.loadWaveform(voice, static_cast<int>(kRate), false);
    engine.setBacking(backing, static_cast<int>(kRate));
    engine.setBackingAudible(true);
    engine.setBackingGainDb(0.0f);

    engine.seek(1.75);
    engine.play();
    auto out = playBlocks(engine, 8);
    for (int i = 0; i < out.getNumSamples(); ++i)
        CHECK(std::abs(out.getSample(0, i)) < 1.0e-6f); // they cancel wherever they are together

    // A loop of 0.3 s, shorter than what is played: through several wraps,
    // and in the middle of a block.
    engine.setLoopRange(1.0, 1.3);
    engine.setLoopEnabled(true);
    engine.seek(1.0);
    out = playBlocks(engine, 80); // 0.85 s
    for (int i = 0; i < out.getNumSamples(); ++i)
        CHECK(std::abs(out.getSample(0, i)) < 1.0e-6f);
    // ...and with the voice not heard, the accompaniment alone walks the loop.
    engine.setVoiceAudible(false);
    engine.seek(1.0);
    out = playBlocks(engine, 80);
    const int loopStart = static_cast<int>(1.0 * kRate), loopLength = static_cast<int>(0.3 * kRate);
    for (int i = 0; i < out.getNumSamples(); ++i)
        CHECK(std::abs(out.getSample(0, i) - backing.getSample(0, loopStart + i % loopLength)) < 1.0e-6f);
}

// 21. Shorter than the voice: the voice alone from where it ends, and the
// transport goes on to the voice's end.
void shorterThanTheVoice()
{
    const int length = static_cast<int>(kRate * 2);
    const int shorter = static_cast<int>(kRate * 1.2) + 137;
    const auto voice = tone(1, length, 220.0, 0.5);
    const auto backing = tone(2, shorter, 330.0, 0.4);
    AudioEngine engine;
    engine.prepareToPlay(kBlock, kRate);
    engine.loadWaveform(voice, static_cast<int>(kRate), false);
    engine.setBacking(backing, static_cast<int>(kRate));
    engine.setBackingAudible(true);
    engine.setBackingGainDb(0.0f);
    engine.seek(1.0);
    engine.play();
    const int start = static_cast<int>(kRate);
    const auto out = playBlocks(engine, 60); // to 1.64 s
    CHECK(largestDifference(out, 0, 0, out.getNumSamples(), [&](int i) {
              const int at = start + i;
              return static_cast<double>(voice.getSample(0, at)) +
                     (at < shorter ? backing.getSample(0, at) : 0.0f);
          }) < 1.0e-6);
    CHECK(std::abs(engine.getDuration() - 2.0) < 1.0e-9);
    // Longer than the voice: the transport still ends with the voice.
    engine.setBacking(tone(2, length * 2, 330.0, 0.4), static_cast<int>(kRate));
    engine.seek(1.9);
    playBlocks(engine, 20);
    CHECK(!engine.isPlaying());
}

// 23. A sound laid over once: from its start to its end whatever the
// transport does, no playhead moved, gone when it has played.
void aSoundLaidOverOnce()
{
    const int length = static_cast<int>(kRate * 6);
    const auto voice = tone(1, length, 220.0, 0.25);
    const auto scale = keyfinder::scaleTone(60, 0.0, kRate);
    AudioEngine engine;
    engine.prepareToPlay(kBlock, kRate);
    engine.loadWaveform(voice, static_cast<int>(kRate), false);
    engine.setVoiceAudible(false);
    engine.seek(2.0);
    engine.play();
    playBlocks(engine, 3);
    const double before = engine.getPosition();
    engine.playOverlay(scale, static_cast<int>(kRate));
    engine.setOverlayGainDb(0.0f);
    CHECK(engine.isOverlayPlaying());
    CHECK(engine.getPosition() == before); // no playhead moved
    const int blocks = scale.getNumSamples() / kBlock + 2;
    const auto out = playBlocks(engine, blocks);
    CHECK(!engine.isOverlayPlaying());
    // The scale itself, then nothing; the transport went on at its own pace.
    CHECK(largestDifference(out, 0, 0, out.getNumSamples(), [&](int i) {
              return i < scale.getNumSamples() ? static_cast<double>(scale.getSample(0, i)) : 0.0;
          }) < 1.0e-6);
    CHECK(largestDifference(out, 1, 0, out.getNumSamples(), [&](int i) {
              return i < scale.getNumSamples() ? static_cast<double>(scale.getSample(0, i)) : 0.0;
          }) < 1.0e-6);
    CHECK(std::abs(engine.getPosition() - (before + blocks * kBlock / kRate)) < 1.0e-9);
    // No step where it starts or ends, nor between its notes: from one
    // sample to the next never more than a sine at its top frequency moves.
    const double largestStep = keyfinder::kScalePeak * 2.0 * kPi * 523.3 / kRate * 1.05;
    for (int i = 1; i < out.getNumSamples(); ++i)
        CHECK(std::abs(out.getSample(0, i) - out.getSample(0, i - 1)) <= largestStep);

    // Its own level; over a loop it is not started again; a seek leaves it be.
    engine.setOverlayGainDb(-6.0f);
    engine.setLoopRange(2.0, 2.2);
    engine.setLoopEnabled(true);
    engine.seek(2.0);
    engine.playOverlay(scale, static_cast<int>(kRate));
    auto looped = playBlocks(engine, 30); // 0.32 s: one wrap of the loop
    engine.seek(2.05);
    const auto more = playBlocks(engine, 10);
    CHECK(largestDifference(looped, 0, 0, looped.getNumSamples(), [&](int i) {
              return static_cast<double>(gainOf(-6.0f)) * scale.getSample(0, i);
          }) < 1.0e-6);
    CHECK(largestDifference(more, 0, 0, more.getNumSamples(), [&](int i) {
              return static_cast<double>(gainOf(-6.0f)) * scale.getSample(0, looped.getNumSamples() + i);
          }) < 1.0e-6);

    // Stopped with the transport, and not carried over a pause.
    CHECK(engine.isOverlayPlaying());
    engine.pause();
    CHECK(!engine.isOverlayPlaying());
    engine.play();
    const auto after = playBlocks(engine, 4);
    for (int i = 0; i < after.getNumSamples(); ++i)
        CHECK(after.getSample(0, i) == 0.0f);
    engine.playOverlay(scale, static_cast<int>(kRate));
    engine.stop();
    CHECK(!engine.isOverlayPlaying());
    engine.playOverlay(scale, static_cast<int>(kRate));
    engine.stopOverlay();
    CHECK(!engine.isOverlayPlaying());
}

// 24. The device changes rate: both are converted to it, and a place in
// seconds is the same place in both.
void theDeviceChangesRate()
{
    // Two clicks a second apart in each, at the same times.
    const int rate = 48000;
    const int length = rate * 3;
    juce::AudioBuffer<float> voice(1, length), backing(1, length);
    voice.clear();
    backing.clear();
    for (const double at : {1.0, 2.0})
        for (int i = 0; i < 480; ++i)
        {
            const float pulse = static_cast<float>(std::sin(kPi * i / 480.0));
            voice.setSample(0, static_cast<int>(at * rate) + i, pulse * 0.5f);
            backing.setSample(0, static_cast<int>(at * rate) + i, pulse * -0.5f);
        }
    AudioEngine engine;
    engine.prepareToPlay(kBlock, rate);
    engine.loadWaveform(voice, rate, false);
    engine.setBacking(backing, rate);
    engine.setBackingAudible(true);
    engine.setBackingGainDb(0.0f);
    engine.seek(0.9);

    engine.prepareToPlay(kBlock, 44100.0); // the device now runs at 44.1 kHz
    engine.convertForDeviceRate();
    CHECK(std::abs(engine.getDuration() - 3.0) < 1.0e-3);
    CHECK(std::abs(engine.getPosition() - 0.9) < 1.0e-3);
    engine.play();
    auto out = playBlocks(engine, 120); // 0.9 s to 2.29 s at 44.1 kHz
    // Converted alike, they still cancel: neither click is left.
    CHECK(out.getMagnitude(0, 0, out.getNumSamples()) < 1.0e-4f);
    // The accompaniment alone has its clicks at 1 s and 2 s.
    engine.setVoiceAudible(false);
    engine.seek(0.9);
    out = playBlocks(engine, 120);
    int peakAt = 0;
    for (int i = 0; i < out.getNumSamples(); ++i)
        if (std::abs(out.getSample(0, i)) > std::abs(out.getSample(0, peakAt)))
            peakAt = i;
    const double peakSeconds = 0.9 + peakAt / 44100.0;
    const double nearest = std::abs(peakSeconds - 1.005) < std::abs(peakSeconds - 2.005) ? 1.005 : 2.005;
    CHECK(std::abs(peakSeconds - nearest) < 0.002);
    CHECK(out.getMagnitude(0, 0, out.getNumSamples()) > 0.45f);
    std::printf("  at 44.1 kHz: a click of the accompaniment at %.4f s\n", peakSeconds);
}
} // namespace

int main()
{
    juce::ScopedJuceInitialiser_GUI juce;
    offItChangesNothing();
    theTwoAreAdded();
    itFollowsTheTransport();
    shorterThanTheVoice();
    aSoundLaidOverOnce();
    theDeviceChangesRate();
    std::printf("AudioEngineBackingTests: all passed\n");
    return 0;
}
