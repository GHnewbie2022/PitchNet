// A render that lands while playing is swapped in at the playhead. Inside the
// re-rendered part the old and new audio are two renders at unrelated phases,
// and a crossfade there cancels (a click heard once). The swap waits for the
// playhead to leave that part, and then the two buffers agree where they meet.
#include "TestAssert.h"

#include "../Source/Audio/AudioEngine.h"
#include "../Source/Audio/RealtimePitchProcessor.h"
#include "../Source/Models/Project.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
constexpr double kRate = 48000.0;
constexpr double kPi = 3.14159265358979323846;

juce::AudioBuffer<float> tone(int samples, double phase)
{
    juce::AudioBuffer<float> b(1, samples);
    for (int i = 0; i < samples; ++i)
        b.setSample(0, i, static_cast<float>(0.5 * std::sin(2 * kPi * 220.0 * i / kRate + phase)));
    return b;
}

// Plays blocks of 512 and returns what came out (channel 0).
std::vector<float> playBlocks(AudioEngine &engine, int blocks)
{
    std::vector<float> out;
    juce::AudioBuffer<float> block(2, 512);
    for (int k = 0; k < blocks; ++k)
    {
        juce::AudioSourceChannelInfo info(&block, 0, 512);
        engine.getNextAudioBlock(info);
        out.insert(out.end(), block.getReadPointer(0), block.getReadPointer(0) + 512);
    }
    return out;
}

double lowestLevelDb(const std::vector<float> &x)
{
    double lowest = 1e9;
    for (size_t i = 0; i + 256 <= x.size(); i += 64)
    {
        double e = 0;
        for (size_t j = i; j < i + 256; ++j)
            e += static_cast<double>(x[j]) * x[j];
        lowest = std::min(lowest, 10.0 * std::log10(e / 256 + 1e-20));
    }
    return lowest;
}

void swapWaitsForThePlayheadToLeaveTheChangedPart()
{
    juce::ScopedJuceInitialiser_GUI juce;
    AudioEngine engine;
    engine.prepareToPlay(512, kRate);
    const int length = static_cast<int>(kRate * 4);
    auto before = tone(length, 0.0);
    engine.loadWaveform(before, static_cast<int>(kRate), false);
    engine.seek(1.0);
    engine.play();
    playBlocks(engine, 4);

    // The re-render: 0.9 - 1.6 s now the same tone at the opposite phase.
    auto after = before;
    const auto flipped = tone(length, kPi);
    const int from = static_cast<int>(0.9 * kRate), to = static_cast<int>(1.6 * kRate);
    after.copyFrom(0, from, flipped, 0, from, to - from);
    engine.loadWaveform(after, static_cast<int>(kRate), true); // playing: the deferred handoff

    // Play on through the changed part and beyond it.
    const auto out = playBlocks(engine, static_cast<int>(0.8 * kRate / 512));
    const auto report = engine.getSwapReport();
    const double level = 20.0 * std::log10(0.5 / std::sqrt(2.0));
    std::printf("  swap at %.3f s after waiting %.2f s (forced %d); lowest level %.1f dB (tone %.1f dB)\n",
                report.positionSeconds, report.waitedSeconds, report.forcedInside ? 1 : 0,
                lowestLevelDb(out), level);
    CHECK(report.count == 1);
    CHECK(!report.forcedInside);
    CHECK(report.positionSeconds >= 1.6 - 0.05); // after the changed part
    CHECK(lowestLevelDb(out) > level - 1.0);      // no cancellation
}
// A loop inside the re-rendered part never leaves it, so the swap is made
// there after a wait. Its crossfade is only in what was output then: playing
// the swap point again plays the new audio as it is (it used to be written
// into the buffer, a click on every pass until the buffer was rebuilt).
void forcedSwapLeavesTheNewAudioUntouched()
{
    juce::ScopedJuceInitialiser_GUI juce;
    AudioEngine engine;
    engine.prepareToPlay(512, kRate);
    const int length = static_cast<int>(kRate * 4);
    auto before = tone(length, 0.0);
    engine.loadWaveform(before, static_cast<int>(kRate), false);
    engine.setLoopRange(1.0, 1.3);
    engine.setLoopEnabled(true);
    engine.seek(1.0);
    engine.play();
    playBlocks(engine, 2);

    auto after = before;
    const auto flipped = tone(length, kPi);
    const int from = static_cast<int>(0.9 * kRate), to = static_cast<int>(1.6 * kRate);
    after.copyFrom(0, from, flipped, 0, from, to - from);
    engine.loadWaveform(after, static_cast<int>(kRate), true);
    playBlocks(engine, 260); // past the wait: the swap is made inside the part
    const auto report = engine.getSwapReport();
    CHECK(report.count == 1);
    CHECK(report.forcedInside);

    // Play the swap point again.
    engine.pause();
    engine.seek(report.positionSeconds);
    engine.play();
    const auto again = playBlocks(engine, 2);
    const int at = static_cast<int>(std::llround(report.positionSeconds * kRate));
    double worst = 0.0;
    for (int i = 0; i < 1024; ++i)
        worst = std::max(worst, std::abs(static_cast<double>(again[static_cast<size_t>(i)]) -
                                         after.getSample(0, at + i)));
    std::printf("  forced swap at %.3f s; replayed, largest difference from the new audio %.6f\n",
                report.positionSeconds, worst);
    CHECK(worst < 1.0e-6);
}
// The plug-in's cache handoff (RealtimePitchProcessor) waits the same way.
void pluginSwapWaitsForThePlayheadToLeaveTheChangedPart()
{
    juce::ScopedJuceInitialiser_GUI juce;
    Project project;
    auto &audio = project.getAudioData();
    audio.sampleRate = static_cast<int>(kRate);
    const int length = static_cast<int>(kRate * 4);
    audio.waveform = tone(length, 0.0);
    RealtimePitchProcessor processor;
    processor.setProject(&project);
    processor.prepareToPlay(kRate, 512);
    processor.invalidate();
    processor.waitForPendingUpdate();

    juce::AudioBuffer<float> input(1, 512), output(1, 512);
    input.clear();
    juce::int64 at = static_cast<juce::int64>(1.0 * kRate);
    auto block = [&]() {
        juce::AudioPlayHead::PositionInfo info;
        info.setIsPlaying(true);
        info.setTimeInSamples(at);
        processor.processBlock(input, output, &info);
        at += 512;
        return std::vector<float>(output.getReadPointer(0), output.getReadPointer(0) + 512);
    };
    block();
    block();

    const auto flipped = tone(length, kPi);
    const int from = static_cast<int>(0.9 * kRate), to = static_cast<int>(1.6 * kRate);
    audio.waveform.copyFrom(0, from, flipped, 0, from, to - from);
    processor.invalidate();
    processor.waitForPendingUpdate();

    std::vector<float> out;
    for (int k = 0; k < static_cast<int>(0.8 * kRate / 512); ++k)
    {
        const auto b = block();
        out.insert(out.end(), b.begin(), b.end());
    }
    // Still the old (unflipped) tone just before the changed part ends, the
    // new audio after it, and no dip anywhere.
    const int beforeEnd = static_cast<int>(1.55 * kRate - 1.0 * kRate) - 2 * 512;
    const double level = 20.0 * std::log10(0.5 / std::sqrt(2.0));
    double oldMatch = 0.0;
    for (int i = 0; i < 256; ++i)
        oldMatch += std::abs(out[static_cast<size_t>(beforeEnd + i)] -
                             tone(length, 0.0).getSample(0, static_cast<int>(1.55 * kRate) + i));
    std::printf("  plug-in: old audio kept inside the changed part (diff %.6f); lowest level %.1f dB (tone %.1f dB)\n",
                oldMatch / 256, lowestLevelDb(out), level);
    CHECK(oldMatch / 256 < 1.0e-4);
    CHECK(lowestLevelDb(out) > level - 1.0);
}
} // namespace

int main()
{
    pluginSwapWaitsForThePlayheadToLeaveTheChangedPart();
    swapWaitsForThePlayheadToLeaveTheChangedPart();
    forcedSwapLeavesTheNewAudioUntouched();
    std::printf("AudioEngineSwapTests passed\n");
    return 0;
}
