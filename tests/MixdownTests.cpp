// The export of the whole (spec section 15): the voice and the
// accompaniment added, each at its fader's level, as mixdown::mix makes it
// for the App's export - and the voice alone, which is the export it was.
#include "TestAssert.h"

#include "../Source/Audio/Mixdown.h"

#include <cmath>
#include <cstdio>

namespace
{
constexpr double kPi = 3.14159265358979323846;

juce::AudioBuffer<float> tone(int channels, int samples, double hz, double level, double rate)
{
    juce::AudioBuffer<float> b(channels, samples);
    for (int ch = 0; ch < channels; ++ch)
        for (int i = 0; i < samples; ++i)
            b.setSample(ch, i, static_cast<float>(level * (ch == 0 ? 1.0 : 0.6) *
                                                  std::sin(2 * kPi * hz * i / rate)));
    return b;
}

// 25. Each at its fader's gain, added; the peak is told.
void theTwoAdded()
{
    const int rate = 48000, length = rate;
    const auto voice = tone(2, length, 220.0, 0.5, rate);
    const auto backing = tone(2, length, 330.0, 0.7, rate);
    const auto result = mixdown::mix({&voice, rate, -3.0f}, {&backing, rate, -6.0f}, rate, 2);
    CHECK(result.sampleRate == rate && result.audio.getNumChannels() == 2 &&
          result.audio.getNumSamples() == length);
    const float gv = juce::Decibels::decibelsToGain(-3.0f), gb = juce::Decibels::decibelsToGain(-6.0f);
    float peak = 0.0f;
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < length; ++i)
        {
            const float wanted = gv * voice.getSample(ch, i) + gb * backing.getSample(ch, i);
            CHECK(std::abs(result.audio.getSample(ch, i) - wanted) < 1.0e-6f);
            peak = std::max(peak, std::abs(wanted));
        }
    CHECK(std::abs(result.peak - peak) < 1.0e-6f);

    // Over full scale is told, not turned down.
    const auto loud = mixdown::mix({&voice, rate, 6.0f}, {&backing, rate, 6.0f}, rate, 2);
    CHECK(loud.peak > 1.0f);
    // A fader at its bottom is silence; no accompaniment is the voice alone.
    const auto silentBacking = mixdown::mix({&voice, rate, 0.0f}, {&backing, rate, -60.0f}, rate, 2);
    const auto noBacking = mixdown::mix({&voice, rate, 0.0f}, {}, rate, 2);
    for (int i = 0; i < length; ++i)
    {
        CHECK(silentBacking.audio.getSample(0, i) == voice.getSample(0, i));
        CHECK(noBacking.audio.getSample(1, i) == voice.getSample(1, i));
    }
}

// 26. At different rates: each brought to the rate asked for, then added; as
// long as the longer.
void atDifferentRates()
{
    const auto voice = tone(1, 44100, 220.0, 0.5, 44100.0);      // 1 s at 44.1 kHz
    const auto backing = tone(1, 48000 * 2, 330.0, 0.4, 48000.0); // 2 s at 48 kHz
    const auto result = mixdown::mix({&voice, 44100, 0.0f}, {&backing, 48000, 0.0f}, 48000, 1);
    CHECK(result.sampleRate == 48000);
    CHECK(std::abs(result.audio.getNumSamples() - 96000) <= 2);
    // The second half is the accompaniment alone, untouched (its own rate).
    for (int i = 60000; i < 90000; ++i)
        CHECK(std::abs(result.audio.getSample(0, i) - backing.getSample(0, i)) < 1.0e-6f);
    // The first half holds both tones: against the accompaniment alone, what
    // is left is the voice's 220 Hz at 0.5.
    double sum = 0.0;
    for (int i = 9600; i < 38400; ++i)
    {
        const double left = result.audio.getSample(0, i) - backing.getSample(0, i);
        sum += left * left;
    }
    const double rms = std::sqrt(sum / 28800.0);
    CHECK(std::abs(rms - 0.5 / std::sqrt(2.0)) < 0.005);
    // Asked for at 44.1 kHz, both are there and it is 2 s long.
    const auto lower = mixdown::mix({&voice, 44100, 0.0f}, {&backing, 48000, 0.0f}, 44100, 1);
    CHECK(std::abs(lower.audio.getNumSamples() - 88200) <= 2);
    for (int i = 4410; i < 39690; ++i)
        CHECK(std::abs(lower.audio.getSample(0, i) -
                       (voice.getSample(0, i) + 0.4 * std::sin(2 * kPi * 330.0 * i / 44100.0))) < 2.0e-3);
}

// 27. A mono voice and a stereo accompaniment: the voice on both sides; to
// one channel, the accompaniment's two averaged.
void aMonoVoiceAndAStereoAccompaniment()
{
    const int rate = 48000, length = 4800;
    const auto voice = tone(1, length, 220.0, 0.5, rate);
    const auto backing = tone(2, length, 330.0, 0.4, rate);
    const auto stereo = mixdown::mix({&voice, rate, 0.0f}, {&backing, rate, 0.0f}, rate, 2);
    const auto mono = mixdown::mix({&voice, rate, 0.0f}, {&backing, rate, 0.0f}, rate, 1);
    CHECK(stereo.audio.getNumChannels() == 2 && mono.audio.getNumChannels() == 1);
    for (int i = 0; i < length; ++i)
    {
        for (int ch = 0; ch < 2; ++ch)
            CHECK(std::abs(stereo.audio.getSample(ch, i) -
                           (voice.getSample(0, i) + backing.getSample(ch, i))) < 1.0e-6f);
        CHECK(std::abs(mono.audio.getSample(0, i) -
                       (voice.getSample(0, i) +
                        0.5f * (backing.getSample(0, i) + backing.getSample(1, i)))) < 1.0e-6f);
    }
}

// 28. The voice alone, at 0 dB with no accompaniment, is what its own export
// makes: the channels, then the rate - sample for sample.
void theVoiceAloneIsItsExport()
{
    const auto voice = tone(2, 44100, 220.0, 0.5, 44100.0);
    for (const int channels : {1, 2})
        for (const int rate : {44100, 48000})
        {
            auto exported = mixdown::toChannels(voice, channels);
            exported = AudioResampler::resample(exported, 44100, rate);
            const auto mixed = mixdown::mix({&voice, 44100, 0.0f}, {}, rate, channels);
            CHECK(mixed.audio.getNumChannels() == exported.getNumChannels() &&
                  mixed.audio.getNumSamples() == exported.getNumSamples());
            for (int ch = 0; ch < channels; ++ch)
                for (int i = 0; i < exported.getNumSamples(); ++i)
                    CHECK(mixed.audio.getSample(ch, i) == exported.getSample(ch, i));
        }
    CHECK(mixdown::faderGain(-60.0f) == 0.0f && mixdown::faderGain(-80.0f) == 0.0f);
    CHECK(mixdown::faderGain(0.0f) == 1.0f);
    CHECK(std::abs(mixdown::faderGain(40.0f) - juce::Decibels::decibelsToGain(12.0f)) < 1.0e-6f);
}
} // namespace

int main()
{
    juce::ScopedJuceInitialiser_GUI juce;
    theTwoAdded();
    atDifferentRates();
    aMonoVoiceAndAStereoAccompaniment();
    theVoiceAloneIsItsExport();
    std::printf("MixdownTests: all passed\n");
    return 0;
}
