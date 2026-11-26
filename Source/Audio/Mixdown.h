#pragma once

#include "../JuceHeader.h"
#include "../Utils/AudioResampler.h"

#include <algorithm>

/**
 * The voice and the accompaniment as one file (spec section 15): the export
 * of the whole, as the App plays the two together - each at the level its
 * fader is at, added. What is exported one track at a time takes no fader:
 * the voice alone is the export it always was.
 */
namespace mixdown
{
/** A fader's dB as a gain: -60 and under is silence, +12 at most. The
 *  playback engine reads its faders with the same rule. */
inline float faderGain(float dB)
{
    dB = juce::jlimit(-60.0f, 12.0f, dB);
    return dB <= -60.0f ? 0.0f : juce::Decibels::decibelsToGain(dB);
}

/**
 * `input` as one or two channels, as an export makes them: to one, the
 * average of all it has; to two, a single channel on both sides, else its
 * first two.
 */
inline juce::AudioBuffer<float> toChannels(const juce::AudioBuffer<float>& input, int outChannels)
{
    outChannels = juce::jlimit(1, 2, outChannels);
    const int inChannels = juce::jmax(1, input.getNumChannels());
    const int numSamples = input.getNumSamples();

    juce::AudioBuffer<float> output(outChannels, numSamples);
    output.clear();

    if (outChannels == 1) {
        float* dst = output.getWritePointer(0);
        for (int i = 0; i < numSamples; ++i) {
            float sum = 0.0f;
            for (int ch = 0; ch < inChannels; ++ch)
                sum += input.getSample(ch, i);
            dst[i] = sum / static_cast<float>(inChannels);
        }
        return output;
    }

    // Stereo output
    if (inChannels == 1) {
        output.copyFrom(0, 0, input, 0, 0, numSamples);
        output.copyFrom(1, 0, input, 0, 0, numSamples);
    } else {
        output.copyFrom(0, 0, input, 0, 0, numSamples);
        output.copyFrom(1, 0, input, 1, 0, numSamples);
    }
    return output;
}

struct Track
{
    const juce::AudioBuffer<float>* audio = nullptr; // null or empty: no such track
    int sampleRate = 0;
    float gainDb = 0.0f;
};

struct Mix
{
    juce::AudioBuffer<float> audio;
    int sampleRate = 0;
    float peak = 0.0f; // the largest sample: over 1 clips in a fixed-point file
};

/**
 * Both tracks brought to `channels` and `sampleRate` (channels first, then
 * the rate, as a single track is exported), each at its fader's gain, added.
 * As long as the longer of the two. Not for the audio thread.
 */
inline Mix mix(const Track& voice, const Track& backing, int sampleRate, int channels)
{
    Mix result;
    result.sampleRate = sampleRate;
    channels = juce::jlimit(1, 2, channels);
    juce::AudioBuffer<float> parts[2];
    float gains[2] = {faderGain(voice.gainDb), faderGain(backing.gainDb)};
    const Track* tracks[2] = {&voice, &backing};
    int length = 0;
    for (int t = 0; t < 2; ++t)
    {
        const auto& track = *tracks[t];
        if (track.audio == nullptr || track.audio->getNumSamples() <= 0 || track.sampleRate <= 0)
            continue;
        parts[t] = toChannels(*track.audio, channels);
        if (track.sampleRate != sampleRate)
            parts[t] = AudioResampler::resample(parts[t], track.sampleRate, sampleRate);
        length = std::max(length, parts[t].getNumSamples());
    }
    result.audio.setSize(channels, length);
    result.audio.clear();
    for (int t = 0; t < 2; ++t)
        for (int ch = 0; ch < channels && parts[t].getNumSamples() > 0; ++ch)
            result.audio.addFrom(ch, 0, parts[t], ch, 0, parts[t].getNumSamples(), gains[t]);
    result.peak = length > 0 ? result.audio.getMagnitude(0, length) : 0.0f;
    return result;
}
} // namespace mixdown
