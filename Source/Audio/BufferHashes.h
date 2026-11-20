#pragma once

#include "../JuceHeader.h"

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

/**
 * Block hashes of an audio buffer, to find where two versions of the same
 * timeline differ without keeping a copy of the old one: a render swapped in
 * while playing must not be crossfaded inside the part it changed (two renders
 * at unrelated phases cancel), only where the two are the same audio.
 */
namespace bufferhashes
{
constexpr int kBlock = 2048;

inline std::vector<uint64_t> of(const juce::AudioBuffer<float>& buffer)
{
    std::vector<uint64_t> hashes;
    if (buffer.getNumChannels() <= 0)
        return hashes;
    const int samples = buffer.getNumSamples();
    hashes.reserve(static_cast<size_t>(samples / kBlock + 1));
    for (int start = 0; start < samples; start += kBlock)
    {
        uint64_t h = 1469598103934665603ull; // FNV-1a over the samples' bits
        const int end = std::min(samples, start + kBlock);
        for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
        {
            const float* x = buffer.getReadPointer(channel);
            for (int i = start; i < end; ++i)
            {
                uint32_t bits = 0;
                std::memcpy(&bits, x + i, sizeof(bits));
                h = (h ^ bits) * 1099511628211ull;
            }
        }
        hashes.push_back(h);
    }
    return hashes;
}

// [start, end) in samples where the two differ (block resolution); a
// different length counts as differing everywhere; {-1, -1} when the same.
inline std::pair<int64_t, int64_t> differing(const std::vector<uint64_t>& a,
                                             const std::vector<uint64_t>& b,
                                             int64_t lengthSamples)
{
    if (a.size() != b.size())
        return {0, lengthSamples};
    int64_t first = -1, last = -1;
    for (size_t k = 0; k < a.size(); ++k)
        if (a[k] != b[k])
        {
            if (first < 0)
                first = static_cast<int64_t>(k) * kBlock;
            last = static_cast<int64_t>(k + 1) * kBlock;
        }
    return {first, last};
}
} // namespace bufferhashes
