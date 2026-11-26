#include "AccompanimentChroma.h"

#include "../../Utils/AudioResampler.h"

#include <algorithm>
#include <cmath>
#include <complex>

namespace keyfinder
{
namespace
{
constexpr double kRate = 12000.0;
constexpr int kFftOrder = 13;
constexpr int kWindow = 1 << kFftOrder; // 8192 samples: 1.46 Hz a bin
constexpr int kHop = 1024;
constexpr int kLowNote = 33;  // A1, 55 Hz
constexpr int kHighNote = 96; // C7, 2093 Hz
constexpr int kBinsPerSemitone = 3;
constexpr int kLogBins = (kHighNote - kLowNote + 1) * kBinsPerSemitone;
constexpr int kMedian = 17; // frames along time, bins along frequency
constexpr double kPi = 3.14159265358979323846;

// The channels averaged, the loudest sample at 1, at 12 kHz.
std::vector<float> monoAt12k(const juce::AudioBuffer<float>& audio, double sampleRate)
{
    const int channels = audio.getNumChannels();
    const int samples = audio.getNumSamples();
    if (channels <= 0 || samples <= 0 || sampleRate <= 0.0)
        return {};
    juce::AudioBuffer<float> mono(1, samples);
    mono.clear();
    for (int ch = 0; ch < channels; ++ch)
        mono.addFrom(0, 0, audio, ch, 0, samples, 1.0f / static_cast<float>(channels));
    const float peak = mono.getMagnitude(0, samples);
    if (peak > 1.0e-9f)
        mono.applyGain(1.0f / peak);
    if (std::abs(sampleRate - kRate) > 1.0e-6)
        mono = AudioResampler::resample(mono, sampleRate, kRate);
    const float* data = mono.getReadPointer(0);
    return std::vector<float>(data, data + mono.getNumSamples());
}

double midiOf(double hz) { return 69.0 + 12.0 * std::log2(std::max(hz, 1.0e-9) / 440.0); }

// One row of the map from FFT bins to a bin of the logarithmic grid.
struct LogBin
{
    int first = 0;
    std::vector<float> weights;
};

// Edge values are repeated outwards, as scipy's median_filter does by default.
int reflected(int index, int size)
{
    while (index < 0 || index >= size)
        index = index < 0 ? -index - 1 : 2 * size - index - 1;
    return index;
}

float medianOf(std::array<float, kMedian>& values)
{
    std::nth_element(values.begin(), values.begin() + kMedian / 2, values.end());
    return values[kMedian / 2];
}
} // namespace

Chroma analyse(const juce::AudioBuffer<float>& audio, double sampleRate)
{
    Chroma chroma;
    chroma.frameSeconds = static_cast<double>(kHop) / kRate;
    const auto samples = monoAt12k(audio, sampleRate);
    const int total = static_cast<int>(samples.size());
    if (total < kWindow)
        return chroma;
    const int frames = 1 + (total - kWindow) / kHop;
    const int bins = kWindow / 2 + 1;

    std::vector<float> window(static_cast<size_t>(kWindow));
    for (int i = 0; i < kWindow; ++i)
        window[static_cast<size_t>(i)] =
            static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * i / (kWindow - 1)));

    juce::dsp::FFT fft(kFftOrder);
    std::vector<float> work(static_cast<size_t>(2 * kWindow));
    const auto magnitudes = [&](int frame)
    {
        const float* in = samples.data() + static_cast<size_t>(frame) * kHop;
        for (int i = 0; i < kWindow; ++i)
            work[static_cast<size_t>(i)] = in[i] * window[static_cast<size_t>(i)];
        std::fill(work.begin() + kWindow, work.end(), 0.0f);
        fft.performFrequencyOnlyForwardTransform(work.data());
    };

    // How far the recording is tuned from A = 440: the peaks of the whole
    // song's spectrum, each by how far it is from a semitone, as angles
    // (a deviation of -0.5 and of +0.5 are the same thing), by their energy.
    std::vector<double> whole(static_cast<size_t>(bins), 0.0);
    for (int frame = 0; frame < frames; ++frame)
    {
        magnitudes(frame);
        for (int b = 0; b < bins; ++b)
            whole[static_cast<size_t>(b)] += work[static_cast<size_t>(b)];
    }
    const double binHz = kRate / kWindow;
    {
        std::complex<double> mean(0.0, 0.0);
        for (int b = 1; b + 1 < bins; ++b)
        {
            const double hz = b * binHz;
            const double below = whole[static_cast<size_t>(b - 1)];
            const double here = whole[static_cast<size_t>(b)];
            const double above = whole[static_cast<size_t>(b + 1)];
            if (hz <= 100.0 || hz >= 2000.0 || !(here > below && here > above))
                continue;
            const double offset = 0.5 * (below - above) / (below - 2.0 * here + above);
            const double midi = midiOf(hz + offset * binHz);
            mean += here * here * std::polar(1.0, 2.0 * kPi * (midi - std::round(midi)));
        }
        chroma.tuneSemitones = std::abs(mean) > 0.0 ? std::arg(mean) / (2.0 * kPi) : 0.0;
    }

    // FFT bins to the logarithmic grid: each of its bins a triangle a third
    // of a semitone wide, or as wide as one FFT bin where those are wider
    // (the lowest notes), every one summing to 1.
    std::vector<double> binMidi(static_cast<size_t>(bins));
    for (int b = 0; b < bins; ++b)
        binMidi[static_cast<size_t>(b)] = midiOf(b * binHz);
    const auto binWidth = [&](int b)
    {
        if (b <= 0)
            return binMidi[1] - binMidi[0];
        if (b >= bins - 1)
            return binMidi[static_cast<size_t>(bins - 1)] - binMidi[static_cast<size_t>(bins - 2)];
        return 0.5 * (binMidi[static_cast<size_t>(b + 1)] - binMidi[static_cast<size_t>(b - 1)]);
    };
    std::vector<LogBin> logBins(static_cast<size_t>(kLogBins));
    for (int i = 0; i < kLogBins; ++i)
    {
        const double centre = kLowNote + chroma.tuneSemitones +
                              static_cast<double>(i - 1) / kBinsPerSemitone;
        const double hz = 440.0 * std::pow(2.0, (centre - 69.0) / 12.0);
        const int at = std::min(bins - 1, static_cast<int>(std::ceil(hz / binHz - 1.0e-12)));
        const double width = std::max(1.0 / kBinsPerSemitone, binWidth(at));
        auto& row = logBins[static_cast<size_t>(i)];
        double sum = 0.0;
        std::vector<double> weights;
        int first = -1;
        for (int b = 1; b < bins; ++b)
        {
            const double weight = 1.0 - std::abs(binMidi[static_cast<size_t>(b)] - centre) / width;
            if (weight <= 0.0)
            {
                if (first >= 0)
                    break;
                continue;
            }
            if (first < 0)
                first = b;
            weights.push_back(weight);
            sum += weight;
        }
        row.first = std::max(0, first);
        if (sum > 0.0)
            for (const double weight : weights)
                row.weights.push_back(static_cast<float>(weight / sum));
    }

    std::vector<float> spectrum(static_cast<size_t>(frames) * kLogBins, 0.0f);
    for (int frame = 0; frame < frames; ++frame)
    {
        magnitudes(frame);
        float* out = spectrum.data() + static_cast<size_t>(frame) * kLogBins;
        for (int i = 0; i < kLogBins; ++i)
        {
            const auto& row = logBins[static_cast<size_t>(i)];
            float value = 0.0f;
            for (size_t k = 0; k < row.weights.size(); ++k)
                value += work[static_cast<size_t>(row.first) + k] * row.weights[k];
            out[i] = value;
        }
    }

    // What holds still in time is harmonic, what is spread over frequency at
    // one moment is a hit: each bin keeps the harmonic share of itself.
    const auto at = [&](int frame, int bin) -> float
    {
        return spectrum[static_cast<size_t>(frame) * kLogBins + static_cast<size_t>(bin)];
    };
    chroma.frames.assign(static_cast<size_t>(frames), {});
    std::array<float, kMedian> values{};
    for (int frame = 0; frame < frames; ++frame)
        for (int note = kLowNote; note <= kHighNote; ++note)
        {
            const int bin = (note - kLowNote) * kBinsPerSemitone + 1; // the semitone's middle bin
            for (int k = 0; k < kMedian; ++k)
                values[static_cast<size_t>(k)] = at(reflected(frame + k - kMedian / 2, frames), bin);
            const double harmonic = medianOf(values);
            for (int k = 0; k < kMedian; ++k)
                values[static_cast<size_t>(k)] = at(frame, reflected(bin + k - kMedian / 2, kLogBins));
            const double percussive = medianOf(values);
            const double mask = harmonic * harmonic /
                                (harmonic * harmonic + percussive * percussive + 1.0e-12);
            chroma.frames[static_cast<size_t>(frame)][static_cast<size_t>(note % 12)] +=
                static_cast<float>(at(frame, bin) * mask);
        }
    return chroma;
}

std::array<double, 12> sum(const Chroma& chroma, int first, int last)
{
    std::array<double, 12> classes{};
    first = std::max(0, first);
    last = std::min(last, static_cast<int>(chroma.frames.size()));
    for (int frame = first; frame < last; ++frame)
        for (size_t c = 0; c < 12; ++c)
            classes[c] += chroma.frames[static_cast<size_t>(frame)][c];
    return classes;
}

std::array<double, 12> scaleShares(const std::array<double, 12>& classes)
{
    std::array<double, 12> shares{};
    double total = 0.0;
    for (const double value : classes)
        total += value;
    if (total <= 0.0)
        return shares;
    for (int root = 0; root < 12; ++root)
    {
        double inScale = 0.0;
        for (int c = 0; c < 12; ++c)
            if (inMajorScale(c, root))
                inScale += classes[static_cast<size_t>(c)];
        shares[static_cast<size_t>(root)] = inScale / total;
    }
    return shares;
}
} // namespace keyfinder
