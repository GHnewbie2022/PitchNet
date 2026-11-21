// Spec Q12: notes rendered with Rubber Band R3 (RubberBandShift.h).
#include "TestAssert.h"

#include "../Source/Audio/Synthesis/RubberBandShift.h"
#include "../Source/Models/ProjectSerializer.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr int kRate = 44100;

std::vector<float> tone(double hz, int samples)
{
  std::vector<float> x(static_cast<size_t>(samples));
  for (int i = 0; i < samples; ++i)
    x[static_cast<size_t>(i)] = static_cast<float>(
        0.3 * std::sin(2 * kPi * hz * i / kRate) + 0.1 * std::sin(2 * kPi * 2 * hz * i / kRate));
  return x;
}

double peakHz(const std::vector<float> &x, int from, int to)
{
  // Plain DFT around the expected range, 0.1 Hz steps.
  double best = 0.0, bestHz = 0.0;
  for (double hz = 380.0; hz < 470.0; hz += 0.1)
  {
    double re = 0.0, im = 0.0;
    for (int i = from; i < to; ++i)
    {
      const double w = 0.5 - 0.5 * std::cos(2 * kPi * (i - from) / (to - from));
      re += w * x[static_cast<size_t>(i)] * std::cos(2 * kPi * hz * i / kRate);
      im += w * x[static_cast<size_t>(i)] * std::sin(2 * kPi * hz * i / kRate);
    }
    if (re * re + im * im > best)
    {
      best = re * re + im * im;
      bestHz = hz;
    }
  }
  return bestHz;
}

// With and without formant preservation (OptionFormantPreserved), which
// the large-shift listening test compares.
void shiftsByTheRatio(bool preserveFormant)
{
  const int samples = kRate * 2;
  const auto source = tone(440.0, samples);
  const float ratio = static_cast<float>(std::pow(2.0, -1.0 / 12.0));
  const auto out = rubberbandshift::shift(source, std::vector<float>(source.size(), ratio), kRate,
                                          nullptr, preserveFormant);
  CHECK(out.size() == source.size());
  const double hz = peakHz(out, kRate / 2, kRate / 2 + 8192);
  std::printf("  440 Hz down a semitone%s: %.1f Hz (expected %.1f)\n",
              preserveFormant ? ", formants kept" : "", hz, 440.0 * ratio);
  CHECK(std::abs(hz - 440.0 * ratio) < 1.0);
}

// Where the ratio is 1 the output lines up with the input (the stretcher's
// delay is removed), so it can be laid in place of the recording.
void linesUpWithTheInput(bool preserveFormant)
{
  const int samples = kRate * 2;
  const auto source = tone(440.0, samples);
  std::vector<float> ratio(source.size(), 1.0f);
  for (int i = kRate; i < kRate + kRate / 2; ++i)
    ratio[static_cast<size_t>(i)] = 0.98f;
  const auto out = rubberbandshift::shift(source, ratio, kRate, nullptr, preserveFormant);
  CHECK(out.size() == source.size());
  int bestLag = 0;
  double best = -1.0;
  const int from = kRate / 4, to = kRate / 2;
  for (int lag = -200; lag <= 200; ++lag)
  {
    double c = 0.0;
    for (int i = from; i < to; ++i)
      c += static_cast<double>(out[static_cast<size_t>(i + lag)]) * source[static_cast<size_t>(i)];
    if (c > best)
    {
      best = c;
      bestLag = lag;
    }
  }
  double eo = 0.0, es = 0.0;
  for (int i = from; i < to; ++i)
  {
    eo += static_cast<double>(out[static_cast<size_t>(i)]) * out[static_cast<size_t>(i)];
    es += static_cast<double>(source[static_cast<size_t>(i)]) * source[static_cast<size_t>(i)];
  }
  const double levelDb = 10.0 * std::log10(eo / es);
  std::printf("  unshifted part%s: lag %d samples, level %+.2f dB\n",
              preserveFormant ? ", formants kept" : "", bestLag, levelDb);
  CHECK(bestLag == 0);
  CHECK(std::abs(levelDb) < 0.5);
}

// A vowel at 680 Hz (煙幕's 地 is 674 Hz) shifted up 3 semitones with its
// formants kept: each harmonic should come out at the input's envelope
// where it now sits. Rubber Band's own envelope assumes a voice under about
// 450 Hz; given the recorded pitch, the envelope is smoothed past the
// harmonics instead (mean error 15.2 dB and level -12.5 dB without it).
void keepsTheFormantsOfAHighVoice()
{
  const int samples = kRate * 2;
  const double f0 = 680.0, ratio = std::pow(2.0, 3.0 / 12.0);
  const auto envelope = [](double hz) {
    const auto peak = [&](double centre, double width) {
      return 1.0 / (1.0 + std::pow((hz - centre) / width, 2));
    };
    return (0.2 + peak(700, 150) + 0.6 * peak(1200, 200) + 0.3 * peak(2800, 300)) /
           (1.0 + hz / 3000.0);
  };
  std::vector<float> source(static_cast<size_t>(samples), 0.0f);
  for (int k = 1; k * f0 < 9000; ++k)
    for (int i = 0; i < samples; ++i)
      source[static_cast<size_t>(i)] += static_cast<float>(
          0.1 * envelope(k * f0) * std::sin(2 * kPi * k * f0 * i / kRate + k));
  const std::vector<float> ratios(source.size(), static_cast<float>(ratio));
  const std::vector<float> pitch(source.size(), static_cast<float>(f0));
  const auto out = rubberbandshift::shift(source, ratios, kRate, nullptr, true, &pitch);
  CHECK(out.size() == source.size());
  const auto amplitude = [&](double hz) {
    double re = 0.0, im = 0.0, weight = 0.0;
    for (int i = kRate / 2; i < kRate * 3 / 2; ++i)
    {
      const double w = 0.5 - 0.5 * std::cos(2 * kPi * (i - kRate / 2) / kRate);
      weight += w;
      re += w * out[static_cast<size_t>(i)] * std::cos(2 * kPi * hz * i / kRate);
      im += w * out[static_cast<size_t>(i)] * std::sin(2 * kPi * hz * i / kRate);
    }
    return 2.0 * std::sqrt(re * re + im * im) / weight;
  };
  double error = 0.0, got = 0.0, wanted = 0.0;
  int count = 0;
  for (int k = 1; k * f0 * ratio < 5000; ++k)
  {
    const double hz = k * f0 * ratio, a = amplitude(hz), want = 0.1 * envelope(hz);
    error += std::abs(20.0 * std::log10(a / want));
    got += a * a;
    wanted += want * want;
    ++count;
  }
  error /= count;
  const double levelDb = 10.0 * std::log10(got / wanted);
  std::printf("  680 Hz up 3 semitones, formants kept: harmonic error %.2f dB, level %+.2f dB\n",
              error, levelDb);
  CHECK(error < 3.5);
  CHECK(std::abs(levelDb) < 3.5);
}

void theChoiceIsSaved()
{
  Project project;
  Note note(10, 40, 60.0f);
  note.setSynthEngine(Note::SynthEngine::RubberBand);
  project.addNote(note);
  project.addNote(Note(40, 70, 62.0f));
  Project reopened;
  CHECK(ProjectSerializer::fromJson(reopened, juce::JSON::parse(juce::JSON::toString(
                                                  ProjectSerializer::toJson(project)))));
  CHECK(reopened.getNotes().size() == 2);
  CHECK(reopened.getNotes()[0].getSynthEngine() == Note::SynthEngine::RubberBand);
  CHECK(reopened.getNotes()[1].getSynthEngine() == Note::SynthEngine::Vocoder);
}
} // namespace

int main()
{
  juce::ScopedJuceInitialiser_GUI juce;
  for (const bool preserveFormant : {false, true})
  {
    shiftsByTheRatio(preserveFormant);
    linesUpWithTheInput(preserveFormant);
  }
  keepsTheFormantsOfAHighVoice();
  theChoiceIsSaved();
  std::printf("RubberBandShiftTests passed\n");
  return 0;
}
