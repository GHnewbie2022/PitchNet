// Spec Q12: "vocoder + recorded highs" (HighBandRestore.h).
#include "TestAssert.h"

#include "../Source/Audio/Synthesis/HarmonicFloor.h"
#include "../Source/Audio/Synthesis/HighBandRestore.h"
#include "../Source/Models/ProjectSerializer.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr int kRate = 44100;

std::vector<float> tones(int samples, double lowHz, double highHz)
{
  std::vector<float> x(static_cast<size_t>(samples));
  for (int i = 0; i < samples; ++i)
    x[static_cast<size_t>(i)] = static_cast<float>(0.3 * std::sin(2 * kPi * lowHz * i / kRate) +
                                                   0.1 * std::sin(2 * kPi * highHz * i / kRate));
  return x;
}

double amplitude(const std::vector<float> &x, double hz, int from, int to)
{
  double re = 0.0, im = 0.0, weight = 0.0;
  for (int i = from; i < to; ++i)
  {
    const double w = 0.5 - 0.5 * std::cos(2 * kPi * (i - from) / (to - from));
    weight += w;
    re += w * x[static_cast<size_t>(i)] * std::cos(2 * kPi * hz * i / kRate);
    im += w * x[static_cast<size_t>(i)] * std::sin(2 * kPi * hz * i / kRate);
  }
  return 2.0 * std::sqrt(re * re + im * im) / weight;
}

// Below the crossover the render stays; above it the recording's content
// replaces the render's. Outside the weighted span nothing changes.
void takesTheRecordingAboveTheCrossover()
{
  const int samples = kRate;
  const auto render = tones(samples, 1000.0, 12000.0);
  const auto recording = tones(samples, 1500.0, 13000.0);
  std::vector<float> weight(static_cast<size_t>(samples), 0.0f);
  const int first = kRate / 4, last = kRate * 3 / 4;
  for (int i = first; i < last; ++i)
    weight[static_cast<size_t>(i)] = 1.0f;
  auto out = render;
  CHECK(highband::restore(out, recording, weight, kRate));
  const int from = first + 2048, to = last - 2048;
  const double low = amplitude(out, 1000.0, from, to);
  const double recordingLow = amplitude(out, 1500.0, from, to);
  const double renderHigh = amplitude(out, 12000.0, from, to);
  const double recordingHigh = amplitude(out, 13000.0, from, to);
  std::printf("  1 kHz (render) %.3f, 1.5 kHz (recording) %.4f, 12 kHz (render) %.4f, "
              "13 kHz (recording) %.3f\n",
              low, recordingLow, renderHigh, recordingHigh);
  CHECK(std::abs(low - 0.3) < 0.003);
  CHECK(recordingLow < 0.001);
  CHECK(renderHigh < 0.001);
  CHECK(std::abs(recordingHigh - 0.1) < 0.003);
  for (int i = 0; i < first; ++i)
    CHECK(out[static_cast<size_t>(i)] == render[static_cast<size_t>(i)]);
  for (int i = last; i < samples; ++i)
    CHECK(out[static_cast<size_t>(i)] == render[static_cast<size_t>(i)]);

  // No weight anywhere: untouched.
  auto same = render;
  CHECK(highband::restore(same, recording, std::vector<float>(render.size(), 0.0f), kRate));
  CHECK(same == render);
}

// A voice's harmonics pressed down to the noise around them (HarmonicFloor.h).
std::vector<float> noise(int samples, float level, unsigned seed)
{
  std::mt19937 generator(seed);
  std::normal_distribution<float> distribution(0.0f, level);
  std::vector<float> x(static_cast<size_t>(samples));
  for (auto &sample : x)
    sample = distribution(generator);
  return x;
}

double rms(const std::vector<float> &x, int from, int to)
{
  double sum = 0.0;
  for (int i = from; i < to; ++i)
    sum += static_cast<double>(x[static_cast<size_t>(i)]) * x[static_cast<size_t>(i)];
  return std::sqrt(sum / (to - from));
}

void pressesHarmonicsDownToTheNoise()
{
  const int samples = kRate;
  const int first = kRate / 4, last = kRate * 3 / 4;
  const int from = first + 4096, to = last - 4096;
  std::vector<float> weight(static_cast<size_t>(samples), 0.0f);
  for (int i = first; i < last; ++i)
    weight[static_cast<size_t>(i)] = 1.0f;
  const auto voice = [&](double pitch, int harmonics)
  {
    auto x = noise(samples, 0.01f, 7);
    for (int i = 0; i < samples; ++i)
    {
      for (int h = 1; h <= harmonics; ++h)
        x[static_cast<size_t>(i)] += static_cast<float>(0.1 * std::sin(2 * kPi * h * pitch * i / kRate));
      x[static_cast<size_t>(i)] += static_cast<float>(0.05 * std::sin(2 * kPi * 6000.0 * i / kRate));
    }
    return x;
  };

  // Harmonics of 300 Hz under noise, up to 3.6 kHz: the ones under the top
  // go down to the noise, the one over it and a tone further up stay, and
  // nothing changes outside the weighted span.
  const auto before = voice(300.0, 12);
  auto audio = before;
  CHECK(harmonicfloor::flatten(audio, weight, kRate, 300.0, 3500.0));
  std::printf("  300 Hz %.4f, 1.5 kHz %.4f, 3.3 kHz %.4f (were 0.1); 3.6 kHz %.4f (0.1), "
              "6 kHz %.4f (0.05)\n",
              amplitude(audio, 300.0, from, to), amplitude(audio, 1500.0, from, to),
              amplitude(audio, 3300.0, from, to), amplitude(audio, 3600.0, from, to),
              amplitude(audio, 6000.0, from, to));
  CHECK(amplitude(audio, 300.0, from, to) < 0.01);
  CHECK(amplitude(audio, 1500.0, from, to) < 0.01);
  CHECK(amplitude(audio, 3300.0, from, to) < 0.01);
  CHECK(std::abs(amplitude(audio, 3600.0, from, to) - 0.1) < 0.005);
  CHECK(std::abs(amplitude(audio, 6000.0, from, to) - 0.05) < 0.003);
  for (int i = 0; i < first; ++i)
    CHECK(audio[static_cast<size_t>(i)] == before[static_cast<size_t>(i)]);
  for (int i = last; i < samples; ++i)
    CHECK(audio[static_cast<size_t>(i)] == before[static_cast<size_t>(i)]);

  // Noise alone has no harmonics to lose: it is not raised, and barely
  // lowered (its peaks over the median inside the bands).
  const auto hiss = noise(samples, 0.05f, 11);
  auto flattened = hiss;
  CHECK(harmonicfloor::flatten(flattened, weight, kRate, 300.0, 3500.0));
  const double change = 20.0 * std::log10(rms(flattened, from, to) / rms(hiss, from, to));
  std::printf("  noise alone: %.2f dB\n", change);
  CHECK(change <= 0.0 && change > -0.5);

  // A low voice takes a longer window to tell its harmonics apart.
  auto low = voice(120.0, 5);
  CHECK(harmonicfloor::flatten(low, weight, kRate, 120.0, 3500.0));
  std::printf("  120 Hz %.4f, 240 Hz %.4f (were 0.1)\n", amplitude(low, 120.0, from, to),
              amplitude(low, 240.0, from, to));
  CHECK(amplitude(low, 120.0, from, to) < 0.02);
  CHECK(amplitude(low, 240.0, from, to) < 0.02);

  // Too low to tell apart at all, no pitch, no weight: untouched.
  auto same = before;
  CHECK(harmonicfloor::flatten(same, weight, kRate, 30.0, 3500.0));
  CHECK(same == before);
  CHECK(harmonicfloor::flatten(same, weight, kRate, 0.0, 3500.0));
  CHECK(same == before);
  CHECK(harmonicfloor::flatten(same, std::vector<float>(before.size(), 0.0f), kRate, 300.0, 3500.0));
  CHECK(same == before);
}

// The choice is stored with the note. A new note is on Auto, nothing is
// written for that, and a note with nothing stored reads back as Auto.
void theChoiceIsSaved()
{
  Project project;
  CHECK(Note(10, 20, 60.0f).getSynthEngine() == Note::SynthEngine::Auto);
  Note chosen(10, 20, 60.0f);
  chosen.setSynthEngine(Note::SynthEngine::VocoderRecordedHighs);
  project.addNote(chosen);
  Note note(30, 40, 62.0f);
  note.setSynthEngine(Note::SynthEngine::Vocoder);
  project.addNote(note);
  project.addNote(Note(50, 60, 64.0f));
  const auto json = ProjectSerializer::toJson(project);
  CHECK(!juce::JSON::toString(json).containsIgnoreCase("auto"));
  Project reopened;
  CHECK(ProjectSerializer::fromJson(reopened, json));
  CHECK(reopened.getNotes().size() == 3);
  if (reopened.getNotes().size() == 3)
  {
    CHECK(reopened.getNotes()[0].getSynthEngine() == Note::SynthEngine::VocoderRecordedHighs);
    CHECK(reopened.getNotes()[1].getSynthEngine() == Note::SynthEngine::Vocoder);
    CHECK(reopened.getNotes()[2].getSynthEngine() == Note::SynthEngine::Auto);
  }
}
} // namespace

int main()
{
  juce::ScopedJuceInitialiser_GUI juce;
  takesTheRecordingAboveTheCrossover();
  pressesHarmonicsDownToTheNoise();
  theChoiceIsSaved();
  std::printf("HighBandRestoreTests passed\n");
  return 0;
}
