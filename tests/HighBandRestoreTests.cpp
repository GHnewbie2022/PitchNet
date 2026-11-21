// Spec Q12: "vocoder + recorded highs" (HighBandRestore.h).
#include "TestAssert.h"

#include "../Source/Audio/Synthesis/HighBandRestore.h"
#include "../Source/Models/ProjectSerializer.h"

#include <cmath>
#include <cstdio>
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

// It is the default: a new note has it, nothing is written for it, and a
// note with nothing stored reads back as it. The vocoder on its own (the
// listening-test tool's) is stored by name.
void itIsTheDefault()
{
  Project project;
  CHECK(Note(10, 20, 60.0f).getSynthEngine() == Note::SynthEngine::VocoderRecordedHighs);
  project.addNote(Note(10, 20, 60.0f));
  Note note(30, 40, 62.0f);
  note.setSynthEngine(Note::SynthEngine::Vocoder);
  project.addNote(note);
  const auto json = ProjectSerializer::toJson(project);
  CHECK(!juce::JSON::toString(json).contains("vocoderRecordedHighs"));
  Project reopened;
  CHECK(ProjectSerializer::fromJson(reopened, json));
  CHECK(reopened.getNotes().size() == 2);
  if (reopened.getNotes().size() == 2)
  {
    CHECK(reopened.getNotes()[0].getSynthEngine() == Note::SynthEngine::VocoderRecordedHighs);
    CHECK(reopened.getNotes()[1].getSynthEngine() == Note::SynthEngine::Vocoder);
  }
}
} // namespace

int main()
{
  juce::ScopedJuceInitialiser_GUI juce;
  takesTheRecordingAboveTheCrossover();
  itIsTheDefault();
  std::printf("HighBandRestoreTests passed\n");
  return 0;
}
