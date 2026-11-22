// Spec Q7 / Q12: vocal fry at the end of a phrase (FryDetector.h).
#include "TestAssert.h"

#include "../Source/Utils/FryDetector.h"

#include <cmath>
#include <vector>

namespace
{
// A curve of `frames` frames at `hz`, appended.
void add(std::vector<float> &curve, int frames, float hz)
{
  curve.insert(curve.end(), static_cast<size_t>(frames), hz);
}

// A voice at 200 Hz dropping to 100 Hz for a short run, then silence.
void theEndOfAPhraseIsFound()
{
  std::vector<float> curve;
  add(curve, 20, 0.0f);
  add(curve, 40, 200.0f);
  add(curve, 8, 100.0f);
  add(curve, 30, 0.0f);
  const auto found = fry::candidates(curve);
  CHECK(found.size() == 1);
  CHECK(found[0].first == 60 && found[0].second == 68);
}

// The App's curve is resampled: the jump leaves one frame part-way between
// the two pitches (194 -> 136 -> 100 Hz in the recording the rule was tuned
// on). The run after it is still found.
void aFramePartWayDoesNotHideIt()
{
  std::vector<float> curve;
  add(curve, 40, 194.0f);
  add(curve, 1, 136.0f);
  add(curve, 7, 100.0f);
  add(curve, 30, 0.0f);
  const auto found = fry::candidates(curve);
  CHECK(found.size() == 1);
  CHECK(found[0].first == 41 && found[0].second == 48);
}

// A low run between two stretches of the same voice is a detection error
// (a sub-harmonic), not fry.
void aDropTheVoiceComesBackFromIsNot()
{
  std::vector<float> curve;
  add(curve, 40, 200.0f);
  add(curve, 8, 100.0f);
  add(curve, 40, 200.0f);
  CHECK(fry::candidates(curve).empty());
}

// A sung leap down: the low note is long, or the drop is under 7 semitones.
void aLeapDownIsNot()
{
  std::vector<float> longLow;
  add(longLow, 40, 200.0f);
  add(longLow, 60, 100.0f);
  add(longLow, 30, 0.0f);
  CHECK(fry::candidates(longLow).empty());

  std::vector<float> small;
  add(small, 40, 200.0f);
  add(small, 8, 140.0f); // 6.2 semitones
  add(small, 30, 0.0f);
  CHECK(fry::candidates(small).empty());
}

// What follows the run in the recording decides (rule 4). Irregular pulses
// that go on: fry. The voice going on periodic at its own pitch, the sound
// stopping, or a pause and then a consonant: not fry.
void whatFollowsDecides()
{
  constexpr int kRate = 44100;
  constexpr double kPi = 3.14159265358979323846;
  const auto build = [&](int kind)
  {
    AudioData data;
    data.sampleRate = kRate;
    add(data.rawF0, 40, 200.0f);
    add(data.rawF0, 8, 100.0f);
    add(data.rawF0, 40, 0.0f);
    const int total = 88 * 512;
    data.originalWaveform.setSize(1, total);
    auto *x = data.originalWaveform.getWritePointer(0);
    unsigned seed = 12345u;
    int nextPulse = 48 * 512;
    for (int i = 0; i < total; ++i)
    {
      const int frame = i / 512;
      float value = 0.0f;
      if (frame < 40)
        value = 0.3f * static_cast<float>(std::sin(2 * kPi * 200.0 * i / kRate));
      else if (frame < 48)
        value = 0.2f * static_cast<float>(std::sin(2 * kPi * 100.0 * i / kRate));
      else if (kind == 1) // the voice goes on, quieter, periodic
        value = 0.1f * static_cast<float>(std::sin(2 * kPi * 220.0 * i / kRate));
      else if (kind == 3 && frame >= 51) // a 35 ms pause, then a consonant's noise
      {
        seed = seed * 1664525u + 1013904223u;
        value = 0.1f * (static_cast<float>((seed >> 8) & 0xffff) / 32768.0f - 1.0f);
      }
      else if (kind == 0 && i >= nextPulse) // pulses 6 to 20 ms apart, each ringing out
      {
        seed = seed * 1664525u + 1013904223u;
        nextPulse = i + kRate * (6 + static_cast<int>((seed >> 16) % 15)) / 1000;
      }
      x[i] = value;
    }
    if (kind == 0)
    {
      // each pulse is a short damped ring at 900 Hz
      std::vector<int> starts;
      seed = 12345u;
      for (int at = 48 * 512; at < total;)
      {
        starts.push_back(at);
        seed = seed * 1664525u + 1013904223u;
        at += kRate * (6 + static_cast<int>((seed >> 16) % 15)) / 1000;
      }
      for (const int start : starts)
        for (int i = start; i < std::min(total, start + 400); ++i)
          x[i] += 0.6f * static_cast<float>(std::exp(-(i - start) / 80.0) *
                                            std::sin(2 * kPi * 900.0 * (i - start) / kRate));
    }
    return data;
  };
  CHECK(fry::detect(build(0)).size() == 1); // irregular pulses go on
  CHECK(fry::detect(build(1)).empty());     // periodic voice goes on
  CHECK(fry::detect(build(2)).empty());     // silence
  CHECK(fry::detect(build(3)).empty());     // a pause, then noise
}

// A note on Auto is rendered with the vocoder and the recording's highs;
// where its stretch of the recording holds 3 frames or more of fry, that
// fry keeps the recording. A note with a choice of its own keeps it.
void autoPicksByTheNote()
{
  const fry::Ranges ranges{{60, 68}};
  Note onFry(50, 70, 55.0f);
  Note before(20, 60, 55.0f);
  Note edge(40, 62, 55.0f); // 2 frames of it
  CHECK(fry::holds(onFry, ranges));
  CHECK(!fry::holds(before, ranges));
  CHECK(!fry::holds(edge, ranges));
  CHECK(fry::resolved(onFry) == Note::SynthEngine::VocoderRecordedHighs);

  // The fry, on to the end of its note (frame 70, a voiced frame at 69
  // included) and over the unvoiced frames after it, up to the next voiced
  // one.
  std::vector<float> curve(100, 0.0f);
  for (int i = 20; i < 68; ++i)
    curve[static_cast<size_t>(i)] = 200.0f;
  curve[69] = 100.0f;
  curve[80] = 200.0f;
  std::vector<Note> notes{before, onFry};
  const auto kept = fry::kept(notes, curve, ranges);
  CHECK(kept.size() == 1);
  CHECK(kept[0].first == 60);
  CHECK(kept[0].second == 80);

  // Less than 9 frames of voice before the fry: the whole note is kept.
  // With no other edit it plays the recording as an unedited note does and
  // is not rendered, so nothing of it is listed; with a volume edit it is
  // rendered and kept through the mask, from its first frame.
  std::vector<Note> shortLead{Note(20, 54, 55.0f), Note(54, 70, 55.0f)};
  CHECK(fry::playsRecording(shortLead[1], ranges));
  CHECK(fry::kept(shortLead, curve, ranges).empty());
  shortLead[1].setVolumeDb(3.0f);
  CHECK(!fry::playsRecording(shortLead[1], ranges));
  const auto whole = fry::kept(shortLead, curve, ranges);
  CHECK(whole.size() == 1);
  CHECK(whole[0].first == 54);
  CHECK(whole[0].second == 80);
  CHECK(fry::keepsWhole(shortLead[1], ranges));
  CHECK(!fry::keepsWhole(shortLead[0], ranges));
  CHECK(!fry::keepsWhole(onFry, ranges));

  // A note with a choice of its own is rendered whole with it.
  notes[1].setSynthEngine(Note::SynthEngine::RubberBand);
  CHECK(fry::kept(notes, curve, ranges).empty());
  CHECK(fry::resolved(notes[1]) == Note::SynthEngine::RubberBand);
}
} // namespace

int main()
{
  theEndOfAPhraseIsFound();
  aFramePartWayDoesNotHideIt();
  aDropTheVoiceComesBackFromIsNot();
  aLeapDownIsNot();
  whatFollowsDecides();
  autoPicksByTheNote();
  std::printf("FryDetectorTests passed\n");
  return 0;
}
