// Spec Q7 / Q12: vocal fry at the end of a phrase and as a word starts
// (FryDetector.h).
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

// A word starting in fry: a short run an octave under the voice after it.
// What is kept runs from the low run to that voice, the frame part-way
// between them included, and back over the unvoiced frames before it, 6 at
// most.
void theStartOfAWordIsFound()
{
  std::vector<float> curve;
  add(curve, 20, 0.0f);
  add(curve, 2, 100.0f);
  add(curve, 1, 140.0f);
  add(curve, 40, 200.0f);
  const auto found = fry::onsets(curve);
  CHECK(found.size() == 1);
  CHECK(found[0].first == 14 && found[0].second == 23);
  CHECK(fry::candidates(curve).empty());
}

// As it reads in most of the sung ones (煙幕 at 31.57 s: 254 Hz, 6 unvoiced
// frames, 223, 110, 144, 215 Hz): the voice catches for one frame before it
// drops. That frame and the unvoiced ones before it are kept too; the walk
// back stops at the voice before.
void theVoiceCatchingFirstIsKept()
{
  std::vector<float> curve;
  add(curve, 30, 254.0f);
  add(curve, 4, 0.0f);
  add(curve, 1, 223.0f);
  add(curve, 1, 110.0f);
  add(curve, 1, 144.0f);
  add(curve, 30, 215.0f);
  const auto found = fry::onsets(curve);
  CHECK(found.size() == 1);
  CHECK(found[0].first == 30 && found[0].second == 37);
}

// Fry between two words with no break (the speech at 53.81 s: 268 Hz, five
// frames near 134 Hz, 258 Hz): the voice before it stays rendered. A low
// run before the unvoiced frames that lead in is part of the same creak
// (59.30 s: 197, 117, 97, 97, 96 Hz, 3 unvoiced, five frames at 105 Hz).
void fryBetweenTwoWords()
{
  std::vector<float> joined;
  add(joined, 30, 268.0f);
  add(joined, 5, 134.0f);
  add(joined, 30, 258.0f);
  const auto found = fry::onsets(joined);
  CHECK(found.size() == 1);
  CHECK(found[0].first == 30 && found[0].second == 35);

  std::vector<float> creak;
  add(creak, 30, 197.0f);
  add(creak, 4, 97.0f);
  add(creak, 3, 0.0f);
  add(creak, 5, 105.0f);
  add(creak, 30, 206.0f);
  const auto both = fry::onsets(creak);
  CHECK(both.size() == 1);
  CHECK(both[0].first == 31 && both[0].second == 42); // 6 frames back from 37
}

// A sung note under a leap is longer than 9 frames; a drop of less than 7
// semitones is a step of the melody; and the voice after has to be longer
// than the low run.
void aLeapUpIsNot()
{
  std::vector<float> longLow;
  add(longLow, 20, 0.0f);
  add(longLow, 10, 100.0f);
  add(longLow, 40, 200.0f);
  CHECK(fry::onsets(longLow).empty());

  std::vector<float> small;
  add(small, 20, 0.0f);
  add(small, 4, 140.0f); // 6.2 semitones
  add(small, 40, 200.0f);
  CHECK(fry::onsets(small).empty());

  std::vector<float> shortVoice;
  add(shortVoice, 20, 0.0f);
  add(shortVoice, 5, 100.0f);
  add(shortVoice, 4, 200.0f);
  add(shortVoice, 20, 0.0f);
  CHECK(fry::onsets(shortVoice).empty());
}

// Notes on Auto keep an onset they reach into; the frames of a note with a
// choice of its own are rendered with it; with no note on Auto there,
// nothing is kept.
void autoKeepsTheOnset()
{
  const fry::Ranges ranges{{34, 43}};
  std::vector<Note> notes{Note(10, 38, 55.0f), Note(38, 70, 55.0f)};
  auto kept = fry::keptOnsets(notes, ranges);
  CHECK(kept.size() == 1);
  CHECK(kept[0].first == 34 && kept[0].second == 43);

  notes[0].setSynthEngine(Note::SynthEngine::RubberBand);
  kept = fry::keptOnsets(notes, ranges);
  CHECK(kept.size() == 1);
  CHECK(kept[0].first == 38 && kept[0].second == 43);

  notes[1].setSynthEngine(Note::SynthEngine::VocoderRecordedHighs);
  CHECK(fry::keptOnsets(notes, ranges).empty());

  std::vector<Note> elsewhere{Note(50, 70, 55.0f)};
  CHECK(fry::keptOnsets(elsewhere, ranges).empty());
}

// The curve the vocoder sings can reach the voice a frame after the
// detector's own (煙幕 at 230.4 s: 337 against 424 Hz). Up to where it has
// arrived, 2 frames on at most, the vocoder is given the voice's pitch.
void theVoiceArrivesWithTheSungCurve()
{
  std::vector<float> raw{0.0f, 189.0f, 253.0f, 424.0f, 421.0f, 415.0f};
  std::vector<float> dense{0.0f, 189.0f, 253.0f, 337.0f, 421.0f, 415.0f};
  CHECK(fry::voiceArrivesAt(raw, dense, 3) == 4);
  CHECK(fry::voiceArrivesAt(raw, raw, 3) == 3); // arrives with the detector's

  // Never more than 2 frames on, nor past the curves or into unvoiced frames.
  std::vector<float> slow{0.0f, 189.0f, 253.0f, 300.0f, 310.0f, 320.0f, 424.0f};
  std::vector<float> high(slow.size(), 424.0f);
  CHECK(fry::voiceArrivesAt(high, slow, 3) == 5);
  CHECK(fry::voiceArrivesAt(raw, dense, 6) == 6);
  std::vector<float> unvoiced{0.0f, 189.0f, 253.0f, 0.0f, 421.0f};
  CHECK(fry::voiceArrivesAt(unvoiced, dense, 3) == 3);
}
} // namespace

// The user's own choice of "keep the fry": what Auto does where the detector
// finds fry in the note, and all of the note where it finds none.
void theUsersChoiceOfKeptFry()
{
  const fry::Ranges ends{{60, 68}};
  std::vector<float> curve(100, 0.0f);
  for (int i = 20; i < 68; ++i)
    curve[static_cast<size_t>(i)] = 200.0f;
  curve[69] = 100.0f;
  curve[80] = 200.0f;
  const auto told = Note::SynthEngine::VocoderRecordedHighsFry;

  // On a note the detector finds fry in: as on Auto.
  std::vector<Note> notes{Note(20, 50, 55.0f), Note(50, 70, 55.0f)};
  const auto onAuto = fry::kept(notes, curve, ends);
  notes[1].setSynthEngine(told);
  CHECK(fry::keepsFry(notes[1]));
  CHECK(fry::resolved(notes[1]) == Note::SynthEngine::VocoderRecordedHighs);
  CHECK(fry::kept(notes, curve, ends) == onAuto);
  CHECK(onAuto.size() == 1 && onAuto[0].first == 60 && onAuto[0].second == 80);
  const fry::Found found{ends, {}};
  CHECK(!fry::toldWithoutFry(notes[1], found));
  CHECK(!fry::wholeAsRecorded(notes[1], found));
  CHECK(fry::keptAsTold(notes, found, true).empty());

  // What the menu marks: the choice, or on Auto what the detector decides.
  Note left(20, 50, 55.0f), fried(50, 70, 55.0f);
  CHECK(fry::shownAs(left, found) == Note::SynthEngine::VocoderRecordedHighs);
  CHECK(fry::shownAs(fried, found) == told);
  fried.setSynthEngine(Note::SynthEngine::VocoderRecordedHighs);
  CHECK(fry::shownAs(fried, found) == Note::SynthEngine::VocoderRecordedHighs);
  fried.setSynthEngine(Note::SynthEngine::RubberBand);
  CHECK(fry::shownAs(fried, found) == Note::SynthEngine::RubberBand);
  left.setSynthEngine(told);
  CHECK(fry::shownAs(left, found) == told);

  // On a note the detector finds none in: all of it is the recording. With
  // no other edit it plays the recording and is not rendered (not listed
  // for the synthesizer, listed for the display); with a volume edit it is
  // rendered, and kept whole.
  CHECK(fry::toldWithoutFry(left, found));
  CHECK(fry::wholeAsRecorded(left, found));
  CHECK(fry::playsRecording(left, found));
  std::vector<Note> plain{left, Note(50, 70, 55.0f)};
  CHECK(fry::keptAsTold(plain, found).empty());
  auto listed = fry::keptAsTold(plain, found, true);
  CHECK(listed.size() == 1 && listed[0].first == 20 && listed[0].second == 50);
  plain[0].setVolumeDb(3.0f);
  CHECK(!fry::playsRecording(plain[0], found));
  listed = fry::keptAsTold(plain, found);
  CHECK(listed.size() == 1 && listed[0].first == 20 && listed[0].second == 50);
  // Auto on the same note keeps nothing: the detector found nothing.
  Note untold(20, 50, 55.0f);
  CHECK(!fry::wholeAsRecorded(untold, found));
  CHECK(!fry::playsRecording(untold, found));

  // Fry as the word starts counts as found too: the note is not taken whole.
  const fry::Found starting{{}, {{34, 43}}};
  Note word(36, 60, 55.0f);
  word.setSynthEngine(told);
  CHECK(fry::holdsAny(word, starting));
  CHECK(!fry::toldWithoutFry(word, starting));
  std::vector<Note> words{Note(10, 30, 55.0f), word};
  const auto onset = fry::keptOnsets(words, starting.onsets);
  CHECK(onset.size() == 1 && onset[0].first == 34 && onset[0].second == 43);
  word.setSynthEngine(Note::SynthEngine::Auto);
  CHECK(fry::shownAs(word, starting) == told);

  // The display lists what stays the recording, the notes that play it
  // anyway included: the note kept whole for its short lead, which the
  // synthesizer's list leaves out.
  std::vector<Note> shortLead{Note(20, 54, 55.0f), Note(54, 70, 55.0f)};
  CHECK(fry::kept(shortLead, curve, ends).empty());
  const auto shown = fry::keptForDisplay(shortLead, curve, found);
  CHECK(shown.size() == 1 && shown[0].first == 54 && shown[0].second == 80);
}

int main()
{
  theEndOfAPhraseIsFound();
  aFramePartWayDoesNotHideIt();
  aDropTheVoiceComesBackFromIsNot();
  aLeapDownIsNot();
  whatFollowsDecides();
  autoPicksByTheNote();
  theStartOfAWordIsFound();
  theVoiceCatchingFirstIsKept();
  fryBetweenTwoWords();
  aLeapUpIsNot();
  autoKeepsTheOnset();
  theVoiceArrivesWithTheSungCurve();
  theUsersChoiceOfKeptFry();
  std::printf("FryDetectorTests passed\n");
  return 0;
}
