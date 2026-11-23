#pragma once

#include "../../JuceHeader.h"
#include "../../Models/Project.h"
#include "../SynthesisEngineType.h"
#include "../Vocoder.h"
#include "PsolaSynthesizer.h"
#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

/**
 * Voiced-Only Blend synthesizer.
 * Resynthesizes dirty regions using the selected engine, then blends:
 *   voiced frames  → synthesized audio
 *   long unvoiced runs, detected breaths and sibilants → original audio
 *
 * Two engines produce that synthesized audio. Vocoder regenerates the region
 * from its mel spectrogram; PsolaSynthesizer transforms the original waveform
 * instead. Everything after the render - blending, splicing, the audio move
 * patches - is identical for both.
 */
class IncrementalSynthesizer {
public:
  using ProgressCallback = std::function<void(const juce::String &message)>;
  using CompleteCallback = std::function<void(bool success)>;

  IncrementalSynthesizer();
  ~IncrementalSynthesizer();

  void setVocoder(Vocoder *v) { vocoder = v; }
  void setProject(Project *p) { project = p; }
  void setSynthesisEngine(SynthesisEngineType type) { synthesisEngine = type; }
  SynthesisEngineType getSynthesisEngine() const { return synthesisEngine; }
  /// Spec Q12: whether R3 notes keep the recorded formants (Rubber Band's
  /// OptionFormantPreserved, its envelope set by the recorded pitch). On:
  /// the user's choice after listening, 2026-11-21 - without it a note tuned
  /// down takes a heavier, bigger voice (heard at -0.29 semitones on 煙幕's
  /// 地, plainly at -3). Off only for listening tests.
  void setRubberBandFormantPreserved(bool preserved) { rubberBandFormantPreserved = preserved; }
  // Listening tests only: false puts R3 back on the stretched recording (the
  // two passes the one-pass render replaced), to tell the two apart by ear.
  void setRubberBandOnePass(bool onePass) { rubberBandOnePass = onePass; }
  // Listening tests only: false keeps what an edited vowel leaves under the
  // sibilant after it (BreathSibilanceProtection.h), to tell the two apart.
  void setVoiceTailRemoved(bool removed) { voiceTailRemoved = removed; }
  // Listening tests only: false writes the render where the vocoder put it,
  // neither shifted nor rotated to meet the recording at its splices.
  void setSpliceAligned(bool aligned) { spliceAligned = aligned; }

  void synthesizeRegion(ProgressCallback onProgress,
                        CompleteCallback onComplete);

  void cancel();
  bool isSynthesizing() const { return isBusy.load(); }

private:
  /// Compute synthesis range: find voiced segments overlapping dirty range,
  /// expand to include complete segments + padding frames.
  std::pair<int, int> computeSynthesisRange(int dirtyStart, int dirtyEnd);

  /// Generate per-sample blend mask from voicedMask and the breath/sibilance
  /// labels. 1.0 = use synthesized, 0.0 = use original, smooth ramps at
  /// transitions.
  std::vector<float> generateBlendMask(int startFrame, int endFrame,
                                       int hopSize);

  /// Per-output-frame map into the original audio: which source frame each
  /// rendered frame reads from. Identity unless a note has been retimed.
  /// Mirrors what applyNoteTimingToMel() does for the vocoder path.
  std::vector<double> buildSourceFrameMap(int startFrame, int endFrame) const;

  /// Per-output-frame target-over-source frequency ratio for the phase
  /// vocoder, given the already-retimed target F0 for the range.
  std::vector<float> buildPitchRatioCurve(int startFrame, int endFrame,
                                          const std::vector<double> &sourceFrameMap,
                                          const std::vector<float> &targetF0) const;

  Vocoder *vocoder = nullptr;
  Project *project = nullptr;
  SynthesisEngineType synthesisEngine = SynthesisEngineType::Vocoder;
  bool rubberBandFormantPreserved = true;
  bool rubberBandOnePass = true;
  bool voiceTailRemoved = true;
  bool spliceAligned = true;
  PsolaSynthesizer psola;

  std::shared_ptr<std::atomic<bool>> cancelFlag;
  std::atomic<uint64_t> jobId{0};
  std::atomic<bool> isBusy{false};

  std::thread applyThread;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(IncrementalSynthesizer)
  // The edited neighbours the last pass rendered too, for the debug recording.
  std::vector<int> journalNeighbours;
};
