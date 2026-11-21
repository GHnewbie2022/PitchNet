# PitchNet changes to Rubber Band Library 4.0.0

Rubber Band Library is GPL v2 or later. These files were changed by the
PitchNet project; each change is marked "PitchNet change" with its date.

- 2026-11-21: `RubberBandStretcher::setFormantPitchHint(double hz)` (R3
  only). With OptionFormantPreserved the formant envelope's cepstral cutoff
  was fixed at `sampleRate / 650`, so for a voice pitched near or above
  450 Hz the harmonics leaked into the envelope and the shifted voice lost
  level and timbre. The hint moves the cutoff to `sampleRate / hint` when
  the hint is above 650 Hz; 0 keeps the library's behaviour.
  Files: `rubberband/RubberBandStretcher.h`, `src/RubberBandStretcher.cpp`,
  `src/finer/R3Stretcher.h`, `src/finer/R3Stretcher.cpp`.

- 2026-11-22: `StretchCalculator` keeps the fraction of its checkpointed
  expected output frame. In real-time mode every change of ratio (time
  ratio or pitch scale) checkpoints the expected output position, and it
  was rounded to a whole frame each time. A ratio that changes with every
  analysis frame, as a pitch curve or a time map does, then added the same
  fraction of a sample each frame: at a time ratio of 1.075 the output ran
  0.9 ms late with every second (50 ms after a minute). With a time ratio
  of 1 the expected position is a whole number anyway, so pitch shifting
  without stretching is unchanged.
  Files: `src/common/StretchCalculator.h`, `src/common/StretchCalculator.cpp`.
