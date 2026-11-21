#pragma once

#include <atomic>
#include <vector>

/**
 * Spec Q12: a note the user chose to render with Rubber Band's R3 engine
 * instead of the vocoder. The vocoder rebuilds the voice from its spectrum
 * and pitch, and a voice with a fine buzz in it (a slight rasp, heard as
 * "麻感") comes out loose and electronic; R3 shifts the recording itself and
 * keeps that texture (listening test on 煙幕 02:48.124, 2026-11-21: R3 kept
 * it best, ahead of PSOLA and Signalsmith Stretch).
 *
 * Rubber Band Library (third_party/rubberband) is GPL v2 or later; PitchNet
 * is AGPL v3, which may combine with it. A closed-source build would need
 * its commercial licence.
 */
namespace rubberbandshift
{
/**
 * `source` (mono, `sampleRate`) with its pitch multiplied by `ratio` (one
 * frequency ratio per sample, as long as `source`), R3 engine. With
 * `preserveFormant` the spectral envelope stays where it was recorded
 * (OptionFormantPreserved); without it the formants move with the pitch.
 * `sourceHz` (optional, one value per sample, 0 where unvoiced) is the
 * recorded pitch: with it the formant envelope is smoothed across the
 * voice's own harmonics (Rubber Band's fixed envelope assumes a voice
 * under about 450 Hz and pulls higher notes down).
 * Time is unchanged: the result has the same length and lines up with
 * `source` sample for sample (the stretcher's delay is removed). Empty if
 * cancelled or the input is unusable.
 */
std::vector<float> shift(const std::vector<float> &source, const std::vector<float> &ratio,
                         int sampleRate, const std::atomic<bool> *cancel = nullptr,
                         bool preserveFormant = false,
                         const std::vector<float> *sourceHz = nullptr);
} // namespace rubberbandshift
