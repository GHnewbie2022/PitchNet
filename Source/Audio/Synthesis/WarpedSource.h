#pragma once

#include "../WarpMap.h"

#include <cstdint>
#include <vector>

/**
 * The recording laid out on the timeline by the warp map - the layer every
 * other kind of processing is blended onto (spec Q1):
 *
 *   not stretched  the recording itself, moved to where the map puts it
 *   stretched      by what the recording holds there (SourceKinds.h):
 *                    voice   signalsmith-stretch, window 35 ms, hop 9 ms
 *                    noise   breath and unvoiced sibilance: the noise made
 *                            again (NoiseStretch.h)
 *                    fry     its own waveform, overlapped (OverlapStretch.h)
 *
 * One stretch for everything - signalsmith's default, a 120 ms window - was
 * heard as a haze over the voice at every ratio and as phasiness at 1.33x:
 * a window that long smears each moment over 60 ms either way (the gaps
 * between words fill in, the ripple of a vibrato in the harmonics blurs).
 * At 35 ms the voice stays as clear as the recording, but a breath turns
 * phasey and fry, whose pulses come further apart than the window can tell,
 * turns into a buzz; each of the two has a stretch that does not. Chosen by
 * ear on 4 sung phrases and 2 spoken ones at 0.80x, 1.10x and 1.33x.
 *
 * A run is signalsmith throughout; inside its stretched segments the noise
 * and the fry are then laid over it, each across 20 ms. Noise meets noise
 * with no phase to match, and two unrelated noises add in power: every
 * crossfade is raised by how unlike its two sides are, up to 3 dB halfway.
 * A breath that an anchor cuts in two starts its own noise a window before
 * the anchor (or ends it a window after), where signalsmith's rendering is
 * still the recording; next to the anchor it is not. Fry meets the voice
 * next to it in phase where it
 * starts (its first window is placed against signalsmith's rendering) and,
 * where it ends, at the place within 40 ms past it where the two differ
 * least.
 *
 * Neighbouring stretched segments are processed as one continuous run whose
 * rate changes at the anchors, so an anchor between two stretched segments
 * is not a seam. A run hands over to the copied recording inside unstretched
 * audio next to it, where the stretch plays at rate 1, with a 10 ms
 * crossfade. Both sides carry the same audio there, but the stretch keeps its
 * own phase: it starts aligned with the recording and can end up to half a
 * period off it, so a fade inside a held vowel can cancel (a dip, heard as a
 * click). The fade goes where the two differ least - a quiet spot, or one
 * where they happen to line up - searched up to handOverSeconds from the run.
 *
 * The random seed is fixed: past a ratio of 2 signalsmith randomises phases,
 * and the same edit must always render the same audio.
 */
namespace warp
{
struct WarpedSourceSettings
{
  double sampleRate = 44100.0;
  // Samples per map frame. Fractional when the recording is rendered at its
  // own rate (spec Q5 C): 48 kHz is 512 * 48000 / 44100 = 557.3 samples.
  double hopSize = 512.0;
  // Unstretched audio a run always renders on each side; runs closer than
  // twice this are joined into one.
  double padSeconds = 0.06;
  // How far into unstretched audio a run may go to find its hand-over (at
  // most halfway to the next run).
  double handOverSeconds = 0.3;
  double fadeSeconds = 0.010;
  long seed = 20261115L;
  // What each frame of the recording is (SourceKind, one per map frame; a
  // frame sits on its centre, as the labels do in the blend mask). Null, or
  // past its end: voice.
  const std::vector<std::uint8_t> *kinds = nullptr;
  double kindFadeSeconds = 0.020;
  // How far past the end of fry its hand-over to the voice may go.
  double kindReachSeconds = 0.040;
};

enum class SourceKind : std::uint8_t
{
  Voice = 0,
  Noise = 1,  // breath, unvoiced sibilance
  Pulses = 2, // vocal fry
};

/**
 * Listening tests only (WarpRender): true stretches everything with
 * signalsmith's default as before the kinds, so the two can be told apart.
 */
void setLegacyStretch(bool legacy);
bool legacyStretch();

struct StretchedRun
{
  int startSample = 0; // timeline samples [startSample, endSample)
  int endSample = 0;
};

/**
 * Renders timeline samples [0, outputSamples) of `source` (mono, at the
 * map's hop) laid out by `map`. Returns the stretched runs, padding
 * included, so a caller can tell which samples are not plain copies.
 */
std::vector<StretchedRun> renderWarpedSource(const float *source,
                                             int sourceSamples,
                                             const WarpMap &map,
                                             float *output, int outputSamples,
                                             const WarpedSourceSettings &settings = {});
} // namespace warp
