#include "WarpLayer.h"

#include "BreathSibilanceDetector.h"
#include "ProjectGain.h"
#include "RecordingChange.h"
#include "BreathSibilanceAnalysis.h"
#include "Synthesis/SourceKinds.h"
#include "Synthesis/WarpedSource.h"
#include "SourceAudio.h"
#include "../Utils/AudioResampler.h"
#include "../Utils/Constants.h"
#include "../Utils/NoteGainCurve.h"
#include "../Utils/PitchCurveProcessor.h"

#include <algorithm>
#include <cmath>

namespace warp
{
void fitToProject(AudioData &audioData)
{
  // Before the mel exists (host-backed ARA state is hydrated later) the map
  // keeps the length it was saved with.
  const double frames = static_cast<double>(audioData.getNumFrames());
  if (frames > 0.0 && audioData.warpMap.length() != frames)
    audioData.warpMap.setLength(frames);
}

void syncNotes(Project &project)
{
  auto &audioData = project.getAudioData();
  fitToProject(audioData);
  const auto &map = audioData.warpMap;
  for (auto &note : project.getNotes())
  {
    const int start = static_cast<int>(
        std::lround(map.destinationAt(note.getSrcStartFrame())));
    int end = static_cast<int>(
        std::lround(map.destinationAt(note.getSrcEndFrame())));
    if (note.getSrcEndFrame() > note.getSrcStartFrame())
      end = std::max(end, start + 1);
    note.setStartFrame(start);
    note.setEndFrame(end);
    note.clearTimingPreview();
  }
}

void rememberCompositeMap(AudioData &audioData)
{
  if (audioData.compositeMapKnown)
    return;
  audioData.compositeMap = audioData.warpMap;
  audioData.compositeMapKnown = true;
}

bool compositeLagsMap(const AudioData &audioData)
{
  return audioData.compositeMapKnown &&
         !audioData.compositeMap.sameMappingAs(audioData.warpMap);
}

void apply(Project &project, const WarpMap &map)
{
  auto &audioData = project.getAudioData();
  rememberCompositeMap(audioData);
  const bool soundChanges = !audioData.warpMap.sameMappingAs(map);
  audioData.warpMap = map;
  fitToProject(audioData);
  syncNotes(project);
  project.setModified(true);
  // Adding or removing an anchor that pins the recording where it already
  // plays changes nothing audible: nothing to lay out again.
  if (!soundChanges)
    return;
  PitchCurveProcessor::rebuildBaseFromNotes(project);
  // Stretched notes keep their curve per timeline frame: fit them to the
  // recording through the new map, or their contour slides. Edited notes
  // too: the contour their edits sit on is the recorded one.
  PitchCurveProcessor::refitUntouchedCurves(project, 0, audioData.getNumFrames(),
                                            true, true);
  audioData.warpLayoutPending = true;
}

const juce::AudioBuffer<float> &reference(const AudioData &audioData)
{
  if (audioData.warpedOriginal.getNumSamples() > 0)
    return audioData.warpedOriginal;
  if (audioData.originalWaveform.getNumSamples() > 0)
    return audioData.originalWaveform;
  return audioData.waveform;
}

void rebuildReference(AudioData &audioData)
{
  fitToProject(audioData);
  // What each frame is stretched as (SourceKinds.h), the same for every
  // stretch below. The labels it reads are not saved with a project: made
  // here if no load path has yet, or a reference built before them would be
  // another stretch than the one built after.
  breathsibilance::ensure(audioData);
  const auto kinds = sourceKinds(audioData);
  // The same layout at the recording's own rate, for playback (spec Q5 C).
  sourceaudio::rebuildWarped(audioData, kinds);
  const auto &original = audioData.originalWaveform;
  if (audioData.warpMap.isIdentity() || original.getNumSamples() <= 0 ||
      original.getNumChannels() <= 0)
  {
    audioData.warpedOriginal.setSize(0, 0);
    return;
  }

  // One stretch, not two (spec Q5 C). Where the recording is held at its
  // own rate, the reference is that stretch converted the way the analysis
  // copy was made from the recording. Stretched a second time from the
  // analysis copy, the reference carried the same audio at another phase:
  // signalsmith keeps its own, and two runs (44.1 and 48 kHz) drifted up to
  // 0.5 ms apart. Playback crossfades from the edited composite to the
  // recording at its own rate 20 ms outside each edit, and there the two
  // cancelled: a dip of 4 to 6 dB for 8 ms in a held voice next to an
  // edited note (煙幕 at 86.2 s stretched 1.33x, heard as a catch).
  if (audioData.warpReferenceFromRecording &&
      audioData.warpedSourceAudio.getNumSamples() > 0 &&
      audioData.sourceSampleRate > 0)
  {
    auto converted = recordingchange::analysisCopy(audioData.warpedSourceAudio,
                                                   audioData.sourceSampleRate);
    if (converted.getNumSamples() == original.getNumSamples())
    {
      audioData.warpedOriginal.setSize(original.getNumChannels(),
                                       original.getNumSamples(), false, true,
                                       false);
      for (int channel = 0; channel < original.getNumChannels(); ++channel)
        audioData.warpedOriginal.copyFrom(channel, 0, converted, 0, 0,
                                          converted.getNumSamples());
      return;
    }
  }

  // The plug-in: the host owns the recording, and playback stretches the
  // host's audio at its own rate, channel by channel (the same call with the
  // same settings, sourceaudio::rebuildWarped). Channel 0 is what the
  // analysis copy was converted from, so the reference is that channel
  // stretched here the same way and converted the same way.
  if (audioData.warpReferenceFromRecording && audioData.analysisSource != nullptr &&
      audioData.analysisSourceRate > 0 &&
      audioData.analysisSource->getNumChannels() > 0 &&
      audioData.analysisSource->getNumSamples() > 0)
  {
    const auto &source = *audioData.analysisSource;
    const int rate = audioData.analysisSourceRate;
    WarpedSourceSettings atSourceRate;
    atSourceRate.sampleRate = rate;
    atSourceRate.hopSize = static_cast<double>(HOP_SIZE) * rate / SAMPLE_RATE;
    atSourceRate.kinds = &kinds;
    juce::AudioBuffer<float> stretched(1, source.getNumSamples());
    renderWarpedSource(source.getReadPointer(0), source.getNumSamples(),
                       audioData.warpMap, stretched.getWritePointer(0),
                       source.getNumSamples(), atSourceRate);
    auto converted = AudioResampler::resample(stretched, rate, SAMPLE_RATE);
    if (converted.getNumSamples() == original.getNumSamples())
    {
      audioData.warpedOriginal.setSize(original.getNumChannels(),
                                       original.getNumSamples(), false, true,
                                       false);
      for (int channel = 0; channel < original.getNumChannels(); ++channel)
        audioData.warpedOriginal.copyFrom(channel, 0, converted, 0, 0,
                                          converted.getNumSamples());
      return;
    }
  }

  // Without the recording at its own rate (an embedded project with no file
  // attached, a plug-in project at the host's rate or not yet given the
  // host's audio), or for a project saved before the one stretch
  // (AudioData::warpReferenceFromRecording): the analysis copy, stretched.
  WarpedSourceSettings settings;
  settings.sampleRate =
      audioData.sampleRate > 0 ? audioData.sampleRate : SAMPLE_RATE;
  settings.hopSize = HOP_SIZE;
  settings.kinds = &kinds;
  const int samples = original.getNumSamples();
  audioData.warpedOriginal.setSize(original.getNumChannels(), samples, false,
                                   true, false);
  for (int channel = 0; channel < original.getNumChannels(); ++channel)
    renderWarpedSource(original.getReadPointer(channel), samples,
                       audioData.warpMap,
                       audioData.warpedOriginal.getWritePointer(channel),
                       samples, settings);
}

void layOut(Project &project)
{
  auto &audioData = project.getAudioData();
  // Every edit is rendered again below, so the reference may be the one
  // stretch from here on, whatever the file was saved with.
  audioData.warpReferenceFromRecording = true;
  rebuildReference(audioData);
  const auto &source = reference(audioData);
  auto &composite = audioData.waveform;
  const int samples = composite.getNumSamples();
  if (samples > 0 && source.getNumChannels() > 0 && &source != &composite)
  {
    const auto gain = NoteGainCurve::build(projectgain::regions(project), 0,
                                           samples, samples, HOP_SIZE);
    for (int channel = 0; channel < composite.getNumChannels(); ++channel)
    {
      const float *from = source.getReadPointer(
          std::min(channel, source.getNumChannels() - 1));
      float *to = composite.getWritePointer(channel);
      const int count = std::min(samples, source.getNumSamples());
      for (int i = 0; i < count; ++i)
        to[i] = from[i] * (gain.empty() ? 1.0f : gain[static_cast<size_t>(i)]);
      std::fill(to + count, to + samples, 0.0f);
    }
  }

  // The composite now holds no rendered edit; every note whose pitch is
  // edited has to be rendered again where it now plays.
  for (auto &note : project.getNotes())
  {
    if (note.isRest())
      continue;
    note.setRenderedEdit(false);
    if (std::abs(project.getFormantShift()) > 0.001f ||
        !note.isNeutralForOriginalWaveform())
    {
      note.markDirty();
      note.markSynthDirty();
    }
  }
  audioData.compositeMap = audioData.warpMap;
  audioData.compositeMapKnown = true;
  audioData.warpLayoutPending = false;
}

int sourceFrameAt(const AudioData &audioData, int frame)
{
  if (audioData.warpMap.isIdentity())
    return frame;
  return static_cast<int>(std::floor(
      audioData.warpMap.sourceAt(static_cast<double>(frame) + 0.5)));
}

bool voicedAt(const AudioData &audioData, int frame)
{
  const int source = sourceFrameAt(audioData, frame);
  return source >= 0 && source < static_cast<int>(audioData.voicedMask.size()) &&
         audioData.voicedMask[static_cast<size_t>(source)];
}

std::uint8_t labelAt(const AudioData &audioData, int frame)
{
  return breathsibilance::labelAt(audioData.breathSibilance,
                                  sourceFrameAt(audioData, frame));
}
} // namespace warp
