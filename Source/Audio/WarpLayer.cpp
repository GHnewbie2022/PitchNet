#include "WarpLayer.h"

#include "BreathSibilanceDetector.h"
#include "ProjectGain.h"
#include "Synthesis/WarpedSource.h"
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
  const auto &original = audioData.originalWaveform;
  if (audioData.warpMap.isIdentity() || original.getNumSamples() <= 0 ||
      original.getNumChannels() <= 0)
  {
    audioData.warpedOriginal.setSize(0, 0);
    return;
  }

  WarpedSourceSettings settings;
  settings.sampleRate =
      audioData.sampleRate > 0 ? audioData.sampleRate : SAMPLE_RATE;
  settings.hopSize = HOP_SIZE;
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
