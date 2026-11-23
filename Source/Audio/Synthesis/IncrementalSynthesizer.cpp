#include "IncrementalSynthesizer.h"
#include "SpliceAlignment.h"
#include "BreathSibilanceProtection.h"
#include "FormantShifter.h"
#include "../BreathSibilanceAnalysis.h"
#include "../NoteTiming.h"
#include "../ProjectGain.h"
#include "../WarpLayer.h"
#include "OriginalWaveformRestore.h"
#include "RubberBandShift.h"
#include "HighBandRestore.h"
#include "../../Utils/FryDetector.h"
#include "../../Utils/NoteGainCurve.h"
#include "../../Utils/PitchCurveProcessor.h"
#include "../../Utils/AppLogger.h"
#include "../../Utils/Constants.h"
#include "../../Utils/Localization.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {
constexpr int kAdjacentFrameTolerance = 1;
// The vocoder's output lags its input by about 4.5 ms (measured on the
// envelope of whole renders; see the warp layer spec, Q6). It is advanced by
// this much before it is blended with the recording, so the two line up.
constexpr double kVocoderLatencySeconds = 0.0045;

struct EditedClusterSelection {
  std::vector<uint8_t> members;
  int startFrame = -1;
  int endFrame = -1;
};

struct CommitFrameRange {
  int start = 0;
  int end = 0;
};

std::vector<float> interpolateMelFrame(
    const std::vector<std::vector<float>> &mel, double sourceFrame) {
  if (mel.empty())
    return {};

  sourceFrame = std::clamp(sourceFrame, 0.0,
                           static_cast<double>(mel.size() - 1));
  const int left = static_cast<int>(std::floor(sourceFrame));
  const int right = std::min(left + 1, static_cast<int>(mel.size()) - 1);
  const float amount = static_cast<float>(sourceFrame - left);
  const size_t bins = std::min(mel[static_cast<size_t>(left)].size(),
                               mel[static_cast<size_t>(right)].size());
  std::vector<float> frame(bins, 0.0f);
  for (size_t bin = 0; bin < bins; ++bin)
    frame[bin] = mel[static_cast<size_t>(left)][bin] * (1.0f - amount) +
                 mel[static_cast<size_t>(right)][bin] * amount;
  return frame;
}

// Every timeline frame reads the mel where the warp map says its audio was
// recorded (WarpMap::sourceFrameAt): one continuous map for the whole track,
// so notes, gaps and breaths are stretched alike.
void applyWarpToMel(const Project &project, int rangeStart, int rangeEnd,
                    std::vector<std::vector<float>> &melRange) {
  const auto &audioData = project.getAudioData();
  if (audioData.warpMap.isIdentity())
    return;
  for (int frame = rangeStart; frame < rangeEnd; ++frame)
    melRange[static_cast<size_t>(frame - rangeStart)] = interpolateMelFrame(
        audioData.melSpectrogram, audioData.warpMap.sourceFrameAt(frame));
}

float interpolateF0(const std::vector<float> &f0, double frame) {
  if (f0.empty())
    return 0.0f;

  frame = std::clamp(frame, 0.0, static_cast<double>(f0.size() - 1));
  const int left = static_cast<int>(std::floor(frame));
  const int right = std::min(left + 1, static_cast<int>(f0.size()) - 1);
  const float leftValue = f0[static_cast<size_t>(left)];
  const float rightValue = f0[static_cast<size_t>(right)];

  // Never interpolate across an unvoiced gap: averaging a real pitch with a
  // zero would halve it and swing the ratio wildly for one frame.
  if (leftValue <= 0.0f)
    return rightValue;
  if (rightValue <= 0.0f)
    return leftValue;

  const float amount = static_cast<float>(frame - left);
  return leftValue * (1.0f - amount) + rightValue * amount;
}

// A note with no edit, or one on Auto kept whole as recorded for the vocal
// fry it holds (FryDetector.h): either plays the recording, and a pass that
// finds it dirty puts the recording back. Called per note on the message
// thread and on the synthesis thread: each keeps its own detection.
bool neutralForRender(const Note &note, const AudioData &audioData) {
  if (note.isNeutralForOriginalWaveform())
    return true;
  thread_local fry::Cache fryCache;
  return fry::playsRecording(note, fryCache.of(audioData));
}

// The waveform a pass commits: the edited notes. Timing is not a note edit
// any more - the warp layer lays the whole composite out again (WarpLayer.h)
// - so a note's commit range is just where it plays now.
std::vector<CommitFrameRange>
collectCommitFrameRanges(const Project &project, bool hasDirtyNoteAnchors,
                         int f0DirtyStart, int f0DirtyEnd) {
  std::vector<CommitFrameRange> ranges;

  if (hasDirtyNoteAnchors) {
    for (const auto &note : project.getNotes()) {
      if (note.isRest() || !note.isDirty())
        continue;
      if (note.getEndFrame() > note.getStartFrame())
        ranges.push_back({note.getStartFrame(), note.getEndFrame()});
    }
  }

  // F0-only tools have no dirty-note anchor, so their edited curve range is
  // the commit range.
  if (!hasDirtyNoteAnchors && f0DirtyStart >= 0 && f0DirtyEnd > f0DirtyStart)
    ranges.push_back({f0DirtyStart, f0DirtyEnd});

  std::sort(ranges.begin(), ranges.end(),
            [](const CommitFrameRange &a, const CommitFrameRange &b) {
              return a.start < b.start;
            });

  std::vector<CommitFrameRange> merged;
  for (const auto &range : ranges) {
    if (merged.empty() || range.start > merged.back().end) {
      merged.push_back(range);
    } else {
      merged.back().end = std::max(merged.back().end, range.end);
    }
  }
  return merged;
}

int findBestSpliceCenter(const float *existing,
                         const std::vector<float> &rendered,
                         int nominalCenter, int searchRadius,
                         int analysisHalfSamples) {
  const int sampleCount = static_cast<int>(rendered.size());
  if (existing == nullptr || sampleCount <= 0)
    return std::clamp(nominalCenter, 0, std::max(0, sampleCount - 1));

  const int firstCenter =
      std::max({1, analysisHalfSamples, nominalCenter - searchRadius});
  const int lastCenter = std::min(
      sampleCount - analysisHalfSamples - 1, nominalCenter + searchRadius);
  if (firstCenter > lastCenter)
    return std::clamp(nominalCenter, 0, sampleCount - 1);

  double bestScore = std::numeric_limits<double>::infinity();
  int bestCenter = std::clamp(nominalCenter, firstCenter, lastCenter);
  const double distanceScale = 1.0 / std::max(1, searchRadius);

  for (int center = firstCenter; center <= lastCenter; ++center) {
    double squaredError = 0.0;
    double signalEnergy = 1.0e-12;
    for (int offset = -analysisHalfSamples;
         offset <= analysisHalfSamples; ++offset) {
      const int sample = center + offset;
      const double weight =
          1.0 - static_cast<double>(std::abs(offset)) /
                    static_cast<double>(analysisHalfSamples + 1);
      const double oldSample = existing[sample];
      const double newSample = rendered[static_cast<size_t>(sample)];
      const double difference = oldSample - newSample;
      squaredError += weight * difference * difference;
      signalEnergy +=
          weight * (oldSample * oldSample + newSample * newSample);
    }

    const double oldValue = existing[center];
    const double newValue = rendered[static_cast<size_t>(center)];
    const double valueDifference = oldValue - newValue;
    const double valueEnergy =
        oldValue * oldValue + newValue * newValue + signalEnergy * 1.0e-3;

    const double oldSlope = existing[center] - existing[center - 1];
    const double newSlope = rendered[static_cast<size_t>(center)] -
                            rendered[static_cast<size_t>(center - 1)];
    const double slopeDifference = oldSlope - newSlope;
    const double slopeEnergy =
        oldSlope * oldSlope + newSlope * newSlope + signalEnergy * 1.0e-4;

    // Waveform agreement dominates. Matching the exact value and first
    // derivative at the splice center further suppresses residual clicks.
    const double score =
        squaredError / signalEnergy +
        0.25 * valueDifference * valueDifference / valueEnergy +
        0.10 * slopeDifference * slopeDifference / slopeEnergy +
        1.0e-6 * std::abs(center - nominalCenter) * distanceScale;
    if (score < bestScore) {
      bestScore = score;
      bestCenter = center;
    }
  }

  return bestCenter;
}

// Commit fades use their entire overlap as the comparison window. Keep the
// nominal boundary unless the improvement is substantial; timing patches keep
// their existing search policy in findBestSpliceCenter().
int findBestCommitSpliceCenter(const float *existing,
                              const std::vector<float> &rendered,
                              int nominalCenter, int searchRadius,
                              int fadeHalf) {
  const int sampleCount = static_cast<int>(rendered.size());
  // A truncated nominal fade cannot be compared fairly with a full candidate.
  if (existing == nullptr || fadeHalf <= 0 || searchRadius <= 0 ||
      nominalCenter < fadeHalf || nominalCenter > sampleCount - fadeHalf)
    return nominalCenter;

  auto scoreAt = [&](int center) {
    double error = 0.0;
    double energy = 0.0;
    for (int offset = -fadeHalf; offset < fadeHalf; ++offset) {
      const double t = static_cast<double>(offset + fadeHalf) / (2 * fadeHalf);
      const double blend = t * t * (3.0 - 2.0 * t);
      // Emphasize where both signals contribute, for either fade direction.
      const double weight = 4.0 * blend * (1.0 - blend);
      const double oldValue = existing[center + offset];
      const double newValue = rendered[static_cast<size_t>(center + offset)];
      const double difference = oldValue - newValue;
      error += weight * difference * difference;
      energy += weight * (oldValue * oldValue + newValue * newValue);
    }
    // Stabilize near-silent windows instead of chasing tiny numerical changes.
    return error / (energy + 2 * fadeHalf * 1.0e-10);
  };

  const double nominalScore = scoreAt(nominalCenter);
  double bestScore = nominalScore;
  int bestCenter = nominalCenter;
  const int radius = std::min(searchRadius, fadeHalf);
  // Visit nearest candidates first so equal scores prefer less displacement.
  for (int distance = 1; distance <= radius; ++distance) {
    for (int direction : {-1, 1}) {
      const int center = nominalCenter + direction * distance;
      if (center < fadeHalf || center > sampleCount - fadeHalf)
        continue;
      const double score = scoreAt(center);
      if (score < bestScore) {
        bestScore = score;
        bestCenter = center;
      }
    }
  }
  // Require both a 10% relative and 0.01 absolute normalized improvement.
  return nominalScore - bestScore > std::max(0.01, nominalScore * 0.10)
             ? bestCenter
             : nominalCenter;
}

EditedClusterSelection
selectConnectedEditedNotes(const Project &project, bool hasDirtyNoteAnchors,
                           int f0DirtyStart, int f0DirtyEnd) {
  const auto &notes = project.getNotes();
  EditedClusterSelection selection;
  selection.members.assign(notes.size(), uint8_t{0});

  const bool hasF0DirtyRange = f0DirtyStart >= 0 && f0DirtyEnd >= 0;
  auto isF0Anchor = [&](const Note &note) {
    return !hasDirtyNoteAnchors && hasF0DirtyRange &&
           note.getStartFrame() < f0DirtyEnd &&
           note.getEndFrame() > f0DirtyStart;
  };
  auto isSeed = [&](const Note &note) {
    if (note.isRest())
      return false;
    if (isF0Anchor(note))
      return true;
    return note.isDirty() && (std::abs(project.getFormantShift()) > 0.001f ||
                              !neutralForRender(note, project.getAudioData()));
  };
  auto isEditedClusterCandidate = [&](const Note &note) {
    if (note.isRest())
      return false;

    // A dirty neutral note is being reset. It must split the edited cluster
    // instead of pulling its neighbours into the new pass.
    if (std::abs(project.getFormantShift()) <= 0.001f &&
        note.isDirty() && neutralForRender(note, project.getAudioData()) &&
        !isF0Anchor(note))
      return false;

    return std::abs(project.getFormantShift()) > 0.001f ||
           isF0Anchor(note) || note.hasRenderedEdit() ||
           !neutralForRender(note, project.getAudioData());
  };
  auto areAdjacent = [&](size_t leftIndex, size_t rightIndex) {
    const auto &left = notes[leftIndex];
    const auto &right = notes[rightIndex];
    return right.getStartFrame() - left.getEndFrame() <=
           kAdjacentFrameTolerance;
  };

  std::vector<size_t> sortedIndices;
  sortedIndices.reserve(notes.size());
  for (size_t i = 0; i < notes.size(); ++i)
    if (!notes[i].isRest())
      sortedIndices.push_back(i);
  std::sort(sortedIndices.begin(), sortedIndices.end(),
            [&](size_t a, size_t b) {
              return notes[a].getStartFrame() < notes[b].getStartFrame();
            });

  // Every dirty/F0-edited note seeds a component. Walk transitively through
  // adjacent edited notes so the vocoder receives one continuous phrase as
  // render context. A separate commit mask later limits waveform replacement
  // to the notes that were actually edited.
  for (size_t seedPosition = 0; seedPosition < sortedIndices.size();
       ++seedPosition) {
    const size_t seedIndex = sortedIndices[seedPosition];
    if (!isSeed(notes[seedIndex]))
      continue;

    selection.members[seedIndex] = 1;

    size_t leftPosition = seedPosition;
    while (leftPosition > 0) {
      const size_t candidateIndex = sortedIndices[leftPosition - 1];
      const size_t currentIndex = sortedIndices[leftPosition];
      if (!isEditedClusterCandidate(notes[candidateIndex]) ||
          !areAdjacent(candidateIndex, currentIndex))
        break;
      selection.members[candidateIndex] = 1;
      --leftPosition;
    }

    size_t rightPosition = seedPosition;
    while (rightPosition + 1 < sortedIndices.size()) {
      const size_t currentIndex = sortedIndices[rightPosition];
      const size_t candidateIndex = sortedIndices[rightPosition + 1];
      if (!isEditedClusterCandidate(notes[candidateIndex]) ||
          !areAdjacent(currentIndex, candidateIndex))
        break;
      selection.members[candidateIndex] = 1;
      ++rightPosition;
    }
  }

  for (size_t i = 0; i < notes.size(); ++i) {
    if (selection.members[i] == 0)
      continue;
    const auto &note = notes[i];
    if (selection.startFrame < 0 ||
        note.getStartFrame() < selection.startFrame)
      selection.startFrame = note.getStartFrame();
    if (selection.endFrame < 0 || note.getEndFrame() > selection.endFrame)
      selection.endFrame = note.getEndFrame();
  }

  return selection;
}
} // namespace

IncrementalSynthesizer::IncrementalSynthesizer() = default;

IncrementalSynthesizer::~IncrementalSynthesizer() { cancel(); }

void IncrementalSynthesizer::cancel() {
  if (cancelFlag)
    cancelFlag->store(true);
}

// ---------------------------------------------------------------------------
// computeSynthesisRange: find voiced segments overlapping dirty range,
// expand to include complete segments + padding.
// ---------------------------------------------------------------------------
std::pair<int, int>
IncrementalSynthesizer::computeSynthesisRange(int dirtyStart, int dirtyEnd) {
  if (!project)
    return {dirtyStart, dirtyEnd};

  const auto &audioData = project->getAudioData();
  const int totalFrames = static_cast<int>(audioData.voicedMask.size());
  if (totalFrames == 0)
    return {dirtyStart, dirtyEnd};

  dirtyStart = std::max(0, dirtyStart);
  dirtyEnd = std::min(totalFrames, dirtyEnd);

  // Give the vocoder enough temporal context to stabilize local phase when
  // doing chunked re-synthesis.
  constexpr int kPadFrames = 24;
  // Bridge short UV gaps so adjacent notes around consonants are synthesized
  // together; this avoids junction phase resets between neighboring notes.
  constexpr int kGapBridgeFrames = 16;

  auto isVoiced = [&](int idx) -> bool {
    return idx >= 0 && idx < totalFrames && warp::voicedAt(audioData, idx);
  };

  // Expand backward to include neighboring voiced segments across short gaps.
  int start = dirtyStart;
  int backGap = 0;
  while (start > 0) {
    if (isVoiced(start - 1)) {
      --start;
      backGap = 0;
      continue;
    }
    if (backGap < kGapBridgeFrames) {
      --start;
      ++backGap;
      continue;
    }
    break;
  }
  start = std::max(0, start - kPadFrames);

  // Expand forward to include neighboring voiced segments across short gaps.
  int end = dirtyEnd;
  int fwdGap = 0;
  while (end < totalFrames) {
    if (isVoiced(end)) {
      ++end;
      fwdGap = 0;
      continue;
    }
    if (fwdGap < kGapBridgeFrames) {
      ++end;
      ++fwdGap;
      continue;
    }
    break;
  }
  end = std::min(totalFrames, end + kPadFrames);

  return {start, end};
}

// ---------------------------------------------------------------------------
// generateBlendMask: per-sample blend factor from voicedMask and the
// breath/sibilance labels.
// 1.0 = synthesized, 0.0 = original, smooth ramps at transitions.
// ---------------------------------------------------------------------------
std::vector<float>
IncrementalSynthesizer::generateBlendMask(int startFrame, int endFrame,
                                          int hopSize) {
  // Timeline frames read the analysis of the audio that plays there, through
  // the warp map (WarpLayer.h).
  const auto &audioData = project->getAudioData();
  const int totalFrames = static_cast<int>(audioData.voicedMask.size());
  const int numFrames = endFrame - startFrame;
  const int numSamples = numFrames * hopSize;
  auto isVoicedAt = [&](int frame) {
    return frame >= 0 && frame < totalFrames && warp::voicedAt(audioData, frame);
  };

  // Step 1: stability-first frame mask.
  // Default to synthesized audio in the whole region to avoid internal
  // orig/synth combing artifacts at note junctions.
  std::vector<float> frameMask(numFrames, 1.0f);

  // Keep original audio only for long unvoiced runs (e.g. clear breaths/silence),
  // not for short UV gaps between notes.
  constexpr int kKeepOriginalUnvoicedFrames = 24;
  if (numFrames > 0 && totalFrames > 0) {
    int i = 0;
    while (i < numFrames) {
      if (isVoicedAt(startFrame + i)) {
        ++i;
        continue;
      }

      const int runStart = i;
      while (i < numFrames) {
        if (isVoicedAt(startFrame + i))
          break;
        ++i;
      }
      const int runEnd = i;
      const int runLen = runEnd - runStart;
      if (runLen >= kKeepOriginalUnvoicedFrames) {
        for (int k = runStart; k < runEnd; ++k)
          frameMask[k] = 0.0f;
      }
    }
  }

  // Step 2: expand to per-sample (sample-and-hold)
  std::vector<float> mask(numSamples, 0.0f);
  for (int i = 0; i < numFrames; ++i) {
    int ss = i * hopSize;
    int se = std::min(ss + hopSize, numSamples);
    for (int s = ss; s < se; ++s)
      mask[s] = frameMask[i];
  }

  // Step 3: smooth transitions with linear ramp at frame boundaries
  constexpr int kMinRampSamples = 512;
  const int kRampSamples = std::max(kMinRampSamples, hopSize * 2);
  for (int i = 0; i < numFrames - 1; ++i) {
    if (frameMask[i] == frameMask[i + 1])
      continue;
    // Transition at frame boundary. The ramp lies inside the unvoiced run
    // (a kept run is always the unvoiced side): centred on the boundary, its
    // second half crossfaded the recording with the render over the voice's
    // onset (or its last moment), where the two meet at unrelated phases and
    // cancel - a 5-10 dB notch as a note starts after a rest. Here it hands
    // over before the voice starts, between breath and breath.
    const int boundary = (i + 1) * hopSize;
    const bool voiceStarts = frameMask[i] < frameMask[i + 1];
    int rampStart = voiceStarts ? std::max(0, boundary - kRampSamples) : boundary;
    int rampEnd =
        voiceStarts ? boundary : std::min(numSamples, boundary + kRampSamples);
    float fromVal = frameMask[i];
    float toVal = frameMask[i + 1];
    for (int s = rampStart; s < rampEnd; ++s) {
      float t = static_cast<float>(s - rampStart) /
                static_cast<float>(rampEnd - rampStart);
      mask[s] = fromVal + (toVal - fromVal) * t;
    }
  }

  // Step 4: detected breaths and sibilants keep the original recording at
  // any length (see BreathSibilanceProtection.h for why and how).
  if (audioData.warpMap.isIdentity()) {
    breathsibilance::protectNoiseInBlendMask(
        mask, audioData.breathSibilance, startFrame, numFrames, hopSize);
  } else {
    std::vector<std::uint8_t> timelineLabels(
        static_cast<size_t>(std::max(0, endFrame)), 0);
    for (int frame = std::max(0, startFrame); frame < endFrame; ++frame)
      timelineLabels[static_cast<size_t>(frame)] =
          warp::labelAt(audioData, frame);
    breathsibilance::protectNoiseInBlendMask(mask, timelineLabels, startFrame,
                                             numFrames, hopSize);
  }

  return mask;
}

// ---------------------------------------------------------------------------
// synthesizeRegion: Voiced-Only Blend approach.
// ---------------------------------------------------------------------------
std::vector<double>
IncrementalSynthesizer::buildSourceFrameMap(int startFrame, int endFrame) const {
  std::vector<double> sourceFrameMap;
  if (!project || endFrame <= startFrame)
    return sourceFrameMap;

  sourceFrameMap.resize(static_cast<size_t>(endFrame - startFrame));
  for (int frame = startFrame; frame < endFrame; ++frame)
    sourceFrameMap[static_cast<size_t>(frame - startFrame)] =
        static_cast<double>(frame);

  const auto &warpMap = project->getAudioData().warpMap;
  if (!warpMap.isIdentity())
    for (int frame = startFrame; frame < endFrame; ++frame)
      sourceFrameMap[static_cast<size_t>(frame - startFrame)] =
          warpMap.sourceFrameAt(frame);
  return sourceFrameMap;
}

std::vector<float> IncrementalSynthesizer::buildPitchRatioCurve(
    int startFrame, int endFrame, const std::vector<double> &sourceFrameMap,
    const std::vector<float> &targetF0) const {
  std::vector<float> ratio;
  if (!project || endFrame <= startFrame)
    return ratio;

  const auto &audioData = project->getAudioData();
  const auto &sourceF0 =
      audioData.denseF0.empty() ? audioData.f0 : audioData.denseF0;

  const int numFrames = endFrame - startFrame;
  ratio.assign(static_cast<size_t>(numFrames), 1.0f);
  if (sourceF0.empty())
    return ratio;

  for (int frame = 0; frame < numFrames; ++frame) {
    const size_t index = static_cast<size_t>(frame);
    if (index >= targetF0.size() || index >= sourceFrameMap.size())
      break;

    const float target = targetF0[index];
    const float source = interpolateF0(sourceF0, sourceFrameMap[index]);

    // An unvoiced frame has no meaningful pitch to move. Those frames come
    // from the original side of the blend anyway, so leaving the ratio at
    // unity keeps the transform out of their way.
    if (target > 0.0f && source > 0.0f)
      ratio[index] = target / source;
  }

  return ratio;
}

void IncrementalSynthesizer::synthesizeRegion(ProgressCallback onProgress,
                                              CompleteCallback onComplete) {
  // PSOLA transforms the original waveform, so it needs neither the model nor
  // a loaded one, and it does not want the mel.
  const bool usePsola = synthesisEngine == SynthesisEngineType::Psola;

  if (!project || (!usePsola && !vocoder)) {
    if (onComplete)
      onComplete(false);
    return;
  }

  auto &audioData = project->getAudioData();
  if (audioData.melSpectrogram.empty() || audioData.f0.empty()) {
    if (onComplete)
      onComplete(false);
    return;
  }

  if (!usePsola && !vocoder->isLoaded()) {
    if (onComplete)
      onComplete(false);
    return;
  }

  if (usePsola && audioData.originalWaveform.getNumSamples() == 0 &&
      audioData.waveform.getNumSamples() == 0) {
    if (onComplete)
      onComplete(false);
    return;
  }

  // A warp edit lays the composite out again first (WarpLayer.h): the
  // recording moved and stretched, every gain applied, and every note whose
  // pitch is edited queued to be rendered where it now plays.
  bool laidOut = false;
  if (audioData.warpLayoutPending) {
    warp::layOut(*project);
    laidOut = true;
    LOG("synthesizeRegion: warp layout rebuilt");
  } else if (!audioData.warpMap.isIdentity() &&
             audioData.warpedOriginal.getNumSamples() == 0) {
    // A loaded project: its composite is laid out already, but the
    // reference it blends onto is derived and not saved.
    warp::rebuildReference(audioData);
  }

  if (!project->hasDirtyNotes() && !project->hasF0DirtyRange()) {
    if (onComplete)
      onComplete(laidOut);
    return;
  }

  // An edited neighbour is rendered again with any note that changed next
  // to it. The base pitch is smoothed across note boundaries, so a change on
  // one note bends the first and last frames of the notes beside it; an
  // untouched neighbour is refitted to keep singing the recording, but an
  // edited one keeps its curve and its earlier render, which then plays the
  // old bend under the new line (two drags in a fast turn left the note
  // between them 2 semitones off at its start: a catch). And a note put back
  // to the recording next to an edited one sung legato would meet that
  // note's earlier render, which nothing can line up with the recording.
  // Rendered again in the same pass, the neighbour follows the new line and
  // its handovers are lined up (the commit's splices and reset-edge anchors).
  {
    constexpr int kBoundaryReach = 12; // frames the base smoothing reaches past a boundary
    auto &notes = project->getNotes();
    std::vector<Note *> neighbours;
    journalNeighbours.clear();
    for (const auto &changed : notes) {
      if (changed.isRest() || !changed.isDirty())
        continue;
      for (auto &other : notes) {
        if (other.isRest() || other.isDirty() || &other == &changed)
          continue;
        if (neutralForRender(other, project->getAudioData()) && !other.hasRenderedEdit())
          continue; // plays the recording (refitted to it)
        if (other.getStartFrame() < changed.getEndFrame() + kBoundaryReach &&
            other.getEndFrame() > changed.getStartFrame() - kBoundaryReach)
          neighbours.push_back(&other);
      }
    }
    for (auto *note : neighbours) {
      note->markDirty();
      note->markSynthDirty();
      journalNeighbours.push_back(note->getStartFrame());
    }
  }

  auto [dirtyStart, dirtyEnd] = project->getDirtyFrameRange();
  const int rawDirtyStart = dirtyStart;
  const int rawDirtyEnd = dirtyEnd;
  if (dirtyStart < 0 || dirtyEnd < 0) {
    if (onComplete)
      onComplete(false);
    return;
  }
  const bool hasDirtyNoteAnchors = project->hasDirtyNotes();
  const auto [f0DirtyStart, f0DirtyEnd] = project->getF0DirtyRange();
  auto commitFrameRanges = collectCommitFrameRanges(
      *project, hasDirtyNoteAnchors, f0DirtyStart, f0DirtyEnd);
  if (commitFrameRanges.empty()) {
    if (onComplete)
      onComplete(false);
    return;
  }
  auto editedClusters = selectConnectedEditedNotes(
      *project, hasDirtyNoteAnchors, f0DirtyStart, f0DirtyEnd);
  if (editedClusters.startFrame >= 0)
    dirtyStart = std::min(dirtyStart, editedClusters.startFrame);
  if (editedClusters.endFrame >= 0)
    dirtyEnd = std::max(dirtyEnd, editedClusters.endFrame);

  // Compute synthesis range after expanding to complete connected edited-note
  // clusters, then add voiced-segment context and padding.
  auto [startFrame, endFrame] = computeSynthesisRange(dirtyStart, dirtyEnd);
  startFrame = std::max(0, startFrame);
  endFrame =
      std::min(static_cast<int>(audioData.melSpectrogram.size()), endFrame);

  if (startFrame >= endFrame) {
    if (onComplete)
      onComplete(false);
    return;
  }

  // Breath/sibilance labels are not saved with a project. Every load path
  // rebuilds them; this catches any path that did not, before the blend
  // mask reads them.
  breathsibilance::ensure(audioData);

  // Generate blend mask before async call (voicedMask is stable here).
  // Both engines share the analysis frame grid; the phase vocoder just may
  // not have a model to ask for it.
  const int hopSize = (vocoder != nullptr && vocoder->isLoaded())
                          ? vocoder->getHopSize()
                          : HOP_SIZE;
  // Kept deliberately, not leftover debugging. Four lines and roughly 250
  // bytes per pass, against a log already dominated by model loading and
  // detection - cheap enough to leave on, and it turns "editing one note
  // rewrote the region" from a discussion into a question a user's debug log
  // answers directly. That is how the commit-range bug in 717ad6f was found.
  //
  // One line per synthesis pass. Two ranges matter and they are easy to
  // confuse: `commit` is what actually replaces waveform, `render` is only
  // the context handed to the engine and is expected to be far wider. If a
  // single note edit ever shows a commit spanning more than that note, the
  // discriminator is `noteAnchors` - `no` means the F0 branch of
  // collectCommitFrameRanges fired and committed the whole edited curve
  // range instead of one note.
  {
    const double framesToSeconds =
        static_cast<double>(hopSize) /
        static_cast<double>(std::max(1, audioData.sampleRate));

    int dirtyNoteCount = 0;
    for (const auto &note : project->getNotes())
      if (!note.isRest() && note.isDirty())
        ++dirtyNoteCount;

    juce::String commitText;
    int committedFrames = 0;
    for (const auto &range : commitFrameRanges) {
      commitText << "[" << range.start << "," << range.end << ") ";
      committedFrames += range.end - range.start;
    }
    if (commitText.isEmpty())
      commitText = "(none)";

    LOG(juce::String("synthesizeRegion: engine=") +
        (usePsola ? "PSOLA" : "Vocoder") +
        "  noteAnchors=" + (hasDirtyNoteAnchors ? "yes" : "NO") +
        "  dirtyNotes=" + juce::String(dirtyNoteCount) +
        "  f0Dirty=[" + juce::String(f0DirtyStart) + "," +
        juce::String(f0DirtyEnd) + ")");
    LOG(juce::String("  dirty=[") + juce::String(rawDirtyStart) + "," +
        juce::String(rawDirtyEnd) + ")  cluster=[" +
        juce::String(editedClusters.startFrame) + "," +
        juce::String(editedClusters.endFrame) + ")  render=[" +
        juce::String(startFrame) + "," + juce::String(endFrame) + ") " +
        juce::String(static_cast<double>(endFrame - startFrame) *
                         framesToSeconds,
                     2) + "s");
    LOG(juce::String("  commit=") + commitText + " " +
        juce::String(static_cast<double>(committedFrames) * framesToSeconds,
                     2) + "s");

#if PITCHNET_SYNTH_NOTE_DIAGNOSTICS
    // The exact inputs to collectCommitFrameRanges' branch, per dirty note.
    // pendingTiming=YES on an edit that moved nothing is the false positive
    // that promotes the commit from one note to the whole F0 dirty range.
    const auto &allNotes = project->getNotes();
    for (size_t i = 0; i < allNotes.size(); ++i) {
      const auto &note = allNotes[i];
      if (note.isRest() || !note.isDirty())
        continue;
      LOG(juce::String("  dirtyNote[") + juce::String(static_cast<int>(i)) +
          "] cur=[" + juce::String(note.getStartFrame()) + "," +
          juce::String(note.getEndFrame()) + ")  src=[" +
          juce::String(note.getSrcStartFrame()) + "," +
          juce::String(note.getSrcEndFrame()) + ")  rendered=[" +
          juce::String(note.getRenderedStartFrame()) + "," +
          juce::String(note.getRenderedEndFrame()) + ")" +
          "  hasRenderedEdit=" + (note.hasRenderedEdit() ? "yes" : "no") +
          "  neutral=" +
          (neutralForRender(note, project->getAudioData()) ? "yes" : "no"));
    }
#endif
  }

  std::vector<float> blendMask = generateBlendMask(startFrame, endFrame, hopSize);

  // Early exit: if blend mask is all-zero, nothing to synthesize
  bool hasVoiced = std::any_of(blendMask.begin(), blendMask.end(),
                               [](float v) { return v > 0.0f; });
  if (!hasVoiced) {
    project->clearAllDirty();
    if (onComplete)
      onComplete(true);
    return;
  }

  // Copy original waveform segment for blending
  // The recording as laid out by the warp map: moved where it is not
  // stretched, signalsmith where it is (WarpLayer.h).
  const auto &origWaveform = warp::reference(audioData);
  int startSample = startFrame * hopSize;
  int numSynthSamples = (endFrame - startFrame) * hopSize;
  int totalOrigSamples = origWaveform.getNumSamples();

  std::vector<float> originalSegment(numSynthSamples, 0.0f);
  {
    const float *origPtr = origWaveform.getReadPointer(0);
    int copyLen = std::min(numSynthSamples,
                           std::max(0, totalOrigSamples - startSample));
    if (copyLen > 0 && startSample >= 0)
      std::copy(origPtr + startSample, origPtr + startSample + copyLen,
                originalSegment.begin());
  }

  std::vector<float> adjustedF0Range =
      project->getAdjustedF0ForRange(startFrame, endFrame);

  if (adjustedF0Range.empty()) {
    if (onComplete)
      onComplete(false);
    return;
  }

  // Snapshot controls before dispatch: workers must not read live note edits.
  std::vector<float> formantShifts(endFrame - startFrame, project->getFormantShift());
  for (const auto& note : project->getNotes()) {
    if (note.isRest()) continue;
    for (int frame = std::max(startFrame, note.getStartFrame());
         frame < std::min(endFrame, note.getEndFrame()); ++frame)
      formantShifts[frame - startFrame] += note.getFormantShift();
  }
  const auto formantF0 = adjustedF0Range;
  const int formantSampleRate = audioData.sampleRate;
  const bool hasGlobalFormant = std::abs(project->getFormantShift()) > 0.001f;

  // Mel is the neural vocoder's input and is not cheap to copy, so the PSOLA
  // path skips building it entirely.
  std::vector<std::vector<float>> melRange;
  if (!usePsola) {
    melRange.assign(audioData.melSpectrogram.begin() + startFrame,
                    audioData.melSpectrogram.begin() + endFrame);
    applyWarpToMel(*project, startFrame, endFrame, melRange);

    if (melRange.empty()) {
      if (onComplete)
        onComplete(false);
      return;
    }
  }

  // Spec Q12: notes set to "vocoder + recorded highs". Their render keeps
  // the vocoder below 10 kHz and takes the recording above it
  // (HighBandRestore.h). Notes with a formant edit too: the recording's
  // highs stay where they were recorded, and that was preferred to the
  // vocoder's and to R3's in all 6 listening tests (formant +2 and -2
  // semitones, unstretched, 0.90x and 1.33x). A formant shift on the whole
  // project renders every note, the ones with no edit of their own included.
  // A note on Auto is one of these (fry::resolved()); the fry it holds at
  // the end of a phrase or as a word starts stays the recording
  // (fryKeptFrames below).
  std::vector<std::pair<int, int>> recordedHighsFrames;
  if (!usePsola)
    for (const auto &note : project->getNotes())
      if (!note.isRest() && note.isDirty() &&
          (std::abs(project->getFormantShift()) > 0.001f ||
           !neutralForRender(note, project->getAudioData())) &&
          fry::resolved(note) == Note::SynthEngine::VocoderRecordedHighs &&
          note.getEndFrame() > startFrame && note.getStartFrame() < endFrame)
        recordedHighsFrames.emplace_back(note.getStartFrame(), note.getEndFrame());

  // Vocal fry at the end of a phrase, and as a word starts, stays the
  // recording in notes left on Auto (FryDetector.h): no engine tried kept
  // its pulses. Timeline frames of this range that play it; they hand over
  // as a restored note does (lined up with the render next to them, then
  // crossfaded where the two agree best). Kept through the blend mask with
  // a 6 ms fade, the voice crossfaded with itself out of phase, which was
  // heard as less smooth.
  std::vector<std::pair<int, int>> fryKeptFrames;
  fry::Ranges onsetHeld; // the onsets kept, for what the vocoder sings under them
  {
    auto fryKept = fry::kept(project->getNotes(), audioData.rawF0,
                             fry::detect(audioData));
    const auto onsetKept =
        fry::keptOnsets(project->getNotes(), fry::onsets(audioData.rawF0));
    fryKept.insert(fryKept.end(), onsetKept.begin(), onsetKept.end());
    onsetHeld = onsetKept;
    if (!fryKept.empty())
      for (int frame = std::max(0, startFrame); frame < endFrame; ++frame) {
        const int source = warp::sourceFrameAt(audioData, frame);
        bool kept = false;
        for (const auto &[from, to] : fryKept)
          kept = kept || (source >= from && source < to);
        if (!kept)
          continue;
        if (!fryKeptFrames.empty() && fryKeptFrames.back().second == frame)
          fryKeptFrames.back().second = frame + 1;
        else
          fryKeptFrames.emplace_back(frame, frame + 1);
      }
  }

  // Spec Q12: notes the user chose to render with Rubber Band R3. They take
  // the recording shifted by R3 instead of the engine's render, by the same
  // ratio PSOLA would apply (target over recorded pitch, per frame).
  std::vector<std::pair<int, int>> rubberBandFrames;
  std::vector<float> rubberBandRatio;
  // The recorded pitch where it is voiced (0 elsewhere), per frame: R3's
  // formant envelope is smoothed across the voice's own harmonics with it.
  std::vector<float> rubberBandSourceHz;
  for (const auto &note : project->getNotes())
    if (!note.isRest() && note.isDirty() && !neutralForRender(note, project->getAudioData()) &&
        fry::resolved(note) == Note::SynthEngine::RubberBand &&
        note.getEndFrame() > startFrame && note.getStartFrame() < endFrame)
      rubberBandFrames.emplace_back(note.getStartFrame(), note.getEndFrame());
  if (!rubberBandFrames.empty()) {
    std::sort(rubberBandFrames.begin(), rubberBandFrames.end());
    // Target over recorded pitch where both are voiced. Unvoiced frames take
    // the ratio across them (interpolated, held at the ends) instead of 1:
    // a step in R3's pitch scale clicks (168.767 s, where a note's last
    // voiced frame went from 0.973 to 1 in one step).
    const auto sourceFrames = buildSourceFrameMap(startFrame, endFrame);
    const auto &recorded = audioData.denseF0.empty() ? audioData.f0 : audioData.denseF0;
    const int frames = endFrame - startFrame;
    rubberBandRatio.assign(static_cast<size_t>(frames), 0.0f);
    rubberBandSourceHz.assign(static_cast<size_t>(frames), 0.0f);
    // A note the warp stretched or moved after its pitch was edited keeps
    // its stored contour, spread over its own frames, and that sits a
    // fraction of a frame off the recording as the track's map plays it.
    // Target over recording then carried that misfit wherever the pitch
    // moves (a glide, vibrato): on the 煙幕 phrase at 165.65 s stretched
    // 1.02x the ratio stepped 9.8 cent rms from one frame to the next,
    // against a flat line inside each note unstretched, and R3 sang it as a
    // quiver from 1.02x up. Such notes are measured against their own
    // unedited curve laid out the same way, which leaves the edit alone.
    const auto unedited =
        PitchCurveProcessor::uneditedPitchOnMovedNotes(*project, startFrame, endFrame);
    for (int f = 0; f < frames; ++f) {
      const auto index = static_cast<size_t>(f);
      // Voicing is the recording's, where this frame reads it (the warp).
      const int absolute = index < sourceFrames.size()
                               ? static_cast<int>(std::lround(sourceFrames[index]))
                               : startFrame + f;
      const bool voiced = absolute >= 0 &&
                          static_cast<size_t>(absolute) < audioData.voicedMask.size() &&
                          audioData.voicedMask[static_cast<size_t>(absolute)];
      const float target = index < adjustedF0Range.size() ? adjustedF0Range[index] : 0.0f;
      const float source = index < sourceFrames.size() && !recorded.empty()
                               ? interpolateF0(recorded, sourceFrames[index])
                               : 0.0f;
      const float reference = index < unedited.size() && unedited[index] > 0.0f
                                  ? midiToFreq(unedited[index])
                                  : source;
      if (voiced && target > 0.0f && reference > 0.0f)
        rubberBandRatio[index] = target / reference;
      if (voiced && source > 0.0f)
        rubberBandSourceHz[index] = source;
    }
    int previous = -1;
    for (int f = 0; f <= frames; ++f) {
      if (f < frames && rubberBandRatio[static_cast<size_t>(f)] <= 0.0f)
        continue;
      const float left = previous >= 0 ? rubberBandRatio[static_cast<size_t>(previous)]
                         : f < frames  ? rubberBandRatio[static_cast<size_t>(f)]
                                       : 1.0f;
      const float right = f < frames ? rubberBandRatio[static_cast<size_t>(f)] : left;
      for (int g = previous + 1; g < f; ++g) {
        const float amount = static_cast<float>(g - previous) / static_cast<float>(f - previous);
        rubberBandRatio[static_cast<size_t>(g)] = left + amount * (right - left);
      }
      previous = f;
    }
  }

  // PSOLA's two input curves: where each output frame reads from in the
  // original audio, and how far its pitch has to move once it gets there.
  // Built here on the message thread while the project is stable.
  DspSynthesisRequest dspRequest;
  if (usePsola) {
    const auto sourceFrameMap = buildSourceFrameMap(startFrame, endFrame);
    auto pitchRatio =
        buildPitchRatioCurve(startFrame, endFrame, sourceFrameMap, adjustedF0Range);

    if (sourceFrameMap.empty() || pitchRatio.empty()) {
      if (onComplete)
        onComplete(false);
      return;
    }

    // Copy only the span the curves actually reach, plus a window of context
    // on each side, rather than the whole track on every keystroke.
    double minSourceFrame = sourceFrameMap.front();
    double maxSourceFrame = sourceFrameMap.front();
    for (double frame : sourceFrameMap) {
      minSourceFrame = std::min(minSourceFrame, frame);
      maxSourceFrame = std::max(maxSourceFrame, frame);
    }

    // Enough margin for the widest grain PSOLA can ask for at the lowest
    // pitch it will track, plus room for the mark search either side.
    const int context = 4096;
    const int totalSourceSamples = origWaveform.getNumSamples();
    const int sourceStart = std::clamp(
        static_cast<int>(std::floor(minSourceFrame)) * hopSize - context, 0,
        std::max(0, totalSourceSamples));
    const int sourceEnd = std::clamp(
        (static_cast<int>(std::ceil(maxSourceFrame)) + 1) * hopSize + context, 0,
        totalSourceSamples);

    if (sourceEnd <= sourceStart || origWaveform.getNumChannels() <= 0) {
      if (onComplete)
        onComplete(false);
      return;
    }

    const float *sourcePtr = origWaveform.getReadPointer(0);
    dspRequest.source.assign(sourcePtr + sourceStart, sourcePtr + sourceEnd);
    dspRequest.sourceStartSample = sourceStart;
    dspRequest.sourceFrame = sourceFrameMap;
    dspRequest.pitchRatio = std::move(pitchRatio);
    dspRequest.hopSize = hopSize;
    dspRequest.numOutputFrames = endFrame - startFrame;
    dspRequest.sampleRate = audioData.sampleRate;

    // PSOLA places one grain per pitch period, so it needs the original pitch
    // itself rather than only the ratio, and the voiced mask to know where a
    // period is meaningful at all. Absolute frame indexing, so a retimed note
    // reading outside the range still resolves.
    dspRequest.sourceF0 =
        audioData.denseF0.empty() ? audioData.f0 : audioData.denseF0;
    dspRequest.sourceVoiced.resize(audioData.voicedMask.size());
    for (size_t frame = 0; frame < audioData.voicedMask.size(); ++frame)
      dspRequest.sourceVoiced[frame] =
          audioData.voicedMask[frame] ? uint8_t{1} : uint8_t{0};
  }

  if (onProgress)
    onProgress(TR("progress.synthesizing"));

  // Cancel previous job
  if (cancelFlag)
    cancelFlag->store(true);
  cancelFlag = std::make_shared<std::atomic<bool>>(false);
  uint64_t currentJobId = ++jobId;
  isBusy = true;

  int capturedStartFrame = startFrame;
  int capturedEndFrame = endFrame;
  const size_t capturedNoteCount = project->getNotes().size();
  auto capturedCancelFlag = cancelFlag;
  auto capturedProject = project;
  // Debug recording: what this pass was asked to do (DebugJournal.h).
  juce::String journalDetail;
  if (project->getDebugJournal().isEnabled()) {
    juce::StringArray dirtyNotes, neighbourNotes, commits;
    for (const auto &note : project->getNotes())
      if (!note.isRest() && note.isDirty())
        dirtyNotes.add(juce::String(note.getStartFrame()));
    for (const int start : journalNeighbours)
      neighbourNotes.add(juce::String(start));
    for (const auto &range : commitFrameRanges)
      commits.add("[" + juce::String(range.start) + "," + juce::String(range.end) + ")");
    journalDetail = "dirty notes " + dirtyNotes.joinIntoString(" ") +
                    "; rendered with them " + neighbourNotes.joinIntoString(" ") +
                    "; f0 dirty [" + juce::String(f0DirtyStart) + "," + juce::String(f0DirtyEnd) +
                    "); render [" + juce::String(startFrame) + "," + juce::String(endFrame) +
                    "); commit " + commits.joinIntoString(" ") + " (frames, 512 samples)";
  }
  // Paired with the synthesizeRegion lines above: without an outcome line a
  // log cannot tell a render that committed from one that was superseded.
  const double renderStartMs = juce::Time::getMillisecondCounterHiRes();
  auto elapsedText = [renderStartMs]() {
    return juce::String((juce::Time::getMillisecondCounterHiRes() -
                         renderStartMs) / 1000.0, 1) + "s";
  };


  // Spec Q12: where the warp map is not the identity, R3 reads the
  // recording itself and does the stretch and the pitch change in one pass
  // (rubberbandshift::warpShift): the slice of the recording this render
  // plays, and where each of its frames reads it. Shifting what Signalsmith
  // had already stretched left a quiver from 1.33x up.
  std::vector<float> warpedRecording;
  std::vector<double> warpedSourceAt;
  if (!rubberBandFrames.empty() && !audioData.warpMap.isIdentity() &&
      audioData.originalWaveform.getNumSamples() > 0) {
    const auto &recording = audioData.originalWaveform;
    const int frames = endFrame - startFrame;
    warpedSourceAt.resize(static_cast<size_t>(frames) + 1);
    for (int f = 0; f <= frames; ++f)
      warpedSourceAt[static_cast<size_t>(f)] =
          audioData.warpMap.sourceAt(static_cast<double>(startFrame + f)) * hopSize;
    // R3 is fed past the end until its delayed tail is out: at the smallest
    // time ratio (1/8) that is about 16000 samples more.
    constexpr int kLead = 64, kTail = 32768;
    const int total = recording.getNumSamples();
    const int sliceStart = std::clamp(
        static_cast<int>(std::floor(warpedSourceAt.front())) - kLead, 0, total);
    const int sliceEnd = std::clamp(
        static_cast<int>(std::ceil(warpedSourceAt.back())) + kTail, sliceStart, total);
    warpedRecording.assign(recording.getReadPointer(0) + sliceStart,
                           recording.getReadPointer(0) + sliceEnd);
    for (auto &position : warpedSourceAt)
      position -= static_cast<double>(sliceStart);
  }

  // Everything past the render is engine-agnostic: both paths hand back the
  // same span of samples and share this callback verbatim.
  auto onRendered =
      [this, capturedCancelFlag, capturedProject, capturedStartFrame,
       capturedEndFrame, hopSize, currentJobId, onComplete, elapsedText,
       formantShifts, formantF0, formantSampleRate, hasGlobalFormant, usePsola,
       hasDirtyNoteAnchors, f0DirtyStart, f0DirtyEnd,
       capturedNoteCount, journalDetail, commitFrameRanges = std::move(commitFrameRanges),
       recordedHighsFrames = std::move(recordedHighsFrames),
       fryKeptFrames = std::move(fryKeptFrames),
       rubberBandFrames = std::move(rubberBandFrames),
       rubberBandRatio = std::move(rubberBandRatio),
       rubberBandFormant = rubberBandFormantPreserved,
       rubberBandSourceHz = std::move(rubberBandSourceHz),
       warpedRecording = std::move(warpedRecording),
       warpedSourceAt = std::move(warpedSourceAt),
       blendMask = std::move(blendMask),
       originalSegment = std::move(originalSegment)](
          std::vector<float> synthesizedAudio) {
        if (currentJobId != jobId.load())
          return;
        if (capturedCancelFlag->load()) {
          LOG("synthesizeRegion: cancelled after " + elapsedText() +
              " (superseded by a newer edit)");
          isBusy = false;
          if (onComplete)
            onComplete(false);
          return;
        }
        if (synthesizedAudio.empty()) {
          LOG("synthesizeRegion: render returned no audio after " +
              elapsedText());
          isBusy = false;
          if (onComplete)
            onComplete(false);
          return;
        }

        std::thread([this, capturedCancelFlag, capturedProject,
                     capturedStartFrame, capturedEndFrame, hopSize,
                     currentJobId, onComplete, elapsedText, hasDirtyNoteAnchors,
                     f0DirtyStart, f0DirtyEnd, capturedNoteCount, journalDetail,
                     commitFrameRanges, blendMask, originalSegment,
                     recordedHighsFrames, fryKeptFrames,
                     rubberBandFrames, rubberBandRatio, rubberBandFormant,
                     rubberBandSourceHz, warpedRecording, warpedSourceAt,
                     formantShifts, formantF0, formantSampleRate, hasGlobalFormant, usePsola,
                     synthesizedAudio = std::move(synthesizedAudio)]() mutable {
          if (currentJobId != jobId.load())
            return;
          if (capturedCancelFlag->load()) {
            isBusy = false;
            if (onComplete)
              juce::MessageManager::callAsync(
                  [onComplete]() { onComplete(false); });
            return;
          }

          auto &audioData = capturedProject->getAudioData();
          if (capturedProject->getNotes().size() != capturedNoteCount) {
            LOG("synthesizeRegion: discarded after " + elapsedText() +
                " (note count changed during render)");
            isBusy = false;
            if (onComplete)
              juce::MessageManager::callAsync(
                  [onComplete]() { onComplete(false); });
            return;
          }
          int totalSamples = audioData.waveform.getNumSamples();
          int startSample = capturedStartFrame * hopSize;
          int expectedSamples =
              (capturedEndFrame - capturedStartFrame) * hopSize;

          if (expectedSamples <= 0) {
            isBusy = false;
            if (onComplete)
              juce::MessageManager::callAsync(
                  [onComplete]() { onComplete(false); });
            return;
          }

          // Back a short render with the original, never with silence.
          //
          // A render that returns fewer samples than the frame count asks for
          // used to be zero-filled to length. Those samples land where the
          // blend mask reads 1.0 - fully synthesized - so the hole is
          // committed with no fade on either side and reads as a dropout.
          // Continuing with the original samples degrades to leaving that
          // stretch unedited, which is what the blend already does for every
          // unvoiced frame. The short fade keeps the handover from being a
          // step in its own right.
          //
          // Whether this ever fires is unknown, hence the log line: if the
          // model is truncating, that is worth finding out from a user's
          // debug log rather than by inference.
          {
            const size_t target = static_cast<size_t>(expectedSamples);
            const size_t rendered = std::min(synthesizedAudio.size(), target);
            if (rendered < target) {
              LOG("IncrementalSynthesizer: short render, " +
                  juce::String(static_cast<int>(rendered)) + " of " +
                  juce::String(expectedSamples) +
                  " samples; padding from original");
            }
            synthesizedAudio.resize(target, 0.0f);
            for (size_t i = rendered; i < target; ++i)
              synthesizedAudio[i] =
                  i < originalSegment.size() ? originalSegment[i] : 0.0f;

            const size_t fade = std::min<size_t>(128, rendered);
            if (rendered < target && fade > 0) {
              for (size_t i = rendered - fade; i < rendered; ++i) {
                if (i >= originalSegment.size())
                  break;
                const float t = static_cast<float>(i - (rendered - fade)) /
                                static_cast<float>(fade);
                synthesizedAudio[i] +=
                    t * (originalSegment[i] - synthesizedAudio[i]);
              }
            }
          }

          // Advance the vocoder's output by its latency so it lines up with
          // the recording it is blended with. The range carries 24 frames of
          // context past the commit, so the tail filled in from the
          // recording is never committed.
          if (!usePsola) {
            const int latency = static_cast<int>(
                std::lround(kVocoderLatencySeconds * formantSampleRate));
            if (latency > 0 &&
                latency < static_cast<int>(synthesizedAudio.size())) {
              synthesizedAudio.erase(synthesizedAudio.begin(),
                                     synthesizedAudio.begin() + latency);
              const size_t size = synthesizedAudio.size();
              synthesizedAudio.resize(size + static_cast<size_t>(latency));
              for (size_t i = size; i < synthesizedAudio.size(); ++i)
                synthesizedAudio[i] =
                    i < originalSegment.size() ? originalSegment[i] : 0.0f;
            }
          }

          if (!formant::process(synthesizedAudio, formantShifts, formantF0,
                                hopSize, formantSampleRate, capturedCancelFlag.get())) {
            if (currentJobId == jobId.load()) {
              isBusy = false;
              if (onComplete)
                juce::MessageManager::callAsync([onComplete]() { onComplete(false); });
            }
            return;
          }

          int samplesToWrite =
              std::min(expectedSamples, totalSamples - startSample);
          if (samplesToWrite <= 0) {
            isBusy = false;
            if (onComplete)
              juce::MessageManager::callAsync(
                  [onComplete]() { onComplete(false); });
            return;
          }

          // Spec Q12: the Rubber Band notes. Over each of them the recording
          // becomes the recording shifted by R3 where the voice is (the blend
          // mask), consonants, breaths and silence staying as recorded (R3
          // smears a note into the silence after it). The mask is smoothed
          // over 20 ms first: switched sample by sample with it, the two
          // signals, whose phases do not agree, met in a step wherever the
          // mask moved fast (a note edge sung legato, a voiced run ending): a
          // pop (02:48.568). Past each edge the R3 audio goes on over the
          // whole handover fade below, then back to the recording over 10 ms,
          // where a neighbour's render covers it. The notes are then laid in
          // like a restored span: the render is dropped there, and at their
          // edges a neighbour's render is lined up with the R3 audio.
          std::vector<std::pair<int, int>> rubberBandSamples;
          if (!rubberBandFrames.empty()) {
            const int ratioFrames = static_cast<int>(rubberBandRatio.size());
            std::vector<float> ratio(originalSegment.size(), 1.0f);
            for (size_t i = 0; i < ratio.size(); ++i)
              ratio[i] = dspSynthesis::samplePitchRatio(
                  rubberBandRatio,
                  (static_cast<double>(i) + 0.5) / hopSize - 0.5, ratioFrames);
            std::vector<float> sourceHz(originalSegment.size(), 0.0f);
            for (size_t i = 0; i < sourceHz.size(); ++i) {
              const size_t frame = i / static_cast<size_t>(hopSize);
              if (frame < rubberBandSourceHz.size())
                sourceHz[i] = rubberBandSourceHz[frame];
            }
            auto shifted = warpedRecording.empty()
                               ? rubberbandshift::shift(originalSegment, ratio,
                                                        formantSampleRate,
                                                        capturedCancelFlag.get(),
                                                        rubberBandFormant, &sourceHz)
                               : rubberbandshift::warpShift(
                                     warpedRecording.data(),
                                     static_cast<int>(warpedRecording.size()),
                                     warpedSourceAt, hopSize, ratio,
                                     formantSampleRate, capturedCancelFlag.get(),
                                     rubberBandFormant, &sourceHz);
            if (shifted.empty() ||
                !formant::process(shifted, formantShifts, formantF0, hopSize,
                                  formantSampleRate, capturedCancelFlag.get())) {
              if (currentJobId == jobId.load()) {
                isBusy = false;
                if (onComplete)
                  juce::MessageManager::callAsync([onComplete]() { onComplete(false); });
              }
              return;
            }
            for (const auto &[from, to] : rubberBandFrames) {
              const int first = std::max(0, (from - capturedStartFrame) * hopSize);
              const int last = std::min(samplesToWrite, (to - capturedStartFrame) * hopSize);
              if (last <= first)
                continue;
              if (!rubberBandSamples.empty() && rubberBandSamples.back().second == first)
                rubberBandSamples.back().second = last;
              else
                rubberBandSamples.emplace_back(first, last);
            }
            // R3 rebuilds the phase: even where it does not shift, its output
            // does not line up with the recording (correlation -0.6 to 0.0 at a
            // ratio of 1). So where R3 audio meets anything else mid-voice it
            // is lined up first, as a render is. Two kinds of edge:
            //  - inner: a neighbour rendered in this pass. Its render is lined
            //    up with the R3 audio by the handover below (the R3 audio goes
            //    on over that whole fade, then back to the recording over
            //    10 ms where the render covers it);
            //  - outer: the commit ends there (an unedited neighbour). The R3
            //    audio hands over to the recording where the two agree best
            //    once lined up: candidate points from just inside the note to
            //    93 ms into the neighbour, each with its own shift and rotation
            //    (splicealign), scored by what is left of their difference -
            //    in silence that is the silence (a note after a rest), sung
            //    legato the closest match. The alignment glides in over the
            //    46 ms before, the crossfade takes 512 samples, and the commit
            //    is carried past it with no render up to the reach of its own
            //    fade, so it ends where both sides are the recording. Ending
            //    the commit at the note edge crossfaded the R3 audio into the
            //    old render there: an uneven joint (會 into 我, 03:50.37).
            constexpr int kReach = 256 + 128 + 64; // the handover fade's reach
            constexpr int kOuterHalf = 256;        // half the handover crossfade
            constexpr int kOuterSearchIn = 256;    // candidates inside the note
            constexpr int kOuterSearchOut = 4096;  // and into the neighbour
            constexpr int kOuterStep = 32;
            constexpr int kOuterGlide = 2048;
            constexpr int kCommitReach = 512 + 128; // the commit fade's reach
            const int ramp = std::max(1, formantSampleRate / 100);
            const int total = static_cast<int>(originalSegment.size());
            const std::vector<float> recordedSegment = originalSegment;
            const auto commitEdgeAt = [&](int sample, bool end) {
              for (const auto &range : commitFrameRanges)
                if (((end ? range.end : range.start) - capturedStartFrame) * hopSize == sample)
                  return true;
              return false;
            };
            const auto maxLagAt = [&](int centre) {
              const int frame = capturedStartFrame + centre / hopSize;
              const auto &f0 = audioData.f0;
              if (frame < 0 || frame >= static_cast<int>(f0.size()) ||
                  !(f0[static_cast<size_t>(frame)] > 0.0f))
                return splicealign::kMaxLagSamples;
              return std::clamp(static_cast<int>(std::ceil(
                                    audioData.sampleRate / f0[static_cast<size_t>(frame)] / 2.0)),
                                splicealign::kMaxLagSamples, 320);
            };
            const auto smooth = [](float x) {
              x = std::clamp(x, 0.0f, 1.0f);
              return x * x * (3.0f - 2.0f * x);
            };
            const auto shiftedHilbert = splicealign::hilbert(shifted);
            // The best handover point between `from` and `to`: its alignment
            // and its score (the lined-up difference over the energy there,
            // the energy itself counting against loud points).
            struct Handover {
              int at = -1;
              splicealign::Alignment alignment;
            };
            const auto bestHandover = [&](int from, int to) {
              Handover best;
              double bestScore = std::numeric_limits<double>::infinity();
              double loudest = 1.0e-12;
              for (int c = from; c <= to; c += kOuterStep)
                if (c - kOuterHalf >= 0 && c + kOuterHalf <= total)
                  for (int i = c - kOuterHalf; i < c + kOuterHalf; ++i)
                    loudest = std::max(loudest, static_cast<double>(std::abs(recordedSegment[static_cast<size_t>(i)])));
              for (int c = from; c <= to; c += kOuterStep) {
                if (c - kOuterHalf < 0 || c + kOuterHalf > total)
                  continue;
                const auto alignment = splicealign::find(recordedSegment.data(), shifted,
                                                         shiftedHilbert, c, kOuterHalf,
                                                         maxLagAt(c));
                double error = 0.0, energy = 1.0e-12;
                for (int i = c - kOuterHalf; i < c + kOuterHalf; ++i) {
                  const double w = 1.0 - std::abs(i - c) / static_cast<double>(kOuterHalf + 1);
                  const double o = recordedSegment[static_cast<size_t>(i)];
                  const double r = splicealign::sample(shifted, shiftedHilbert, i, alignment);
                  error += w * (o - r) * (o - r);
                  energy += w * (o * o + r * r);
                }
                // Mismatch in proportion to what is heard there.
                const double level = std::sqrt(energy / kOuterHalf) / loudest;
                const double score = (error / energy) * level;
                if (score < bestScore) {
                  bestScore = score;
                  best = {c, alignment};
                }
              }
              return best;
            };
            std::vector<float> aligned = shifted;
            std::vector<float> weight(originalSegment.size(), 0.0f);
            std::vector<std::pair<int, int>> renderless; // samples with no render
            struct Extension {
              int edge;     // span edge (samples)
              bool end;     // the span's end, else its start
              int reachTo;  // commit carried to cover this sample
            };
            std::vector<Extension> extensions;
            for (const auto &[first, last] : rubberBandSamples) {
              const bool outerLeft = commitEdgeAt(first, false) && first > 0;
              const bool outerRight = commitEdgeAt(last, true) && last < samplesToWrite;
              const int glide = std::max(1, std::min(kOuterGlide,
                                                     (last - first) / (outerLeft && outerRight ? 2 : 1)));
              Handover right, left;
              if (outerRight)
                right = bestHandover(std::max(first + glide / 2, last - kOuterSearchIn),
                                     std::min(total - kOuterHalf, last + kOuterSearchOut));
              if (outerLeft)
                left = bestHandover(std::max(kOuterHalf, first - kOuterSearchOut),
                                    std::min(last - glide / 2, first + kOuterSearchIn));
              if (right.at >= 0) {
                const int glideStart = std::max(first, right.at - kOuterHalf - glide);
                for (int n = glideStart; n < std::min(total, right.at + kOuterHalf + 64); ++n) {
                  const float g = n < right.at - kOuterHalf
                                      ? smooth(static_cast<float>(n - glideStart) /
                                               std::max(1, right.at - kOuterHalf - glideStart))
                                      : 1.0f;
                  aligned[static_cast<size_t>(n)] = splicealign::sample(
                      shifted, shiftedHilbert, n, {g * right.alignment.lag, g * right.alignment.angle});
                }
                const int reachTo = std::min(total, right.at + kOuterHalf);
                renderless.emplace_back(last, std::min(total, reachTo + 2 * kCommitReach + hopSize));
                extensions.push_back({last, true, reachTo});
              }
              if (left.at >= 0) {
                const int glideEnd = std::min(last, left.at + kOuterHalf + glide);
                for (int n = std::max(0, left.at - kOuterHalf - 64); n < glideEnd; ++n) {
                  const float g = n >= left.at + kOuterHalf
                                      ? smooth(static_cast<float>(glideEnd - n) /
                                               std::max(1, glideEnd - (left.at + kOuterHalf)))
                                      : 1.0f;
                  aligned[static_cast<size_t>(n)] = splicealign::sample(
                      shifted, shiftedHilbert, n, {g * left.alignment.lag, g * left.alignment.angle});
                }
                const int reachTo = std::max(0, left.at - kOuterHalf);
                renderless.emplace_back(std::max(0, reachTo - 2 * kCommitReach - hopSize), first);
                extensions.push_back({first, false, reachTo});
              }
              // R3 over the note; past an outer edge up to its handover
              // (crossfaded over it), past an inner edge over the handover
              // fade's reach and then 10 ms.
              const int from = std::max(0, left.at >= 0 ? left.at - kOuterHalf : first - kReach - ramp);
              const int to = std::min(total, right.at >= 0 ? right.at + kOuterHalf : last + kReach + ramp);
              for (int i = from; i < to; ++i) {
                float w = 1.0f;
                if (left.at >= 0 && i < left.at + kOuterHalf)
                  w = smooth(static_cast<float>(i - (left.at - kOuterHalf)) / (2 * kOuterHalf));
                else if (left.at < 0 && i < first - kReach)
                  w = smooth(static_cast<float>(i - (first - kReach - ramp)) / ramp);
                if (right.at >= 0 && i >= right.at - kOuterHalf)
                  w = std::min(w, smooth(static_cast<float>((right.at + kOuterHalf) - i) / (2 * kOuterHalf)));
                else if (right.at < 0 && i >= last + kReach)
                  w = std::min(w, smooth(static_cast<float>((last + kReach + ramp) - i) / ramp));
                weight[static_cast<size_t>(i)] = std::max(weight[static_cast<size_t>(i)], w);
              }
            }
            // The voice, smoothed: a 20 ms moving average of the blend mask.
            const int smoothing = std::max(1, formantSampleRate / 50);
            std::vector<double> sum(originalSegment.size() + 1, 0.0);
            for (size_t i = 0; i < originalSegment.size(); ++i)
              sum[i + 1] = sum[i] + (i < blendMask.size() ? blendMask[i] : 0.0f);
            for (size_t i = 0; i < originalSegment.size() && i < aligned.size(); ++i) {
              if (weight[i] <= 0.0f)
                continue;
              const auto lo = static_cast<size_t>(std::max<long long>(0, static_cast<long long>(i) - smoothing / 2));
              const auto hi = std::min(originalSegment.size(), i + static_cast<size_t>(smoothing / 2) + 1);
              const float voice = static_cast<float>((sum[hi] - sum[lo]) / static_cast<double>(hi - lo));
              originalSegment[i] += weight[i] * voice * (aligned[i] - originalSegment[i]);
            }
            // No render past an outer edge, and the commit carried over the
            // handover with room for its own fade.
            for (const auto &[from, to] : renderless)
              for (int i = from; i < to && i < static_cast<int>(blendMask.size()); ++i)
                blendMask[static_cast<size_t>(i)] = 0.0f;
            for (const auto &extension : extensions)
              for (auto &range : commitFrameRanges) {
                if (extension.end &&
                    (range.end - capturedStartFrame) * hopSize == extension.edge) {
                  const int need = (extension.reachTo + kCommitReach + hopSize - 1) / hopSize;
                  range.end = std::min(capturedEndFrame, capturedStartFrame + std::max(
                      range.end - capturedStartFrame, need));
                }
                if (!extension.end &&
                    (range.start - capturedStartFrame) * hopSize == extension.edge) {
                  const int need = std::max(0, (extension.reachTo - kCommitReach) / hopSize);
                  range.start = std::max(capturedStartFrame, capturedStartFrame + std::min(
                      range.start - capturedStartFrame, need));
                }
              }
          }

          // Build blended target from model/original. `keptOriginal` is the
          // part of it that is the recording (unvoiced runs, breaths and
          // sibilants, restored spans): the splice alignment below moves and
          // rotates only the render, so this part is written as it is.
          std::vector<float> targetSegment(samplesToWrite, 0.0f);
          std::vector<float> keptOriginal(samplesToWrite, 0.0f);
          for (int i = 0; i < samplesToWrite; ++i) {
            const float b =
                (i < static_cast<int>(blendMask.size())) ? blendMask[i] : 0.0f;
            const float synth = synthesizedAudio[static_cast<size_t>(i)];
            const float orig = originalSegment[static_cast<size_t>(i)];
            keptOriginal[static_cast<size_t>(i)] = (1.0f - b) * orig;
            targetSegment[static_cast<size_t>(i)] =
                b * synth + keptOriginal[static_cast<size_t>(i)];
          }

          // Use the same gain envelope as direct gain edits (GainEditAction):
          // note volumes, sibilant balance and breath levels.
          const auto gainRegions = projectgain::regions(*capturedProject);
          const auto sampleGain = NoteGainCurve::build(
              gainRegions, startSample, samplesToWrite,
              audioData.waveform.getNumSamples(), hopSize);
          // Gains outside the notes alone (breath levels, stray sibilants):
          // a restore of pristine audio around a neutral note must keep them.
          const auto breathRegions =
              projectgain::outsideNoteRegions(*capturedProject);
          const auto breathGain =
              breathRegions.empty()
                  ? std::vector<float>{}
                  : NoteGainCurve::build(breathRegions, startSample,
                                         samplesToWrite,
                                         audioData.waveform.getNumSamples(),
                                         hopSize);

          for (int i = 0; i < samplesToWrite; ++i) {
            targetSegment[static_cast<size_t>(i)] *=
                sampleGain[static_cast<size_t>(i)];
            keptOriginal[static_cast<size_t>(i)] *=
                sampleGain[static_cast<size_t>(i)];
          }
          // The render alone (with its gain), and the share of it each
          // sample keeps (the blend mask, then less where a reset span takes
          // over): the splice alignment moves and rotates the render first and
          // weighs it after, so a shift never carries a fade along with it.
          std::vector<float> synthGained(static_cast<size_t>(samplesToWrite));
          std::vector<float> renderShare(static_cast<size_t>(samplesToWrite));
          for (int i = 0; i < samplesToWrite; ++i) {
            synthGained[static_cast<size_t>(i)] =
                synthesizedAudio[static_cast<size_t>(i)] * sampleGain[static_cast<size_t>(i)];
            renderShare[static_cast<size_t>(i)] =
                (i < static_cast<int>(blendMask.size())) ? blendMask[i] : 0.0f;
          }

          // A neutral dirty note is an explicit reset: its body must be the
          // pristine source, not a vocoder round-trip at neutral pitch.
          //
          // Consecutive neutral notes form one contiguous reset span. Fading
          // at every internal note boundary crossfaded the synthesized pass
          // against the original, and those two match in energy but not in
          // phase, so the blend cancels instead of smoothing - no fade length
          // avoids that. Inside a span there is nothing to smooth between,
          // since both sides are the same source audio, so the span is copied
          // wholesale. Only its outer edges, where it meets genuinely edited
          // audio, need a transition, and those are placed where the two
          // signals agree best rather than on the note boundary.
          constexpr int kResetFadeHalfSamples = 256;
          constexpr int kResetSpliceSearchRadiusSamples = 128;

          struct ResetSpan {
            int startSample;
            int endSample;
            bool rubberBand = false; // a Rubber Band note (Q12) or kept fry, not a restore
          };
          std::vector<ResetSpan> resetSpans;
          if (!hasGlobalFormant) {
            // Notes are not guaranteed to be in frame order, so collect and
            // sort before looking for contiguity.
            std::vector<std::pair<int, int>> neutralFrameRanges;
            for (const auto &note : capturedProject->getNotes()) {
              if (note.isRest() || !note.isDirty() ||
                  !neutralForRender(note, capturedProject->getAudioData()))
                continue;
              neutralFrameRanges.emplace_back(note.getStartFrame(),
                                              note.getEndFrame());
            }
            std::sort(neutralFrameRanges.begin(), neutralFrameRanges.end());

            int spanStartFrame = -1;
            int spanEndFrame = -1;
            auto flushSpan = [&]() {
              if (spanStartFrame < 0)
                return;
              const int first =
                  std::max(0, (spanStartFrame - capturedStartFrame) * hopSize);
              const int last =
                  std::min(samplesToWrite,
                           (spanEndFrame - capturedStartFrame) * hopSize);
              if (last > first)
                resetSpans.push_back({first, last});
              spanStartFrame = -1;
              spanEndFrame = -1;
            };
            for (const auto &range : neutralFrameRanges) {
              // Only exactly contiguous notes join a span. Any gap - a rest,
              // or an edited note between two neutral ones - ends it, so the
              // restore is never silently widened past what it covered before.
              if (spanStartFrame >= 0 && range.first == spanEndFrame) {
                spanEndFrame = std::max(spanEndFrame, range.second);
              } else {
                flushSpan();
                spanStartFrame = range.first;
                spanEndFrame = range.second;
              }
            }
            flushSpan();
          }

          // Where each reset edge hands over from the render to the
          // recording: the commit's alignment below lines the render up there.
          std::vector<int> resetEdgeCentres;

          auto resetSmoothstep = [](float t) {
            t = std::clamp(t, 0.0f, 1.0f);
            return t * t * (3.0f - 2.0f * t);
          };

          // The Rubber Band notes hand over like restored spans; only the
          // restores go on to the exact restore of the recording further down.
          std::vector<ResetSpan> handoverSpans = resetSpans;
          for (const auto &[first, last] : rubberBandSamples)
            handoverSpans.push_back({first, last, true});
          // Vocal fry kept as recorded hands over the same way, with the
          // note's own gains like a Rubber Band note.
          for (const auto &[from, to] : fryKeptFrames) {
            const int first =
                std::max(0, (from - capturedStartFrame) * hopSize);
            const int last =
                std::min(samplesToWrite, (to - capturedStartFrame) * hopSize);
            if (last > first)
              handoverSpans.push_back({first, last, true});
          }
          std::sort(handoverSpans.begin(), handoverSpans.end(),
                    [](const ResetSpan &a, const ResetSpan &b) {
                      return a.startSample < b.startSample;
                    });

          for (size_t spanIndex = 0; spanIndex < handoverSpans.size();
               ++spanIndex) {
            const auto &span = handoverSpans[spanIndex];
            const int bodySamples = span.endSample - span.startSample;
            if (bodySamples <= 0)
              continue;

            const int fadeHalf =
                std::min(kResetFadeHalfSamples, std::max(1, bodySamples / 4));
            const int searchRadius =
                std::min(kResetSpliceSearchRadiusSamples,
                         std::max(0, bodySamples / 2 - fadeHalf));
            const bool fadeLeft = span.startSample > 0;
            const bool fadeRight = span.endSample < samplesToWrite;

            const int leftCenter =
                fadeLeft ? findBestCommitSpliceCenter(
                               originalSegment.data(), targetSegment,
                               span.startSample, searchRadius, fadeHalf)
                         : span.startSample;
            const int rightCenter =
                fadeRight ? findBestCommitSpliceCenter(
                                originalSegment.data(), targetSegment,
                                span.endSample, searchRadius, fadeHalf)
                          : span.endSample;

            // A fade needs room outside the span, so keep it from reaching
            // into a neighbouring span and blending the same samples twice.
            const int previousEnd =
                spanIndex > 0 ? handoverSpans[spanIndex - 1].endSample : 0;
            const int nextStart = spanIndex + 1 < handoverSpans.size()
                                      ? handoverSpans[spanIndex + 1].startSample
                                      : samplesToWrite;
            const int firstSample = std::max(
                {0, previousEnd,
                 fadeLeft ? leftCenter - fadeHalf : span.startSample});
            const int lastSample = std::min(
                {samplesToWrite, nextStart,
                 fadeRight ? rightCenter + fadeHalf : span.endSample});
            if (fadeLeft)
              resetEdgeCentres.push_back(leftCenter);
            if (fadeRight)
              resetEdgeCentres.push_back(rightCenter);

            for (int sample = firstSample; sample < lastSample; ++sample) {
              float leftBlend = 1.0f;
              if (fadeLeft && sample < leftCenter + fadeHalf)
                leftBlend = resetSmoothstep(
                    static_cast<float>(sample - (leftCenter - fadeHalf)) /
                    static_cast<float>(2 * fadeHalf));

              float rightBlend = 1.0f;
              if (fadeRight && sample >= rightCenter - fadeHalf)
                rightBlend = resetSmoothstep(
                    static_cast<float>((rightCenter + fadeHalf) - sample) /
                    static_cast<float>(2 * fadeHalf));

              const float blend = std::min(leftBlend, rightBlend);
              const auto index = static_cast<size_t>(sample);
              // A Rubber Band note keeps its own gains (volume, sibilants);
              // a restored note only the breath levels around it.
              const float restored =
                  span.rubberBand ? originalSegment[index] * sampleGain[index]
                  : breathGain.empty() ? originalSegment[index]
                                       : originalSegment[index] * breathGain[index];
              targetSegment[index] += blend * (restored - targetSegment[index]);
              keptOriginal[index] += blend * (restored - keptOriginal[index]);
              renderShare[index] *= 1.0f - blend;
            }
          }

          // Commit only the edited interval. Preserve the 1024-sample
          // smoothstep fade (~23 ms at 44.1 kHz), allowing a conservative
          // displacement of at most 128 samples (~2.9 ms) when agreement
          // across the full overlap meaningfully improves.
          constexpr int kCommitFadeHalfSamples = 512;
          constexpr int kSpliceSearchRadiusSamples = 128;
          std::vector<float> commitMask(static_cast<size_t>(samplesToWrite),
                                        0.0f);
          const float *existingSamples =
              audioData.waveform.getReadPointer(0, startSample);
          auto smoothstep = [](float t) {
            t = std::clamp(t, 0.0f, 1.0f);
            return t * t * (3.0f - 2.0f * t);
          };
          // What is committed: the render, shifted and phase-rotated at each
          // splice to match the audio it fades into (SpliceAlignment.h).
          std::vector<float> committed = targetSegment;
          const auto renderHilbert = splicealign::hilbert(targetSegment);
          // The render alone is what gets moved and rotated: shifting the
          // kept recording too smeared sibilants with a gliding fractional
          // delay (a consonant wobbled after a whole-song render).
          const auto synthHilbert = splicealign::hilbert(synthGained);
          const bool journalOn = capturedProject->getDebugJournal().isEnabled();
          std::vector<debugjournal::Splice> journalSplices;
          for (const auto &range : commitFrameRanges) {
            const int bodyStart =
                (range.start - capturedStartFrame) * hopSize;
            const int bodyEnd = (range.end - capturedStartFrame) * hopSize;
            const int bodySamples = bodyEnd - bodyStart;
            if (bodySamples <= 0)
              continue;

            // Quarter of the body rather than a half: the fade is centred on
            // a searched point, not on the boundary, so it needs room to be
            // displaced into without colliding with the fade at the far end.
            const int fadeHalf =
                std::min(kCommitFadeHalfSamples,
                         std::max(1, bodySamples / 4));
            const bool fadeLeft = range.start > 0;
            const bool fadeRight = range.end * hopSize < totalSamples;
            const int nonOverlappingSearchRadius =
                std::max(0, bodySamples / 2 - fadeHalf);
            const int searchRadius =
                std::min(kSpliceSearchRadiusSamples, nonOverlappingSearchRadius);
            const int leftCenter = fadeLeft
                                       ? findBestCommitSpliceCenter(
                                             existingSamples, targetSegment,
                                             bodyStart, searchRadius, fadeHalf)
                                       : bodyStart;
            const int rightCenter = fadeRight
                                        ? findBestCommitSpliceCenter(
                                              existingSamples, targetSegment,
                                              bodyEnd, searchRadius, fadeHalf)
                                        : bodyEnd;
            const int leftFadeStart = leftCenter - fadeHalf;
            const int leftFadeEnd = leftCenter + fadeHalf;
            const int rightFadeStart = rightCenter - fadeHalf;
            const int rightFadeEnd = rightCenter + fadeHalf;
            const int firstSample =
                std::max(0, fadeLeft ? leftFadeStart : bodyStart);
            const int lastSample = std::min(
                samplesToWrite, fadeRight ? rightFadeEnd : bodyEnd);

            // Each splice its own shift and angle; between them they glide.
            // The vocoder starts its own phase, so in a voice the render can
            // sit anywhere within a pitch period of the recording: the shift
            // is searched over half a period either way (kMaxLagSamples, about
            // 1 ms, left a low voice's render up to half a period off, and the
            // crossfade dipped where an edited note met an unedited one sung
            // legato).
            auto periodAt = [&](int centre) {
              const int frame = (startSample + centre) / hopSize;
              const auto &f0 = audioData.f0;
              if (frame < 0 || frame >= static_cast<int>(f0.size()) ||
                  !(f0[static_cast<size_t>(frame)] > 0.0f))
                return 0.0;
              return static_cast<double>(audioData.sampleRate) /
                     f0[static_cast<size_t>(frame)];
            };
            constexpr int kMaxPeriodLag = 320; // half the period of a 70 Hz voice
            auto maxLagAt = [&](int centre) {
              const double period = periodAt(centre);
              return period > 0.0
                         ? std::clamp(static_cast<int>(std::ceil(period / 2.0)),
                                      splicealign::kMaxLagSamples, kMaxPeriodLag)
                         : splicealign::kMaxLagSamples;
            };
            auto leftAlign = fadeLeft ? splicealign::find(existingSamples,
                                                          targetSegment,
                                                          renderHilbert,
                                                          leftCenter, fadeHalf,
                                                          maxLagAt(leftCenter))
                                      : splicealign::Alignment{};
            auto rightAlign = fadeRight ? splicealign::find(existingSamples,
                                                            targetSegment,
                                                            renderHilbert,
                                                            rightCenter, fadeHalf,
                                                            maxLagAt(rightCenter))
                                        : splicealign::Alignment{};
            if (!fadeLeft)
              leftAlign = rightAlign;
            if (!fadeRight)
              rightAlign = leftAlign;
            // A reset span inside the range (a note put back to the
            // recording next to an edited one rendered in the same pass)
            // hands the render over to the recording at its edge, mid-voice
            // when the notes are sung legato. That handover is a splice too:
            // the render is lined up with the recording there as well (only
            // the render moves; the recording is written as it is).
            struct Anchor {
              int at;
              splicealign::Alignment alignment;
            };
            std::vector<Anchor> anchors{{leftCenter, leftAlign}};
            for (const int centre : resetEdgeCentres) {
              if (centre <= leftCenter || centre >= rightCenter)
                continue;
              double renderEnergy = 0.0, originalEnergy = 0.0;
              for (int i = std::max(0, centre - kResetFadeHalfSamples);
                   i < std::min(samplesToWrite, centre + kResetFadeHalfSamples); ++i) {
                const float r = synthGained[static_cast<size_t>(i)];
                const float o = originalSegment[static_cast<size_t>(i)];
                renderEnergy += static_cast<double>(r) * r;
                originalEnergy += static_cast<double>(o) * o;
              }
              // Silence on either side: nothing to line up.
              constexpr double kAudible = 2 * kResetFadeHalfSamples * 1.0e-6;
              if (renderEnergy < kAudible || originalEnergy < kAudible)
                continue;
              anchors.push_back({centre, splicealign::find(
                                             originalSegment.data(), synthGained,
                                             synthHilbert, centre,
                                             kResetFadeHalfSamples, maxLagAt(centre))});
            }
            anchors.push_back({rightCenter, rightAlign});
            std::sort(anchors.begin() + 1, anchors.end() - 1,
                      [](const Anchor &a, const Anchor &b) { return a.at < b.at; });
            // From each anchor to the next: a shift of a whole period is the
            // same in a voice, so of the next anchor's equivalent shifts the
            // one nearest this one's (the glide between them stretches the
            // render as little as possible: each period of glide is a pitch
            // change), and the angle the short way round.
            constexpr double kPi = 3.14159265358979323846;
            for (size_t k = 1; k < anchors.size(); ++k) {
              auto &next = anchors[k].alignment;
              const auto &previous = anchors[k - 1].alignment;
              const double period = periodAt(anchors[k].at);
              if (period > 0.0 && (fadeLeft || k + 1 < anchors.size())) {
                while (next.lag - previous.lag > period / 2.0)
                  next.lag -= period;
                while (next.lag - previous.lag < -period / 2.0)
                  next.lag += period;
              }
              while (next.angle - previous.angle > kPi)
                next.angle -= 2.0 * kPi;
              while (next.angle - previous.angle < -kPi)
                next.angle += 2.0 * kPi;
            }
            if (journalOn)
              for (size_t k = 0; k < anchors.size(); ++k)
                journalSplices.push_back(
                    {startSample + anchors[k].at,
                     k == 0 ? (fadeLeft ? "left" : "left (no fade)")
                            : k + 1 == anchors.size() ? (fadeRight ? "right" : "right (no fade)")
                                                      : "reset-edge",
                     anchors[k].alignment.lag, anchors[k].alignment.angle});
            size_t segment = 0;
            for (int sample = firstSample; sample < lastSample; ++sample) {
              while (segment + 2 < anchors.size() && sample >= anchors[segment + 1].at)
                ++segment;
              const auto &from = anchors[segment];
              const auto &to = anchors[segment + 1];
              const double t = std::clamp(
                  static_cast<double>(sample - from.at) / std::max(1, to.at - from.at),
                  0.0, 1.0);
              const splicealign::Alignment at{
                  from.alignment.lag + t * (to.alignment.lag - from.alignment.lag),
                  from.alignment.angle + t * (to.alignment.angle - from.alignment.angle)};
              committed[static_cast<size_t>(sample)] =
                  renderShare[static_cast<size_t>(sample)] *
                      splicealign::sample(synthGained, synthHilbert, sample, at) +
                  keptOriginal[static_cast<size_t>(sample)];
            }

            for (int sample = firstSample; sample < lastSample; ++sample) {
              float leftBlend = 1.0f;
              if (fadeLeft && sample < leftFadeEnd)
                leftBlend = smoothstep(
                    static_cast<float>(sample - leftFadeStart) /
                    static_cast<float>(2 * fadeHalf));

              float rightBlend = 1.0f;
              if (fadeRight && sample >= rightFadeStart)
                rightBlend = smoothstep(
                    static_cast<float>(rightFadeEnd - sample) /
                    static_cast<float>(2 * fadeHalf));

              const float blend = std::min(leftBlend, rightBlend);
              auto &mask = commitMask[static_cast<size_t>(sample)];
              mask = std::max(mask, blend);
            }
          }

          // Spec Q12, "vocoder + recorded highs": over those notes the band
          // above 10 kHz comes from the recording as the warp layer laid it
          // out (with the same gain), faded in and out over 10 ms at their
          // edges. Done last, on what is about to be written: the splice
          // alignment moves and rotates the render where it meets the
          // recording, and highs put back before it were carried along (the
          // notes next to an unedited one came out 2.5 dB down above 11 kHz
          // and unrelated to the recording there).
          if (!recordedHighsFrames.empty()) {
            const int total = static_cast<int>(committed.size());
            std::vector<float> share(committed.size(), 0.0f);
            std::vector<float> recorded(committed.size(), 0.0f);
            for (int i = 0; i < total; ++i)
              recorded[static_cast<size_t>(i)] =
                  originalSegment[static_cast<size_t>(i)] * sampleGain[static_cast<size_t>(i)];
            const int ramp = std::max(1, formantSampleRate / 100);
            for (const auto &[from, to] : recordedHighsFrames) {
              const int first = (from - capturedStartFrame) * hopSize;
              const int last = (to - capturedStartFrame) * hopSize;
              for (int i = std::max(0, first - ramp / 2); i < std::min(total, last + ramp / 2); ++i) {
                const float in = static_cast<float>(i - (first - ramp / 2)) / ramp;
                const float out = static_cast<float>((last + ramp / 2) - i) / ramp;
                share[static_cast<size_t>(i)] = std::max(
                    share[static_cast<size_t>(i)], std::clamp(std::min(in, out), 0.0f, 1.0f));
              }
            }
            if (!highband::restore(committed, recorded, share, formantSampleRate,
                                   capturedCancelFlag.get())) {
              if (currentJobId == jobId.load()) {
                isBusy = false;
                if (onComplete)
                  juce::MessageManager::callAsync([onComplete]() { onComplete(false); });
              }
              return;
            }
          }

          // The composite as it was before this pass wrote (debug recording).
          std::vector<float> journalBefore;
          if (journalOn && audioData.waveform.getNumChannels() > 0)
            journalBefore.assign(audioData.waveform.getReadPointer(0, startSample),
                                 audioData.waveform.getReadPointer(0, startSample) + samplesToWrite);

          for (int channel = 0; channel < audioData.waveform.getNumChannels();
               ++channel) {
            auto *destination =
                audioData.waveform.getWritePointer(channel, startSample);
            for (int sample = 0; sample < samplesToWrite; ++sample) {
              const float blend = commitMask[static_cast<size_t>(sample)];
              if (blend <= 0.0f)
                continue;
              const float current = destination[sample];
              const float rendered = committed[static_cast<size_t>(sample)];
              destination[sample] = current + blend * (rendered - current);
            }
          }

          // Exact restore next to unedited audio.
          //
          // Reverting a note (undo or Restore) copies the original into its
          // body, but the splices above still crossfade each edge between the
          // original, this pass's render and whatever the composite already
          // held - which includes the reverted edit's own earlier crossfade.
          // So a revert never gave back the original around the edges, and
          // every edit/revert cycle left slightly different audio there (up to
          // -21 dBFS in an A/B of two identical edit/undo exports).
          //
          // Where no edited audio is nearby, the composite around the note
          // should simply be the original, so copy it sample for sample over
          // the note and past each such edge far enough to cover any earlier
          // splice. A straight copy, not a blend: blending with weight 1.0 is
          // not bit-exact in float. Edges next to edited audio keep their
          // crossfade; there the body is still copied up to where it starts.
          //
          // Timing patches protect only their own samples (including fades).
          // A global formant shift still means no note is neutral.
          {
            // Covers the widest splice any version has written past a note
            // edge: commit fade (512 before, 384 now) + 128 search, and the
            // reset fade 256 + 128 search.
            constexpr int kExactRestoreReachSamples = 2 * 512;
            static_assert(kExactRestoreReachSamples >=
                              kCommitFadeHalfSamples + kSpliceSearchRadiusSamples,
                          "exact restore must cover the commit splice");
            static_assert(kExactRestoreReachSamples >=
                              kResetFadeHalfSamples +
                                  kResetSpliceSearchRadiusSamples,
                          "exact restore must cover the reset splice");

            const auto &original = warp::reference(audioData);
            const int originalSamples = original.getNumSamples();
            if (!hasGlobalFormant && !resetSpans.empty() && originalSamples > 0 &&
                original.getNumChannels() > 0) {
              std::vector<OriginalWaveformRestore::Range> timingProtectedRanges;
              // A breath with its level changed is not pristine either; its
              // gain curve is smoothed 128 samples past its ends.
              for (const auto &region : breathRegions)
                timingProtectedRanges.push_back({region.start - 128,
                                                 region.end + 128});
              const auto &allNotes = capturedProject->getNotes();

              // True if audio of an edited note (current edit, or rendered
              // audio still in the composite) can reach [absStart, absEnd).
              // Notes being reverted in this pass are excluded: their
              // renderedEdit flag is only cleared after this block.
              auto editedAudioNear = [&](int absStart, int absEnd) {
                for (const auto &note : allNotes) {
                  if (note.isRest())
                    continue;
                  const bool neutral = neutralForRender(note, capturedProject->getAudioData());
                  if (note.isDirty() && neutral)
                    continue;
                  if (neutral && !note.hasRenderedEdit())
                    continue;
                  // Where the note plays and where its render was written:
                  // timeline frames. Its frames in the recording are not a
                  // place in the composite (WarpLayer.h); counted here, a
                  // note the warp had moved reached notes it does not
                  // touch, the copy below then stopped 1024 samples short
                  // of a restored note's end, and what the pass had left
                  // there met it with a step (煙幕 at 229.9 s stretched
                  // 0.80x: 0.059 between two samples, heard as a click).
                  const int first = std::min(note.getStartFrame(),
                                             note.getRenderedStartFrame());
                  const int last = std::max(note.getEndFrame(),
                                            note.getRenderedEndFrame());
                  const int footprintStart =
                      first * hopSize - kExactRestoreReachSamples;
                  const int footprintEnd =
                      last * hopSize + kExactRestoreReachSamples;
                  if (footprintStart < absEnd && footprintEnd > absStart)
                    return true;
                }
                return false;
              };

              const int writeLimit =
                  std::min(samplesToWrite, originalSamples - startSample);
              for (const auto &span : resetSpans) {
                const int absStart = startSample + span.startSample;
                const int absEnd = startSample + span.endSample;
                const bool leftExact = !editedAudioNear(
                    absStart - kExactRestoreReachSamples, absStart);
                const bool rightExact = !editedAudioNear(
                    absEnd, absEnd + kExactRestoreReachSamples);

                const int copyStart = std::max(
                    0, leftExact ? span.startSample - kExactRestoreReachSamples
                                 : span.startSample + kExactRestoreReachSamples);
                const int copyEnd = std::min(
                    writeLimit, rightExact
                                    ? span.endSample + kExactRestoreReachSamples
                                    : span.endSample - kExactRestoreReachSamples);
                if (copyEnd <= copyStart)
                  continue;

                for (int channel = 0;
                     channel < audioData.waveform.getNumChannels(); ++channel) {
                  const int sourceChannel =
                      std::min(channel, original.getNumChannels() - 1);
                  OriginalWaveformRestore::copy(
                      audioData.waveform.getWritePointer(channel),
                      original.getReadPointer(sourceChannel),
                      startSample + copyStart, startSample + copyEnd,
                      timingProtectedRanges);
                }
              }
            }
          }

          // Debug recording: what this pass changed in the composite, with a
          // little around it, before and after.
          if (journalOn && !journalBefore.empty()) {
            const float *after = audioData.waveform.getReadPointer(0, startSample);
            int first = -1, last = -1;
            for (int i = 0; i < samplesToWrite; ++i)
              if (after[i] != journalBefore[static_cast<size_t>(i)]) {
                if (first < 0)
                  first = i;
                last = i;
              }
            debugjournal::Entry entry;
            entry.name = "render pass";
            entry.detail = journalDetail;
            entry.splices = std::move(journalSplices);
            if (first >= 0) {
              constexpr int kAround = 2048;
              first = std::max(0, first - kAround);
              last = std::min(samplesToWrite - 1, last + kAround);
              entry.audioStart = startSample + first;
              entry.audioBefore.assign(journalBefore.begin() + first, journalBefore.begin() + last + 1);
              entry.audioAfter.assign(after + first, after + last + 1);
            } else {
              entry.detail += "; wrote nothing";
            }
            capturedProject->getDebugJournal().recordRender(std::move(entry));
          }

          // Persist only semantic render state. Previously this loop stored a
          // copy of the rendered samples on every note; the composite waveform
          // is now authoritative, while this flag keeps adjacency selection and
          // project edit detection independent of audio caches.
          const bool hasF0DirtyRange = f0DirtyStart >= 0 && f0DirtyEnd >= 0;
          auto overlapsF0DirtyRange = [&](const Note &note) {
            return hasF0DirtyRange && note.getStartFrame() < f0DirtyEnd &&
                   note.getEndFrame() > f0DirtyStart;
          };
          auto isCommitAnchor = [&](const Note &note) {
            if (note.isRest())
              return false;
            if (note.isDirty())
              return true;

            // F0-range edits may not mark a specific note dirty. In that case,
            // notes overlapping the F0 dirty range are the edited anchors.
            return !hasDirtyNoteAnchors && overlapsF0DirtyRange(note);
          };

          auto &notes = capturedProject->getNotes();
          for (auto &note : notes) {
            if (note.isRest())
              continue;

            const int noteStart = note.getStartFrame();
            const int noteEnd = note.getEndFrame();
            const int overlapStart = std::max(capturedStartFrame, noteStart);
            const int overlapEnd = std::min(capturedEndFrame, noteEnd);
            if (overlapEnd <= overlapStart)
              continue;

            const bool isF0Anchor =
                !hasDirtyNoteAnchors && overlapsF0DirtyRange(note);
            if (!isCommitAnchor(note))
              continue;

            const bool isNeutralReset =
                !hasGlobalFormant && note.isDirty() && neutralForRender(note, capturedProject->getAudioData()) &&
                !isF0Anchor;
            note.setRenderedEdit(!isNeutralReset);
            note.setSynthDirty(false);
          }

          LOG("synthesizeRegion: committed after " + elapsedText());
          isBusy = false;
          juce::MessageManager::callAsync(
              [capturedProject, onComplete]() {
                capturedProject->clearAllDirty();
                if (onComplete) onComplete(true);
              });
        }).detach();
      };

  // Under vocal fry kept as a word starts (FryDetector.h) the vocoder is
  // given the pitch of the voice after it, not the fry's own low one, up to
  // where the curve it sings has arrived at that voice. What it renders
  // there is not heard - the recording plays - but it reached the handover
  // still singing the fry: a low period or two past the kept frames, and the
  // tail of its own, louder fry under the crossfade. At 70 to 300 Hz that
  // was 5 to 14 dB over the recording around the handover (煙幕 at 230.4 s
  // stretched 1.10x), as strong as one pulse of the fry, and it was heard
  // as one grain more. Given the voice's pitch it meets the handover already
  // singing the voice. Unvoiced frames stay unvoiced.
  if (!usePsola && !onsetHeld.empty()) {
    for (const auto &[from, to] : onsetHeld) {
      const int arrived = fry::voiceArrivesAt(audioData.rawF0, audioData.denseF0, to);
      int firstAfter = -1;
      for (int frame = std::max(0, startFrame); frame < endFrame; ++frame) {
        const int source = warp::sourceFrameAt(audioData, frame);
        if (source < arrived)
          continue;
        if (source < arrived + 3 &&
            adjustedF0Range[static_cast<size_t>(frame - startFrame)] > 0.0f)
          firstAfter = frame;
        if (firstAfter >= 0 || source >= arrived + 3)
          break;
      }
      if (firstAfter < 0)
        continue;
      const float pitch = adjustedF0Range[static_cast<size_t>(firstAfter - startFrame)];
      for (int frame = std::max(0, startFrame); frame < firstAfter; ++frame) {
        const int source = warp::sourceFrameAt(audioData, frame);
        auto &value = adjustedF0Range[static_cast<size_t>(frame - startFrame)];
        if (source >= from && source < arrived && value > 0.0f)
          value = pitch;
      }
    }
  }

  if (usePsola) {
    psola.renderAsync(std::move(dspRequest), std::move(onRendered),
                      capturedCancelFlag);
  } else {
    // Pass the cancel flag so the worker skips stale queued jobs instead of
    // running full inference on renders a newer edit already superseded.
    vocoder->inferAsync(std::move(melRange), std::move(adjustedF0Range),
                        std::move(onRendered), capturedCancelFlag);
  }
}
