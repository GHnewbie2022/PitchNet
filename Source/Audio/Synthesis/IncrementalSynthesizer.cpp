#include "IncrementalSynthesizer.h"
#include "BreathSibilanceProtection.h"
#include "FormantShifter.h"
#include "../BreathSibilanceAnalysis.h"
#include "../NoteTiming.h"
#include "../ProjectGain.h"
#include "../WarpLayer.h"
#include "OriginalWaveformRestore.h"
#include "../../Utils/NoteGainCurve.h"
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
                              !note.isNeutralForOriginalWaveform());
  };
  auto isEditedClusterCandidate = [&](const Note &note) {
    if (note.isRest())
      return false;

    // A dirty neutral note is being reset. It must split the edited cluster
    // instead of pulling its neighbours into the new pass.
    if (std::abs(project.getFormantShift()) <= 0.001f &&
        note.isDirty() && note.isNeutralForOriginalWaveform() &&
        !isF0Anchor(note))
      return false;

    return std::abs(project.getFormantShift()) > 0.001f ||
           isF0Anchor(note) || note.hasRenderedEdit() ||
           !note.isNeutralForOriginalWaveform();
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
    // Transition at frame boundary
    int center = (i + 1) * hopSize;
    int rampStart = std::max(0, center - kRampSamples / 2);
    int rampEnd = std::min(numSamples, center + kRampSamples / 2);
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
          (note.isNeutralForOriginalWaveform() ? "yes" : "no"));
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
  // Paired with the synthesizeRegion lines above: without an outcome line a
  // log cannot tell a render that committed from one that was superseded.
  const double renderStartMs = juce::Time::getMillisecondCounterHiRes();
  auto elapsedText = [renderStartMs]() {
    return juce::String((juce::Time::getMillisecondCounterHiRes() -
                         renderStartMs) / 1000.0, 1) + "s";
  };


  // Everything past the render is engine-agnostic: both paths hand back the
  // same span of samples and share this callback verbatim.
  auto onRendered =
      [this, capturedCancelFlag, capturedProject, capturedStartFrame,
       capturedEndFrame, hopSize, currentJobId, onComplete, elapsedText,
       formantShifts, formantF0, formantSampleRate, hasGlobalFormant, usePsola,
       hasDirtyNoteAnchors, f0DirtyStart, f0DirtyEnd,
       capturedNoteCount, commitFrameRanges = std::move(commitFrameRanges),
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
                     f0DirtyStart, f0DirtyEnd, capturedNoteCount,
                     commitFrameRanges, blendMask, originalSegment,
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

          // Build blended target from model/original.
          std::vector<float> targetSegment(samplesToWrite, 0.0f);
          for (int i = 0; i < samplesToWrite; ++i) {
            const float b =
                (i < static_cast<int>(blendMask.size())) ? blendMask[i] : 0.0f;
            const float synth = synthesizedAudio[static_cast<size_t>(i)];
            const float orig = originalSegment[static_cast<size_t>(i)];
            targetSegment[static_cast<size_t>(i)] =
                b * synth + (1.0f - b) * orig;
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
          };
          std::vector<ResetSpan> resetSpans;
          if (!hasGlobalFormant) {
            // Notes are not guaranteed to be in frame order, so collect and
            // sort before looking for contiguity.
            std::vector<std::pair<int, int>> neutralFrameRanges;
            for (const auto &note : capturedProject->getNotes()) {
              if (note.isRest() || !note.isDirty() ||
                  !note.isNeutralForOriginalWaveform())
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

          auto resetSmoothstep = [](float t) {
            t = std::clamp(t, 0.0f, 1.0f);
            return t * t * (3.0f - 2.0f * t);
          };

          for (size_t spanIndex = 0; spanIndex < resetSpans.size();
               ++spanIndex) {
            const auto &span = resetSpans[spanIndex];
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
                spanIndex > 0 ? resetSpans[spanIndex - 1].endSample : 0;
            const int nextStart = spanIndex + 1 < resetSpans.size()
                                      ? resetSpans[spanIndex + 1].startSample
                                      : samplesToWrite;
            const int firstSample = std::max(
                {0, previousEnd,
                 fadeLeft ? leftCenter - fadeHalf : span.startSample});
            const int lastSample = std::min(
                {samplesToWrite, nextStart,
                 fadeRight ? rightCenter + fadeHalf : span.endSample});

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
              const float restored =
                  breathGain.empty() ? originalSegment[index]
                                     : originalSegment[index] * breathGain[index];
              targetSegment[index] += blend * (restored - targetSegment[index]);
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

          for (int channel = 0; channel < audioData.waveform.getNumChannels();
               ++channel) {
            auto *destination =
                audioData.waveform.getWritePointer(channel, startSample);
            for (int sample = 0; sample < samplesToWrite; ++sample) {
              const float blend = commitMask[static_cast<size_t>(sample)];
              if (blend <= 0.0f)
                continue;
              const float current = destination[sample];
              const float rendered = targetSegment[static_cast<size_t>(sample)];
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
                  const bool neutral = note.isNeutralForOriginalWaveform();
                  if (note.isDirty() && neutral)
                    continue;
                  if (neutral && !note.hasRenderedEdit())
                    continue;
                  const int first = std::min(
                      {note.getStartFrame(), note.getSrcStartFrame(),
                       note.getRenderedStartFrame()});
                  const int last = std::max(
                      {note.getEndFrame(), note.getSrcEndFrame(),
                       note.getRenderedEndFrame()});
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
                !hasGlobalFormant && note.isDirty() && note.isNeutralForOriginalWaveform() &&
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
