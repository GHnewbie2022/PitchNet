#include "Note.h"

#include <algorithm>
#include "../Utils/Constants.h"
#include <atomic>
#include <cmath>

std::uint64_t Note::nextUid()
{
    static std::atomic<std::uint64_t> counter{0};
    return ++counter;
}

Note::Note(int startFrame, int endFrame, float midiNote)
    : srcStartFrame(startFrame), srcEndFrame(endFrame),
      startFrame(startFrame), endFrame(endFrame), midiNote(midiNote),
      lastNonMacroMidiNote(midiNote), originalMidiNote(midiNote),
      renderedStartFrame(startFrame), renderedEndFrame(endFrame)
{
}

std::vector<float> Note::getAdjustedF0() const
{
    if (f0Values.empty() || pitchOffset == 0.0f)
        return f0Values;

    // Convert semitone offset to frequency ratio
    float ratio = std::pow(2.0f, pitchOffset / 12.0f);

    std::vector<float> adjusted;
    adjusted.reserve(f0Values.size());

    for (float f0 : f0Values)
    {
        if (f0 > 0.0f)
            adjusted.push_back(f0 * ratio);
        else
            adjusted.push_back(0.0f);
    }

    return adjusted;
}

std::vector<float> Note::computeF0FromDelta() const
{
    int numFrames = endFrame - startFrame;
    std::vector<float> result(numFrames, 0.0f);
    const auto& activeDelta = getActiveDeltaPitch();

    for (int i = 0; i < numFrames; ++i)
    {
        // Get delta pitch for this frame (or 0 if not available)
        float delta = (i < static_cast<int>(activeDelta.size())) ? activeDelta[i] : 0.0f;

        // Actual MIDI = base + offset + delta
        float actualMidi = midiNote + pitchOffset + delta;

        // Convert to Hz
        result[i] = midiToFreq(actualMidi);
    }

    return result;
}

bool Note::containsFrame(int frame) const
{
    return frame >= startFrame && frame < endFrame;
}

bool Note::isPitchUntouched() const
{
    constexpr float eps = 0.001f;
    return std::abs(midiNote - originalMidiNote) <= eps &&
           std::abs(pitchOffset) <= eps && !hasBakedDeltaPitch() &&
           std::abs(tiltLeft) <= eps && std::abs(tiltRight) <= eps &&
           frozenTilt.empty() && std::abs(vibrato - 1.0f) <= eps &&
           std::abs(pitchDrift - 1.0f) <= eps && smoothLeftFrames == 0 &&
           smoothRightFrames == 0 && std::abs(deltaScale - 1.0f) <= eps &&
           std::abs(deltaOffset) <= eps;
}

bool Note::isNeutralForOriginalWaveform() const
{
    constexpr float kPitchEpsilon = 0.001f;
    constexpr float kGainEpsilon = 0.001f;

    // Timing is not part of it: the warp layer moves and stretches the
    // recording itself (WarpLayer.h), so a retimed note with its recorded
    // pitch still plays the (laid out) recording.
    return std::abs(midiNote - originalMidiNote) <= kPitchEpsilon &&
           !hasBakedDeltaPitch() &&
           std::abs(pitchOffset) <= kPitchEpsilon &&
           std::abs(formantShift) <= kPitchEpsilon &&
           std::abs(volumeDb) <= kGainEpsilon &&
           std::abs(sibilantBalance) <= kGainEpsilon &&
           std::abs(tiltLeft) <= kPitchEpsilon &&
           std::abs(tiltRight) <= kPitchEpsilon &&
           std::none_of(frozenTilt.begin(), frozenTilt.end(),
                        [](float v) { return std::abs(v) > kPitchEpsilon; }) &&
           std::abs(vibrato - 1.0f) <= kPitchEpsilon &&
           std::abs(pitchDrift - 1.0f) <= kPitchEpsilon &&
           smoothLeftFrames == 0 &&
           smoothRightFrames == 0 &&
           std::abs(deltaScale - 1.0f) <= kPitchEpsilon &&
           std::abs(deltaOffset) <= kPitchEpsilon;
}

void Note::setAnalysisSpans(std::vector<SourceSpan> spans)
{
    std::sort(spans.begin(), spans.end());
    spans.erase(std::unique(spans.begin(), spans.end()), spans.end());
    spans.erase(std::remove_if(spans.begin(), spans.end(),
                               [](const SourceSpan& s) { return s.second <= s.first; }),
                spans.end());
    analysisSpans = std::move(spans);
}

void Note::trimAnalysisSpans()
{
    analysisSpans.erase(
        std::remove_if(analysisSpans.begin(), analysisSpans.end(),
                       [this](const SourceSpan& s)
                       { return s.second <= srcStartFrame || s.first >= srcEndFrame; }),
        analysisSpans.end());
}

void Note::shiftAnalysisSpans(int frames)
{
    for (auto& span : analysisSpans)
    {
        span.first += frames;
        span.second += frames;
    }
}

const Note::SourceSpan* Note::exactAnalysisSpan() const
{
    if (analysisSpans.size() != 1)
        return nullptr;
    const auto& span = analysisSpans.front();
    if (span.first != srcStartFrame || span.second != srcEndFrame ||
        !(span.midi > 0.0f) || span.originalDelta.empty())
        return nullptr;
    return &span;
}

bool Note::restoreAnalysisPitch()
{
    const auto* span = exactAnalysisSpan();
    if (!span)
        return false;
    const float midi = span->midi;
    const auto curve = span->originalDelta;
    setMidiNote(midi);
    setOriginalMidiNote(midi);
    setPitchOffset(0.0f);
    setTiltLeft(0.0f);
    setTiltRight(0.0f);
    setVibrato(1.0f);
    setPitchDrift(1.0f);
    setSmoothLeftFrames(0);
    setSmoothRightFrames(0);
    setDeltaScale(1.0f);
    setDeltaOffset(0.0f);
    setFrozenTilt({});
    setBakedDeltaPitch({});
    setOriginalDeltaPitch(curve);
    setDeltaPitch(curve);
    markDirty();
    markSynthDirty();
    return true;
}
