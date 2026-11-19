#pragma once

#include "../../JuceHeader.h"
#include <memory>
#include <optional>
#include <utility>
#include <vector>

struct AudioData;

namespace MidiReference
{
struct Note
{
    double startSeconds;
    double endSeconds;
    int midiPitch;
};

struct Track
{
    int number;
    juce::String name;
    std::vector<Note> notes;
};

struct Guide
{
    juce::String filename;
    std::vector<Track> tracks;
};

struct State
{
    std::shared_ptr<const Guide> guide;
    size_t selectedTrack = 0;
    bool visible = false;
};

// Track numbers are one-based file indices. Tracks without notes are omitted.
std::vector<Track> readTracks(juce::InputStream& input);

// All times are relative to the guide's start. Different pitches tied for
// the greatest overlap have no unambiguous target.
std::optional<int> findPitchWithGreatestOverlap(const std::vector<Note>& notes,
                                                double startSeconds, double endSeconds);

// The guide's span on the audio's time axis: it starts at the timeline offset,
// or at the first ARA playback region, which may begin inside the audio.
std::pair<double, double> getGuideSpan(const AudioData& audio);

// The guide pitch a double-click snaps a note spanning these audio times to:
// the note is clipped to the span and compared in guide time.
std::optional<int> findSnapPitch(const std::vector<Note>& notes, std::pair<double, double> span,
                                 double noteStartSeconds, double noteEndSeconds);
}
