#pragma once

#include "../JuceHeader.h"
#include "Project.h"

/**
 * Handles Project serialization to/from JSON format.
 *
 * Design principles:
 * - Decoupled from Project class (Project doesn't know about serialization details)
 * - Uses JUCE's built-in JSON support (no external dependencies)
 * - Stateless utility class
 */
class ProjectSerializer {
public:
    // 6: the warp layer (WarpMap.h). Older formats kept per-note timing that
    // the warp map cannot reproduce, so they are refused (spec Q3) and
    // lastLoadWasLegacy() says why.
    // 7: a project file may link its recording instead of embedding the
    // audio (spec Q11, RecordingFingerprint.h): "recording" and
    // "editPieces" in place of "waveform", "originalWaveform" and
    // "melSpectrogram". Version 6 files still open as before.
    // 8: a stretched project's edits are blended onto the recording
    // stretched once, at its own rate ("referenceFromRecording" in "warp",
    // AudioData::warpReferenceFromRecording). A build before it stretches
    // the analysis copy on its own and would lay the saved edits over a
    // reference they do not meet, so it refuses the file. Versions 6 and 7
    // still open: stretched, they keep their own reference until the warp
    // is next laid out.
    static constexpr int FORMAT_VERSION = 8;
    static constexpr int FIRST_WARP_FORMAT_VERSION = 6;

    /** True when the last fromJson()/fromBinaryArchive() call refused a
     *  project saved by a PitchNet without the warp layer. */
    static bool lastLoadWasLegacy();

    enum class BinaryArchiveMode {
        // Standalone documents and conventional plug-in state must remain
        // usable without an external audio source.
        selfContained,

        // ARA region state can recover immutable source samples from the host,
        // cheaply rebuild source-derived features, and restore edited audio
        // from the separately persisted processed-region render. Keep edit
        // state, but omit all project-level waveform and mel buffers.
        hostBackedARA
    };

    /**
     * Save project to JSON file.
     */
    static bool saveToFile(const Project& project, const juce::File& file);

    /**
     * Load project from JSON file.
     */
    static bool loadFromFile(Project& project, const juce::File& file);
    // Same, reporting progress (0..1) between the read, parse and build stages.
    // Safe to call off the message thread; the callback runs on that thread.
    static bool loadFromFile(Project& project, const juce::File& file,
                             const std::function<void(double)>& onProgress);

    /**
     * Convert project to JSON object.
     */
    static juce::var toJson(const Project& project,
                            bool includeAnalysisCache = false,
                            bool includePitchData = true);

    /**
     * Load project from JSON object.
     */
    static bool fromJson(Project& project, const juce::var& json);

    /**
     * Serialize project metadata as compact JSON plus heavy analysis/render
     * caches as raw binary buffers. Intended for DAW/plugin state archives.
     */
    static bool toBinaryArchive(const Project& project,
                                juce::MemoryBlock& destData,
                                BinaryArchiveMode mode =
                                    BinaryArchiveMode::selfContained);

    /**
     * Same archive, written straight into a caller-owned stream. Lets the
     * plug-in state path emit the archive once instead of building a whole
     * second copy of it in an intermediate MemoryBlock.
     */
    static bool toBinaryArchive(const Project& project,
                                juce::OutputStream& out,
                                BinaryArchiveMode mode =
                                    BinaryArchiveMode::selfContained);

    /**
     * Load a project from toBinaryArchive(). Falls back to false for unknown
     * input so callers can try legacy JSON.
     */
    static bool fromBinaryArchive(Project& project, const void* data,
                                  size_t sizeInBytes);

private:
    // Note serialization
    static juce::var noteToJson(const Note& note, bool includeAnalysisCache,
                                bool includePitchData);
    // legacyPitchCenter is the project-wide strength, used for notes saved
    // before each note stored its own (0 when no correction was applied).
    static bool noteFromJson(Note& note, const juce::var& json,
                             float legacyPitchCenter);

    // Pitch data serialization
    static juce::var pitchDataToJson(const AudioData& audioData);
    static bool pitchDataFromJson(AudioData& audioData, const juce::var& json);
    static juce::var audioBufferToJson(const juce::AudioBuffer<float>& buffer);
    static bool audioBufferFromJson(juce::AudioBuffer<float>& buffer, const juce::var& json);
    static juce::var melSpectrogramToJson(const std::vector<std::vector<float>>& mel);
    static bool melSpectrogramFromJson(std::vector<std::vector<float>>& mel, const juce::var& json);

    // Array helpers (compact string format)
    static juce::String floatArrayToString(const std::vector<float>& arr, int precision = 4);
    static std::vector<float> stringToFloatArray(const juce::String& str);
    static juce::String boolArrayToString(const std::vector<bool>& arr);
    static std::vector<bool> stringToBoolArray(const juce::String& str);

    ProjectSerializer() = delete;
};
