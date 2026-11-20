#pragma once

#include "../JuceHeader.h"
#include "Note.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

/**
 * Debug recording: a journal kept inside the project of what was done to it
 * and what each render wrote, so a click or catch heard later can be traced
 * back to the step and the render pass that made it.
 *
 * Off by default. Turned on, it adds a start entry with the notes as they
 * are (never clearing what was recorded before: turning it on again by
 * mistake loses nothing), then one entry for every history step (apply / undo / redo, with the notes it
 * changed, before and after), every render pass (what was asked to render,
 * the neighbours rendered with it, where it wrote and how its splices were
 * lined up, and the composite before and after it wrote), and every mark the
 * user sets with the M key at the playhead while listening. Saved with the
 * project (ProjectSerializer, "debugJournal"); turned off it stops recording
 * and keeps what it has. Every entry carries the date and time it happened,
 * so batches of work line up with when they were reported.
 *
 * Render passes run on a worker thread, so entries go in under a lock.
 */
namespace debugjournal
{
struct NoteState
{
    int start = 0, end = 0;
    float midi = 0.0f, offset = 0.0f, tiltLeft = 0.0f, tiltRight = 0.0f,
          tiltPivot = 0.5f, vibrato = 1.0f, drift = 1.0f, formant = 0.0f,
          volume = 0.0f, sibilant = 0.0f;
    bool baked = false, rendered = false;

    static NoteState of(const Note& note)
    {
        NoteState s;
        s.start = note.getStartFrame();
        s.end = note.getEndFrame();
        s.midi = note.getMidiNote();
        s.offset = note.getPitchOffset();
        s.tiltLeft = note.getTiltLeft();
        s.tiltRight = note.getTiltRight();
        s.tiltPivot = note.getTiltPivot();
        s.vibrato = note.getVibrato();
        s.drift = note.getPitchDrift();
        s.formant = note.getFormantShift();
        s.volume = note.getVolumeDb();
        s.sibilant = note.getSibilantBalance();
        s.baked = note.hasBakedDeltaPitch();
        s.rendered = note.hasRenderedEdit();
        return s;
    }

    // The edit, not the render flag (that changes when a render lands).
    bool sameEdit(const NoteState& o) const
    {
        return start == o.start && end == o.end && midi == o.midi && offset == o.offset &&
               tiltLeft == o.tiltLeft && tiltRight == o.tiltRight && tiltPivot == o.tiltPivot &&
               vibrato == o.vibrato && drift == o.drift && formant == o.formant &&
               volume == o.volume && sibilant == o.sibilant && baked == o.baked;
    }

    juce::var toJson() const
    {
        juce::Array<juce::var> a{start, end, midi, offset, tiltLeft, tiltRight, tiltPivot,
                                 vibrato, drift, formant, volume, sibilant, baked ? 1 : 0,
                                 rendered ? 1 : 0};
        return a;
    }

    static NoteState fromJson(const juce::var& v)
    {
        NoteState s;
        if (auto* a = v.getArray(); a != nullptr && a->size() >= 14)
        {
            s.start = (*a)[0];
            s.end = (*a)[1];
            s.midi = (*a)[2];
            s.offset = (*a)[3];
            s.tiltLeft = (*a)[4];
            s.tiltRight = (*a)[5];
            s.tiltPivot = (*a)[6];
            s.vibrato = (*a)[7];
            s.drift = (*a)[8];
            s.formant = (*a)[9];
            s.volume = (*a)[10];
            s.sibilant = (*a)[11];
            s.baked = static_cast<int>((*a)[12]) != 0;
            s.rendered = static_cast<int>((*a)[13]) != 0;
        }
        return s;
    }
};

// One splice of a render pass: where (composite sample), what kind, and how
// the render was lined up there.
struct Splice
{
    int at = 0;
    juce::String kind; // "left", "right", "reset-edge"
    double lag = 0.0, angle = 0.0;
};

struct Entry
{
    juce::int64 time = 0;    // when it happened (ms since 1970, local clock)
    juce::String kind;       // start, apply, undo, redo, render, mark
    juce::String name;       // the action's name, or the render's summary
    double playhead = 0.0;   // seconds on the timeline
    std::vector<NoteState> before, after; // changed notes (start: all, in after)
    juce::String detail;     // render: what was asked, rendered, written
    std::vector<Splice> splices;
    int audioStart = 0;      // composite sample of audioBefore[0] / audioAfter[0]
    std::vector<float> audioBefore, audioAfter;
};

class Journal
{
public:
    // Composite audio kept across all render entries, in samples (about 2.5
    // minutes at 44.1 kHz per side): beyond it the oldest passes keep their
    // text and drop their audio.
    static constexpr size_t kMaxAudioSamples = 12'000'000;

    bool isEnabled() const { return enabled.load(); }
    bool isEmpty() const
    {
        const std::lock_guard<std::mutex> lock(mutex);
        return entries.empty();
    }

    // Recording on: a start entry with every note as it is now, which the
    // next steps are diffed against. What was recorded before stays.
    void start(const std::vector<Note>& notes, double playhead)
    {
        addStart(notes, playhead, "recording started");
        enabled.store(true);
    }

    // A project opened with recording on: its notes as loaded, so the steps
    // after this are told apart from those of the last session.
    void resume(const std::vector<Note>& notes, double playhead)
    {
        if (enabled.load())
            addStart(notes, playhead, "project opened, recording on");
    }

    void stop() { enabled.store(false); }

    // After a history step: the notes it changed (by start and end frame,
    // so splits and merges show as removed and added notes).
    void recordHistory(const std::vector<Note>& notes, const juce::String& kind,
                       const juce::String& name, double playhead)
    {
        if (!enabled.load())
            return;
        const auto current = snapshot(notes);
        const std::lock_guard<std::mutex> lock(mutex);
        Entry e;
        e.time = now();
        e.kind = kind;
        e.name = name;
        e.playhead = playhead;
        for (const auto& old : lastNotes)
        {
            const auto match = std::find_if(current.begin(), current.end(), [&](const NoteState& n)
                                            { return n.start == old.start && n.end == old.end; });
            if (match == current.end() || !match->sameEdit(old))
                e.before.push_back(old);
        }
        for (const auto& n : current)
        {
            const auto match = std::find_if(lastNotes.begin(), lastNotes.end(), [&](const NoteState& o)
                                            { return o.start == n.start && o.end == n.end; });
            if (match == lastNotes.end() || !match->sameEdit(n))
                e.after.push_back(n);
        }
        lastNotes = current;
        entries.push_back(std::move(e));
    }

    // Anything else worth a line (a new render swapped in while playing).
    void recordEvent(const juce::String& kind, const juce::String& name, double playhead,
                     const juce::String& detail = {})
    {
        if (!enabled.load())
            return;
        const std::lock_guard<std::mutex> lock(mutex);
        Entry e;
        e.time = now();
        e.kind = kind;
        e.name = name;
        e.playhead = playhead;
        e.detail = detail;
        entries.push_back(std::move(e));
    }

    void recordMark(double playhead)
    {
        if (!enabled.load())
            return;
        const std::lock_guard<std::mutex> lock(mutex);
        Entry e;
        e.time = now();
        e.kind = "mark";
        e.name = "heard here";
        e.playhead = playhead;
        entries.push_back(std::move(e));
    }

    void recordRender(Entry e)
    {
        if (!enabled.load())
            return;
        const std::lock_guard<std::mutex> lock(mutex);
        e.time = now();
        e.kind = "render";
        audioSamples += e.audioBefore.size() + e.audioAfter.size();
        // Oldest passes give up their audio first.
        for (auto& old : entries)
        {
            if (audioSamples <= kMaxAudioSamples)
                break;
            audioSamples -= old.audioBefore.size() + old.audioAfter.size();
            old.audioBefore.clear();
            old.audioAfter.clear();
        }
        entries.push_back(std::move(e));
    }

    std::vector<Entry> copyEntries() const
    {
        const std::lock_guard<std::mutex> lock(mutex);
        return entries;
    }

    juce::var toJson() const
    {
        const std::lock_guard<std::mutex> lock(mutex);
        auto* obj = new juce::DynamicObject();
        obj->setProperty("enabled", enabled.load());
        juce::Array<juce::var> list;
        for (const auto& e : entries)
        {
            auto* item = new juce::DynamicObject();
            item->setProperty("time", e.time);
            item->setProperty("kind", e.kind);
            item->setProperty("name", e.name);
            item->setProperty("playhead", e.playhead);
            item->setProperty("before", notesJson(e.before));
            item->setProperty("after", notesJson(e.after));
            if (e.detail.isNotEmpty())
                item->setProperty("detail", e.detail);
            if (!e.splices.empty())
            {
                juce::Array<juce::var> splices;
                for (const auto& s : e.splices)
                    splices.add(juce::Array<juce::var>{s.at, s.kind, s.lag, s.angle});
                item->setProperty("splices", splices);
            }
            if (!e.audioBefore.empty() || !e.audioAfter.empty())
            {
                item->setProperty("audioStart", e.audioStart);
                item->setProperty("audioBefore", encode(e.audioBefore));
                item->setProperty("audioAfter", encode(e.audioAfter));
            }
            list.add(juce::var(item));
        }
        obj->setProperty("entries", list);
        return juce::var(obj);
    }

    void fromJson(const juce::var& json)
    {
        const std::lock_guard<std::mutex> lock(mutex);
        entries.clear();
        audioSamples = 0;
        if (auto* list = json.getProperty("entries", juce::var()).getArray())
            for (const auto& item : *list)
            {
                Entry e;
                e.time = static_cast<juce::int64>(item.getProperty("time", 0));
                e.kind = item.getProperty("kind", "").toString();
                e.name = item.getProperty("name", "").toString();
                e.playhead = item.getProperty("playhead", 0.0);
                e.before = notesFromJson(item.getProperty("before", juce::var()));
                e.after = notesFromJson(item.getProperty("after", juce::var()));
                e.detail = item.getProperty("detail", "").toString();
                if (auto* splices = item.getProperty("splices", juce::var()).getArray())
                    for (const auto& s : *splices)
                        if (auto* a = s.getArray(); a != nullptr && a->size() >= 4)
                            e.splices.push_back({(*a)[0], (*a)[1].toString(), (*a)[2], (*a)[3]});
                e.audioStart = item.getProperty("audioStart", 0);
                e.audioBefore = decode(item.getProperty("audioBefore", "").toString());
                e.audioAfter = decode(item.getProperty("audioAfter", "").toString());
                audioSamples += e.audioBefore.size() + e.audioAfter.size();
                entries.push_back(std::move(e));
            }
        // Recording goes on from the notes as they were last recorded.
        lastNotes.clear();
        for (const auto& e : entries)
            if (e.kind == "start")
                lastNotes = e.after;
        enabled.store(static_cast<bool>(json.getProperty("enabled", false)));
    }

    // The notes as the journal last saw them, after a load (the project's
    // own notes are the truth; this only seeds the next diff).
    void resync(const std::vector<Note>& notes)
    {
        const std::lock_guard<std::mutex> lock(mutex);
        lastNotes = snapshot(notes);
    }

private:
    static std::vector<NoteState> snapshot(const std::vector<Note>& notes)
    {
        std::vector<NoteState> out;
        for (const auto& n : notes)
            if (!n.isRest())
                out.push_back(NoteState::of(n));
        return out;
    }

    static juce::var notesJson(const std::vector<NoteState>& notes)
    {
        juce::Array<juce::var> a;
        for (const auto& n : notes)
            a.add(n.toJson());
        return a;
    }

    static std::vector<NoteState> notesFromJson(const juce::var& v)
    {
        std::vector<NoteState> out;
        if (auto* a = v.getArray())
            for (const auto& n : *a)
                out.push_back(NoteState::fromJson(n));
        return out;
    }

    // 16-bit samples scaled to the segment's peak, base64: about 2.7 bytes a
    // sample, 96 dB below the peak - plenty to see a dip or a click.
    static juce::String encode(const std::vector<float>& samples)
    {
        if (samples.empty())
            return {};
        float peak = 0.0f;
        for (float s : samples)
            peak = std::max(peak, std::abs(s));
        const float scale = peak > 0.0f ? peak : 1.0f;
        juce::MemoryBlock block;
        block.append(&scale, sizeof(float));
        for (float s : samples)
        {
            const auto q = static_cast<int16_t>(std::lround(std::clamp(s / scale, -1.0f, 1.0f) * 32767.0f));
            block.append(&q, sizeof(int16_t));
        }
        return block.toBase64Encoding();
    }

    static std::vector<float> decode(const juce::String& text)
    {
        if (text.isEmpty())
            return {};
        juce::MemoryBlock block;
        if (!block.fromBase64Encoding(text) || block.getSize() < sizeof(float))
            return {};
        float scale = 1.0f;
        std::memcpy(&scale, block.getData(), sizeof(float));
        const size_t count = (block.getSize() - sizeof(float)) / sizeof(int16_t);
        std::vector<float> out(count);
        const auto* data = static_cast<const char*>(block.getData()) + sizeof(float);
        for (size_t i = 0; i < count; ++i)
        {
            int16_t q = 0;
            std::memcpy(&q, data + i * sizeof(int16_t), sizeof(int16_t));
            out[i] = q / 32767.0f * scale;
        }
        return out;
    }

    static juce::int64 now() { return juce::Time::currentTimeMillis(); }

    void addStart(const std::vector<Note>& notes, double playhead, const juce::String& name)
    {
        const auto all = snapshot(notes);
        const std::lock_guard<std::mutex> lock(mutex);
        lastNotes = all;
        Entry e;
        e.time = now();
        e.kind = "start";
        e.name = name;
        e.playhead = playhead;
        e.after = all;
        entries.push_back(std::move(e));
    }

    std::atomic<bool> enabled{false};
    mutable std::mutex mutex;
    std::vector<Entry> entries;
    std::vector<NoteState> lastNotes;
    size_t audioSamples = 0;
};
} // namespace debugjournal
