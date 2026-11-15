#pragma once

#include "Note.h"

#include <cstdint>
#include <vector>

/**
 * A project's notes. The same std::vector, but every live list registers
 * itself so a NoteRef can find which list a note lives in, and whether that
 * list still exists.
 */
class NoteList : public std::vector<Note>
{
public:
    NoteList();
    NoteList(const NoteList& other);
    NoteList(NoteList&& other) noexcept;
    NoteList& operator=(const NoteList& other) = default;
    NoteList& operator=(NoteList&& other) noexcept = default;
    ~NoteList();
};

/**
 * A note an undo action changed, found again when the action is undone or
 * redone. Actions used to keep the Note* itself, but the project's notes
 * sit in one vector: a split or merge inserts into and erases from it, which
 * moves the notes after it (or the whole list), and undoing past one wrote
 * through a stale pointer - into the wrong note, or into freed memory.
 *
 * get() returns the note while it is still where it was, otherwise the note
 * with the same uid in the same list, otherwise nullptr (the note is gone,
 * the action skips it).
 */
class NoteRef
{
public:
    NoteRef() = default;
    NoteRef(Note* note); // NOLINT: implicit, so actions convert Note* lists

    Note* get() const;
    Note* operator->() const { return get(); }
    explicit operator bool() const { return get() != nullptr; }

private:
    mutable Note* hint = nullptr;
    std::uint64_t uid = 0;
    const NoteList* owner = nullptr;
};

namespace noteref
{
std::vector<NoteRef> refs(const std::vector<Note*>& notes);
/** The notes, in the same order; nullptr where one is gone. */
std::vector<Note*> resolve(const std::vector<NoteRef>& refs);
/** Only the notes that still exist. */
std::vector<Note*> present(const std::vector<NoteRef>& refs);
}
