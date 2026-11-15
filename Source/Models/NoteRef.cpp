#include "NoteRef.h"

#include <algorithm>
#include <functional>
#include <mutex>

namespace
{
std::mutex& registryMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::vector<const NoteList*>& registry()
{
    static std::vector<const NoteList*> lists;
    return lists;
}

void enrol(const NoteList* list)
{
    std::lock_guard<std::mutex> lock(registryMutex());
    registry().push_back(list);
}

void withdraw(const NoteList* list)
{
    std::lock_guard<std::mutex> lock(registryMutex());
    auto& lists = registry();
    lists.erase(std::remove(lists.begin(), lists.end(), list), lists.end());
}

bool holds(const NoteList& list, const Note* note)
{
    if (list.empty() || note == nullptr)
        return false;
    const std::less<const Note*> before;
    return !before(note, list.data()) && before(note, list.data() + list.size());
}
} // namespace

NoteList::NoteList() { enrol(this); }
NoteList::NoteList(const NoteList& other) : std::vector<Note>(other) { enrol(this); }
NoteList::NoteList(NoteList&& other) noexcept : std::vector<Note>(std::move(other))
{
    enrol(this);
}
NoteList::~NoteList() { withdraw(this); }

NoteRef::NoteRef(Note* note) : hint(note)
{
    if (note == nullptr)
        return;
    uid = note->getUid();
    std::lock_guard<std::mutex> lock(registryMutex());
    for (const auto* list : registry())
        if (holds(*list, note))
        {
            owner = list;
            break;
        }
}

Note* NoteRef::get() const
{
    if (hint == nullptr)
        return nullptr;
    // A note outside every project's list (a scratch copy): nothing can move
    // it under us, so the pointer is all there is.
    if (owner == nullptr)
        return hint;

    std::lock_guard<std::mutex> lock(registryMutex());
    const auto& lists = registry();
    if (std::find(lists.begin(), lists.end(), owner) == lists.end())
        return nullptr; // the project is gone
    auto& list = const_cast<NoteList&>(*owner);
    if (holds(list, hint) && hint->getUid() == uid)
        return hint;
    for (auto& note : list)
        if (note.getUid() == uid)
        {
            hint = &note;
            return hint;
        }
    return nullptr;
}

namespace noteref
{
std::vector<NoteRef> refs(const std::vector<Note*>& notes)
{
    return std::vector<NoteRef>(notes.begin(), notes.end());
}

std::vector<Note*> resolve(const std::vector<NoteRef>& refs)
{
    std::vector<Note*> notes;
    notes.reserve(refs.size());
    for (const auto& ref : refs)
        notes.push_back(ref.get());
    return notes;
}

std::vector<Note*> present(const std::vector<NoteRef>& refs)
{
    std::vector<Note*> notes;
    for (const auto& ref : refs)
        if (auto* note = ref.get())
            notes.push_back(note);
    return notes;
}
} // namespace noteref
