// The changes of key on the piano roll's time ruler, and the lock on a note
// (spec section 15). Members of PianoRollComponent; the edits themselves are
// keyflags:: (Undo/KeyFlagEdits.h) and LockAction.

#include "../PianoRollComponent.h"

#include "../../Undo/KeyFlagEdits.h"
#include "../../Undo/LockAction.h"
#include "../../Undo/PitchUndoManager.h"
#include "../../Utils/Localization.h"
#include "../Components/AppFont.h"

namespace
{
constexpr float kFlagWidth = 24.0f;
constexpr float kFlagHeight = 12.0f;
constexpr float kFlagTop = 1.0f;
const juce::Colour kFlagColour{0xFF6FB6FFu};

const char *keyName(int root)
{
  static const char *names[12] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
  return root < 0 ? "-" : names[root % 12];
}

juce::String clockText(double seconds)
{
  const int whole = juce::jmax(0, static_cast<int>(seconds));
  return juce::String(whole / 60) + ":" + juce::String(whole % 60).paddedLeft('0', 2);
}
} // namespace

std::vector<PianoRollComponent::KeyFlag> PianoRollComponent::keyFlags() const
{
  std::vector<KeyFlag> flags;
  // A song in one key shows none: its key is the toolbar's.
  if (project == nullptr || project->getKeyChanges().empty())
    return flags;
  const auto stretches = keyflags::stretches(*project);
  const float left = static_cast<float>(pianoKeysWidth);
  for (size_t i = 0; i < stretches.size(); ++i)
  {
    KeyFlag flag;
    flag.stretch = static_cast<int>(i);
    flag.root = stretches[i].root;
    flag.x = left + timeToX(stretches[i].startSeconds) - static_cast<float>(scrollX);
    flags.push_back(flag);
  }
  // The stretch the left edge is in keeps its label in view, pinned to the
  // edge until the next flag pushes it out.
  for (size_t i = 0; i < flags.size(); ++i)
  {
    auto &flag = flags[i];
    float x = flag.x;
    if (x < left)
    {
      const float next = i + 1 < flags.size() ? flags[i + 1].x : left + kFlagWidth + 4.0f;
      if (next <= left)
        x = -kFlagWidth * 2.0f; // wholly out of view
      else
        x = std::min(left, next - kFlagWidth - 2.0f);
      flag.atOwnPlace = false;
    }
    flag.bounds = {x, kFlagTop, kFlagWidth, kFlagHeight};
  }
  return flags;
}

int PianoRollComponent::keyFlagAt(juce::Point<float> position) const
{
  if (position.y >= static_cast<float>(timelineHeight) || position.x < static_cast<float>(pianoKeysWidth))
    return -1;
  const auto flags = keyFlags();
  // The later flag is drawn over the earlier one.
  for (auto it = flags.rbegin(); it != flags.rend(); ++it)
    if (it->bounds.expanded(1.0f, 4.0f).contains(position) && it->bounds.getRight() > pianoKeysWidth)
      return it->stretch;
  return -1;
}

void PianoRollComponent::drawKeyFlags(juce::Graphics &g)
{
  const auto flags = keyFlags();
  if (flags.empty())
    return;
  juce::Graphics::ScopedSaveState state(g);
  g.reduceClipRegion(pianoKeysWidth, 0, getWidth() - pianoKeysWidth, timelineHeight);
  g.setFont(AppFont::getBoldFont(9.5f));
  for (const auto &flag : flags)
  {
    if (flag.bounds.getRight() <= pianoKeysWidth || flag.bounds.getX() >= getWidth())
      continue;
    const bool dragged = keyFlagDragIndex >= 0 && flag.stretch == keyFlagDragIndex + 1;
    // The first stretch's key is the project's own: it does not move.
    const auto colour = flag.stretch == 0 ? kFlagColour.withMultipliedBrightness(0.72f) : kFlagColour;
    if (flag.atOwnPlace && flag.stretch > 0)
    {
      g.setColour(colour);
      g.fillRect(flag.x, 0.0f, 1.0f, static_cast<float>(timelineHeight));
    }
    g.setColour(dragged ? colour.brighter(0.25f) : colour);
    g.fillRoundedRectangle(flag.bounds, 2.5f);
    g.setColour(juce::Colour(0xFF101418u));
    g.drawText(keyName(flag.root), flag.bounds, juce::Justification::centred, false);
  }
}

int PianoRollComponent::keyRootAtViewLeft() const
{
  if (project == nullptr || project->getKeyChanges().empty())
    return selectedScaleRootNote;
  const double seconds = std::max(0.0, xToTime(static_cast<float>(scrollX)));
  return project->keyRootAt(keyflags::sourceFrameAtSeconds(project->getAudioData(), seconds));
}

bool PianoRollComponent::commitKeyEdit(const KeyState &before, const KeyState &after,
                                       const juce::String &name)
{
  if (project == nullptr || before == after)
    return false;
  after.applyTo(*project);
  if (undoManager != nullptr)
    undoManager->addAction(std::make_unique<KeyChangeAction>(*project, before, after, name));
  refreshScaleFromProject();
  if (onKeyChangesEdited)
    onKeyChangesEdited();
  return true;
}

bool PianoRollComponent::setStretchKey(int stretch, int root)
{
  if (project == nullptr)
    return false;
  return commitKeyEdit(KeyState::of(*project), keyflags::rekeyed(*project, stretch, root),
                       TR("key.set"));
}

bool PianoRollComponent::removeKeyChange(int changeIndex)
{
  if (project == nullptr)
    return false;
  return commitKeyEdit(KeyState::of(*project), keyflags::removed(*project, changeIndex),
                       TR("key.remove_change"));
}

bool PianoRollComponent::addKeyChangeAt(double seconds, int root)
{
  if (project == nullptr)
    return false;
  const int frame = keyflags::sourceFrameNearSeconds(project->getAudioData(), seconds);
  return commitKeyEdit(KeyState::of(*project), keyflags::added(*project, frame, root),
                       TR("key.add_change"));
}

// ---------------------------------------------------------------------------
// The mouse on the ruler

bool PianoRollComponent::keyFlagMouseDown(const juce::MouseEvent &e)
{
  if (project == nullptr || e.y >= timelineHeight || e.x < pianoKeysWidth)
    return false;
  const int stretch = keyFlagAt(e.position);
  if (e.mods.isPopupMenu())
  {
    // A flag's own menu; on the bare ruler, a change of key added there.
    if (stretch >= 0)
      showKeyFlagMenu(stretch, e.getScreenPosition());
    else
      showAddKeyChangeMenu(std::max(0.0, xToTime(e.x - pianoKeysWidth + static_cast<float>(scrollX))),
                           e.getScreenPosition());
    return true;
  }
  if (stretch <= 0)
    return false; // the bare ruler, or the first stretch's flag: the playhead's
  const auto flags = keyFlags();
  const auto &flag = flags[static_cast<size_t>(stretch)];
  if (!flag.atOwnPlace)
    return false; // pinned to the edge, away from where the change is
  keyFlagDragIndex = stretch - 1;
  keyFlagDragGrab = e.position.x - flag.x;
  keyFlagDragBefore = project->getKeyChanges();
  repaint();
  return true;
}

void PianoRollComponent::keyFlagMouseDrag(const juce::MouseEvent &e)
{
  if (project == nullptr || keyFlagDragIndex < 0)
    return;
  const double seconds =
      xToTime(e.position.x - keyFlagDragGrab - pianoKeysWidth + static_cast<float>(scrollX));
  const int frame = keyflags::sourceFrameNearSeconds(project->getAudioData(), std::max(0.0, seconds));
  // Not past the changes on either side, nor off the recording.
  project->setKeyChanges(keyflags::moved(*project, keyFlagDragIndex, frame).changes);
  if (onKeyChangesEdited)
    onKeyChangesEdited();
  repaint();
}

void PianoRollComponent::keyFlagMouseUp()
{
  if (keyFlagDragIndex < 0)
    return;
  keyFlagDragIndex = -1;
  if (project != nullptr && project->getKeyChanges() != keyFlagDragBefore)
  {
    // Where it was let go is the step; the drag itself was shown as it went.
    const auto after = KeyState::of(*project);
    auto before = after;
    before.changes = keyFlagDragBefore;
    if (undoManager != nullptr)
      undoManager->addAction(
          std::make_unique<KeyChangeAction>(*project, before, after, TR("key.move_change")));
    if (onKeyChangesEdited)
      onKeyChangesEdited();
  }
  keyFlagDragBefore.clear();
  repaint();
}

void PianoRollComponent::showKeyFlagMenu(int stretch, juce::Point<int> screenPosition)
{
  if (project == nullptr)
    return;
  const auto stretches = keyflags::stretches(*project);
  if (stretch < 0 || stretch >= static_cast<int>(stretches.size()))
    return;
  juce::PopupMenu menu;
  menu.addSectionHeader(TR("key.menu_stretch").replace("{n}", juce::String(stretch + 1)));
  for (int root = 0; root < 12; ++root)
    menu.addItem(1 + root, keyName(root), true, stretches[static_cast<size_t>(stretch)].root == root);
  if (stretch > 0)
  {
    menu.addSeparator();
    menu.addItem(100, TR("key.remove_change"));
  }
  juce::Component::SafePointer<PianoRollComponent> safeThis(this);
  menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea({screenPosition.x, screenPosition.y, 1, 1}),
                     [safeThis, stretch](int chosen)
                     {
                       if (safeThis == nullptr || chosen <= 0)
                         return;
                       if (chosen == 100)
                         safeThis->removeKeyChange(stretch - 1);
                       else
                         safeThis->setStretchKey(stretch, chosen - 1);
                     });
}

void PianoRollComponent::showAddKeyChangeMenu(double seconds, juce::Point<int> screenPosition)
{
  if (project == nullptr || keyflags::sourceFrames(project->getAudioData()) <= 0)
    return;
  const int frame = keyflags::sourceFrameNearSeconds(project->getAudioData(), seconds);
  const bool possible = keyflags::canAdd(*project, frame);
  const int rootThere = project->keyRootAt(frame);
  juce::PopupMenu menu;
  menu.addSectionHeader(TR("key.menu_add").replace("{t}", clockText(seconds)));
  // To the key that is there already would change nothing.
  for (int root = 0; root < 12; ++root)
    menu.addItem(1 + root, keyName(root), possible && root != rootThere, root == rootThere);
  juce::Component::SafePointer<PianoRollComponent> safeThis(this);
  menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea({screenPosition.x, screenPosition.y, 1, 1}),
                     [safeThis, seconds](int chosen)
                     {
                       if (safeThis != nullptr && chosen > 0)
                         safeThis->addKeyChangeAt(seconds, chosen - 1);
                     });
}

// ---------------------------------------------------------------------------
// The lock

bool PianoRollComponent::toggleLock(const std::vector<Note *> &notes)
{
  std::vector<Note *> present;
  for (auto *note : notes)
    if (note != nullptr && !note->isRest())
      present.push_back(note);
  if (project == nullptr || present.empty())
    return false;
  const bool lock = LockAction::wanted(present);
  auto action = std::make_unique<LockAction>(present, lock, lock ? TR("edit.lock") : TR("edit.unlock"));
  for (auto *note : present)
    note->setLocked(lock);
  project->setModified(true);
  if (undoManager != nullptr)
    undoManager->addAction(std::move(action));
  if (onLockChanged)
    onLockChanged();
  repaint();
  return true;
}

bool PianoRollComponent::toggleLockOnSelection()
{
  return toggleLock(getSelectedNotes());
}

juce::Rectangle<float> PianoRollComponent::getLockButtonBounds(const Note &note) const
{
  constexpr float buttonGap = 4.0f;
  const auto bounds = getSynthButtonBounds(note);
  return {bounds.getRight() + buttonGap, bounds.getY(), static_cast<float>(lockButtonWidth),
          static_cast<float>(lockButtonHeight)};
}

juce::Rectangle<int> PianoRollComponent::getLockButtonLocalBounds(const Note &note) const
{
  const auto bounds = getLockButtonBounds(note).translated(
      static_cast<float>(pianoKeysWidth) - static_cast<float>(scrollX),
      static_cast<float>(headerHeight) - static_cast<float>(scrollY));
  return {static_cast<int>(std::round(bounds.getX())), static_cast<int>(std::round(bounds.getY())),
          lockButtonWidth, lockButtonHeight};
}
