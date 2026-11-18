#include "SplitHandler.h"
#include "../../PianoRollComponent.h"
#include "../NoteSplitter.h"

SplitHandler::SplitHandler(PianoRollComponent &owner)
    : InteractionHandler(owner) {}

bool SplitHandler::mouseDown(const juce::MouseEvent &e, float worldX,
                             float worldY) {
  if (!owner_.isCanvasPoint(e))
    return false;

  if (mergeFirstNote && mergeSecondNote &&
      owner_.noteSplitter->mergeNotes(mergeFirstNote, mergeSecondNote))
  {
    clearGuide();
    return true;
  }

  Note *note = owner_.findNoteForSplitAt(worldX, worldY);
  if (note) {
    owner_.noteSplitter->splitNoteAtX(note, worldX);
    return true;
  }
  return false;
}

void SplitHandler::mouseMove(const juce::MouseEvent &e, float worldX,
                             float worldY) {
  if (!owner_.isCanvasPoint(e)) {
    clearGuide();
    return;
  }

  if (!owner_.project) {
    clearGuide();
    return;
  }

  const auto mergeNotes =
      owner_.noteSplitter->findMergeableNotesAt(worldX, worldY);
  Note *const nextMergeFirst = mergeNotes.first;
  Note *const nextMergeSecond = mergeNotes.second;
  float nextTop = 0.0f, nextBottom = 0.0f;
  Note *const nextGuideNote =
      nextMergeFirst ? nullptr
                     : owner_.findNoteForSplitAt(worldX, worldY, &nextTop,
                                                 &nextBottom);
  splitGuideTop = nextTop;
  splitGuideBottom = nextBottom;
  const float nextGuideX = nextGuideNote ? worldX : -1.0f;

  const bool hoverChanged =
      nextMergeFirst != mergeFirstNote || nextMergeSecond != mergeSecondNote ||
      nextGuideNote != splitGuideNote || nextGuideX != splitGuideX;

  mergeFirstNote = nextMergeFirst;
  mergeSecondNote = nextMergeSecond;
  splitGuideNote = nextGuideNote;
  splitGuideX = nextGuideX;

  if (hoverChanged)
    owner_.repaint();
}

void SplitHandler::cancel() { clearGuide(); }

void SplitHandler::clearGuide() {
  const bool hadHover = splitGuideX >= 0 || splitGuideNote != nullptr ||
                        mergeFirstNote != nullptr || mergeSecondNote != nullptr;
  splitGuideX = -1.0f;
  splitGuideNote = nullptr;
  mergeFirstNote = nullptr;
  mergeSecondNote = nullptr;
  if (hadHover)
    owner_.repaint();
}
