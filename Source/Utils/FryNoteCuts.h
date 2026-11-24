#pragma once

#include "../Models/Project.h"

#include <vector>

/**
 * Notes cut where the vocal fry a word starts with begins (spec Q12).
 *
 * The detector keeps such fry as recorded from its lead to the voice after
 * it (fry::onsets). A note can start before those frames: the analysis put
 * its edge some frames into the unvoiced lead, further back than the
 * kOnsetLead frames the kept ones reach (煙幕 at 80.33 s and 230.30 s, 3
 * frames each). The kept frames then lie in the middle of the note, and the
 * note's two green squares, which mark a stretch at its start and one at its
 * end (Note::setKeptRange), have nowhere to stand for them: left in their
 * corners, the first drag of either one dropped the fry. Cut at the first
 * kept frame, the note that holds the fry starts with it, and its left
 * square stands at the end of what is kept.
 *
 * The cut is the App's own split (NoteSplitter::splitNoteAtFrame, with no
 * history step): the pitch edit, the way of rendering and the stretches kept
 * by hand go to both pieces, and what the note sings stays. Both pieces keep
 * the note's centre (centred on its own frames, as a split by hand is, the
 * piece that starts in fry sat 2 semitones under the voice), so the notes
 * lay the pitch curves out as they did. The split still writes them, and the
 * analysed curve of every other note, over again, a digit off in the fourth
 * place: the curves and the other notes are put back as they were read,
 * value for value, so that a project opens as it did before notes were cut,
 * but for the notes cut (煙幕: the unstretched renders the same sample for
 * sample, the guide rule's 621 other notes the same in every setting). A
 * note that was as the analysis made it gives two analysed notes ("Restore
 * note separation" leaves them cut); one the user had merged or split before
 * keeps its own account of the analysis.
 *
 * The piece before the fry is a note with no voiced frame. Moved on its own,
 * the note after it leaves it behind, and the base takes a step between the
 * two: the guide rule (WarpRender --guide-fix) moves it with the note it led
 * into.
 *
 * Made where a project's notes come to be with their pitch curves laid out,
 * before anything shows them:
 *   - after an analysis (EditorController::analyzeAudioImpl);
 *   - as a project is read with its frames known: JSON that carries them,
 *     the plug-in's state with its audio (ProjectSerializer);
 *   - as a linked project's recording is read (recordingchange::hydrate:
 *     the curves need its frames), and again when that recording was edited
 *     outside and its pitch detected anew (recordingchange::openLinked).
 * Not made where the plug-in gives a project its host's audio or applies a
 * changed recording: its piano roll may be showing that project, and holds
 * its notes by address. Cutting again changes nothing.
 */
namespace frycut
{
struct Cut
{
    int noteStart = 0; // source start frame of the note to cut
    int at = 0;        // source frame its right piece starts at
};

/** The cuts still to make: each onset whose kept frames start after the
 *  start of the note they are in and end before its end. */
std::vector<Cut> wanted(const Project& project);

/** Makes them. Returns how many notes were cut: 0 too when the project's
 *  pitch curves are not laid out yet. */
int apply(Project& project);

/** For listening tools only: with `false`, apply() cuts nothing, and a
 *  project opens as it did before notes were cut (WarpRender,
 *  PITCHNET_FRY_CUT=0). */
void setEnabled(bool enabled);
} // namespace frycut
