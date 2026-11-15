// Offline check for the warp layer: runs PitchNet's own analysis and
// synthesis on a file, applies warp anchors and a pitch shift, and writes the
// result to a WAV file. Never plays audio.
//
//   WarpRender <in.wav> <out.wav> [--anchors s:d,s:d,...] [--shift semitones]
//              [--shift-range from:to] [--engine vocoder|psola]
//              [--detector fcpe|swift|rmvpe] [--dump-notes]
//              [--snapshot prefix]
//
// Anchors are source:timeline positions in seconds. --shift moves the pitch
// of every note whose recorded start lies in --shift-range (seconds, default
// the whole file). Run from the repository root so Resources/models is found.
// --snapshot draws the piano roll in the Warp and Timing tools to
// <prefix>_*.png, with a drag of the first anchor pushed past its limit.
#include "../../Source/Audio/EditorController.h"
#include "../../Source/Audio/WarpLayer.h"
#include "../../Source/Utils/Constants.h"
#include "../../Source/Utils/OnnxRuntime.h"
#include "../../Source/Utils/PitchCurveProcessor.h"
#include "../../Source/UI/PianoRollComponent.h"
#include "../../Source/UI/PianoRoll/PianoRollInteractionContext.h"

#include "../../Source/JuceHeader.h"

#include <algorithm>
#include <atomic>
#include <vector>
#include <cstdio>
#include <memory>

namespace
{
bool waitFor(const std::atomic<bool> &flag, int timeoutMs)
{
  const auto start = juce::Time::getMillisecondCounter();
  while (!flag.load())
  {
    juce::MessageManager::getInstance()->runDispatchLoopUntil(20);
    if (juce::Time::getMillisecondCounter() - start > static_cast<juce::uint32>(timeoutMs))
      return false;
  }
  return true;
}

bool writePng(juce::Component &component, const juce::File &file)
{
  const auto image = component.createComponentSnapshot(
      component.getLocalBounds(), true, 2.0f);
  file.deleteFile();
  juce::FileOutputStream stream(file);
  juce::PNGImageFormat png;
  return stream.openedOk() && png.writeImageToStream(image, stream);
}

juce::MouseEvent mouseAt(juce::Component &component, juce::Point<float> at,
                         juce::Point<float> downAt, juce::ModifierKeys mods)
{
  auto source = juce::Desktop::getInstance().getMainMouseSource();
  const auto now = juce::Time::getCurrentTime();
  return juce::MouseEvent(source, at, mods, juce::MouseInputSource::defaultPressure,
                          0.0f, 0.0f, 0.0f, 0.0f, &component, &component, now,
                          downAt, now, 1, false);
}

void snapshots(Project &project, const juce::String &prefix)
{
  PianoRollComponent roll;
  roll.setSize(1400, 640);
  roll.setProject(&project);
  roll.setPixelsPerSecond(300.0f);
  roll.setScrollX(0.0);
  // Centre on the median note: one stray low note would pull a min..max
  // range off the singing.
  std::vector<float> pitches;
  for (const auto &note : project.getNotes())
    if (!note.isRest())
      pitches.push_back(note.getMidiNote());
  if (!pitches.empty())
  {
    std::nth_element(pitches.begin(), pitches.begin() + pitches.size() / 2,
                     pitches.end());
    const float median = pitches[pitches.size() / 2];
    roll.centerOnPitchRange(median - 6.0f, median + 6.0f);
  }
  const auto dir = juce::File::getCurrentWorkingDirectory();
  auto shot = [&](const juce::String &name)
  {
    const auto file = dir.getChildFile(prefix + "_" + name + ".png");
    std::printf("snapshot %s %s\n", file.getFullPathName().toRawUTF8(),
                writePng(roll, file) ? "ok" : "FAILED");
  };

  auto view = [&](const char *label)
  {
    std::printf("%s: %.1f px/s, %.1f px/semitone, view %.2f-%.2f s\n", label,
                roll.getPixelsPerSecond(), roll.getPixelsPerSemitone(),
                roll.getScrollX() / roll.getPixelsPerSecond(),
                (roll.getScrollX() + roll.getVisibleContentWidth()) /
                    roll.getPixelsPerSecond());
  };
  view("start");
  roll.zoomHorizontallyBy(1.5f);
  view("G");
  roll.zoomHorizontallyBy(1.0f / 1.5f);
  view("H");
  roll.zoomVerticallyBy(1.25f);
  view("Shift+G");
  roll.zoomVerticallyBy(1.0f / 1.25f);
  view("Shift+H");
  roll.zoomToFullLength();
  view("Shift+F");
  shot("zoom_full");
  int picked = 0;
  for (auto &note : project.getNotes())
    if (!note.isRest() && picked < 3)
    {
      note.setSelected(true);
      ++picked;
    }
  roll.zoomToNotes();
  view("Shift+S (first 3 notes)");
  shot("zoom_selected");
  for (auto &note : project.getNotes())
    note.setSelected(false);
  roll.setPixelsPerSecond(300.0f);
  roll.setPlayheadTime(2.0);
  roll.zoomToNotes();
  view("Shift+S (nothing selected, playhead 2.00 s)");
  roll.setPixelsPerSecond(300.0f);
  roll.setScrollX(0.0);
  roll.setEditMode(EditMode::Select);
  shot("select");
  roll.setEditMode(EditMode::Warp);
  shot("warp");

  const auto &anchors = project.getAudioData().warpMap.anchors();
  if (anchors.empty())
  {
    // No anchors yet: drag the end of the first sung note (spec section 4).
    const Note *first = nullptr;
    for (const auto &note : project.getNotes())
      if (!note.isRest())
      {
        first = &note;
        break;
      }
    if (first == nullptr)
      return;
    auto &context = roll.getInteractionContext();
    const float edgeX = static_cast<float>(first->getEndFrame()) * HOP_SIZE /
                            SAMPLE_RATE * roll.getPixelsPerSecond() -
                        static_cast<float>(roll.getScrollX()) +
                        PianoRollComponent::pianoKeysWidth;
    const float rowY = context.midiToY(first->getAdjustedMidiNote()) +
                       roll.getPixelsPerSemitone() * 0.5f -
                       static_cast<float>(roll.getScrollY()) +
                       PianoRollComponent::headerHeight;
    const juce::Point<float> edge(edgeX, rowY), offRow(edgeX, rowY + 300.0f);
    const juce::ModifierKeys left(juce::ModifierKeys::leftButtonModifier);
    roll.mouseMove(mouseAt(roll, edge, edge, {}));
    shot("boundary_hover");
    std::printf("note edge at %.3f s, row %.1f px\n",
                first->getEndFrame() * HOP_SIZE / double(SAMPLE_RATE), rowY);
    roll.mouseDown(mouseAt(roll, edge, edge, left));
    roll.mouseDrag(mouseAt(roll, {edgeX + 90.0f, rowY}, edge, left));
    shot("boundary_drag");
    roll.mouseUp(mouseAt(roll, {edgeX + 90.0f, rowY}, edge, {}));
    std::printf("after boundary drag:");
    for (const auto &anchor : project.getAudioData().warpMap.anchors())
      std::printf(" %.3f->%.3f", anchor.source * HOP_SIZE / SAMPLE_RATE,
                  anchor.destination * HOP_SIZE / SAMPLE_RATE);
    std::printf("\n");
    // A click 200 px above the notes row places a single anchor instead.
    const auto count = project.getAudioData().warpMap.anchors().size();
    const float newEdgeX = edgeX + 90.0f;
    const juce::Point<float> below(newEdgeX + 40.0f, rowY - 200.0f);
    roll.mouseDown(mouseAt(roll, below, below, left));
    roll.mouseUp(mouseAt(roll, below, below, {}));
    std::printf("click off the row added %d anchor(s)\n",
                static_cast<int>(project.getAudioData().warpMap.anchors().size() - count));
    juce::ignoreUnused(offRow);
    return;
  }
  // Screen x of the first anchor: world x minus scroll plus the piano keys.
  const float worldX = static_cast<float>(
      anchors.front().destination * HOP_SIZE / SAMPLE_RATE * 300.0);
  const float x = worldX - static_cast<float>(roll.getScrollX()) +
                  PianoRollComponent::pianoKeysWidth;
  const juce::Point<float> down(x, 300.0f);
  const juce::ModifierKeys left(juce::ModifierKeys::leftButtonModifier);
  roll.mouseMove(mouseAt(roll, down, down, {}));
  shot("hover");
  roll.mouseDown(mouseAt(roll, down, down, left));
  roll.mouseDrag(mouseAt(roll, {x + 120.0f, 300.0f}, down, left));
  shot("drag");
  roll.mouseDrag(mouseAt(roll, {40.0f, 300.0f}, down, left));
  shot("limit");
  roll.mouseUp(mouseAt(roll, {40.0f, 300.0f}, down, {}));
  shot("released");

  // Shift + Alt + click on the canvas sets the playhead (MainComponent does
  // setPlayheadTime from onSeek).
  roll.onSeek = [&](double time)
  {
    roll.setPlayheadTime(time);
    std::printf("playhead set to %.3f s\n", time);
  };
  {
    const float clickX = static_cast<float>(1.25 * roll.getPixelsPerSecond() -
                                            roll.getScrollX()) +
                         PianoRollComponent::pianoKeysWidth;
    const juce::Point<float> at(clickX, 400.0f);
    const juce::ModifierKeys shiftAlt(juce::ModifierKeys::leftButtonModifier |
                                      juce::ModifierKeys::shiftModifier |
                                      juce::ModifierKeys::altModifier);
    const auto anchorsBefore = project.getAudioData().warpMap.anchors().size();
    roll.mouseDown(mouseAt(roll, at, at, shiftAlt));
    roll.mouseUp(mouseAt(roll, at, at, {}));
    std::printf("anchors unchanged by Shift+Alt+click: %s\n",
                project.getAudioData().warpMap.anchors().size() == anchorsBefore
                    ? "yes" : "NO");
    shot("playhead");
  }

  auto dumpAnchors = [&](const char *label)
  {
    std::printf("%s:", label);
    for (const auto &anchor : project.getAudioData().warpMap.anchors())
      std::printf(" %.3f->%.3f", anchor.source * HOP_SIZE / SAMPLE_RATE,
                  anchor.destination * HOP_SIZE / SAMPLE_RATE);
    std::printf("  pending %d\n", project.getAudioData().warpLayoutPending ? 1 : 0);
  };
  dumpAnchors("after drag");

  // Type a value into the ratio strip: the segment between the 2nd and 3rd
  // anchors (world x of 2.5 s), "150%".
  project.getAudioData().warpLayoutPending = false;
  const float stripX = static_cast<float>(2.5 * 300.0 - roll.getScrollX()) +
                       PianoRollComponent::pianoKeysWidth;
  const juce::Point<float> strip(stripX, PianoRollComponent::headerHeight + 8.0f);
  roll.mouseDoubleClick(mouseAt(roll, strip, strip, left));
  for (auto *child : roll.getChildren())
    if (auto *editor = dynamic_cast<juce::TextEditor *>(child))
    {
      std::printf("editor shows \"%s\"\n", editor->getText().toRawUTF8());
      shot("input");
      editor->setText("150%", false);
      editor->onReturnKey();
      break;
    }
  dumpAnchors("after typing 150%");
  shot("typed");

  // Alt + click on the last anchor deletes it.
  project.getAudioData().warpLayoutPending = false;
  const auto &now = project.getAudioData().warpMap.anchors();
  const float lastX = static_cast<float>(now.back().destination * HOP_SIZE /
                                         SAMPLE_RATE * 300.0 - roll.getScrollX()) +
                      PianoRollComponent::pianoKeysWidth;
  const juce::Point<float> last(lastX, 300.0f);
  const juce::ModifierKeys altLeft(juce::ModifierKeys::leftButtonModifier |
                                   juce::ModifierKeys::altModifier);
  roll.mouseDown(mouseAt(roll, last, last, altLeft));
  roll.mouseUp(mouseAt(roll, last, last, juce::ModifierKeys::altModifier));
  dumpAnchors("after Alt+click");
  shot("deleted");
}

bool writeWav(const juce::File &file, const juce::AudioBuffer<float> &buffer,
              double rate)
{
  file.deleteFile();
  juce::WavAudioFormat format;
  std::unique_ptr<juce::OutputStream> stream(file.createOutputStream());
  if (!stream)
    return false;
  std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(
      stream.get(), rate, static_cast<unsigned int>(buffer.getNumChannels()), 24,
      {}, 0));
  if (!writer)
    return false;
  stream.release();
  return writer->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples());
}
} // namespace

int main(int argc, char **argv)
{
  juce::ScopedJuceInitialiser_GUI juce;
  if (argc < 3)
  {
    std::fprintf(stderr, "usage: WarpRender in.wav out.wav [options]\n");
    return 2;
  }
  const juce::File input = juce::File::getCurrentWorkingDirectory().getChildFile(
      juce::String::fromUTF8(argv[1]));
  const juce::File output = juce::File::getCurrentWorkingDirectory().getChildFile(
      juce::String::fromUTF8(argv[2]));
  juce::String anchorsArg, engine = "vocoder", detector = "fcpe", snapshotPrefix;
  double shift = 0.0, shiftFrom = 0.0, shiftTo = 1.0e9;
  bool dumpNotes = false;
  for (int i = 3; i < argc; ++i)
  {
    const juce::String arg = juce::String::fromUTF8(argv[i]);
    const juce::String next = i + 1 < argc ? juce::String::fromUTF8(argv[i + 1]) : "";
    if (arg == "--anchors") { anchorsArg = next; ++i; }
    else if (arg == "--shift") { shift = next.getDoubleValue(); ++i; }
    else if (arg == "--shift-range")
    {
      shiftFrom = next.upToFirstOccurrenceOf(":", false, false).getDoubleValue();
      shiftTo = next.fromFirstOccurrenceOf(":", false, false).getDoubleValue();
      ++i;
    }
    else if (arg == "--engine") { engine = next; ++i; }
    else if (arg == "--detector") { detector = next; ++i; }
    else if (arg == "--dump-notes") dumpNotes = true;
    else if (arg == "--snapshot") { snapshotPrefix = next; ++i; }
  }

  // ONNX Runtime is loaded from <executable>/../Frameworks on macOS.
  juce::String ortError;
  if (!OnnxRuntime::initialise(&ortError))
  {
    std::fprintf(stderr, "ONNX Runtime: %s\n", ortError.toRawUTF8());
    return 1;
  }
  EditorController controller(false);
  controller.setPitchDetectorType(detector == "swift" ? PitchDetectorType::SwiftF0
                                  : detector == "rmvpe" ? PitchDetectorType::RMVPE
                                                        : PitchDetectorType::FCPE);
  controller.setSynthesisEngineType(engine == "psola" ? SynthesisEngineType::Psola
                                                      : SynthesisEngineType::Vocoder);

  controller.reloadInferenceModels(false);
  // Models load in the background; analysing before they are ready shows a
  // "model failed to load" dialog.
  {
    const auto start = juce::Time::getMillisecondCounter();
    while (!controller.isSelectedPitchDetectorLoaded() || controller.isInferenceBusy())
    {
      juce::MessageManager::getInstance()->runDispatchLoopUntil(50);
      if (juce::Time::getMillisecondCounter() - start > 120000)
      {
        std::fprintf(stderr, "pitch model did not load\n");
        return 1;
      }
    }
  }

  std::atomic<bool> loaded{false}, failed{false};
  controller.loadAudioFileAsync(
      input, [](double, const juce::String &) {},
      [&](const juce::AudioBuffer<float> &) { loaded = true; },
      [&]() { failed = true; loaded = true; });
  if (!waitFor(loaded, 600000) || failed.load() || !controller.getProject())
  {
    std::fprintf(stderr, "analysis failed\n");
    return 1;
  }
  auto &project = *controller.getProject();
  auto &audioData = project.getAudioData();
  const double framesPerSecond = static_cast<double>(SAMPLE_RATE) / HOP_SIZE;
  std::printf("analysed: %d frames, %d notes\n", audioData.getNumFrames(),
              static_cast<int>(project.getNotes().size()));

  // Warp anchors.
  warp::fitToProject(audioData);
  warp::WarpMap map = audioData.warpMap;
  for (const auto &pair : juce::StringArray::fromTokens(anchorsArg, ",", ""))
  {
    if (pair.trim().isEmpty())
      continue;
    const double source =
        pair.upToFirstOccurrenceOf(":", false, false).getDoubleValue() * framesPerSecond;
    const double destination =
        pair.fromFirstOccurrenceOf(":", false, false).getDoubleValue() * framesPerSecond;
    const int index = map.addAnchorAtSource(source);
    if (index < 0)
    {
      std::fprintf(stderr, "anchor at %.3f s refused\n", source / framesPerSecond);
      continue;
    }
    const double placed = map.moveAnchor(index, destination);
    std::printf("anchor %.3f s -> %.3f s (asked %.3f s)\n", source / framesPerSecond,
                placed / framesPerSecond, destination / framesPerSecond);
  }
  if (!map.isIdentity())
    warp::apply(project, map);

  // Pitch shift.
  if (shift != 0.0)
  {
    int shifted = 0;
    for (auto &note : project.getNotes())
    {
      const double start = note.getSrcStartFrame() / framesPerSecond;
      if (note.isRest() || start < shiftFrom || start >= shiftTo)
        continue;
      note.setMidiNote(note.getMidiNote() + static_cast<float>(shift));
      note.markDirty();
      note.markSynthDirty();
      ++shifted;
    }
    PitchCurveProcessor::rebuildBaseFromNotes(project);
    std::printf("shifted %d notes by %.2f semitones\n", shifted, shift);
  }

  if (dumpNotes)
    for (const auto &note : project.getNotes())
      std::printf("note src [%d, %d) -> [%d, %d) midi %.2f%s\n",
                  note.getSrcStartFrame(), note.getSrcEndFrame(),
                  note.getStartFrame(), note.getEndFrame(), note.getMidiNote(),
                  note.isRest() ? " rest" : "");

  if (snapshotPrefix.isNotEmpty())
  {
    snapshots(project, snapshotPrefix);
    controller.requestShutdown();
    return 0;
  }

  std::atomic<bool> pending{false};
  std::atomic<bool> rendered{false}, ok{false};
  const bool anyEdit = audioData.warpLayoutPending || project.hasDirtyNotes();
  if (anyEdit)
  {
    controller.resynthesizeIncrementalAsync(
        project, nullptr,
        [&](bool success) { ok = success; rendered = true; }, pending, false);
    if (!waitFor(rendered, 600000))
    {
      std::fprintf(stderr, "render timed out\n");
      return 1;
    }
    std::printf("render %s\n", ok.load() ? "ok" : "FAILED");
  }

  if (!writeWav(output, audioData.waveform, audioData.sampleRate))
  {
    std::fprintf(stderr, "could not write %s\n", output.getFullPathName().toRawUTF8());
    return 1;
  }
  std::printf("wrote %s (%.2f s at %d Hz)\n", output.getFullPathName().toRawUTF8(),
              audioData.waveform.getNumSamples() / double(audioData.sampleRate),
              audioData.sampleRate);
  controller.requestShutdown();
  return ok.load() || !anyEdit ? 0 : 1;
}
