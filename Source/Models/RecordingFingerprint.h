#pragma once

#include "../JuceHeader.h"

#include <cstdint>
#include <cstring>
#include <vector>

/**
 * Q11: the project links its recording instead of embedding it, the way
 * Cubase keeps VariAudio and AudioWarp edits as "original file + edits".
 *
 * Fingerprint: the recording's shape (rate, channels, length) and a hash of
 * every block of kBlockSamples samples (all channels), as the file reads.
 * When the file is edited outside PitchNet (RX) and the project is opened
 * again, the blocks whose hash changed are the parts to analyse and render
 * again; a different shape is refused (every edit is placed by frame).
 *
 * Piece: a run of the 44.1 kHz composite (AudioData::waveform) that differs
 * from the recording laid out on the timeline - the rendered edits. The
 * composite is saved as these runs and rebuilt on open as the recording with
 * the runs laid over it.
 */
namespace recording
{
constexpr int kBlockSamples = 1024;

struct Fingerprint
{
  int sampleRate = 0;
  int channels = 0;
  juce::int64 samples = 0;
  int blockSamples = kBlockSamples;
  std::vector<std::uint32_t> hashes;

  bool isValid() const { return sampleRate > 0 && channels > 0 && samples > 0 && !hashes.empty(); }
  bool sameShape(const Fingerprint &other) const
  {
    return sampleRate == other.sampleRate && channels == other.channels &&
           samples == other.samples;
  }
  bool operator==(const Fingerprint &other) const
  {
    return sameShape(other) && blockSamples == other.blockSamples && hashes == other.hashes;
  }
};

struct Piece
{
  int start = 0; // composite (timeline) sample
  std::vector<float> samples;
};

inline juce::String encodeBytes(const void *data, size_t bytes)
{
  return juce::MemoryBlock(data, bytes).toBase64Encoding();
}

inline juce::MemoryBlock decodeBytes(const juce::String &text)
{
  juce::MemoryBlock block;
  if (!block.fromBase64Encoding(text))
    block.reset();
  return block;
}

inline juce::var toJson(const Fingerprint &fingerprint)
{
  auto *object = new juce::DynamicObject();
  object->setProperty("sampleRate", fingerprint.sampleRate);
  object->setProperty("channels", fingerprint.channels);
  object->setProperty("samples", fingerprint.samples);
  object->setProperty("blockSamples", fingerprint.blockSamples);
  object->setProperty("hashes", encodeBytes(fingerprint.hashes.data(),
                                            fingerprint.hashes.size() * sizeof(std::uint32_t)));
  return juce::var(object);
}

inline Fingerprint fingerprintFromJson(const juce::var &value)
{
  Fingerprint fingerprint;
  if (!value.isObject())
    return fingerprint;
  fingerprint.sampleRate = static_cast<int>(value.getProperty("sampleRate", 0));
  fingerprint.channels = static_cast<int>(value.getProperty("channels", 0));
  fingerprint.samples = static_cast<juce::int64>(value.getProperty("samples", 0));
  fingerprint.blockSamples = static_cast<int>(value.getProperty("blockSamples", kBlockSamples));
  const auto bytes = decodeBytes(value.getProperty("hashes", "").toString());
  fingerprint.hashes.resize(bytes.getSize() / sizeof(std::uint32_t));
  if (!fingerprint.hashes.empty())
    std::memcpy(fingerprint.hashes.data(), bytes.getData(),
                fingerprint.hashes.size() * sizeof(std::uint32_t));
  const auto expected = fingerprint.blockSamples > 0
                            ? (fingerprint.samples + fingerprint.blockSamples - 1) /
                                  fingerprint.blockSamples
                            : 0;
  if (fingerprint.blockSamples <= 0 ||
      static_cast<juce::int64>(fingerprint.hashes.size()) != expected)
    return {};
  return fingerprint;
}

inline juce::var toJson(const std::vector<Piece> &pieces)
{
  juce::Array<juce::var> list;
  for (const auto &piece : pieces)
  {
    auto *object = new juce::DynamicObject();
    object->setProperty("start", piece.start);
    object->setProperty("samples", encodeBytes(piece.samples.data(),
                                               piece.samples.size() * sizeof(float)));
    list.add(juce::var(object));
  }
  return list;
}

inline std::vector<Piece> piecesFromJson(const juce::var &value)
{
  std::vector<Piece> pieces;
  if (const auto *list = value.getArray())
    for (const auto &item : *list)
    {
      Piece piece;
      piece.start = static_cast<int>(item.getProperty("start", -1));
      const auto bytes = decodeBytes(item.getProperty("samples", "").toString());
      piece.samples.resize(bytes.getSize() / sizeof(float));
      if (piece.start < 0 || piece.samples.empty())
        continue;
      std::memcpy(piece.samples.data(), bytes.getData(), piece.samples.size() * sizeof(float));
      pieces.push_back(std::move(piece));
    }
  return pieces;
}
} // namespace recording
