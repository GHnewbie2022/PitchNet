#pragma once

#include <cctype>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

/**
 * Parses a time typed into the toolbar's time field, in seconds.
 *
 *   "01:02:345"  minutes : seconds : milliseconds (the field's own format)
 *   "01:02.345"  minutes : seconds . fraction
 *   "62.345"     seconds . fraction
 *   "62"         seconds
 *
 * The part after the last ':' of three parts, or after '.', is a decimal
 * fraction: "1:02:5" is 62.5 s, "1:02:005" is 62.005 s. Surrounding spaces
 * are ignored. Anything else, or a negative time, gives nothing.
 *
 * JUCE-free so it can be unit tested on its own.
 */
namespace timecode
{
inline std::optional<double> parse(std::string text)
{
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())))
    text.erase(text.begin());
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
    text.pop_back();
  if (text.empty())
    return std::nullopt;

  std::vector<std::string> parts;
  std::string current;
  for (char c : text)
  {
    if (c == ':')
    {
      parts.push_back(current);
      current.clear();
    }
    else
      current += c;
  }
  parts.push_back(current);
  if (parts.size() > 3)
    return std::nullopt;

  auto digitsOnly = [](const std::string &s)
  {
    if (s.empty())
      return false;
    for (char c : s)
      if (!std::isdigit(static_cast<unsigned char>(c)))
        return false;
    return true;
  };
  // "12" or "12.345": whole and fractional seconds.
  auto seconds = [&](const std::string &s) -> std::optional<double>
  {
    const auto dot = s.find('.');
    const std::string whole = s.substr(0, dot);
    const std::string frac = dot == std::string::npos ? "" : s.substr(dot + 1);
    if (!digitsOnly(whole) || (dot != std::string::npos && !digitsOnly(frac)))
      return std::nullopt;
    double value = std::atof(whole.c_str());
    if (!frac.empty())
      value += std::atof(("0." + frac).c_str());
    return value;
  };

  if (parts.size() == 3)
  {
    if (!digitsOnly(parts[0]) || !digitsOnly(parts[1]) || !digitsOnly(parts[2]))
      return std::nullopt;
    return std::atof(parts[0].c_str()) * 60.0 + std::atof(parts[1].c_str()) +
           std::atof(("0." + parts[2]).c_str());
  }
  if (parts.size() == 2)
  {
    if (!digitsOnly(parts[0]))
      return std::nullopt;
    const auto s = seconds(parts[1]);
    if (!s)
      return std::nullopt;
    return std::atof(parts[0].c_str()) * 60.0 + *s;
  }
  return seconds(parts[0]);
}
} // namespace timecode
