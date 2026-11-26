#pragma once

#include "../../JuceHeader.h"
#include "../Components/AppFont.h"
#include "../../Utils/UI/Theme.h"
#include "../../Utils/Localization.h"
#include "../../Models/Project.h"

#include <utility>
#include <vector>

/**
 * What the welcome screen and the wizard are drawn with (spec section 15):
 * the card they sit on, the pill buttons of a dialog, the cards a way is
 * chosen with, and the keys of the key check.
 */
namespace wizardui
{
inline const juce::Colour kCard{0xFF232323u};      // the dialog's own surface
inline const juce::Colour kInset{0xFF151515u};     // a well inside it
inline const juce::Colour kRaised{0xFF2E2E2Eu};    // something to press
inline const juce::Colour kOutline{0xFF4A4A4Au};
inline const juce::Colour kText{0xFFEFEFEFu};
inline const juce::Colour kChosen{0xFF6FB6FFu};    // the key chosen, the way marked
inline constexpr float kCorner = 10.0f;

inline const char* noteName(int root)
{
    static const char* names[12] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    return names[(root % 12 + 12) % 12];
}

/** The modes on offer, as the toolbar's key menu lists them, and their names. */
inline const std::vector<std::pair<ScaleMode, const char*>>& modes()
{
    static const std::vector<std::pair<ScaleMode, const char*>> all = {
        {ScaleMode::Major, "scale.major"}, {ScaleMode::Minor, "scale.minor"},
        {ScaleMode::Blues, "scale.blues"}, {ScaleMode::Dorian, "scale.dorian"},
        {ScaleMode::HarmonicMinor, "scale.harmonic_minor"}, {ScaleMode::Locrian, "scale.locrian"},
        {ScaleMode::Lydian, "scale.lydian"}, {ScaleMode::MajorPentatonic, "scale.major_pentatonic"},
        {ScaleMode::MelodicMinor, "scale.melodic_minor"}, {ScaleMode::MinorPentatonic, "scale.minor_pentatonic"},
        {ScaleMode::Mixolydian, "scale.mixolydian"}, {ScaleMode::Phrygian, "scale.phrygian"},
        {ScaleMode::PhrygianDominant, "scale.phrygian_dominant"}, {ScaleMode::WholeTone, "scale.whole_tone"}};
    return all;
}
inline juce::String modeName(ScaleMode mode)
{
    for (const auto& item : modes())
        if (item.first == mode)
            return TR(item.second);
    return TR("scale.major");
}

/** "3:07" */
inline juce::String clock(double seconds)
{
    const int whole = juce::jmax(0, static_cast<int>(seconds));
    return juce::String(whole / 60) + ":" + juce::String(whole % 60).paddedLeft('0', 2);
}

/** The dim behind an overlay and the card on it. */
inline void paintCard(juce::Graphics& g, juce::Rectangle<int> whole, juce::Rectangle<int> card)
{
    g.setColour(APP_COLOR_OVERLAY_DIM);
    g.fillRect(whole);
    juce::DropShadow(APP_COLOR_OVERLAY_SHADOW, 24, {0, 8}).drawForRectangle(g, card);
    g.setColour(kCard);
    g.fillRoundedRectangle(card.toFloat(), kCorner);
}

/** A dialog's button: the one that goes on is lit. */
class PillButton : public juce::TextButton
{
public:
    explicit PillButton(bool goesOn = false) : primary(goesOn)
    {
        setMouseCursor(juce::MouseCursor::PointingHandCursor);
    }
    void paintButton(juce::Graphics& g, bool over, bool down) override
    {
        auto colour = primary ? juce::Colour(0xFF3F7FC4u) : juce::Colour(0xFF5B5B5Bu);
        if (!isEnabled())
            colour = juce::Colour(0xFF3A3A3Au);
        else if (down)
            colour = colour.darker(0.2f);
        else if (over)
            colour = colour.brighter(0.12f);
        g.setColour(colour);
        g.fillRoundedRectangle(getLocalBounds().toFloat(), static_cast<float>(getHeight()) * 0.5f);
        g.setColour(kText.withAlpha(isEnabled() ? 1.0f : 0.4f));
        g.setFont(AppFont::getFont(14.0f));
        g.drawFittedText(getButtonText(), getLocalBounds().reduced(10, 0), juce::Justification::centred, 1);
    }

private:
    bool primary = false;
};

/** A line of text to press: "Open a project…", a recent file. */
class LinkButton : public juce::TextButton
{
public:
    LinkButton() { setMouseCursor(juce::MouseCursor::PointingHandCursor); }
    void paintButton(juce::Graphics& g, bool over, bool) override
    {
        g.setColour((over ? kText : APP_COLOR_TEXT_MUTED).withAlpha(isEnabled() ? 1.0f : 0.4f));
        g.setFont(AppFont::getFont(13.0f));
        g.drawText(getButtonText(), getLocalBounds(), justification, true);
    }
    juce::Justification justification = juce::Justification::centredLeft;
};

/** A way to choose: its name, and a line or two of what it does. One not
 *  on offer yet is drawn with a dashed edge and cannot be pressed. */
class ChoiceCard : public juce::Button
{
public:
    ChoiceCard() : juce::Button({}) { setMouseCursor(juce::MouseCursor::PointingHandCursor); }
    void setTexts(const juce::String& name, const juce::String& what)
    {
        title = name;
        detail = what;
        setTitle(name);
        repaint();
    }
    void paintButton(juce::Graphics& g, bool over, bool down) override
    {
        const auto bounds = getLocalBounds().toFloat().reduced(1.0f);
        if (isEnabled())
        {
            g.setColour(down ? kRaised.darker(0.15f) : over ? kRaised.brighter(0.08f) : kRaised);
            g.fillRoundedRectangle(bounds, kCorner);
            g.setColour(over ? kChosen : kOutline);
            g.drawRoundedRectangle(bounds, kCorner, over ? 1.5f : 1.0f);
        }
        else
        {
            juce::Path edge;
            edge.addRoundedRectangle(bounds, kCorner);
            const float dashes[] = {5.0f, 4.0f};
            juce::Path dashed;
            juce::PathStrokeType(1.0f).createDashedStroke(dashed, edge, dashes, 2);
            g.setColour(kOutline);
            g.fillPath(dashed);
        }
        auto text = getLocalBounds().reduced(16, 14);
        g.setColour(isEnabled() ? kText : APP_COLOR_TEXT_MUTED);
        g.setFont(AppFont::getBoldFont(16.0f));
        g.drawText(title, text.removeFromTop(24), juce::Justification::centredLeft, true);
        text.removeFromTop(4);
        g.setColour(APP_COLOR_TEXT_MUTED);
        g.setFont(AppFont::getFont(13.0f));
        g.drawFittedText(detail, text, juce::Justification::topLeft, 4, 0.9f);
    }

private:
    juce::String title, detail;
};

/** A key of the key check: pressed, it is heard and chosen. The one chosen
 *  is ringed; the one sounding is filled. */
class KeyButton : public juce::Button
{
public:
    explicit KeyButton(int rootToShow, bool large = false)
        : juce::Button(noteName(rootToShow)), root(rootToShow), big(large)
    {
        setMouseCursor(juce::MouseCursor::PointingHandCursor);
        setTitle(noteName(rootToShow));
    }
    int getRoot() const { return root; }
    void setRoot(int rootToShow)
    {
        root = rootToShow;
        setTitle(noteName(root));
        repaint();
    }
    void setChosen(bool isChosen)
    {
        if (chosen != isChosen)
        {
            chosen = isChosen;
            repaint();
        }
    }
    void setSounding(bool isSounding)
    {
        if (sounding != isSounding)
        {
            sounding = isSounding;
            repaint();
        }
    }
    bool isChosen() const { return chosen; }
    /** What is written on it, in place of the root's own name ("F Minor"). */
    void setText(const juce::String& text)
    {
        setTitle(text); // setRoot puts the root's own name there
        if (label != text)
        {
            label = text;
            repaint();
        }
    }
    void paintButton(juce::Graphics& g, bool over, bool down) override
    {
        const auto bounds = getLocalBounds().toFloat().reduced(1.0f);
        const float corner = big ? 8.0f : 5.0f;
        auto fill = sounding ? kChosen.withAlpha(0.35f) : kRaised;
        if (down)
            fill = fill.darker(0.15f);
        else if (over)
            fill = fill.brighter(0.1f);
        g.setColour(fill);
        g.fillRoundedRectangle(bounds, corner);
        g.setColour(chosen ? kChosen : kOutline);
        g.drawRoundedRectangle(bounds, corner, chosen ? 2.0f : 1.0f);
        g.setColour(kText);
        g.setFont(big ? AppFont::getBoldFont(17.0f) : AppFont::getFont(12.5f));
        g.drawFittedText(label.isNotEmpty() ? label : juce::String(noteName(root)),
                         getLocalBounds().reduced(6, 0), juce::Justification::centred, 1, 0.75f);
    }

private:
    int root = 0;
    juce::String label;
    bool big = false, chosen = false, sounding = false;
};
} // namespace wizardui
