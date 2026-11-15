#pragma once

#include "PitchPopupMenu.h"
#include "WarpToolIcon.h"
#include "../../Utils/Localization.h"
#include "BinaryData.h"
#include <array>

namespace canvasPopupMenu
{
class LookAndFeel final : public juce::LookAndFeel_V4
{
public:
    LookAndFeel()
    {
        setColour(juce::PopupMenu::backgroundColourId, juce::Colours::transparentBlack);
    }

    void drawPopupMenuBackground(juce::Graphics&, int, int) override {}
    void drawResizableFrame(juce::Graphics&, int, int,
                            const juce::BorderSize<int>&) override {}
    int getPopupMenuBorderSize() override { return 0; }
    // The content paints its own rounded background and outline.
    int getMenuWindowFlags() override { return 0; }
};

inline LookAndFeel& getLookAndFeel()
{
    static LookAndFeel instance;
    return instance;
}

// One row of the menu: the tool's icon (its normal, hovered and pressed
// frames) and its name.
class ToolRow final : public juce::Button
{
public:
    ToolRow(const juce::String& name, juce::Image icon)
        : juce::Button(name), sprite(std::move(icon)) {}

    static constexpr int iconLeft = 8, iconWidth = 22, iconHeight = 21, gap = 10;

    void paintButton(juce::Graphics& g, bool over, bool down) override
    {
        if (over || down)
        {
            g.setColour(juce::Colour(0xFF171717u));
            g.fillRoundedRectangle(getLocalBounds().toFloat().reduced(4, 1), 5);
        }
        if (sprite.isValid())
        {
            const int frameHeight = sprite.getHeight() / 3;
            g.drawImage(sprite, iconLeft, (getHeight() - iconHeight) / 2, iconWidth,
                        iconHeight, 0, (down ? 2 : over ? 1 : 0) * frameHeight,
                        sprite.getWidth(), frameHeight);
        }
        g.setColour(APP_COLOR_TEXT_PRIMARY);
        g.setFont(AppFont::getFont(14));
        g.drawText(getButtonText(),
                   getLocalBounds().withTrimmedLeft(iconLeft + iconWidth + gap),
                   juce::Justification::centredLeft);
    }

private:
    juce::Image sprite;
};

class Content final : public juce::PopupMenu::CustomComponent
{
public:
    // The tools only, one per row: undo, redo and select all have their
    // shortcuts and toolbar buttons, and are not part of a vocal edit's flow.
    static juce::Rectangle<int> getTargetArea(juce::Point<int> mousePosition)
    {
        // The Select tool's icon lands under the pointer. JUCE places the
        // menu below the target rectangle's bottom edge.
        const juce::Point<int> selectIcon(ToolRow::iconLeft + ToolRow::iconWidth / 2,
                                          padding + rowHeight / 2);
        return { mousePosition.x - selectIcon.x,
                 mousePosition.y - selectIcon.y - 1, 1, 1 };
    }

    explicit Content(std::shared_ptr<int> result)
        : juce::PopupMenu::CustomComponent(false), choice(std::move(result))
    {
        const juce::String names[] = { TR("canvas.select"), TR("canvas.split"),
                                       TR("canvas.pitch_drawing"), TR("canvas.warp") };
        const void* data[] = { BinaryData::selectmenu_png, BinaryData::splitmenu_png,
                               BinaryData::drawingmenu_png };
        const int sizes[] = { BinaryData::selectmenu_pngSize, BinaryData::splitmenu_pngSize,
                              BinaryData::drawingmenu_pngSize };
        int textWidth = 0;
        for (int i = 0; i < 4; ++i)
        {
            // The warp tool is drawn like the toolbar's (WarpToolIcon.h).
            rows[i] = std::make_unique<ToolRow>(
                names[i], i < 3 ? juce::ImageCache::getFromMemory(data[i], sizes[i])
                                : warpToolIcon::makeMenuSprite());
            rows[i]->onClick = [this, i]
            {
                *choice = i + 1;
                triggerMenuItem();
            };
            addAndMakeVisible(*rows[i]);
            textWidth = juce::jmax(textWidth, juce::GlyphArrangement::getStringWidthInt(
                                                  AppFont::getFont(14), names[i]));
        }
        width = ToolRow::iconLeft + ToolRow::iconWidth + ToolRow::gap + textWidth + 16;
    }

    void getIdealSize(int& idealWidth, int& idealHeight) override
    {
        idealWidth = width;
        idealHeight = 2 * padding + 4 * rowHeight;
    }

    void paint(juce::Graphics& g) override
    {
        pitchPopupMenu::getLookAndFeel().drawPopupMenuBackground(g, getWidth(),
                                                                 getHeight());
    }

    void resized() override
    {
        for (int i = 0; i < 4; ++i)
            rows[i]->setBounds(0, padding + i * rowHeight, getWidth(), rowHeight);
    }

private:
    static constexpr int padding = 4, rowHeight = 28;
    int width = 0;
    std::shared_ptr<int> choice;
    std::array<std::unique_ptr<ToolRow>, 4> rows;
};
}
