#pragma once

#include "../../JuceHeader.h"

namespace warpToolIcon
{
// The warp tool's icon - a warp anchor (a vertical line under a tab) with
// arrows pointing both ways - in a 46 x 22 box, the size of one row of the
// toolbar's 2x tool sprites. Scale or move it with `transform`.
inline void draw(juce::Graphics& g, juce::Colour colour,
                 const juce::AffineTransform& transform = {})
{
    // Filled one part at a time: overlapping parts in a single path could
    // wind against each other and cut holes.
    g.setColour(colour);
    juce::Path tab, line, arrows;
    tab.addTriangle(16.0f, 1.5f, 30.0f, 1.5f, 23.0f, 8.5f);
    line.addRectangle(21.5f, 6.0f, 3.0f, 15.0f);
    arrows.addArrow({ 18.0f, 14.0f, 3.0f, 14.0f }, 2.6f, 8.0f, 6.0f);
    arrows.addArrow({ 28.0f, 14.0f, 43.0f, 14.0f }, 2.6f, 8.0f, 6.0f);
    g.fillPath(tab, transform);
    g.fillPath(line, transform);
    g.fillPath(arrows, transform);
}

// The toolbar sprite: four 22 px rows at 2x (off, off hovered, on, on
// hovered), like the PNG sprites of the other tools.
inline juce::Image makeToolbarSprite()
{
    constexpr int width = 46, row = 22;
    juce::Image image(juce::Image::ARGB, width, row * 4, true);
    juce::Graphics g(image);
    const juce::Colour colours[] = { juce::Colour(0xFF9A9A9Au), juce::Colour(0xFFFFFFFFu),
                                     juce::Colour(0xFFFF5600u), juce::Colour(0xFFF9855Du) };
    for (int i = 0; i < 4; ++i)
        draw(g, colours[i],
             juce::AffineTransform::translation(0.0f, static_cast<float>(i * row)));
    return image;
}

// The canvas right-click menu's sprite: three 42 px frames at 2x (normal,
// hovered, pressed) on rounded tiles, matching select-menu.png and the
// other tool buttons there.
inline juce::Image makeMenuSprite()
{
    constexpr int width = 43, frame = 42;
    juce::Image image(juce::Image::ARGB, width, frame * 3, true);
    juce::Graphics g(image);
    const juce::Colour tile[] = { juce::Colour(0xFF2E2E2Du), juce::Colour(0xFF5B5B5Bu),
                                  juce::Colour(0xFF5B5B5Bu) };
    const juce::Colour ink[] = { juce::Colour(0xFF9A9A9Au), juce::Colour(0xFF9A9A9Au),
                                 juce::Colour(0xFF2E2E2Du) };
    // The icon is wide: fit its 40 px of arrows into the tile with a margin.
    constexpr float scale = 0.8f;
    const float iconWidth = 40.0f * scale, iconHeight = 19.5f * scale;
    for (int i = 0; i < 3; ++i)
    {
        const float top = static_cast<float>(i * frame);
        g.setColour(tile[i]);
        g.fillRoundedRectangle(0.0f, top, static_cast<float>(width),
                               static_cast<float>(frame), 4.0f);
        const float x = (static_cast<float>(width) - iconWidth) * 0.5f - 3.0f * scale;
        const float y = top + (static_cast<float>(frame) - iconHeight) * 0.5f - 1.5f * scale;
        draw(g, ink[i], juce::AffineTransform::scale(scale).translated(x, y));
    }
    return image;
}
} // namespace warpToolIcon
