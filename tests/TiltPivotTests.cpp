// Tilt about a pivot (Cubase's seesaw): each side ramps from its edge to zero
// at the pivot; a rotation keeps one straight line through it; the note's
// mean offset matches Note::tiltMeanFor.
#include "../Source/Models/Note.h"
#include "../Source/Utils/PitchToolOperations.h"
#include "TestAssert.h"
#include <cmath>
#include <iostream>
#include <numeric>

using PitchToolOperations::applyAllTransformations;

int main()
{
    const std::vector<float> flat(101, 0.0f);
    const auto near = [](float a, float b) { return std::abs(a - b) < 1.0e-4f; };

    // Left tilt about a pivot at 30%: left edge +2, pivot and right side 0.
    {
        const auto t = applyAllTransformations(flat, 2.0f, 0.0f, 1.0f, 0, 0, {}, 1.0f, 0.3f);
        CHECK(near(t[0], 2.0f));
        CHECK(near(t[30], 0.0f));
        CHECK(near(t[15], 1.0f));
        CHECK(near(t[100], 0.0f) && near(t[60], 0.0f));
    }
    // Right tilt about the middle: right edge -1, left half untouched.
    {
        const auto t = applyAllTransformations(flat, 0.0f, -1.0f, 1.0f, 0, 0, {}, 1.0f, 0.5f);
        CHECK(near(t[100], -1.0f) && near(t[75], -0.5f));
        CHECK(near(t[0], 0.0f) && near(t[50], 0.0f));
    }
    // Rotation about 25%: left = -k p, right = k (1 - p) is one straight line.
    {
        const float p = 0.25f, k = 2.0f;
        const auto t = applyAllTransformations(flat, -k * p, k * (1.0f - p), 1.0f, 0, 0, {},
                                               1.0f, p);
        for (int i = 0; i <= 100; ++i)
            CHECK(near(t[static_cast<size_t>(i)], k * (i / 100.0f - p)));
    }
    // The mean offset, which keeps the note's centre honest.
    {
        const float l = 1.5f, r = -0.7f, p = 0.4f;
        const auto t = applyAllTransformations(flat, l, r, 1.0f, 0, 0, {}, 1.0f, p);
        const float mean = std::accumulate(t.begin(), t.end(), 0.0f) / t.size();
        CHECK(std::abs(mean - Note::tiltMeanFor(l, r, p)) < 0.02f);
    }
    // The pivot is kept with the note and clamped to it.
    {
        Note note(0, 10, 60.0f);
        CHECK(near(note.getTiltPivot(), 0.5f));
        note.setTiltPivot(1.4f);
        CHECK(near(note.getTiltPivot(), 1.0f));
    }
    // A tilt frozen when the pivot moves sings exactly what it did.
    {
        const auto live = applyAllTransformations(flat, 1.2f, -0.4f, 1.0f, 0, 0, {}, 1.0f, 0.35f);
        const auto frozen = PitchToolOperations::tiltOffsets(flat.size(), 1.2f, -0.4f, 0.35f);
        const auto moved = applyAllTransformations(flat, 0.0f, 0.0f, 1.0f, 0, 0, {}, 1.0f, 0.8f,
                                                   &frozen);
        for (size_t i = 0; i < flat.size(); ++i)
            CHECK(near(live[i], moved[i]));
        // Laid over a note stretched to twice its length, the shape keeps.
        const std::vector<float> longer(201, 0.0f);
        const auto stretched = applyAllTransformations(longer, 0.0f, 0.0f, 1.0f, 0, 0, {}, 1.0f,
                                                       0.5f, &frozen);
        CHECK(near(stretched[0], live[0]) && near(stretched[200], live[100]) &&
              near(stretched[70], live[35]));
    }
    std::cout << "Tilt about a pivot passed\n";
}
