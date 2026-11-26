#pragma once

#include "../JuceHeader.h"
#include "../Models/Project.h"
#include "../Utils/Localization.h"
#include "Components/StyledWidgets.h"
#include <functional>
#include <optional>

class ScaleSelectionControl : public CompactSelectionButton
{
public:
    ScaleSelectionControl();

    void setProject(Project* projectToUse);
    void showPopup();
    void setProjectKeyScope(bool value) { projectKeyScope = value; refreshLabel(); }
    Project* getProject() const { return project; }
    void setScaleRoot(int rootNote);
    /** In a song that changes key (spec section 15): the stretch the
     *  playhead is in, and its root. The label shows that root, and a root
     *  chosen from the menu is that stretch's (said through
     *  onStretchRootChosen). 0: the first stretch, the project's own key. */
    void setStretch(int index, int root);
    std::function<void(int stretch, int root)> onStretchRootChosen;
    void setScaleMode(ScaleMode mode);
    int getPreferredWidth() const;

    std::function<void(int)> onScaleRootChanged;
    std::function<void(ScaleMode)> onScaleModeChanged;
    std::function<void(std::optional<int>)> onScaleRootPreviewChanged;
    std::function<void(std::optional<ScaleMode>)> onScaleModePreviewChanged;
    std::function<void()> onPreferredWidthChanged;

    void previewScaleRoot(std::optional<int> rootNote);
    void previewScaleMode(std::optional<ScaleMode> mode);

    /** Re-reads the scale name after a language change. */
    void refreshLocalisedText();

private:
    void refreshLabel();

    Project* project = nullptr;
    bool projectKeyScope = false;
    int stretchIndex = 0, stretchRoot = 0;

    LocalisationWatcher languageWatcher{[this] { refreshLocalisedText(); }};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ScaleSelectionControl)
};
