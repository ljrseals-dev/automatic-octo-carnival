#pragma once
#include <juce_core/juce_core.h>
#include "AuraGEngine.h"

class PresetManager
{
public:
    PresetManager();

    juce::StringArray getPresetNames() const;

    // Load built-in preset by index (matches getPresetNames order)
    void loadPresetByIndex(int idx, AuraGEngine& engine) const;

    // File I/O
    bool savePresetToFile(const juce::File&, const AuraGEngine& engine) const;
    bool loadPresetFromFile(const juce::File&, AuraGEngine& engine) const;

private:
    struct Preset
    {
        juce::String name;
        // 32-float latent vector
        std::array<float, AuraGEngine::kLatentDim> latent{};
    };

    std::vector<Preset> builtins;

    void buildFactoryPresets();
};
