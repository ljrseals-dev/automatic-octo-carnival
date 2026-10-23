#include "PresetManager.h"

PresetManager::PresetManager()
{
    buildFactoryPresets();
}

// ------------------------------------------------------------------
// Factory presets — 8 curated latent vectors
// ------------------------------------------------------------------
void PresetManager::buildFactoryPresets()
{
    // Latent layout:
    //   [0..7]   pitch/gesture (populated live)
    //   [8..15]  timbre (populated live)
    //   [16..23] expression macros
    //   [24..27] transcription
    //   [28..31] reserved
    //
    // For presets, we only set dims 16..24 (and use dims 28/29 for
    // waveguide freq/damp since they're reserved).
    auto add = [&](const juce::String& name,
                   std::array<float, 8> expr,
                   float wgFreq, float wgDamp)
    {
        Preset p;
        p.name = name;
        for (int i = 0; i < 8; ++i) p.latent[16 + i] = expr[(size_t) i];
        p.latent[28] = wgFreq / 1200.0f;  // normalized
        p.latent[29] = wgDamp;
        builtins.push_back(p);
    };

    add("Init",           { 0.5f, 0.3f, 0.2f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f }, 196.0f, 0.995f);
    add("Fender Clean",   { 0.8f, 0.1f, 0.2f, 0.7f, 0.2f, 0.4f, 0.6f, 0.1f }, 196.0f, 0.998f);
    add("Marshall Crunch",{ 0.5f, 0.4f, 0.1f, 0.5f, 0.9f, 0.3f, 0.4f, 0.4f }, 196.0f, 0.995f);
    add("Vox Chime",      { 0.6f, 0.3f, 0.1f, 0.9f, 0.4f, 0.5f, 0.5f, 0.2f }, 220.0f, 0.997f);
    add("Mesa Lead",      { 0.3f, 0.6f, 0.1f, 0.4f, 1.0f, 0.6f, 0.3f, 0.6f }, 196.0f, 0.990f);
    add("Acoustic Warm",  { 0.6f, 0.2f, 0.6f, 0.6f, 0.2f, 0.4f, 0.7f, 0.2f }, 196.0f, 0.999f);
    add("Lo-Fi Grit",     { 0.4f, 0.5f, 0.3f, 0.3f, 0.7f, 0.3f, 0.4f, 0.5f }, 150.0f, 0.985f);
    add("Ambient Pad",    { 0.5f, 0.7f, 0.5f, 0.7f, 0.3f, 0.8f, 0.5f, 0.3f }, 130.0f, 0.999f);
}

// ------------------------------------------------------------------
// Preset names
// ------------------------------------------------------------------
juce::StringArray PresetManager::getPresetNames() const
{
    juce::StringArray names;
    for (auto& p : builtins) names.add(p.name);
    return names;
}

// ------------------------------------------------------------------
// Load builtin
// ------------------------------------------------------------------
void PresetManager::loadPresetByIndex(int idx, AuraGEngine& engine) const
{
    if (idx < 0 || idx >= (int) builtins.size()) return;
    engine.writeLatent(builtins[(size_t) idx].latent.data());
}

// ------------------------------------------------------------------
// Save to file
// ------------------------------------------------------------------
bool PresetManager::savePresetToFile(const juce::File& file,
                                     const AuraGEngine& engine) const
{
    float latent[AuraGEngine::kLatentDim];
    engine.readLatent(latent);

    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    obj->setProperty("name", file.getFileNameWithoutExtension());
    obj->setProperty("version", "0.1.0");

    juce::Array<juce::var> arr;
    for (int i = 0; i < AuraGEngine::kLatentDim; ++i)
        arr.add((double) latent[i]);
    obj->setProperty("latent", juce::var(arr));

    juce::var root(obj.get());
    auto json = juce::JSON::toString(root, true);

    juce::TemporaryFile tmp(file);
    if (!tmp.getFile().replaceWithText(json)) return false;
    return tmp.overwriteTargetFileWithTemporary();
}

// ------------------------------------------------------------------
// Load from file
// ------------------------------------------------------------------
bool PresetManager::loadPresetFromFile(const juce::File& file,
                                       AuraGEngine& engine) const
{
    if (!file.existsAsFile()) return false;
    auto parsed = juce::JSON::parse(file.loadFileAsString());
    if (!parsed.isObject()) return false;

    auto* obj = parsed.getDynamicObject();
    if (!obj) return false;

    auto latentVar = obj->getProperty("latent");
    if (!latentVar.isArray()) return false;

    auto* arr = latentVar.getArray();
    if (arr->size() != AuraGEngine::kLatentDim) return false;

    std::array<float, AuraGEngine::kLatentDim> latent{};
    for (int i = 0; i < AuraGEngine::kLatentDim; ++i)
        latent[(size_t) i] = (float) (double) arr->getUnchecked(i);

    engine.writeLatent(latent.data());
    return true;
}
