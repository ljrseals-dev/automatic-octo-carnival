#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include "AuraGEngine.h"
#include "PresetManager.h"
#include "MidiOut.h"

class AuraGProcessor : public juce::AudioProcessor
{
public:
    AuraGProcessor();
    ~AuraGProcessor() override;

    // ---- AudioProcessor overrides ----
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "AURA-G"; }
    bool acceptsMidi()  const override { return true; }
    bool producesMidi() const override { return true; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 2.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return "Default"; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock&) override;
    void setStateInformation(const void*, int) override;

    // ---- AURA-G public API ----
    AuraGEngine&         engine()         { return engineInstance; }
    PresetManager&       presets()        { return presetManager; }
    MidiOut&             midiOut()        { return midiOutInstance; }
    juce::AudioProcessorValueTreeState& apvts() { return apvtsState; }

    // Called by the editor when a preset is loaded
    void syncParametersFromEngine();

    // Called by the editor when the user tweaks any slider
    void syncEngineFromParameters();

private:
    void applyParameterToEngine(const juce::String& id, float value);

    AuraGEngine engineInstance;
    PresetManager presetManager;
    MidiOut midiOutInstance;

    juce::AudioProcessorValueTreeState apvtsState;

    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AuraGProcessor)
};
