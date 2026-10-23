#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include "PluginProcessor.h"
#include "PluginLookAndFeel.h"

class AuraGEditor : public juce::AudioProcessorEditor,
                    private juce::Timer
{
public:
    explicit AuraGEditor(AuraGProcessor&);
    ~AuraGEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;
    void setupSlider(juce::Slider&, juce::Label&, const juce::String& label);
    void setupCombo(juce::ComboBox&, juce::Label&, const juce::String& label);

    AuraGProcessor& proc;
    AuraGLookAndFeel lookAndFeel;

    // Main knobs
    juce::Slider ampGain, fxMix, physMix, drive, sag, sat;
    juce::Slider toneBass, toneMid, toneTreble;
    juce::Slider wgFreq, wgDamp;

    // Labels
    juce::Label ampGainL, fxMixL, physMixL, driveL, sagL, satL;
    juce::Label toneBassL, toneMidL, toneTrebleL;
    juce::Label wgFreqL, wgDampL;

    // Preset
    juce::ComboBox presetBox;
    juce::Label    presetLabel;
    juce::TextButton savePresetBtn { "Save" };
    juce::TextButton loadPresetBtn { "Load" };

    // Expression macros
    juce::Slider expr[8];
    juce::Label  exprL[8];

    // Header
    juce::Label titleLabel;

    // Attachments
    using SA = juce::AudioProcessorValueTreeState::SliderAttachment;
    using CA = juce::AudioProcessorValueTreeState::ComboBoxAttachment;
    std::unique_ptr<SA> ampGainA, fxMixA, physMixA, driveA, sagA, satA;
    std::unique_ptr<SA> toneBassA, toneMidA, toneTrebleA;
    std::unique_ptr<SA> wgFreqA, wgDampA;
    std::vector<std::unique_ptr<SA>> exprA;

    // File chooser
    std::unique_ptr<juce::FileChooser> chooser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AuraGEditor)
};
