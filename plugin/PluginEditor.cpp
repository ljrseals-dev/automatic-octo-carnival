#include "PluginEditor.h"

// ------------------------------------------------------------------
// Construction
// ------------------------------------------------------------------
AuraGEditor::AuraGEditor(AuraGProcessor& p)
    : AudioProcessorEditor(&p), proc(p)
{
    setLookAndFeel(&lookAndFeel);

    // ---- Header ----
    titleLabel.setText("AURA-G", juce::dontSendNotification);
    titleLabel.setJustificationType(juce::Justification::centred);
    titleLabel.setFont(juce::Font(28.0f, juce::Font::bold));
    addAndMakeVisible(titleLabel);

    // ---- Main knobs ----
    setupSlider(ampGain,  ampGainL,  "Amp Gain");
    setupSlider(fxMix,    fxMixL,    "FX Mix");
    setupSlider(physMix,  physMixL,  "Phys Mix");
    setupSlider(drive,    driveL,    "Drive");
    setupSlider(sag,      sagL,      "Sag");
    setupSlider(sat,      satL,      "Saturation");

    // ---- Tone stack ----
    setupSlider(toneBass,   toneBassL,   "Bass");
    setupSlider(toneMid,    toneMidL,    "Mid");
    setupSlider(toneTreble, toneTrebleL, "Treble");

    // ---- Waveguide ----
    setupSlider(wgFreq, wgFreqL, "String Freq");
    setupSlider(wgDamp, wgDampL, "Damping");

    // ---- Presets ----
    presetLabel.setText("Preset", juce::dontSendNotification);
    presetLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(presetLabel);

    presetBox.addItem("Init", 1);
    const auto names = proc.presets().getPresetNames();
    for (int i = 0; i < (int) names.size(); ++i)
        presetBox.addItem(names[(size_t) i], i + 2);

    presetBox.onChange = [this]() {
        const int idx = presetBox.getSelectedId() - 2;
        if (idx < 0) return;
        proc.presets().loadPresetByIndex(idx, proc.engine());
        proc.syncParametersFromEngine();
    };
    addAndMakeVisible(presetBox);

    savePresetBtn.onClick = [this]() {
        chooser = std::make_unique<juce::FileChooser>(
            "Save preset as...",
            juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
                .getChildFile("aura-g-presets"),
            "*.json");
        chooser->launchAsync(
            juce::FileBrowserComponent::saveMode |
            juce::FileBrowserComponent::canSelectFiles,
            [this](const juce::FileChooser& fc) {
                auto f = fc.getResult();
                if (f.getFullPathName().isEmpty()) return;
                if (!f.hasFileExtension("json"))
                    f = f.withFileExtension("json");
                proc.presets().savePresetToFile(f, proc.engine());
            });
    };
    addAndMakeVisible(savePresetBtn);

    loadPresetBtn.onClick = [this]() {
        chooser = std::make_unique<juce::FileChooser>(
            "Load preset...",
            juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
            "*.json");
        chooser->launchAsync(
            juce::FileBrowserComponent::openMode |
            juce::FileBrowserComponent::canSelectFiles,
            [this](const juce::FileChooser& fc) {
                auto f = fc.getResult();
                if (!f.existsAsFile()) return;
                proc.presets().loadPresetFromFile(f, proc.engine());
                proc.syncParametersFromEngine();
            });
    };
    addAndMakeVisible(loadPresetBtn);

    // ---- Expression macros ----
    for (int i = 0; i < 8; ++i)
    {
        setupSlider(expr[i], exprL[i], "Expr " + juce::String(i + 1));
    }

    // ---- APVTS attachments ----
    auto& s = proc.apvts();
    ampGainA    = std::make_unique<SA>(s, "amp_gain",    ampGain);
    fxMixA      = std::make_unique<SA>(s, "fx_mix",      fxMix);
    physMixA    = std::make_unique<SA>(s, "phys_mix",    physMix);
    driveA      = std::make_unique<SA>(s, "drive",       drive);
    sagA        = std::make_unique<SA>(s, "sag",         sag);
    satA        = std::make_unique<SA>(s, "saturation",  sat);
    toneBassA   = std::make_unique<SA>(s, "tone_bass",   toneBass);
    toneMidA    = std::make_unique<SA>(s, "tone_mid",    toneMid);
    toneTrebleA = std::make_unique<SA>(s, "tone_treble", toneTreble);
    wgFreqA     = std::make_unique<SA>(s, "wg_freq",     wgFreq);
    wgDampA     = std::make_unique<SA>(s, "wg_damp",     wgDamp);

    exprA.reserve(8);
    for (int i = 0; i < 8; ++i)
    {
        const auto id = "expr_" + std::to_string(i);
        exprA.push_back(std::make_unique<SA>(s, id.c_str(), expr[i]));
    }

    setSize(720, 560);
    startTimerHz(15);
}

AuraGEditor::~AuraGEditor()
{
    stopTimer();
    setLookAndFeel(nullptr);
}

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------
void AuraGEditor::setupSlider(juce::Slider& sl, juce::Label& lb,
                              const juce::String& text)
{
    sl.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    sl.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 20);
    addAndMakeVisible(sl);

    lb.setText(text, juce::dontSendNotification);
    lb.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(lb);
}

// ------------------------------------------------------------------
// Painting
// ------------------------------------------------------------------
void AuraGEditor::paint(juce::Graphics& g)
{
    // Background
    juce::ColourGradient bg(juce::Colour(0xff1a1a1f), 0.0f, 0.0f,
                            juce::Colour(0xff0a0a0d),
                            0.0f, (float) getHeight(), false);
    g.setGradientFill(bg);
    g.fillAll();

    // Panel outlines
    g.setColour(juce::Colour(0xff2a2a33));
    g.drawRoundedRectangle(getLocalBounds().reduced(4).toFloat(), 8.0f, 1.0f);

    // Title underline
    g.setColour(juce::Colour(0xff7aa2ff));
    g.drawLine(20.0f, 46.0f, (float) getWidth() - 20.0f, 46.0f, 2.0f);
}

// ------------------------------------------------------------------
// Layout
// ------------------------------------------------------------------
void AuraGEditor::resized()
{
    auto r = getLocalBounds().reduced(12);
    r.removeFromTop(40);                     // header

    titleLabel.setBounds(r.getX(), 8, r.getWidth(), 36);

    // ---- Preset row ----
    auto presetRow = r.removeFromTop(32);
    presetLabel.setBounds(presetRow.removeFromLeft(60));
    presetBox  .setBounds(presetRow.removeFromLeft(220));
    presetRow.removeFromLeft(10);
    savePresetBtn.setBounds(presetRow.removeFromLeft(70));
    presetRow.removeFromLeft(6);
    loadPresetBtn.setBounds(presetRow.removeFromLeft(70));
    r.removeFromTop(10);

    auto placeRow = [&](juce::Rectangle<int> row,
                        std::initializer_list<std::pair<juce::Slider*,
                                                       juce::Label*>> items)
    {
        const int n = (int) items.size();
        const int w = row.getWidth() / juce::jmax(1, n);
        for (auto& [sl, lb] : items)
        {
            auto cell = row.removeFromLeft(w);
            lb->setBounds(cell.removeFromTop(18));
            sl->setBounds(cell.reduced(4));
        }
    };

    // ---- Row 1: main knobs ----
    placeRow(r.removeFromTop(110),
             { {&ampGain, &ampGainL}, {&fxMix, &fxMixL}, {&physMix, &physMixL},
               {&drive,   &driveL},   {&sag,   &sagL},   {&sat,     &satL} });
    r.removeFromTop(6);

    // ---- Row 2: tone stack + waveguide ----
    placeRow(r.removeFromTop(110),
             { {&toneBass, &toneBassL}, {&toneMid, &toneMidL},
               {&toneTreble, &toneTrebleL}, {&wgFreq, &wgFreqL},
               {&wgDamp, &wgDampL} });
    r.removeFromTop(6);

    // ---- Row 3: expression macros ----
    placeRow(r.removeFromTop(110),
             { {&expr[0], &exprL[0]}, {&expr[1], &exprL[1]},
               {&expr[2], &exprL[2]}, {&expr[3], &exprL[3]},
               {&expr[4], &exprL[4]}, {&expr[5], &exprL[5]},
               {&expr[6], &exprL[6]}, {&expr[7], &exprL[7]} });
}

// ------------------------------------------------------------------
// Timer (update UI on preset load)
// ------------------------------------------------------------------
void AuraGEditor::timerCallback()
{
    // Light-touch refresh: nothing heavy, just a redraw trigger
    repaint();
}
