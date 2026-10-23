#include "PluginProcessor.h"
#include "PluginEditor.h"

// ------------------------------------------------------------------
// Parameter layout
// ------------------------------------------------------------------
juce::AudioProcessorValueTreeState::ParameterLayout
AuraGProcessor::createLayout()
{
    using P = juce::AudioParameterFloat;
    using R = juce::NormalisableRange<float>;

    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    auto make = [&](const char* id, const char* name,
                    float lo, float hi, float def, float skew = 1.0f)
    {
        R range(lo, hi, 0.001f);
        if (skew != 1.0f) range.setSkewForCentre(def);
        params.push_back(std::make_unique<P>(id, name, range, def));
    };

    make("amp_gain",    "Amp Gain",     0.0f, 1.0f, 0.5f);
    make("fx_mix",      "FX Mix",       0.0f, 1.0f, 0.3f);
    make("phys_mix",    "Physical Mix", 0.0f, 1.0f, 0.2f);
    make("drive",       "Drive",        0.0f, 4.0f, 1.0f);
    make("sag",         "Sag",          0.0f, 1.0f, 0.3f);
    make("saturation",  "Saturation",   0.0f, 1.0f, 0.7f);
    make("tone_bass",   "Bass",       -12.0f, 12.0f, 0.0f);
    make("tone_mid",    "Mid",        -12.0f, 12.0f, 0.0f);
    make("tone_treble", "Treble",     -12.0f, 12.0f, 0.0f);
    make("wg_freq",     "String Freq", 60.0f, 1200.0f, 196.0f);
    make("wg_damp",     "Damping",      0.5f, 0.999f, 0.995f);

    for (int i = 0; i < 8; ++i)
    {
        const auto id = "expr_" + std::to_string(i);
        const auto nm = "Expression " + std::to_string(i);
        make(id.c_str(), nm.c_str(), 0.0f, 1.0f, 0.5f);
    }

    return { params.begin(), params.end() };
}

// ------------------------------------------------------------------
// Construction
// ------------------------------------------------------------------
AuraGProcessor::AuraGProcessor()
    : AudioProcessor(BusesProperties()
        .withInput ("Input",  juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvtsState(*this, nullptr, "AURAG_STATE", createLayout())
{
    // Sync engine to initial parameter values
    syncEngineFromParameters();
}

AuraGProcessor::~AuraGProcessor() = default;

// ------------------------------------------------------------------
// Audio lifecycle
// ------------------------------------------------------------------
void AuraGProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    engineInstance.prepare(sampleRate, samplesPerBlock);
    midiOutInstance.prepare(sampleRate);
}

void AuraGProcessor::releaseResources()
{
    engineInstance.reset();
}

// ------------------------------------------------------------------
// processBlock
// ------------------------------------------------------------------
void AuraGProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                  juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals noDenormals;

    const int n  = buffer.getNumSamples();
    const int ch = buffer.getNumChannels();

    // Push current parameter values to the engine
    syncEngineFromParameters();

    // Run DSP
    engineInstance.processBlock(buffer.getArrayOfWritePointers(), ch, n);

    // Emit MIDI from the transcription engine if any events pending
    midiOutInstance.drainInto(midi, n, getSampleRate());
}

// ------------------------------------------------------------------
// Engine <-> parameters sync
// ------------------------------------------------------------------
void AuraGProcessor::applyParameterToEngine(const juce::String& id, float v)
{
    auto& e = engineInstance;
    if      (id == "amp_gain")    e.setAmpGain(v);
    else if (id == "fx_mix")      e.setFxMix(v);
    else if (id == "phys_mix")    e.setPhysicalMix(v);
    else if (id == "drive")       e.setDrive(v);
    else if (id == "sag")         e.setSag(v);
    else if (id == "saturation")  e.setSaturation(v);
    else if (id == "tone_bass")   e.setToneBass(v);
    else if (id == "tone_mid")    e.setToneMid(v);
    else if (id == "tone_treble") e.setToneTreble(v);
    else if (id == "wg_freq")     e.setWaveguideFreq(v);
    else if (id == "wg_damp")     e.setWaveguideDamp(v);
    else if (id.startsWith("expr_"))
    {
        const int idx = id.fromLastOccurrenceOf("_", false, false).getIntValue();
        e.setExpression(idx, v);
    }
}

void AuraGProcessor::syncEngineFromParameters()
{
    for (auto* p : getParameters())
    {
        if (auto* rp = dynamic_cast<juce::RangedAudioParameter*>(p))
        {
            const auto id = rp->paramID;
            const auto v  = rp->getValue();   // 0..1 normalized
            // Convert to actual value using the parameter's range
            float actual = v;
            if (auto* fp = dynamic_cast<juce::AudioParameterFloat*>(rp))
                actual = fp->convertFrom0to1(v);
            applyParameterToEngine(id, actual);
        }
    }
}

void AuraGProcessor::syncParametersFromEngine()
{
    // Called by the editor after preset load; pushes engine values
    // back into the APVTS so UI sliders reflect the new state.
    float latent[AuraGEngine::kLatentDim];
    engineInstance.readLatent(latent);

    auto setParam = [&](const char* id, float actual) {
        if (auto* p = apvtsState.getParameter(id))
        {
            const float norm = p->convertTo0to1(actual);
            p->setValueNotifyingHost(norm);
        }
    };

    setParam("amp_gain",    latent[16]);
    setParam("fx_mix",      latent[17]);
    setParam("phys_mix",    latent[18]);
    setParam("drive",       latent[19] * 4.0f);
    setParam("sag",         latent[20]);
    setParam("saturation",  latent[21]);
    setParam("tone_bass",   (latent[22] - 0.5f) * 24.0f);
    setParam("tone_mid",    (latent[23] - 0.5f) * 24.0f);
    setParam("tone_treble", (latent[24] - 0.5f) * 24.0f);
}

// ------------------------------------------------------------------
// Editor / state
// ------------------------------------------------------------------
juce::AudioProcessorEditor* AuraGProcessor::createEditor()
{
    return new AuraGEditor(*this);
}

void AuraGProcessor::getStateInformation(juce::MemoryBlock& dest)
{
    auto state = apvtsState.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, dest);
}

void AuraGProcessor::setStateInformation(const void* data, int size)
{
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, size));
    if (xml && xml->hasTagName(apvtsState.state.getType()))
    {
        apvtsState.replaceState(juce::ValueTree::fromXml(*xml));
        syncEngineFromParameters();
    }
}

// ------------------------------------------------------------------
// Plugin instantiation
// ------------------------------------------------------------------
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AuraGProcessor();
}
