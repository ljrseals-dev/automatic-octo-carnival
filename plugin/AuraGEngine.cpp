#include "AuraGEngine.h"

// ------------------------------------------------------------------
// Lifecycle
// ------------------------------------------------------------------
void AuraGEngine::prepare(double sr, int maxBlockSize) noexcept
{
    sampleRate = sr > 0.0 ? sr : 48000.0;
    maxBlock   = maxBlockSize > 0 ? maxBlockSize : 512;

    const int dl = juce::jlimit(2, kMaxDelay,
                                (int) std::round(sampleRate / 60.0));
    delayLine.assign((size_t) dl, 0.0f);
    delayLen  = dl;
    writePos  = 0;
    lastSample = 0.0f;

    bassEq.reset();
    midEq.reset();
    trebleEq.reset();
    envState = 0.0f;
    hpX1 = hpY1 = lpX1 = lpY1 = 0.0f;

    updateBiquads();
}

void AuraGEngine::reset() noexcept
{
    std::fill(delayLine.begin(), delayLine.end(), 0.0f);
    writePos = 0;
    lastSample = 0.0f;
    bassEq.reset();
    midEq.reset();
    trebleEq.reset();
    envState = 0.0f;
    hpX1 = hpY1 = lpX1 = lpY1 = 0.0f;
}

// ------------------------------------------------------------------
// Biquad coefficient design (peaking EQ, RBJ cookbook)
// ------------------------------------------------------------------
static void designPeak(AuraGEngine::Biquad& q,
                       float sr, float f0, float Q, float gainDb)
{
    const float A     = std::pow(10.0f, gainDb / 40.0f);
    const float w0    = AuraGEngine::kTwoPi * f0 / sr;
    const float alpha = std::sin(w0) / (2.0f * Q);
    const float cosw  = std::cos(w0);

    const float b0 = 1.0f + alpha * A;
    const float b1 = -2.0f * cosw;
    const float b2 = 1.0f - alpha * A;
    const float a0 = 1.0f + alpha / A;
    const float a1 = -2.0f * cosw;
    const float a2 = 1.0f - alpha / A;

    q.b0 = b0 / a0;
    q.b1 = b1 / a0;
    q.b2 = b2 / a0;
    q.a1 = a1 / a0;
    q.a2 = a2 / a0;
}

void AuraGEngine::updateBiquads() noexcept
{
    const float b = toneBass.load();
    const float m = toneMid.load();
    const float t = toneTreble.load();

    if (b != lastBassDb)
    {
        designPeak(bassEq, (float) sampleRate, 100.0f, 0.7f, b);
        lastBassDb = b;
    }
    if (m != lastMidDb)
    {
        designPeak(midEq, (float) sampleRate, 800.0f, 1.0f, m);
        lastMidDb = m;
    }
    if (t != lastTrebleDb)
    {
        designPeak(trebleEq, (float) sampleRate, 3000.0f, 0.7f, t);
        lastTrebleDb = t;
    }
}

// ------------------------------------------------------------------
// Latent bus
// ------------------------------------------------------------------
void AuraGEngine::readLatent(float* dst) const noexcept
{
    for (int i = 0; i < kLatentDim; ++i)
        dst[i] = latent[i].load(std::memory_order_relaxed);
}

void AuraGEngine::writeLatent(const float* src) noexcept
{
    for (int i = 0; i < kLatentDim; ++i)
        latent[i].store(src[i], std::memory_order_relaxed);
}

// ------------------------------------------------------------------
// Audio processing
// ------------------------------------------------------------------
void AuraGEngine::processBlock(float* const* channels,
                               int numChannels,
                               int numSamples) noexcept
{
    if (numChannels <= 0 || numSamples <= 0) return;
    if (delayLine.empty()) reset();

    updateBiquads();

    // Snapshot params for this block
    const float gAmp  = ampGain.load();
    const float gFx   = fxMix.load();
    const float gPhys = physMix.load();
    const float drv   = drive.load();
    const float sg    = sag.load();
    const float sat   = saturation.load();
    const float fWg   = wgFreq.load();
    const float dWg   = wgDamp.load();

    // Normalize mixer weights so they sum to 1
    const float sum = gAmp + gFx + gPhys + 1e-6f;
    const float wAmp = gAmp  / sum;
    const float wFx  = gFx   / sum;
    const float wPhy = gPhys / sum;

    // Update waveguide delay length from current frequency
    const int wanted = juce::jlimit(2, kMaxDelay,
                                    (int) std::round(sampleRate / fWg));
    if (wanted != delayLen)
    {
        delayLine.assign((size_t) wanted, 0.0f);
        delayLen = wanted;
        writePos = 0;
        lastSample = 0.0f;
    }

    for (int i = 0; i < numSamples; ++i)
    {
        // ---- Input (mono-sum across channels) ----
        float x = 0.0f;
        for (int c = 0; c < numChannels; ++c)
            x += channels[c][i];
        x /= (float) numChannels;

        // ================= DDSP AMP =================
        // Preamp: tanh drive
        float pre = std::tanh(drv * x);

        // Tone stack: three peaking EQs
        pre = bassEq.process(pre);
        pre = midEq.process(pre);
        pre = trebleEq.process(pre);

        // Power amp: envelope-follower sag + soft clip
        const float absPre = std::fabs(pre);
        envState = 0.999f * envState + 0.001f * absPre;
        const float gainReduction = 1.0f / (1.0f + sg * envState);
        const float satPre = pre * gainReduction * sat;
        const float satPre3 = satPre * satPre * satPre;
        float ampOut = satPre - satPre3 * 0.333333f;
        if (ampOut >  1.0f) ampOut =  1.0f;
        if (ampOut < -1.0f) ampOut = -1.0f;

        // Output transformer: gentle HP + LP (one-pole each)
        const float hpAlpha = 0.995f;
        const float hpOut   = hpAlpha * (hpY1 + ampOut - hpX1);
        hpX1 = ampOut;
        hpY1 = hpOut;

        const float lpAlpha = 0.5f;
        const float lpOut   = lpY1 + lpAlpha * (hpOut - lpY1);
        lpY1 = lpOut;

        // ================= WAVEGUIDE =================
        const float delayed = delayLine[(size_t) writePos];
        const float filtered = 0.5f * (delayed + lastSample);
        delayLine[(size_t) writePos] = filtered + lpOut * 0.05f; // excite
        writePos = (writePos + 1) % delayLen;
        lastSample = filtered;
        const float physOut = dWg * delayed;

        // ================= MIXER =================
        // Simple one-pole FX path (placeholder for future RNN FX)
        static thread_local float fxState = 0.0f;
        fxState = 0.7f * fxState + 0.3f * lpOut;
        const float fxOut = fxState;

        float y = wAmp * lpOut + wFx * fxOut + wPhy * physOut;

        // Output safety clip
        if (y >  1.0f) y =  1.0f;
        if (y < -1.0f) y = -1.0f;

        // ---- Write to all channels ----
        for (int c = 0; c < numChannels; ++c)
            channels[c][i] = y;
    }
}
