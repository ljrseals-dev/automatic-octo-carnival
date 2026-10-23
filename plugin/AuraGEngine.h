#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <vector>

/**
 * AuraGEngine — C++ port of the AURA-G audio path.
 *
 * Signal chain:
 *   input -> DDSP amp -> waveguide excitation -> mixer -> output
 *
 * Also maintains a small LatentBus (32 floats) mirroring the Python
 * implementation, so upstream (UI / preset) code can drive all modules
 * from a single control vector.
 *
 * Thread-safety:
 *   - Audio-thread-only state lives in private members.
 *   - Parameter setters use atomics so the message thread can post values
 *     without locks.
 */
class AuraGEngine
{
public:
    static constexpr int   kLatentDim   = 32;
    static constexpr int   kMaxDelay    = 2048;
    static constexpr float kTwoPi       = 6.28318530718f;

    // ---- Lifecycle ----
    void prepare(double sampleRate, int maxBlockSize) noexcept;
    void reset() noexcept;

    // ---- Parameter setters (message-thread safe) ----
    void setAmpGain(float v)      noexcept { ampGain.store(jlimit(0.0f, 1.0f, v)); }
    void setFxMix(float v)        noexcept { fxMix.store(jlimit(0.0f, 1.0f, v)); }
    void setPhysicalMix(float v)  noexcept { physMix.store(jlimit(0.0f, 1.0f, v)); }
    void setDrive(float v)        noexcept { drive.store(jlimit(0.0f, 4.0f, v)); }
    void setSag(float v)          noexcept { sag.store(jlimit(0.0f, 1.0f, v)); }
    void setSaturation(float v)   noexcept { saturation.store(jlimit(0.0f, 1.0f, v)); }
    void setToneBass(float v)     noexcept { toneBass.store(jlimit(-12.0f, 12.0f, v)); }
    void setToneMid(float v)      noexcept { toneMid.store(jlimit(-12.0f, 12.0f, v)); }
    void setToneTreble(float v)   noexcept { toneTreble.store(jlimit(-12.0f, 12.0f, v)); }
    void setWaveguideFreq(float v) noexcept { wgFreq.store(jlimit(60.0f, 1200.0f, v)); }
    void setWaveguideDamp(float v) noexcept { wgDamp.store(jlimit(0.5f, 0.999f, v)); }
    void setExpression(int idx, float v) noexcept
    {
        if (idx >= 0 && idx < 8)
            expression[idx].store(jlimit(0.0f, 1.0f, v));
    }

    // ---- Latent bus I/O ----
    void readLatent(float* dst) const noexcept;
    void writeLatent(const float* src) noexcept;

    // ---- Audio processing ----
    void processBlock(float* const* channels, int numChannels, int numSamples) noexcept;

private:
    // ---- DSP helpers ----
    struct Biquad
    {
        float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f, a1 = 0.0f, a2 = 0.0f;
        float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f;

        inline float process(float x) noexcept
        {
            const float w = x - a1 * y1 - a2 * y2;
            const float y = b0 * w + b1 * x1 + b2 * x2;
            x2 = x1; x1 = w;
            y2 = y1; y1 = y;
            return y;
        }
        void reset() noexcept { x1 = x2 = y1 = y2 = 0.0f; }
    };

    void updateBiquads() noexcept;

    // ---- State ----
    double sampleRate = 48000.0;
    int    maxBlock   = 512;

    std::atomic<float> ampGain{0.5f};
    std::atomic<float> fxMix{0.3f};
    std::atomic<float> physMix{0.2f};
    std::atomic<float> drive{1.0f};
    std::atomic<float> sag{0.3f};
    std::atomic<float> saturation{0.7f};
    std::atomic<float> toneBass{0.0f};
    std::atomic<float> toneMid{0.0f};
    std::atomic<float> toneTreble{0.0f};
    std::atomic<float> wgFreq{196.0f};
    std::atomic<float> wgDamp{0.995f};
    std::array<std::atomic<float>, 8> expression{};

    // Latent bus mirror (32 floats)
    std::array<std::atomic<float>, kLatentDim> latent{};

    // Tone stack biquads
    Biquad bassEq, midEq, trebleEq;
    float  lastBassDb = 0.0f, lastMidDb = 0.0f, lastTrebleDb = 0.0f;

    // Power amp envelope
    float envState = 0.0f;

    // Waveguide
    std::vector<float> delayLine;
    int    delayLen  = 245;
    int    writePos  = 0;
    float  lastSample = 0.0f;

    // Output transformer
    float hpX1 = 0.0f, hpY1 = 0.0f;
    float lpX1 = 0.0f, lpY1 = 0.0f;

    static inline float jlimit(float lo, float hi, float v) noexcept
    {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};
