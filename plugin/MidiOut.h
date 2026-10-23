#pragma once
#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <vector>

/**
 * MidiOut — a lock-free-ish MIDI event queue.
 *
 * The transcription engine (or preset morphing) pushes note events;
 * processBlock drains them into the outgoing MidiBuffer.
 */
class MidiOut
{
public:
    struct Event
    {
        int sampleOffset;   // samples from block start
        int channel;        // 1..16
        int note;           // 0..127
        int velocity;       // 0..127
        bool isNoteOn;
    };

    void prepare(double sampleRate);
    void reset();

    void pushNoteOn (int note, int velocity, int channel = 1, int sampleOffset = 0);
    void pushNoteOff(int note, int velocity = 0, int channel = 1, int sampleOffset = 0);

    // Called from processBlock; appends pending events to midi, clears queue
    void drainInto(juce::MidiBuffer& midi, int blockSize, double sampleRate);

private:
    static constexpr int kMaxPending = 256;

    struct Slot
    {
        std::atomic<int>  sampleOffset { -1 };
        std::atomic<int>  channel      { 0 };
        std::atomic<int>  note         { 0 };
        std::atomic<int>  velocity     { 0 };
        std::atomic<bool> isNoteOn     { false };
    };

    std::array<Slot, kMaxPending> slots;
    std::atomic<int> head { 0 };
    std::atomic<int> tail { 0 };

    double currentSampleRate = 48000.0;
};
