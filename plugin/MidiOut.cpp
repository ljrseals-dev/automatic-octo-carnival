#include "MidiOut.h"

void MidiOut::prepare(double sr)
{
    currentSampleRate = sr > 0.0 ? sr : 48000.0;
    reset();
}

void MidiOut::reset()
{
    head.store(0);
    tail.store(0);
    for (auto& s : slots)
    {
        s.sampleOffset.store(-1);
        s.channel.store(0);
        s.note.store(0);
        s.velocity.store(0);
        s.isNoteOn.store(false);
    }
}

void MidiOut::pushNoteOn(int note, int velocity, int channel, int sampleOffset)
{
    const int h = head.load(std::memory_order_relaxed);
    const int t = tail.load(std::memory_order_acquire);
    if (((h + 1) % kMaxPending) == t) return;  // full, drop

    auto& s = slots[(size_t) h];
    s.sampleOffset.store(sampleOffset, std::memory_order_relaxed);
    s.channel.store(channel, std::memory_order_relaxed);
    s.note.store(note & 0x7F, std::memory_order_relaxed);
    s.velocity.store(velocity & 0x7F, std::memory_order_relaxed);
    s.isNoteOn.store(true, std::memory_order_relaxed);

    head.store((h + 1) % kMaxPending, std::memory_order_release);
}

void MidiOut::pushNoteOff(int note, int velocity, int channel, int sampleOffset)
{
    const int h = head.load(std::memory_order_relaxed);
    const int t = tail.load(std::memory_order_acquire);
    if (((h + 1) % kMaxPending) == t) return;

    auto& s = slots[(size_t) h];
    s.sampleOffset.store(sampleOffset, std::memory_order_relaxed);
    s.channel.store(channel, std::memory_order_relaxed);
    s.note.store(note & 0x7F, std::memory_order_relaxed);
    s.velocity.store(velocity & 0x7F, std::memory_order_relaxed);
    s.isNoteOn.store(false, std::memory_order_relaxed);

    head.store((h + 1) % kMaxPending, std::memory_order_release);
}

void MidiOut::drainInto(juce::MidiBuffer& midi, int blockSize, double /*sr*/)
{
    int t = tail.load(std::memory_order_relaxed);
    const int h = head.load(std::memory_order_acquire);

    while (t != h)
    {
        auto& s = slots[(size_t) t];
        const int off = juce::jlimit(0, blockSize - 1, s.sampleOffset.load());
        const int ch  = juce::jlimit(1, 16, s.channel.load());
        const int nt  = s.note.load();
        const int vel = s.velocity.load();
        const bool on = s.isNoteOn.load();

        const auto msg = on
            ? juce::MidiMessage::noteOn (ch, nt, (juce::uint8) vel)
            : juce::MidiMessage::noteOff(ch, nt, (juce::uint8) vel);
        midi.addEvent(msg, off);

        t = (t + 1) % kMaxPending;
    }
    tail.store(h, std::memory_order_release);
}
