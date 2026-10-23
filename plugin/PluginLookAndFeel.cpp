#include "PluginLookAndFeel.h"

AuraGLookAndFeel::AuraGLookAndFeel()
{
    setColour(juce::Slider::textBoxTextColourId,        juce::Colours::white);
    setColour(juce::Slider::textBoxOutlineColourId,     juce::Colours::transparentBlack);
    setColour(juce::Slider::textBoxBackgroundColourId,  juce::Colour(0xff0a0a0d));
    setColour(juce::Label::textColourId,                juce::Colour(0xffb8c4ff));
    setColour(juce::ComboBox::backgroundColourId,       juce::Colour(0xff15151c));
    setColour(juce::ComboBox::textColourId,             juce::Colours::white);
    setColour(juce::ComboBox::outlineColourId,          juce::Colour(0xff2a2a33));
    setColour(juce::ComboBox::arrowColourId,            juce::Colour(0xff7aa2ff));
    setColour(juce::TextButton::buttonColourId,         juce::Colour(0xff20202a));
    setColour(juce::TextButton::textColourOnId,         juce::Colours::white);
    setColour(juce::TextButton::textColourOffId,        juce::Colour(0xffb8c4ff));
}

void AuraGLookAndFeel::drawRotarySlider(juce::Graphics& g,
                                        int x, int y, int width, int height,
                                        float pos, float startAngle,
                                        float endAngle, juce::Slider&)
{
    auto bounds = juce::Rectangle<int>(x, y, width, height).toFloat().reduced(6.0f);
    const auto radius = juce::jmin(bounds.getWidth(), bounds.getHeight()) * 0.5f;
    const auto centre = bounds.getCentre();
    const auto angle  = startAngle + pos * (endAngle - startAngle);

    // Track
    juce::Path track;
    track.addCentredArc(centre.x, centre.y, radius - 4.0f, radius - 4.0f,
                        0.0f, startAngle, endAngle, true);
    g.setColour(juce::Colour(0xff2a2a33));
    g.strokePath(track, juce::PathStrokeType(4.0f, juce::PathStrokeType::curved,
                                             juce::PathStrokeType::rounded));

    // Value arc
    juce::Path value;
    value.addCentredArc(centre.x, centre.y, radius - 4.0f, radius - 4.0f,
                        0.0f, startAngle, angle, true);
    g.setColour(juce::Colour(0xff7aa2ff));
    g.strokePath(value, juce::PathStrokeType(4.0f, juce::PathStrokeType::curved,
                                             juce::PathStrokeType::rounded));

    // Knob body
    juce::ColourGradient knobGrad(juce::Colour(0xff33333d),
                                  centre.x, centre.y - radius + 8.0f,
                                  juce::Colour(0xff101015),
                                  centre.x, centre.y + radius - 8.0f, false);
    g.setGradientFill(knobGrad);
    g.fillEllipse(juce::Rectangle<float>(radius * 1.6f, radius * 1.6f)
                      .withCentre(centre));

    // Indicator
    juce::Path pointer;
    const float pointerLen = radius * 0.6f;
    pointer.addRoundedRectangle(-1.5f, -pointerLen, 3.0f, pointerLen * 0.9f, 1.5f);
    pointer.applyTransform(juce::AffineTransform::rotation(angle)
                               .translated(centre.x, centre.y));
    g.setColour(juce::Colour(0xffd6ddff));
    g.fillPath(pointer);
}

void AuraGLookAndFeel::drawComboBox(juce::Graphics& g, int width, int height,
                                    bool, int, int, int, int, juce::ComboBox& box)
{
    juce::Rectangle<int> area(0, 0, width, height);
    g.setColour(box.findColour(juce::ComboBox::backgroundColourId));
    g.fillRoundedRectangle(area.toFloat(), 4.0f);
    g.setColour(box.findColour(juce::ComboBox::outlineColourId));
    g.drawRoundedRectangle(area.toFloat().reduced(0.5f), 4.0f, 1.0f);

    juce::Path arrow;
    const float cx = (float) width - 12.0f;
    const float cy = (float) height * 0.5f;
    arrow.startNewSubPath(cx - 4.0f, cy - 2.0f);
    arrow.lineTo(cx, cy + 3.0f);
    arrow.lineTo(cx + 4.0f, cy - 2.0f);
    g.setColour(box.findColour(juce::ComboBox::arrowColourId));
    g.strokePath(arrow, juce::PathStrokeType(1.6f));
}

juce::Font AuraGLookAndFeel::getLabelFont(juce::Label&)
{
    return juce::Font(12.0f, juce::Font::plain);
}

juce::Font AuraGLookAndFeel::getTextButtonFont(juce::TextButton&, int)
{
    return juce::Font(13.0f, juce::Font::plain);
}
