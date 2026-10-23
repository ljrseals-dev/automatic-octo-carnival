cmake_minimum_required(VERSION 3.22)
project(AuraGPlugin VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "" FORCE)

# ---- JUCE via FetchContent ----
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        7.0.9
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(JUCE)

# ---- Plugin target ----
juce_add_plugin(AuraG
    COMPANY_NAME              "AURA-G"
    COMPANY_WEBSITE           "https://example.com/aura-g"
    COMPANY_EMAIL             "info@example.com"
    IS_SYNTH                  FALSE
    NEEDS_MIDI_INPUT          TRUE
    NEEDS_MIDI_OUTPUT         TRUE
    IS_MIDI_EFFECT            FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD   TRUE
    PLUGIN_MANUFACTURER_CODE  Aurag
    PLUGIN_CODE               Ag01
    FORMATS                   VST3 AU Standalone
    PRODUCT_NAME              "AURA-G"
    VST3_CATEGORIES           Fx Distortion
    AU_MAIN_TYPE              kAudioUnitType_Effect
)

target_sources(AuraG PRIVATE
    PluginProcessor.cpp
    PluginEditor.cpp
    PluginLookAndFeel.cpp
    AuraGEngine.cpp
    PresetManager.cpp
    MidiOut.cpp
)

target_include_directories(AuraG PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(AuraG PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_STRICT_REFCOUNTEDPOINTER=1
    JUCE_MODAL_LOOPS_PERMITTED=0
)

target_link_libraries(AuraG PRIVATE
    juce::juce_audio_utils
    juce::juce_dsp
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
)

# Bundle the presets JSON with the plugin resources
juce_add_binary_data(AuraGBinaryData SOURCES
    resources/presets.json
)
target_link_libraries(AuraG PRIVATE AuraGBinaryData)
