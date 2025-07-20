#!/usr/bin/env python3
"""
VocalVerse XTTS Voice Cloning Test
Test script for cross-lingual voice cloning using Coqui XTTS
"""

import os
import time
from TTS.api import TTS

def test_voice_cloning():
    print("🎭 VocalVerse XTTS Voice Cloning Test")
    print("=" * 50)
    
    try:
        print("📦 Loading XTTS v2 model (this might take a few minutes on first run)...")
        start_time = time.time()
        
        # Initialize XTTS v2 for multilingual voice cloning
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds!")
        
        # Print supported languages
        print(f"🌍 Supported languages: {len(tts.languages)} languages")
        print("Some supported languages:", tts.languages[:10])
        
        return tts
        
    except Exception as e:
        print(f"❌ Error loading XTTS: {e}")
        return None

def clone_voice_demo(tts, speaker_audio_path, text, target_language="en"):
    """Demo function for voice cloning"""
    try:
        print(f"\n🎤 Cloning voice for text: '{text}'")
        print(f"🌐 Target language: {target_language}")
        
        output_path = f"cloned_voice_{target_language}.wav"
        
        # Generate speech with cloned voice
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_audio_path,
            language=target_language,
            file_path=output_path
        )
        
        print(f"✅ Voice cloning completed! Output saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Voice cloning error: {e}")
        return None

if __name__ == "__main__":
    # Test basic model loading
    tts_model = test_voice_cloning()
    
    if tts_model:
        print("\n🎯 XTTS is ready for VocalVerse integration!")
        print("\nNext steps:")
        print("1. Record a voice sample (30 seconds)")
        print("2. Provide text to be cloned")
        print("3. Choose target language (ur, ar, zh-cn, etc.)")
        print("4. Generate cloned voice!")
        
        # Example usage (uncomment when you have a voice sample):
        # voice_sample = "path/to/your/voice.wav"
        # if os.path.exists(voice_sample):
        #     clone_voice_demo(tts_model, voice_sample, "Hello world!", "ur")
    else:
        print("❌ XTTS setup failed. Please check the installation.") 