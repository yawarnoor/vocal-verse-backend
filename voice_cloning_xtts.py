"""
VocalVerse XTTS Voice Cloning Module
Local voice cloning using Coqui XTTS v2 for cross-lingual synthesis
"""

import os
import tempfile
import time
import logging
from TTS.api import TTS

logger = logging.getLogger(__name__)

class XTTSVoiceCloner:
    def __init__(self):
        self.tts = None
        self.model_loaded = False
        self.supported_languages = [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 
            'cs', 'ar', 'zh-cn', 'ja', 'hu', 'ko'
        ]
    
    def load_model(self):
        """Load XTTS v2 model"""
        if self.model_loaded:
            return True
            
        try:
            logger.info("🎭 Loading XTTS v2 model for voice cloning...")
            start_time = time.time()
            
            # Initialize XTTS v2 - GPU can be disabled for compatibility
            # PyTorch 2.1.0 doesn't have weights_only security restrictions
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
            
            load_time = time.time() - start_time
            self.model_loaded = True
            logger.info(f"✅ XTTS v2 model loaded successfully in {load_time:.2f} seconds!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load XTTS model: {str(e)}")
            return False
    
    def clone_voice(self, speaker_audio_path, text, target_language="en"):
        """
        Clone voice using XTTS
        
        Args:
            speaker_audio_path: Path to speaker audio file (30 seconds recommended)
            text: Text to synthesize in the cloned voice
            target_language: Target language code (en, ur, ar, zh-cn, etc.)
        
        Returns:
            Path to generated audio file or None if failed
        """
        if not self.model_loaded:
            if not self.load_model():
                return None
        
        try:
            logger.info(f"🎤 Cloning voice for text: '{text[:50]}...'")
            logger.info(f"🌐 Target language: {target_language}")
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                output_path = tmp_file.name
            
            # Generate speech with cloned voice
            self.tts.tts_to_file(
                text=text,
                speaker_wav=speaker_audio_path,
                language=target_language,
                file_path=output_path
            )
            
            logger.info(f"✅ Voice cloning completed! Output: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Voice cloning failed: {str(e)}")
            return None
    
    def get_supported_languages(self):
        """Return list of supported languages"""
        return self.supported_languages
    
    def is_language_supported(self, language_code):
        """Check if language is supported"""
        return language_code in self.supported_languages

# Global instance
xtts_cloner = XTTSVoiceCloner() 