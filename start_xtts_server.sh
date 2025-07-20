#!/bin/bash

echo "🎭 Starting VocalVerse with XTTS Voice Cloning Support"
echo "=================================================="

# Activate the XTTS virtual environment
echo "📦 Activating XTTS environment..."
source venv_xtts/bin/activate

# Check if TTS is installed
echo "🔍 Checking XTTS installation..."
python -c "from TTS.api import TTS; print('✅ XTTS is ready!')" || {
    echo "❌ XTTS not installed. Installing now..."
    pip install TTS
}

echo "🚀 Starting Flask server with XTTS support..."
echo "Backend will be available at: http://localhost:3000"
echo "New endpoint: http://localhost:3000/clone_xtts"
echo ""
echo "🎯 Features available:"
echo "  ✅ Speech Transcription (Whisper)"
echo "  ✅ Translation (107 languages)"
echo "  ✅ Voice Cloning (External API)"
echo "  🎭 XTTS Voice Cloning (Local, Free, Cross-lingual)"
echo ""

# Start the Flask app
python app.py 