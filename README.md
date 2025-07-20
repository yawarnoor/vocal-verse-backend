# VocalVerse Backend

Voice processing backend for VocalVerse - supports transcription, translation, and voice cloning.

## Features

- 🎤 **Speech Transcription** - Convert audio to text using OpenAI Whisper
- 🌍 **Translation** - Translate text between 100+ languages using Free Translate API
- 🎭 **Voice Cloning** - Clone voices using Hugging Face API
- 🎯 **XTTS Voice Cloning** - Local voice cloning with Coqui XTTS v2

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd vocal-verse-backend
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install FFmpeg:
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
   - **Windows**: Download from https://ffmpeg.org/download.html

## Running the Server

### Standard Mode

```bash
python app.py
```

### With XTTS Support

```bash
chmod +x start_xtts_server.sh
./start_xtts_server.sh
```

The server will run on `http://localhost:3000`

## API Endpoints

### POST /transcribe

Convert audio to text.

- **Body**: `audio` file (WAV, MP3, WEBM)
- **Response**: `{"transcription": "text"}`

### POST /translate

Translate text between languages.

- **Body**: `{"text": "hello", "target_lang": "es"}`
- **Response**: `{"translated_text": "hola", ...}`

### POST /clone

Clone voice using external API.

- **Body**: `voice_sample` file + `text` form data
- **Response**: `{"audio": "base64_audio"}`

### POST /clone_xtts

Local voice cloning with XTTS.

- **Body**: `voice_sample` file + `text` + `target_language` form data
- **Response**: `{"audio": "base64_audio"}`

### GET /status

Check server status and available features.

- **Response**: Server status and feature availability

### GET /test

Basic health check.

- **Response**: `{"message": "Server is running"}`

## Deployment

### Railway

1. Install Railway CLI:

```bash
npm install -g @railway/cli
```

2. Login and link project:

```bash
railway login
railway link
```

3. Deploy:

```bash
railway up
```

### Environment Variables

No environment variables required - all features use free APIs or local models.

## File Structure

- `app.py` - Main Flask application
- `voice_cloning_xtts.py` - XTTS voice cloning module
- `test_xtts.py` - XTTS testing script
- `start_xtts_server.sh` - Script to start server with XTTS
- `railway.json` - Railway deployment configuration
- `requirements.txt` - Python dependencies

## Notes

- Model cache and uploads directories are automatically created
- Large AI models are downloaded on first use and cached locally
- WEBM audio files are automatically converted to WAV using FFmpeg
- All endpoints support CORS for frontend integration
