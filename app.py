import os
import tempfile
import time
from flask import Flask, request, jsonify
import io
from flask_cors import CORS
import traceback
import logging
import base64
import requests

# Try to import AI packages
try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import torch
    import torchaudio
    ai_packages_available = True
    print("✅ AI packages available")
except ImportError as e:
    ai_packages_available = False
    print(f"⚠️ AI packages not available: {e}")

try:
    import numpy as np
    numpy_available = True
except ImportError:
    numpy_available = False
    print("⚠️ NumPy not available")

try:
    import soundfile as sf
    soundfile_available = True
except ImportError:
    soundfile_available = False
    print("⚠️ soundfile not available")

# Try to import XTTS voice cloning
try:
    from voice_cloning_xtts import xtts_cloner
    xtts_available = True
    print("✅ XTTS voice cloning available")
except ImportError as e:
    xtts_available = False
    print(f"⚠️ XTTS not available: {e}")

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)  # Changed to INFO to reduce noise
logger = logging.getLogger(__name__)

# Global variables for models
model = None
processor = None

def load_whisper_model():
    """Load only the Whisper model for transcription"""
    try:
        if not ai_packages_available:
            logger.warning("AI packages not available - skipping Whisper model loading")
            return False
            
        logger.info("Loading Whisper model...")
        global model, processor
        
        # Try loading from local cache first
        try:
            logger.info("Attempting to load Whisper model from local cache...")
            # Try to use a better model for improved accuracy
            model_name = "openai/whisper-small"  # Much better accuracy than tiny
            model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                local_files_only=True,
                cache_dir="./model_cache"
            )
            processor = WhisperProcessor.from_pretrained(
                model_name,
                local_files_only=True,
                cache_dir="./model_cache"
            )
            logger.info("✅ Whisper-small model loaded successfully from local cache")
            return True
        except Exception as cache_error:
            logger.warning(f"whisper-small cache loading failed: {cache_error}")
            
            # Try tiny model from cache as fallback
            try:
                logger.info("Trying whisper-tiny from cache...")
                model_name = "openai/whisper-tiny"
                model = WhisperForConditionalGeneration.from_pretrained(
                    model_name,
                    local_files_only=True,
                    cache_dir="./model_cache"
                )
                processor = WhisperProcessor.from_pretrained(
                    model_name,
                    local_files_only=True,
                    cache_dir="./model_cache"
                )
                logger.info("✅ Whisper-tiny model loaded from local cache")
                return True
            except Exception as tiny_cache_error:
                logger.warning(f"Local cache completely failed: {tiny_cache_error}")
                logger.info("Attempting to download Whisper model...")
                
                # Try downloading if local cache fails
                try:
                    # Try small model first, fallback to tiny if needed
                    try:
                        model_name = "openai/whisper-small"
                        model = WhisperForConditionalGeneration.from_pretrained(
                            model_name,
                            cache_dir="./model_cache"
                        )
                        processor = WhisperProcessor.from_pretrained(
                            model_name,
                            cache_dir="./model_cache"
                        )
                        logger.info("✅ Whisper-small model downloaded and loaded successfully")
                        return True
                    except Exception as small_download_error:
                        logger.warning(f"whisper-small download failed: {small_download_error}, falling back to whisper-tiny")
                        model_name = "openai/whisper-tiny"
                        model = WhisperForConditionalGeneration.from_pretrained(
                            model_name,
                            cache_dir="./model_cache"
                        )
                        processor = WhisperProcessor.from_pretrained(
                            model_name,
                            cache_dir="./model_cache"
                        )
                        logger.info("✅ Whisper-tiny model downloaded and loaded successfully")
                        return True
                except Exception as download_error:
                    logger.error(f"❌ Failed to download any Whisper model: {download_error}")
                    # Don't exit, continue without the model
                    return False
                
    except Exception as e:
        logger.error(f"❌ Error loading Whisper model: {str(e)}")
        return False

def get_supported_languages():
    """Get supported languages from Free Translate API"""
    try:
        response = requests.get("https://ftapi.pythonanywhere.com/languages", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning("Could not fetch supported languages")
            return {}
    except Exception as e:
        logger.warning(f"Error fetching supported languages: {e}")
        return {}

def get_language_mapping():
    """Create mapping from common language names to API codes"""
    return {
        # Major languages
        'Arabic': 'ar',
        'Bengali': 'bn', 
        'Chinese (Simplified)': 'zh-cn',
        'Chinese (Traditional)': 'zh-tw',
        'Dutch': 'nl',
        'English': 'en',
        'French': 'fr',
        'German': 'de',
        'Hindi': 'hi',
        'Italian': 'it',
        'Japanese': 'ja',
        'Korean': 'ko',
        'Portuguese': 'pt',
        'Russian': 'ru',
        'Spanish': 'es',
        'Turkish': 'tr',
        'Urdu': 'ur',
        'Vietnamese': 'vi',
        
        # Additional supported languages
        'Afrikaans': 'af',
        'Albanian': 'sq',
        'Armenian': 'hy',
        'Bulgarian': 'bg',
        'Croatian': 'hr',
        'Czech': 'cs',
        'Danish': 'da',
        'Finnish': 'fi',
        'Greek': 'el',
        'Hebrew': 'he',
        'Hungarian': 'hu',
        'Indonesian': 'id',
        'Irish': 'ga',
        'Norwegian': 'no',
        'Persian': 'fa',
        'Polish': 'pl',
        'Romanian': 'ro',
        'Serbian': 'sr',
        'Slovak': 'sk',
        'Swedish': 'sv',
        'Thai': 'th',
        'Ukrainian': 'uk',
        'Welsh': 'cy',
        
        # Regional languages
        'Gujarati': 'gu',
        'Kannada': 'kn',
        'Malayalam': 'ml',
        'Marathi': 'mr',
        'Punjabi': 'pa',
        'Tamil': 'ta',
        'Telugu': 'te',
        'Sindhi': 'sd',
        'Pashto': 'ps',
        'Hausa': 'ha',
        'Swahili': 'sw',
        'Yoruba': 'yo',
        'Zulu': 'zu'
    }

def load_translation_model():
    """Test Free Translate API availability"""
    try:
        logger.info("Testing Free Translate API...")
        
        # Test the API with a simple translation
        test_response = requests.get(
            "https://ftapi.pythonanywhere.com/translate?sl=en&dl=es&text=hello", 
            timeout=10
        )
        
        if test_response.status_code == 200:
            logger.info("✅ Free Translate API is available and working")
            return True
        else:
            logger.warning("⚠️  Free Translate API test failed")
            return False
    except Exception as e:
        logger.warning(f"⚠️  Free Translate API not available: {e}")
        return False

# Load Whisper model (try to load, but don't exit if it fails)
whisper_available = load_whisper_model()
if not whisper_available:
    logger.warning("⚠️  Whisper model not available - transcription features will be disabled")

# Test Free Translate API availability
translation_available = load_translation_model()

# Get supported languages if API is available
supported_languages = {}
if translation_available:
    supported_languages = get_supported_languages()
    logger.info(f"Free Translate API supports {len(supported_languages)} languages")

def read_audio(audio_file):
    try:
        # Get the original filename and content type for debugging
        original_filename = getattr(audio_file, 'filename', 'unknown')
        content_type = getattr(audio_file, 'content_type', 'unknown')
        logger.info(f"Processing audio: {original_filename}, type: {content_type}")
        
        # Reset file pointer and read data
        audio_file.seek(0)
        audio_data = audio_file.read()
        
        # Determine the appropriate file extension based on content type
        if 'webm' in content_type or 'webm' in original_filename.lower():
            file_ext = '.webm'
        elif 'mp3' in content_type or 'mp3' in original_filename.lower():
            file_ext = '.mp3'
        elif 'wav' in content_type or 'wav' in original_filename.lower():
            file_ext = '.wav'
        else:
            file_ext = '.wav'  # Default fallback
        
        # Create a temporary file with the correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        logger.info(f"Created temporary file: {temp_file_path}")
        
        # If it's a WEBM file, convert it FIRST before trying torchaudio
        if file_ext == '.webm':
            try:
                logger.info("Converting WEBM to WAV using system FFmpeg...")
                
                # Convert WEBM to WAV using system ffmpeg
                temp_wav_path = temp_file_path.replace('.webm', '_converted.wav')
                
                import subprocess
                result = subprocess.run([
                    'ffmpeg', '-i', temp_file_path, 
                    '-acodec', 'pcm_s16le',  # High quality PCM 
                    '-ar', '16000',          # 16kHz sample rate (Whisper standard)
                    '-ac', '1',              # Mono channel
                    '-af', 'highpass=f=200,lowpass=f=3000',  # Filter noise frequencies
                    temp_wav_path, '-y'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    logger.info(f"Successfully converted WEBM to WAV: {temp_wav_path}")
                    # Replace the original file path with converted one
                    os.unlink(temp_file_path)  # Clean up original WEBM
                    temp_file_path = temp_wav_path
                    file_ext = '.wav'
                else:
                    logger.error(f"FFmpeg conversion failed: {result.stderr}")
                    raise Exception(f"WEBM conversion failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.error("FFmpeg conversion timed out")
                raise Exception("WEBM conversion timed out")
            except FileNotFoundError:
                logger.error("FFmpeg not found on system")
                raise Exception("FFmpeg not installed - cannot process WEBM files")
            except Exception as conv_error:
                logger.error(f"WEBM conversion error: {conv_error}")
                raise Exception(f"Could not convert WEBM file: {conv_error}")
        
        try:
            # Try soundfile first (more reliable), then fallback to torchaudio
            try:
                import soundfile as sf
                logger.info("Attempting to load audio with soundfile...")
                audio_data, sample_rate = sf.read(temp_file_path, dtype='float32')
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                
                logger.info(f"Successfully loaded audio with soundfile: shape={audio_data.shape}, sample_rate={sample_rate}")
                return audio_data, sample_rate
                
            except Exception as sf_error:
                logger.warning(f"soundfile failed: {sf_error}, trying torchaudio...")
                
                # Fallback to torchaudio
                waveform, sample_rate = torchaudio.load(temp_file_path)
                logger.info(f"Successfully loaded audio with torchaudio: {waveform.shape}, sample_rate: {sample_rate}")
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Convert to numpy array
                audio_data = waveform.numpy().flatten()
                
                return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Both soundfile and torchaudio failed to load {temp_file_path}: {str(e)}")
            raise Exception(f"Could not load audio file after conversion. Error: {str(e)}")
                
        finally:
            # Clean up the temporary file(s)
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up temp file: {cleanup_error}")
            
    except Exception as e:
        logger.error(f"❌ Error in read_audio: {str(e)}")
        raise

def resample(audio_data, original_sample_rate, target_sample_rate=16000):
    audio_tensor = torch.from_numpy(audio_data).float()
    resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
    return resampler(audio_tensor)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        logger.info("Transcribe request received")
        
        # Check if Whisper model is available
        if not whisper_available or not model or not processor:
            logger.warning("Whisper model not available")
            return jsonify({'error': 'Transcription feature not available. Whisper model not loaded.'}), 503
            
        if 'audio' not in request.files:
            logger.warning("No audio file provided in the request")
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        logger.info(f"Received audio file: {audio_file.filename}, Content-Type: {audio_file.content_type}")

        try:
            audio_data, original_sample_rate = read_audio(audio_file)
            logger.info(f"Audio data shape: {audio_data.shape}, Sample rate: {original_sample_rate}")
        except Exception as e:
            logger.error(f"Error reading audio file: {str(e)}")
            return jsonify({'error': f'Error reading audio file: {str(e)}'}), 400

        # Resample the audio to 16000 Hz
        try:
            resampled_audio_data = resample(audio_data, original_sample_rate).numpy()
            logger.info(f"Resampled audio data shape: {resampled_audio_data.shape}")
        except Exception as e:
            logger.error(f"Error resampling audio: {str(e)}")
            return jsonify({'error': 'Error processing audio'}), 500

        # Process with Whisper model
        try:
            inputs = processor(resampled_audio_data, sampling_rate=16000, return_tensors="pt")
            # Force English language transcription with better parameters
            predicted_ids = model.generate(
                inputs["input_features"],
                language="en",           # Force English language
                task="transcribe",       # Transcribe (not translate)
                do_sample=False,         # Use deterministic generation
                num_beams=5,            # Use beam search for better quality
                no_repeat_ngram_size=3  # Avoid repetitive text
            )
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            logger.info("Transcription completed successfully in English")
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            return jsonify({'error': 'Error during transcription'}), 500

        return jsonify({'transcription': transcription})

    except Exception as e:
        logger.error(f"Unexpected error in transcribe: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/translate', methods=['POST'])
def translate():
    try:
        logger.info("Translate request received")
        
        # Check if Free Translate API is available
        if not translation_available:
            logger.warning("Free Translate API not available")
            return jsonify({'error': 'Translation feature not available. API service is down.'}), 503
            
        data = request.json
        if 'text' not in data or 'target_lang' not in data:
            logger.warning("Text or target language not provided in the request")
            return jsonify({'error': 'Text or target language not provided'}), 400

        text = data['text']
        target_lang = data['target_lang']
        source_lang = 'en'  # Always English as source (as requested)
        
        # Comprehensive mapping from NLLB/old codes to Free Translate API codes
        nllb_to_api_mapping = {
            # Major languages
            'eng_Latn': 'en',        # English
            'urd_Arab': 'ur',        # Urdu
            'ara_Arab': 'ar',        # Arabic
            'aeb_Arab': 'ar',        # Arabic (Tunisian) -> Arabic
            'arb_Arab': 'ar',        # Arabic (Standard) -> Arabic
            'ars_Arab': 'ar',        # Arabic (Najdi) -> Arabic
            'ary_Arab': 'ar',        # Arabic (Moroccan) -> Arabic
            'arz_Arab': 'ar',        # Arabic (Egyptian) -> Arabic
            'fra_Latn': 'fr',        # French
            'deu_Latn': 'de',        # German
            'spa_Latn': 'es',        # Spanish
            'ita_Latn': 'it',        # Italian
            'por_Latn': 'pt',        # Portuguese
            'rus_Cyrl': 'ru',        # Russian
            'jpn_Jpan': 'ja',        # Japanese
            'kor_Hang': 'ko',        # Korean
            'zho_Hans': 'zh-cn',     # Chinese Simplified
            'zho_Hant': 'zh-tw',     # Chinese Traditional
            'hin_Deva': 'hi',        # Hindi
            'ben_Beng': 'bn',        # Bengali
            'tur_Latn': 'tr',        # Turkish
            'vie_Latn': 'vi',        # Vietnamese
            'nld_Latn': 'nl',        # Dutch
            'pol_Latn': 'pl',        # Polish
            'ukr_Cyrl': 'uk',        # Ukrainian
            'ron_Latn': 'ro',        # Romanian
            'ell_Grek': 'el',        # Greek
            'heb_Hebr': 'he',        # Hebrew
            'tha_Thai': 'th',        # Thai
            'swe_Latn': 'sv',        # Swedish
            'dan_Latn': 'da',        # Danish
            'nor_Latn': 'no',        # Norwegian
            'fin_Latn': 'fi',        # Finnish
            'hun_Latn': 'hu',        # Hungarian
            'ces_Latn': 'cs',        # Czech
            'slk_Latn': 'sk',        # Slovak
            'slv_Latn': 'sl',        # Slovenian
            'hrv_Latn': 'hr',        # Croatian
            'srp_Cyrl': 'sr',        # Serbian
            'bul_Cyrl': 'bg',        # Bulgarian
            'est_Latn': 'et',        # Estonian
            'lav_Latn': 'lv',        # Latvian
            'lit_Latn': 'lt',        # Lithuanian
            'mlt_Latn': 'mt',        # Maltese
            'isl_Latn': 'is',        # Icelandic
            'gle_Latn': 'ga',        # Irish
            'cym_Latn': 'cy',        # Welsh
            'eus_Latn': 'eu',        # Basque
            'cat_Latn': 'ca',        # Catalan
            'glg_Latn': 'gl',        # Galician
            
            # South Asian languages
            'snd_Arab': 'sd',        # Sindhi
            'pus_Arab': 'ps',        # Pashto
            'fas_Arab': 'fa',        # Persian
            'guj_Gujr': 'gu',        # Gujarati
            'pan_Guru': 'pa',        # Punjabi
            'mar_Deva': 'mr',        # Marathi
            'tam_Taml': 'ta',        # Tamil
            'tel_Telu': 'te',        # Telugu
            'kan_Knda': 'kn',        # Kannada
            'mal_Mlym': 'ml',        # Malayalam
            'sin_Sinh': 'si',        # Sinhala
            'nep_Deva': 'ne',        # Nepali
            'asm_Beng': 'as',        # Assamese -> not in API, fallback to Bengali
            'ori_Orya': 'or',        # Odia
            
            # African languages
            'hau_Latn': 'ha',        # Hausa
            'yor_Latn': 'yo',        # Yoruba
            'ibo_Latn': 'ig',        # Igbo
            'swa_Latn': 'sw',        # Swahili
            'som_Latn': 'so',        # Somali
            'amh_Ethi': 'am',        # Amharic
            'afr_Latn': 'af',        # Afrikaans
            'zul_Latn': 'zu',        # Zulu
            'xho_Latn': 'xh',        # Xhosa
            'sot_Latn': 'st',        # Sesotho
            'sna_Latn': 'sn',        # Shona
            
            # Southeast Asian languages
            'ind_Latn': 'id',        # Indonesian
            'msa_Latn': 'ms',        # Malay
            'tgl_Latn': 'tl',        # Filipino
            'khm_Khmr': 'km',        # Khmer
            'lao_Laoo': 'lo',        # Lao
            'mya_Mymr': 'my',        # Myanmar
            
            # Central Asian languages
            'uzn_Latn': 'uz',        # Uzbek
            'kaz_Cyrl': 'kk',        # Kazakh
            'kir_Cyrl': 'ky',        # Kyrgyz
            'tgk_Cyrl': 'tg',        # Tajik
            'tur_Latn': 'tr',        # Turkish
            'aze_Latn': 'az',        # Azerbaijani
            'hye_Armn': 'hy',        # Armenian
            'kat_Geor': 'ka',        # Georgian
            
            # Other languages
            'mri_Latn': 'mi',        # Maori
            'haw_Latn': 'haw',       # Hawaiian
            'ceb_Latn': 'ceb',       # Cebuano
            'hmn_Latn': 'hmn',       # Hmong
            'mon_Cyrl': 'mn',        # Mongolian
            'uig_Arab': 'ug',        # Uyghur
            'bod_Tibt': 'bo',        # Tibetan -> not in API
            'lat_Latn': 'la',        # Latin
            'yid_Hebr': 'yi',        # Yiddish
            'epo_Latn': 'eo',        # Esperanto
            'jav_Latn': 'jw',        # Javanese
            'sun_Latn': 'su',        # Sundanese
            'mad_Latn': 'mg',        # Malagasy
            'nya_Latn': 'ny',        # Chichewa
            'cor_Latn': 'co',        # Corsican
            'fry_Latn': 'fy',        # Frisian
            'sco_Latn': 'gd',        # Scots Gaelic
            'bre_Latn': 'br',        # Breton -> not in API
            'ltz_Latn': 'lb',        # Luxembourgish
            'mkd_Cyrl': 'mk',        # Macedonian
            'bel_Cyrl': 'be',        # Belarusian
            'bos_Latn': 'bs',        # Bosnian
            'als_Latn': 'sq',        # Albanian
        }
        
        # Convert NLLB code to API code if needed
        if target_lang in nllb_to_api_mapping:
            original_target = target_lang
            target_lang = nllb_to_api_mapping[target_lang]
            logger.info(f"Mapped {original_target} -> {target_lang}")
        
        # Validate that target language is supported by the API
        if target_lang not in supported_languages:
            logger.warning(f"Unsupported target language: {target_lang}")
            return jsonify({'error': f'Language "{target_lang}" is not supported. Please check /status for supported languages.'}), 400
        
        logger.info(f"Translating: {text} from {source_lang} to {target_lang}")

        # Translate using Free Translate API
        try:
            # Build API URL (always English to target language)
            api_url = f"https://ftapi.pythonanywhere.com/translate?sl={source_lang}&dl={target_lang}&text={requests.utils.quote(text)}"
            
            # Make API request
            response = requests.get(api_url, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                translated_text = result.get('destination-text', '')
                detected_source = result.get('source-language', source_lang)
                
                logger.info(f"Translation completed successfully: {source_lang} -> {target_lang}")
                
                return jsonify({
                    'translated_text': translated_text,
                    'source_language': detected_source,
                    'target_language': target_lang,
                    'original_text': text
                })
            else:
                logger.error(f"Free Translate API error: {response.status_code}")
                return jsonify({'error': f'Translation API error: {response.status_code}'}), 500
                
        except requests.exceptions.Timeout:
            logger.error("Translation API timeout")
            return jsonify({'error': 'Translation request timed out'}), 504
        except Exception as e:
            logger.error(f"Error during translation: {str(e)}")
            return jsonify({'error': f'Translation error: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Unexpected error in translate: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/clone', methods=['POST'])
def clone_voice():
    try:
        logger.info("Clone voice request received")
        if 'voice_sample' not in request.files or 'text' not in request.form:
            logger.warning("Voice sample or text not provided in the request")
            return jsonify({'error': 'Voice sample or text not provided'}), 400

        voice_sample = request.files['voice_sample']
        text = request.form['text']

        logger.info(f"Received voice sample: {voice_sample.filename}, text: {text}")

        # Convert the voice sample to base64
        voice_sample_base64 = base64.b64encode(voice_sample.read()).decode('utf-8')

        # Prepare the payload for the Hugging Face API
        payload = {
            "data": [
                text,
                {"name": voice_sample.filename, "data": f"data:audio/mp3;base64,{voice_sample_base64}"},
                {"name": "audio.wav", "data": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="},
            ]
        }

        logger.info("Sending request to Hugging Face API")
        # Make a request to the Hugging Face API
        response = requests.post("https://bilalsardar-voice-cloning.hf.space/run/predict", json=payload)
        
        logger.info(f"Hugging Face API response status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Error from Hugging Face API: {response.text}")
            return jsonify({'error': 'Voice cloning failed', 'details': response.text}), 500

        result = response.json()
        logger.info("Successfully received response from Hugging Face API")
        cloned_audio = result['data'][0]['data'].split(',')[1]  # Extract base64 audio data

        return jsonify({'audio': cloned_audio})

    except Exception as e:
        logger.error(f"Unexpected error in clone_voice: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/clone_xtts', methods=['POST'])
def clone_voice_xtts():
    """🎭 Clone voice using local XTTS model - FREE Cross-Lingual Voice Cloning!"""
    try:
        if not xtts_available:
            return jsonify({'error': 'XTTS voice cloning not available. Make sure to activate venv_xtts environment.'}), 503
            
        if 'voice_sample' not in request.files or 'text' not in request.form:
            logger.error("Missing voice_sample or text in request")
            return jsonify({'error': 'Missing voice_sample (file) or text (form data)'}), 400

        voice_sample = request.files['voice_sample']
        text = request.form['text']
        target_language = request.form.get('language', 'en')  # Frontend sends 'language'

        logger.info(f"🎭 XTTS Voice Cloning - File: {voice_sample.filename}, Text: {text[:50]}..., Language: {target_language}")

        # Save uploaded voice sample temporarily with original extension
        file_extension = os.path.splitext(voice_sample.filename or 'audio.wav')[1] or '.wav'
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_original:
            voice_sample.save(tmp_original.name)
            original_path = tmp_original.name

        try:
            # Convert audio to proper WAV format using our read_audio function
            logger.info("🔄 Converting audio to WAV format for XTTS...")
            
            # Create a file-like object for read_audio function
            with open(original_path, 'rb') as file:
                # Create a BytesIO object that mimics uploaded file behavior
                import io
                file_obj = io.BytesIO(file.read())
                file_obj.filename = os.path.basename(original_path)
                file_obj.content_type = 'audio/wav' if original_path.endswith('.wav') else 'audio/webm'
                
                audio_data, sample_rate = read_audio(file_obj)
            
            # Save as proper WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                wav_path = tmp_wav.name
            
            # Use soundfile to save as WAV
            import soundfile as sf
            sf.write(wav_path, audio_data, sample_rate)
            logger.info(f"✅ Audio converted to WAV: {wav_path}")

            # Clone voice using XTTS
            cloned_audio_path = xtts_cloner.clone_voice(
                speaker_audio_path=wav_path,
                text=text,
                target_language=target_language
            )

            if cloned_audio_path and os.path.exists(cloned_audio_path):
                # Read the generated audio and convert to base64
                with open(cloned_audio_path, 'rb') as audio_file:
                    audio_data = audio_file.read()
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')

                # Clean up temporary files
                os.unlink(original_path)
                os.unlink(wav_path)
                os.unlink(cloned_audio_path)

                logger.info("✅ XTTS voice cloning successful!")
                return jsonify({
                    'audio': audio_base64,
                    'method': 'XTTS-v2',
                    'language': target_language,
                    'message': f'🎉 Voice cloned in {target_language} using your own voice!'
                })
            else:
                # Clean up
                os.unlink(original_path)
                os.unlink(wav_path)
                return jsonify({'error': 'XTTS voice cloning failed'}), 500

        except Exception as xtts_error:
            # Clean up on error
            try:
                if os.path.exists(original_path):
                    os.unlink(original_path)
                if 'wav_path' in locals() and os.path.exists(wav_path):
                    os.unlink(wav_path)
            except:
                pass  # Ignore cleanup errors
            logger.error(f"XTTS error: {str(xtts_error)}")
            return jsonify({'error': f'XTTS error: {str(xtts_error)}'}), 500

    except Exception as e:
        logger.error(f"Unexpected error in clone_voice_xtts: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500
    
@app.route('/test', methods=['GET'])
def test():
    logger.info("Test request received")
    return jsonify({'message': 'Server is running'}), 200

@app.route('/status', methods=['GET'])
def status():
    """Return the status of available features"""
    logger.info("Status request received")
    return jsonify({
        'server': 'running',
        'features': {
            'transcription': whisper_available and model is not None and processor is not None,
            'translation': translation_available,  # Now using Free Translate API
            'voice_cloning': True  # This uses external API so should always be available
        },
        'supported_languages': supported_languages if translation_available else {},
        'language_mapping': get_language_mapping() if translation_available else {}
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=False)