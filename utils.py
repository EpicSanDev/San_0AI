import os
import json
import logging
from datetime import datetime
import numpy as np
from pathlib import Path
import re
import tempfile
import wave
import pyaudio

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('san_ai.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('SanAI')

def ensure_directories():
    """Crée les répertoires nécessaires s'ils n'existent pas"""
    dirs = ['static/audio', 'voice_profiles', 'conversations', 'temp']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def clean_text(text):
    """Nettoie et normalise le texte"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def save_json(data, filepath):
    """Sauvegarde des données en JSON avec gestion d'erreurs"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde JSON: {e}")
        return False

def load_json(filepath):
    """Charge des données JSON avec gestion d'erreurs"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"Erreur lors du chargement JSON: {e}")
    return {}

def get_audio_duration(audio_file):
    """Obtient la durée d'un fichier audio"""
    try:
        with wave.open(audio_file, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        logging.error(f"Erreur lors de la lecture de la durée audio: {e}")
        return 0

def create_temp_audio_file(audio_data, sample_rate=44100):
    """Crée un fichier audio temporaire"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    try:
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        return temp_file.name
    except Exception as e:
        logging.error(f"Erreur lors de la création du fichier audio temporaire: {e}")
        os.unlink(temp_file.name)
        return None

def format_response(text, audio_path=None, success=True):
    """Formate une réponse standard"""
    return {
        "text": text,
        "audio_path": audio_path,
        "success": success,
        "timestamp": str(datetime.now())
    }

def calculate_audio_features(audio_data):
    """Calcule les caractéristiques audio de base"""
    try:
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        features = {
            "rms": np.sqrt(np.mean(np.square(audio_array))),
            "peak": np.max(np.abs(audio_array)),
            "zero_crossings": np.sum(np.diff(np.signbit(audio_array)))
        }
        return features
    except Exception as e:
        logging.error(f"Erreur lors du calcul des caractéristiques audio: {e}")
        return None

def is_valid_audio(audio_features, thresholds):
    """Vérifie si l'audio est valide selon des seuils"""
    if not audio_features:
        return False
        
    return (
        audio_features["rms"] > thresholds.get("min_rms", 100) and
        audio_features["peak"] > thresholds.get("min_peak", 1000) and
        audio_features["zero_crossings"] > thresholds.get("min_zero_crossings", 100)
    )

def extract_datetime_info(text):
    """Extrait les informations de date et heure d'un texte"""
    date_patterns = {
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})': '%d/%m/%Y',
        r'(\d{1,2})\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(\d{4})': '%d %B %Y'
    }
    
    time_patterns = {
        r'(\d{1,2})[h:](\d{2})': '%H:%M',
        r'(\d{1,2})\s*heures?\s*(\d{2})?': '%H:%M'
    }
    
    extracted_dates = []
    extracted_times = []
    
    for pattern, date_format in date_patterns.items():
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            try:
                date_str = match.group()
                date_obj = datetime.strptime(date_str, date_format)
                extracted_dates.append(date_obj)
            except:
                continue
                
    for pattern, time_format in time_patterns.items():
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            try:
                time_str = match.group()
                time_obj = datetime.strptime(time_str, time_format)
                extracted_times.append(time_obj)
            except:
                continue
                
    return {
        'dates': extracted_dates,
        'times': extracted_times
    }

def get_available_audio_devices():
    """Liste les périphériques audio disponibles"""
    p = pyaudio.PyAudio()
    devices = []
    
    try:
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:  # Only input devices
                    devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': int(device_info['defaultSampleRate'])
                    })
            except:
                continue
    finally:
        p.terminate()
        
    return devices

def find_best_audio_device():
    """Trouve le meilleur périphérique audio disponible"""
    devices = get_available_audio_devices()
    
    if not devices:
        return None
        
    # Préférer les périphériques avec des caractéristiques optimales
    preferred_devices = [d for d in devices if (
        d['channels'] >= 1 and
        d['sample_rate'] >= 44100 and
        'microphone' in d['name'].lower()
    )]
    
    if preferred_devices:
        return max(preferred_devices, key=lambda x: (x['channels'], x['sample_rate']))
    
    # Sinon, retourner le premier disponible
    return devices[0]
