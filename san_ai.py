from flask import Flask, request, jsonify, render_template, send_file
from flask_socketio import SocketIO, emit
from voice_profile_manager import VoiceProfileManager
from memory_assistant import MemoryAssistant  # Ajout de l'import manquant
import queue
import threading
import whisper
from gtts import gTTS
import speech_recognition as sr
import datetime
import json
import os
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import random
from transformers import pipeline
from textblob import TextBlob
import numpy as np
from datetime import timedelta
import wave
import requests
from datetime import datetime, timedelta
import importlib.util
import glob
from collections import defaultdict, Counter
import re
from dateutil.parser import parse
from datetime import datetime, timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_time(time_str):
    """Convertit une chaîne de temps en objet datetime"""
    try:
        # Pour les formats relatifs (dans X heures/minutes)
        if 'dans' in time_str.lower():
            value = int(''.join(filter(str.isdigit, time_str)))
            unit = time_str.lower()
            now = datetime.now()
            if 'heure' in unit:
                return now + timedelta(hours=value)
            elif 'minute' in unit:
                return now + timedelta(minutes=value)
        
        # Pour les formats absolus (à XX:XX)
        return parse(time_str)
    except:
        raise ValueError("Format de temps non reconnu")

class VoiceHandler:
    def __init__(self):
        self.model = whisper.load_model("medium")  # Changer à medium pour plus de précision
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.audio_dir = "static/audio"
        # Créer le dossier audio s'il n'existe pas
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)
        self.last_transcription_time = datetime.now()
        self.min_silence_duration = timedelta(seconds=1)  # Durée minimale de silence entre les phrases
        self.temp_dir = tempfile.gettempdir()
        self.sample_rate = 16000
        self.channels = 1
        self.min_audio_length = 0.5  # Durée minimale en secondes
        self.energy_threshold = 1000  # Seuil de détection du son
        self.recognizer.energy_threshold = self.energy_threshold
        self.recognizer.dynamic_energy_threshold = True
        self.voice_speed = 1.0
        self.voice_pitch = 1.0
        self.noise_reduction = True
        self.silence_threshold = 500  # Augmenter la sensibilité
        self.noise_threshold = 0.02  # Seuil de bruit (0.0 à 1.0)
        self.min_signal_level = 0.1  # Niveau minimal du signal
        self.frame_duration = 0.5  # Durée d'une frame en secondes

        # Ajout de nouveaux paramètres pour améliorer la reconnaissance
        self.chunk_size = 1024
        self.format = sr.AudioData
        self.rate = 44100  # Augmentation de la fréquence d'échantillonnage
        self.device_index = None  # Auto-sélection du meilleur périphérique
        self.timeout = 5  # Timeout en secondes
        self.phrase_threshold = 0.3  # Seuil de détection de phrase
        self.non_speaking_duration = 0.5  # Durée de silence pour fin de phrase
        
        # Configuration avancée de Whisper
        self.whisper_config = {
            'language': 'fr',
            'task': 'transcribe',
            'temperature': 0.2,
            'no_speech_threshold': 0.3,  # Plus sensible
            'logprob_threshold': -1.0,
            'condition_on_previous_text': True,
            'best_of': 3  # Prend la meilleure parmi 3 tentatives
        }
        
        try:
            # Test du microphone
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                print("Microphone initialisé avec succès")
        except Exception as e:
            print(f"Erreur d'initialisation du microphone: {e}")
            print("Périphériques audio disponibles:")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone {index}: {name}")

    def speech_to_text(self, audio_file):
        try:
            print("Début de la transcription...")
            
            # Vérification du fichier audio
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"Fichier audio non trouvé: {audio_file}")
            
            # Vérification de la taille du fichier
            if os.path.getsize(audio_file) < 100:  # Taille minimale en octets
                raise ValueError("Fichier audio trop petit ou corrompu")
            
            # Application de la réduction de bruit
            if self.noise_reduction:
                filtered_file = self._reduce_noise(audio_file)
                if filtered_file:
                    audio_file = filtered_file
                    
            print("Traitement de l'audio avec Whisper...")
            result = self.model.transcribe(
                audio_file,
                **self.whisper_config
            )
            
            text = result["text"].strip()
            confidence = result.get("confidence", 0)
            
            print(f"Texte reconnu: '{text}' (confiance: {confidence:.2f})")
            
            # Validation du résultat
            if not text or confidence < 0.5:
                print("Confiance trop faible, nouvelle tentative...")
                # Deuxième tentative avec des paramètres plus permissifs
                self.whisper_config['temperature'] = 0.4
                result = self.model.transcribe(audio_file, **self.whisper_config)
                text = result["text"].strip()
                self.whisper_config['temperature'] = 0.2  # Retour aux paramètres normaux
            
            return text

        except Exception as e:
            print(f"Erreur détaillée dans speech_to_text: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""

    def text_to_speech(self, text, lang='fr'):
        try:
            tts = gTTS(
                text=text,
                lang=lang,
                slow=False if self.voice_speed >= 1.0 else True
            )
            
            filename = f"{self.audio_dir}/speech_{datetime.now().timestamp()}.mp3"
            tts.save(filename)
            
            # Appliquer les modifications de voix
            self._process_audio(filename)
            
            return filename.replace(self.audio_dir + '/', '')

        except Exception as e:
            print(f"Erreur dans text_to_speech amélioré: {e}")
            return None

    def _reduce_noise(self, audio_file):
        try:
            import numpy as np
            from scipy.io import wavfile
            from scipy import signal
            
            # Lecture du fichier audio
            sample_rate, data = wavfile.read(audio_file)
            
            # Convertir en float32 pour le traitement
            if data.dtype != np.float32:
                data = data.astype(np.float32) / np.iinfo(data.dtype).max
            
            # Calcul de l'enveloppe du signal
            envelope = np.abs(signal.hilbert(data))
            
            # Détection du niveau de bruit
            noise_level = np.percentile(envelope, 10)
            
            # Vérifier si le signal est principalement du bruit
            signal_strength = np.mean(envelope)
            if signal_strength < self.min_signal_level or signal_strength < noise_level * 1.5:
                print("Signal trop faible ou trop bruité")
                return None
            
            # Application d'un filtre passe-bande pour la voix (80Hz-3000Hz)
            nyquist = sample_rate / 2
            low = 80 / nyquist
            high = 3000 / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_data = signal.filtfilt(b, a, data)
            
            # Suppression du bruit
            filtered_data[envelope < noise_level * 2] = 0
            
            # Normalisation
            filtered_data = filtered_data / np.max(np.abs(filtered_data))
            
            # Sauvegarde du fichier filtré
            temp_file = f"{self.temp_dir}/filtered_{datetime.now().timestamp()}.wav"
            wavfile.write(temp_file, sample_rate, (filtered_data * 32767).astype(np.int16))
            
            return temp_file
            
        except Exception as e:
            print(f"Erreur dans la réduction du bruit: {e}")
            return audio_file

    def _process_audio(self, filename):
        try:
            from pydub import AudioSegment
            
            audio = AudioSegment.from_mp3(filename)
            
            # Ajuster la vitesse
            if self.voice_speed != 1.0:
                audio = audio.speedup(playback_speed=self.voice_speed)
            
            # Ajuster le pitch (hauteur de la voix)
            if self.voice_pitch != 1.0:
                audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * self.voice_pitch)
                })
            
            # Sauvegarder les modifications
            audio.export(filename, format="mp3")
            
        except Exception as e:
            print(f"Erreur dans le traitement audio: {e}")

    def start_listening(self):
        self.is_listening = True
        threading.Thread(target=self._process_audio_queue, daemon=True).start()

    def _process_audio_queue(self):
        while self.is_listening:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                text = self.speech_to_text(audio_data)
                if text.strip():
                    socketio.emit('transcription', {'text': text})

    def process_stream(self, audio_data):
        try:
            print("Début du traitement du flux audio...")
            
            if not audio_data:
                raise ValueError("Données audio vides")

            # Décodage et validation des données audio
            decoded_audio = self._decode_audio_data(audio_data)
            if not decoded_audio:
                raise ValueError("Échec du décodage audio")

            # Création du fichier temporaire
            temp_path = self._create_temp_file(decoded_audio)
            
            try:
                # Analyse de la qualité audio
                audio_quality = self._check_audio_quality(temp_path)
                if not audio_quality['is_valid']:
                    raise ValueError(f"Qualité audio insuffisante: {audio_quality['reason']}")

                # Traitement de l'audio
                text = self.speech_to_text(temp_path)
                
                if not text.strip():
                    raise ValueError("Aucun texte détecté dans l'audio")
                
                print(f"Texte détecté avec succès: {text}")
                
                # Émission du résultat
                self._emit_result(text)
                
                return True

            finally:
                self._cleanup_temp_file(temp_path)
                
        except Exception as e:
            error_msg = f"Erreur dans process_stream: {str(e)}"
            print(error_msg)
            socketio.emit('error', {'message': error_msg})
            return False

    def _decode_audio_data(self, audio_data):
        """Décode les données audio depuis base64 ou bytes"""
        try:
            if isinstance(audio_data, str):
                import base64
                if ',' in audio_data:
                    audio_data = audio_data.split(',')[1]
                return base64.b64decode(audio_data)
            return audio_data
        except Exception as e:
            print(f"Erreur de décodage audio: {e}")
            return None

    def _create_temp_file(self, audio_data):
        """Crée un fichier temporaire WAV à partir des données audio"""
        temp_path = os.path.join(self.temp_dir, f'temp_{datetime.now().timestamp()}.wav')
        try:
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
            return temp_path
        except Exception as e:
            print(f"Erreur création fichier temporaire: {e}")
            return None

    def _check_audio_quality(self, audio_path):
        """Vérifie la qualité de l'audio"""
        try:
            with wave.open(audio_path, 'rb') as wf:
                # Vérification de la durée
                duration = wf.getnframes() / float(wf.getframerate())
                if duration < self.min_audio_length:
                    return {'is_valid': False, 'reason': 'Audio trop court'}
                
                # Vérification du niveau sonore
                frames = wf.readframes(wf.getnframes())
                audio_array = np.frombuffer(frames, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(np.square(audio_array)))
                
                if rms < self.noise_threshold:
                    return {'is_valid': False, 'reason': 'Niveau sonore trop faible'}
                
                return {'is_valid': True, 'rms': rms, 'duration': duration}
                
        except Exception as e:
            return {'is_valid': False, 'reason': f'Erreur analyse: {str(e)}'}

    def _emit_result(self, text):
        """Émet le résultat de la reconnaissance vocale"""
        try:
            socketio.emit('transcription', {'text': text})
            response = ai.process_input(text)
            if response:
                audio_path = self.text_to_speech(response)
                if audio_path:
                    socketio.emit('response', {
                        'text': response,
                        'audio_path': audio_path,
                        'success': True
                    })
        except Exception as e:
            print(f"Erreur émission résultat: {e}")

    def _cleanup_temp_file(self, temp_path):
        """Nettoie les fichiers temporaires"""
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"Erreur nettoyage fichier temporaire: {e}")

    def process_continuous_stream(self, audio_file):
        """Traite un flux audio continu"""
        try:
            # Vérifier si le fichier existe et n'est pas vide
            if not os.path.exists(audio_file) or os.path.getsize(audio_file) < 100:
                return None
                
            # Convertir l'audio en texte
            text = self.speech_to_text(audio_file)
            
            if text and text.strip():
                return {'text': text}
                
            return None
            
        except Exception as e:
            print(f"Erreur dans process_continuous_stream: {str(e)}")
            return None

class LanguageModel:
    def __init__(self):
        # Initialisation des attributs de base
        self.models = {
            'small': {
                'name': "asi/gpt-fr-cased-small",
                'model': None,
                'tokenizer': None
            },
            'large': {
                'name': "facebook/xglm-7.5B",
                'model': None,
                'tokenizer': None
            }
        }
        self.current_model = 'large'
        
        # Détection du processeur et configuration du device
        self.is_m1 = self._check_apple_silicon()
        
        if self.is_m1:
            print("Processeur Apple Silicon M1 détecté")
            self.device = "mps"
            self.model_params = {
                'mps': {
                    'device_map': None,
                    'torch_dtype': torch.float16,
                    'low_cpu_mem_usage': True,
                    'use_metal': True
                },
                'cpu': {
                    'device_map': None,
                    'torch_dtype': torch.float32,
                    'low_cpu_mem_usage': True
                }
            }
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_params = {
                'cuda': {
                    'device_map': "auto",
                    'torch_dtype': torch.float16,
                    'max_memory': {0: "8GB"},
                    'load_in_8bit': True
                },
                'cpu': {
                    'device_map': None,
                    'torch_dtype': torch.float32,
                    'low_cpu_mem_usage': True
                }
            }
        
        print(f"Utilisation de: {self.device}")
        
        # Autres paramètres
        self.max_tokens = 512
        self.repetition_penalty = 1.2
        
        # Initialisation du modèle
        self._initialize_default_model()

    def _check_apple_silicon(self):
        try:
            import platform
            return (
                platform.system() == "Darwin" and 
                platform.machine() == "arm64" and 
                hasattr(torch.backends, "mps") and 
                torch.backends.mps.is_available()
            )
        except:
            return False

    def _initialize_default_model(self):
        try:
            model_info = self.models[self.current_model]
            model_info['tokenizer'] = AutoTokenizer.from_pretrained(model_info['name'])
            
            if self.is_m1:
                params = self.model_params['mps']
                model = AutoModelForCausalLM.from_pretrained(
                    model_info['name'],
                    **params
                )
                model = model.to('mps')
                model_info['model'] = model
            else:
                params = self.model_params[self.device]
                model = AutoModelForCausalLM.from_pretrained(
                    model_info['name'],
                    **params
                ).to(self.device)
                model_info['model'] = model
                
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            self._load_fallback_model()

    def _load_fallback_model(self):
        """Charge un modèle de repli plus petit en cas d'erreur"""
        try:
            print("Tentative de chargement du modèle de repli...")
            self.current_model = 'small'
            self._initialize_default_model()
        except Exception as e:
            print(f"Impossible de charger le modèle de repli: {e}")
            raise

    def adjust_parameters(self, speaker, sentiment):
        """Ajuste les paramètres du modèle selon le locuteur et le sentiment"""
        base_temp = 0.7
        if sentiment > 0.5:
            self.repetition_penalty = 1.1  # Plus créatif pour sentiment positif
        else:
            self.repetition_penalty = 1.3  # Plus conservateur pour sentiment négatif

    def switch_model(self, model_size):
        if model_size not in self.models:
            return False
            
        if not self.models[model_size]['model']:
            try:
                model_info = self.models[model_size]
                model_info['tokenizer'] = AutoTokenizer.from_pretrained(model_info['name'])
                params = self.model_params[self.device]
                model_info['model'] = AutoModelForCausalLM.from_pretrained(
                    model_info['name'],
                    **params
                ).to(self.device)
            except Exception as e:
                print(f"Erreur lors du chargement du modèle {model_size}: {e}")
                return False
                
        self.current_model = model_size
        return True

    def generate_response(self, prompt, context=None):
        try:
            model_info = self.models[self.current_model]
            model = model_info['model']
            tokenizer = model_info['tokenizer']

            if context:
                full_prompt = f"{context}\nQuestion: {prompt}\nRéponse:"
            else:
                full_prompt = f"Question: {prompt}\nRéponse:"
                
            inputs = tokenizer(full_prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(model.device)

            if self.is_m1:
                with torch.no_grad():
                    output = model.generate(
                        inputs["input_ids"].to('mps'),
                        max_length=min(256, self.max_tokens),
                        min_length=5,
                        num_return_sequences=1,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        repetition_penalty=self.repetition_penalty,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        no_repeat_ngram_size=3
                    )
            elif self.device == "cpu":
                torch.cuda.empty_cache()
                with torch.no_grad():
                    output = model.generate(
                        inputs["input_ids"],
                        max_length=min(512, self.max_tokens),
                        min_length=5,
                        num_return_sequences=1,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        repetition_penalty=self.repetition_penalty,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        no_repeat_ngram_size=3
                    )
            else:
                output = model.generate(
                    inputs["input_ids"],
                    max_length=512,
                    min_length=5,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            response = response.replace(full_prompt, "").strip()
            return response

        except Exception as e:
            print(f"Erreur lors de la génération de réponse: {e}")
            return "Je suis désolé, je ne peux pas générer une réponse pour le moment."

    def get_embedding(self, text):
        """Génère un embedding pour le texte donné"""
        try:
            model_info = self.models[self.current_model]
            tokenizer = model_info['tokenizer']
            model = model_info['model']
            
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
            return embeddings.cpu().numpy()
            
        except Exception as e:
            print(f"Erreur lors de la génération d'embedding: {e}")
            return None

class KnowledgeBase:
    def __init__(self):
        self.knowledge_file = "knowledge.pkl"
        self.knowledge = {}
        self.vectorizer = TfidfVectorizer()
        self.load_knowledge()
        self.embeddings = {}
        self.language_model = None

    def load_knowledge(self):
        if os.path.exists(self.knowledge_file):
            with open(self.knowledge_file, 'rb') as f:
                self.knowledge = pickle.load(f)
        
    def save_knowledge(self):
        with open(self.knowledge_file, 'wb') as f:
            pickle.dump(self.knowledge, f)

    def add_knowledge(self, question, answer):
        self.knowledge[question.lower()] = answer
        self.save_knowledge()
        if self.language_model:
            self.embeddings[question.lower()] = self.language_model.get_embedding(question)

    def find_similar_question(self, question, threshold=0.7):
        if not self.knowledge:
            return None, 0
            
        questions = list(self.knowledge.keys())
        self.vectorizer.fit(questions + [question])
        question_vector = self.vectorizer.transform([question])
        all_vectors = self.vectorizer.transform(questions)
        
        similarities = cosine_similarity(question_vector, all_vectors)[0]
        max_sim = max(similarities)
        if max_sim >= threshold:
            best_question = questions[similarities.argmax()]
            return best_question, max_sim
            
        return None, 0

class TrainingData:
    def __init__(self):
        self.categories = {
            "greeting": {
                "patterns": [
                    "bonjour", "salut", "hello", "coucou", "bonsoir", "hey",
                    "bonjour san", "salut san", "hello san", "bonsoir san",
                    "bonjour comment vas-tu", "salut ça va"
                ],
                "responses": [
                    "Bonjour! Je suis ravi de vous parler.",
                    "Salut! Comment puis-je vous aider aujourd'hui?",
                    "Bonjour! J'espère que vous allez bien.",
                    "Hey! Je suis là pour vous aider."
                ]
            },
            "mood_query": {
                "patterns": [
                    "comment vas-tu", "ça va", "comment tu te sens",
                    "tout va bien", "comment ça va", "tu vas bien",
                    "comment se passe ta journée"
                ],
                "responses": [
                    "Je vais très bien, merci! Je suis toujours content d'apprendre de nouvelles choses.",
                    "Parfaitement bien! J'adore nos conversations.",
                    "Super bien! J'aime beaucoup discuter avec vous."
                ]
            },
            "goodbye": {
                "patterns": [
                    "au revoir", "bye", "à plus", "à bientôt",
                    "à la prochaine", "salut", "ciao", "adieu",
                    "bonne journée", "bonne soirée"
                ],
                "responses": [
                    "Au revoir! À bientôt j'espère!",
                    "À la prochaine! N'hésitez pas à revenir me voir.",
                    "Au revoir! Passez une excellente journée!"
                ]
            },
            "weather": {
                "patterns": [
                    "quel temps fait-il", "météo", "il fait beau",
                    "il pleut", "température", "climat"
                ],
                "responses": [
                    "Je ne peux pas encore vérifier la météo en temps réel, mais cette fonction arrive bientôt!",
                    "La météo est une information que je ne peux pas encore obtenir.",
                ]
            },
            "time": {
                "patterns": [
                    "quelle heure est-il", "heure", "l'heure",
                    "donne-moi l'heure", "il est quelle heure"
                ],
                "responses": [
                    "Il est actuellement {time}",
                    "L'heure actuelle est {time}"
                ]
            },
            "identity": {
                "patterns": [
                    "qui es-tu", "tu es qui", "présente toi",
                    "que sais-tu faire", "tes capacités",
                    "comment tu t'appelles", "ton nom"
                ],
                "responses": [
                    "Je suis San, votre assistant IA personnel. Je peux vous aider avec diverses tâches comme la conversation, l'apprentissage et la recherche d'informations.",
                    "Je m'appelle San, une IA conçue pour vous assister et apprendre de nos interactions."
                ]
            },
            "help": {
                "patterns": [
                    "aide moi", "help", "que peux-tu faire",
                    "comment ça marche", "aide", "instructions",
                    "guide"
                ],
                "responses": [
                    "Je peux vous aider de plusieurs façons:\n- Répondre à vos questions\n- Apprendre de nouvelles choses\n- Converser avec vous\n- Mémoriser des informations",
                    "Voici mes principales fonctions:\n- Conversation naturelle\n- Apprentissage continu\n- Reconnaissance vocale\n- Mémorisation d'informations"
                ]
            },
            "humor": {
                "patterns": [
                    "raconte une blague", "fais moi rire", "connais tu une blague",
                    "dis quelque chose de drôle", "une histoire drôle"
                ],
                "responses": [
                    "Pourquoi les plongeurs plongent-ils toujours en arrière ? Parce que sinon ils tombent dans le bateau !",
                    "Que fait une fraise sur un cheval ? Tagada tagada !",
                    "Quel est le comble pour un électricien ? Ne pas être au courant !"
                ]
            },
            "motivation": {
                "patterns": [
                    "je suis fatigué", "j'ai besoin de motivation",
                    "encourage moi", "donne moi de l'énergie"
                ],
                "responses": [
                    "La persévérance est la clé du succès ! Continuez, vous êtes sur la bonne voie.",
                    "Chaque petit pas vous rapproche de votre objectif. Vous pouvez le faire !",
                    "La réussite appartient à ceux qui n'abandonnent jamais. Je crois en vous !"
                ]
            }
        }

class EmotionSystem:
    def __init__(self):
        self.emotion_analyzer = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
        self.current_emotion = {
            'joy': 0.5,
            'trust': 0.5,
            'interest': 0.5
        }
        
    def update_emotion(self, text):
        result = self.emotion_analyzer(text)
        score = float(result[0]['score'])
        sentiment = result[0]['label']
        
        # Ajuster les émotions selon le sentiment
        self.current_emotion['joy'] = min(1.0, self.current_emotion['joy'] + (score - 0.5) * 0.1)
        self.current_emotion['interest'] = min(1.0, self.current_emotion['interest'] + 0.05)
        return self.get_emotional_state()

    def get_emotional_state(self):
        dominant_emotion = max(self.current_emotion.items(), key=lambda x: x[1])
        return dominant_emotion[0]

class PersonalitySystem:
    def __init__(self):
        self.traits = {
            'helpful': 0.9,
            'friendly': 0.8,
            'professional': 0.7,
            'curious': 0.8
        }
        self.interests = ['technologie', 'apprentissage', 'conversation', 'résolution de problèmes']
        
    def adjust_response(self, response, emotional_state):
        if emotional_state == 'joy':
            response = f"😊 {response}"
        elif emotional_state == 'trust':
            response = f"🤝 {response}"
        elif emotional_state == 'interest':
            response = f"🤔 {response}"
        return response

class LongTermMemory:
    def __init__(self):
        self.memory_file = "long_term_memory.json"
        self.memories = self.load_memories()
        self.importance_threshold = 0.7

    def load_memories(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return {'interactions': [], 'learned_topics': [], 'user_preferences': {}}

    def save_memories(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memories, f)

    def add_memory(self, interaction, importance=0.5):
        if importance >= self.importance_threshold:
            self.memories['interactions'].append({
                'content': interaction,
                'timestamp': str(datetime.now()),
                'importance': importance
            })
            self.save_memories()

    def get_relevant_memories(self, query, n=3):
        # Recherche les souvenirs pertinents
        relevant = []
        for memory in self.memories['interactions']:
            similarity = cosine_similarity(
                [self.vectorizer.transform([query]).toarray()[0]],
                [self.vectorizer.transform([memory['content']]).toarray()[0]]
            )[0][0]
            if similarity > 0.3:
                relevant.append((memory, similarity))
        return sorted(relevant, key=lambda x: x[1], reverse=True)[:n]

class TaskManager:
    def __init__(self):
        self.tasks = []
        self.reminders = []
        self.task_file = "tasks.json"
        self.load_tasks()

    def load_tasks(self):
        if os.path.exists(self.task_file):
            with open(self.task_file, 'r') as f:
                data = json.load(f)
                self.tasks = data.get('tasks', [])
                self.reminders = data.get('reminders', [])

    def save_tasks(self):
        with open(self.task_file, 'w') as f:
            json.dump({
                'tasks': self.tasks,
                'reminders': self.reminders
            }, f)

    def add_task(self, description, deadline=None):
        task = {
            'description': description,
            'created': str(datetime.now()),
            'deadline': deadline,
            'completed': False
        }
        self.tasks.append(task)
        self.save_tasks()
        return task

    def add_reminder(self, message, trigger_time):
        reminder = {
            'message': message,
            'trigger_time': trigger_time,
            'completed': False
        }
        self.reminders.append(reminder)
        self.save_tasks()
        return reminder

class ReinforcementLearning:
    def __init__(self):
        self.rewards = defaultdict(float)
        self.learning_rate = 0.1
        self.reward_file = "rewards.json"
        self.load_rewards()

    def load_rewards(self):
        if os.path.exists(self.reward_file):
            with open(self.reward_file, 'r') as f:
                self.rewards = defaultdict(float, json.load(f))

    def save_rewards(self):
        with open(self.reward_file, 'w') as f:
            json.dump(dict(self.rewards), f)

    def update_reward(self, action, reward):
        self.rewards[action] = self.rewards[action] + self.learning_rate * (reward - self.rewards[action])
        self.save_rewards()

    def get_best_action(self, possible_actions):
        return max(possible_actions, key=lambda x: self.rewards[x])

class PluginSystem:
    def __init__(self):
        self.plugins = {}
        self.plugin_dir = "plugins"
        self.load_plugins()

    def load_plugins(self):
        if not os.path.exists(self.plugin_dir):
            os.makedirs(self.plugin_dir)
            
        for plugin_file in glob.glob(f"{self.plugin_dir}/*.py"):
            name = os.path.splitext(os.path.basename(plugin_file))[0]
            spec = importlib.util.spec_from_file_location(name, plugin_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.plugins[name] = module

    def execute_plugin(self, name, *args, **kwargs):
        if name in self.plugins:
            return self.plugins[name].execute(*args, **kwargs)
        return None

class ExternalAPI:
    def __init__(self):
        self.api_keys = self.load_api_keys()
        self.cache = {}
        self.cache_duration = timedelta(hours=1)

    def load_api_keys(self):
        try:
            with open('api_keys.json', 'r') as f:
                return json.load(f)
        except:
            return {}

    def get_weather(self, location):
        cache_key = f'weather_{location}'
        if cache_key in self.cache:
            if datetime.now() - self.cache[cache_key]['timestamp'] < self.cache_duration:
                return self.cache[cache_key]['data']

        if 'openweather' in self.api_keys:
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': location,
                'appid': self.api_keys['openweather'],
                'units': 'metric',
                'lang': 'fr'
            }
            response = requests.get(url, params=params)
            if response.ok:
                data = response.json()
                self.cache[cache_key] = {
                    'timestamp': datetime.now(),
                    'data': data
                }
                return data
        return None

class ContextManager:
    def __init__(self):
        self.context_window = 10
        self.contexts = defaultdict(float)
        self.topic_tracker = []
        self.conversation_history = []
        
    def get_context(self):
        """Retourne le contexte actuel sous forme de texte"""
        if not self.conversation_history:
            return ""
            
        # Prendre les N derniers échanges
        recent_history = self.conversation_history[-self.context_window:]
        
        # Construire le contexte avec le sujet actuel
        current_topic = self.topic_tracker[-1] if self.topic_tracker else "général"
        context = f"Sujet actuel: {current_topic}\n\n"
        
        # Ajouter l'historique récent
        context += "\n".join([
            f"Utilisateur: {exchange['user']}\nAssistant: {exchange['response']}"
            for exchange in recent_history
        ])
        
        return context
        
    def update_context(self, user_input, response):
        # Ajouter à l'historique
        self.conversation_history.append({
            'user': user_input,
            'response': response,
            'timestamp': str(datetime.now())
        })
        
        # Limiter la taille de l'historique
        if len(self.conversation_history) > self.context_window * 2:
            self.conversation_history = self.conversation_history[-self.context_window:]
            
        # Extraire les mots clés
        keywords = set(word.lower() for word in user_input.split() if len(word) > 3)
        
        # Mettre à jour les scores de contexte
        for keyword in keywords:
            self.contexts[keyword] += 1.0
            
        # Détecter le changement de sujet
        current_topic = self._detect_topic(keywords)
        self.topic_tracker.append(current_topic)
        
        # Garder seulement les N derniers sujets
        if len(self.topic_tracker) > self.context_window:
            self.topic_tracker.pop(0)

    def get_current_topic(self):
        """Retourne le sujet actuel de la conversation"""
        return self.topic_tracker[-1] if self.topic_tracker else "général"
            
    def _detect_topic(self, keywords):
        # Liste de sujets prédéfinis
        topics = {
            "tech": {"ordinateur", "logiciel", "programme", "application"},
            "météo": {"temps", "pluie", "soleil", "température"},
            "santé": {"santé", "maladie", "douleur", "médecin"},
            # Ajouter d'autres sujets...
        }
        
        # Trouver le sujet le plus proche
        max_overlap = 0
        current_topic = "général"
        
        for topic, topic_keywords in topics.items():
            overlap = len(keywords & topic_keywords)
            if overlap > max_overlap:
                max_overlap = overlap
                current_topic = topic
                
        return current_topic

class EmotionManager(EmotionSystem):
    def __init__(self):
        super().__init__()
        self.emotional_memory = []
        self.empathy_level = 0.7
        
    def update_emotion(self, text, context=None):
        # Analyse émotionnelle plus sophistiquée
        base_emotion = super().update_emotion(text)
        
        # Ajouter le contexte à l'analyse
        if (context):
            context_emotion = self._analyze_context(context)
            base_emotion = self._blend_emotions(base_emotion, context_emotion)
            
        self.emotional_memory.append({
            'emotion': base_emotion,
            'timestamp': datetime.now(),
            'text': text
        })
        
        return base_emotion
        
    def _analyze_context(self, context):
        # Analyser les émotions dans le contexte
        return super().update_emotion(context)
        
    def _blend_emotions(self, emotion1, emotion2):
        # Mélanger deux émotions avec des poids
        return emotion1 if random.random() < self.empathy_level else emotion2

class PrioritizedTaskManager(TaskManager):
    def __init__(self):
        super().__init__()
        self.priority_levels = {
            'haute': 3,
            'moyenne': 2,
            'basse': 1
        }
        
    def add_task(self, description, deadline=None, priority='moyenne'):
        task = {
            'description': description,
            'created': str(datetime.now()),
            'deadline': deadline,
            'completed': False,
            'priority': self.priority_levels.get(priority, 1)
        }
        self.tasks.append(task)
        self.tasks.sort(key=lambda x: (-x['priority'], x.get('deadline', 'inf')))
        self.save_tasks()
        return task

class HumorDetector:
    def __init__(self):
        self.humor_keywords = {
            "blague", "drôle", "rire", "mdr", "lol", "ptdr", "humour",
            "joke", "amusant", "marrant", "rigoler", "rigolo"
        }
        self.joke_patterns = [
            r"pourquoi .*\?",
            r"que fait .*\?",
            r"qu'est-ce qui .*\?"
        ]
    
    def is_humor_request(self, text):
        text_lower = text.lower()
        # Vérifier les mots clés
        if any(word in text_lower for word in self.humor_keywords):
            return True
        # Vérifier les patterns de blagues
        return any(re.match(pattern, text_lower) for pattern in self.joke_patterns)

class SpellChecker:
    def __init__(self):
        self.common_errors = {
            "assisatnt": "assistant",
            "ameloire": "améliorer",
            "san ai": "San AI",
            # Ajouter d'autres corrections courantes
        }
    
    def correct(self, text):
        for error, correction in self.common_errors.items():
            text = text.replace(error, correction)
        return text

class DialogueManager:
    def __init__(self):
        self.conversation_tree = {}
        self.current_context = None
        self.fallback_responses = [
            "Je ne suis pas sûr de comprendre, pouvez-vous reformuler ?",
            "Pourriez-vous être plus précis ?",
            "Je ne saisis pas complètement, pouvez-vous expliquer autrement ?"
        ]

    def manage_dialogue(self, user_input, context):
        # Analyse du contexte
        if not self.current_context:
            self.current_context = self._identify_context(user_input)
        
        # Construction de la réponse
        response = self._build_response(user_input, context)
        
        return response

    def _identify_context(self, text):
        # Ajout de mots-clés pour identifier le contexte
        contexts = {
            'question': ['comment', 'pourquoi', 'quand', 'où', 'qui', 'quoi'],
            'request': ['peux-tu', 'pourrais-tu', 'aide-moi', 'j\'ai besoin'],
            'statement': ['je pense', 'je crois', 'selon moi']
        }
        
        text_lower = text.lower()
        for context_type, keywords in contexts.items():
            if any(keyword in text_lower for keyword in keywords):
                return context_type
        
        return 'general'

    def _build_response(self, user_input, context):
        response_type = self.current_context if self.current_context else 'general'
        
        if response_type == 'question':
            return self._handle_question(user_input, context)
        elif response_type == 'request':
            return self._handle_request(user_input, context)
        elif response_type == 'statement':
            return self._handle_statement(user_input, context)
        
        return random.choice(self.fallback_responses)

    def _handle_question(self, question, context):
        """Gère les questions de l'utilisateur"""
        try:
            # Extraire les mots clés de la question
            keywords = set(word.lower() for word in question.split() if len(word) > 3)

            # Rechercher dans la base de connaissances
            match = self.knowledge_base.find_similar_question(question)
            if (match):
                return match

            # Analyser le type de question
            if any(w in question.lower() for w in ['quoi', 'que', 'qu']):
                return self._handle_what_question(question, keywords)
            elif any(w in question.lower() for w in ['comment', 'combien']):
                return self._handle_how_question(question, keywords)
            elif any(w in question.lower() for w in ['pourquoi', 'pour quoi']):
                return self._handle_why_question(question, keywords)
            else:
                # Utiliser le modèle de langage pour générer une réponse
                return self.language_model.generate_response(question, context)

        except Exception as e:
            print(f"Erreur dans _handle_question: {e}")
            return "Je suis désolé, je ne comprends pas bien votre question. Pourriez-vous la reformuler ?"

    def _handle_request(self, request, context):
        """Gère les demandes et requêtes de l'utilisateur"""
        try:
            request_lower = request.lower()
            
            # Vérifier les demandes de rappel
            if 'rappelle' in request_lower or 'rappel' in request_lower:
                return self._handle_reminder_request(request)
                
            # Vérifier les demandes d'aide
            elif 'aide' in request_lower or 'help' in request_lower:
                return self._handle_help_request(request)
                
            # Vérifier les demandes d'information
            elif any(word in request_lower for word in ['explique', 'montre', 'dis']):
                return self._handle_info_request(request, context)
                
            else:
                # Utiliser le modèle de langage pour des requêtes générales
                response = self.language_model.generate_response(request, context)
                # Ajouter un indicateur d'action
                return f"Je vais vous aider avec ça. {response}"

        except Exception as e:
            print(f"Erreur dans _handle_request: {e}")
            return "Je ne peux pas traiter cette requête pour le moment."

    def _handle_statement(self, statement, context):
        """Gère les affirmations de l'utilisateur"""
        try:
            # Analyser le sentiment
            sentiment = self.analyze_sentiment(statement)['sentiment']
            
            # Classer le type d'affirmation
            if sentiment > 0.3:
                return self._handle_positive_statement(statement)
            elif sentiment < -0.3:
                return self._handle_negative_statement(statement)
            else:
                # Pour les affirmations neutres, essayer de continuer la conversation
                response = self.language_model.generate_response(statement, context)
                
                # Mémoriser l'information si pertinente
                if len(statement.split()) > 3:  # Éviter les affirmations trop courtes
                    self.memory_assistant.store_memory(
                        content=statement,
                        category="statements",
                        importance=0.5
                    )
                    
                return response

        except Exception as e:
            print(f"Erreur dans _handle_statement: {e}")
            return "Je comprends. Pouvez-vous m'en dire plus?"

class ConversationScenario:
    def __init__(self):
        self.scenarios = {
            'help_request': {
                'patterns': ['aide-moi', 'besoin d\'aide', 'comment faire', 'peux-tu m\'aider'],
                'required_info': ['sujet', 'contexte'],
                'response_template': "Je vais vous aider avec {sujet}. Voici la démarche à suivre: {steps}"
            },
            'problem_solving': {
                'patterns': ['j\'ai un problème', 'ça ne marche pas', 'erreur'],
                'required_info': ['problème', 'contexte', 'déjà_essayé'],
                'response_template': "Pour résoudre le problème de {problème}, essayons d'abord {solution}"
            },
            'information_request': {
                'patterns': ['qu\'est-ce que', 'comment', 'pourquoi', 'explique-moi'],
                'required_info': ['sujet'],
                'response_template': "Concernant {sujet}, voici ce que je peux vous dire: {explication}"
            }
        }
        self.current_scenario = None
        self.missing_info = {}

    def identify_scenario(self, text):
        for scenario, data in self.scenarios.items():
            if any(pattern in text.lower() for pattern in data['patterns']):
                return scenario
        return None

    def get_missing_info(self, scenario, context):
        required = set(self.scenarios[scenario]['required_info'])
        available = set(key for key in context if context[key] is not None)
        return required - available

class FeedbackSystem:
    def __init__(self):
        self.feedback_history = []
        self.feedback_threshold = 0.7
        self.improvement_suggestions = []

    def add_feedback(self, response, user_reaction, context):
        feedback = {
            'response': response,
            'reaction': user_reaction,
            'context': context,
            'timestamp': datetime.now()
        }
        self.feedback_history.append(feedback)
        self.analyze_feedback_patterns()

    def analyze_feedback_patterns(self):
        recent_feedback = self.feedback_history[-50:]
        negative_patterns = [f for f in recent_feedback if f['reaction'] == 'negative']
        if len(negative_patterns) / len(recent_feedback) > (1 - self.feedback_threshold):
            self.generate_improvement_suggestion(negative_patterns)

    def generate_improvement_suggestion(self, negative_patterns):
        common_contexts = Counter([f['context'] for f in negative_patterns])
        most_common = common_contexts.most_common(1)[0]
        self.improvement_suggestions.append({
            'context': most_common[0],
            'frequency': most_common[1],
            'timestamp': datetime.now()
        })

class ResponseCache:
    def __init__(self):
        self.cache = {}
        self.max_size = 1000
        self.ttl = timedelta(hours=1)

    def get_response(self, query):
        if query in self.cache:
            entry = self.cache[query]
            if datetime.now() - entry['timestamp'] < self.ttl:
                return entry['response']
            else:
                del self.cache[query]
        return None

    def add_response(self, query, response):
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.items(), key=lambda x: x[1]['timestamp'])
            del self.cache[oldest[0]]
        
        self.cache[query] = {
            'response': response,
            'timestamp': datetime.now(),
            'usage_count': 0
        }

class ConversationStats:
    def __init__(self):
        self.stats = {
            'total_interactions': 0,
            'cache_hits': 0,
            'response_times': [],
            'topic_distribution': Counter(),
            'session_length': []
        }

    def log_interaction(self, topic, response_time):
        self.stats['total_interactions'] += 1
        self.stats['response_times'].append(response_time)
        self.stats['topic_distribution'][topic] += 1

    def log_cache_hit(self):
        self.stats['cache_hits'] += 1

    def log_session_end(self, duration):
        self.stats['session_length'].append(duration)

    def get_analytics(self):
        return {
            'avg_response_time': sum(self.stats['response_times']) / len(self.stats['response_times']),
            'cache_hit_rate': self.stats['cache_hits'] / self.stats['total_interactions'],
            'popular_topics': self.stats['topic_distribution'].most_common(5),
            'avg_session_length': sum(self.stats['session_length']) / len(self.stats['session_length'])
        }

class SanAI:
    def __init__(self):
        self.name = "San"
        self.creation_date = datetime.now()  # Modifié ici
        self.memory_file = "memory.json"
        self.load_memory()
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()
        self.train_basic_model()
        self.conversation_history = []
        self.voice_handler = VoiceHandler()
        self.knowledge_base = KnowledgeBase()
        self.learning_mode = False
        self.language_model = LanguageModel()
        self.knowledge_base.language_model = self.language_model
        self.context_history = []
        self.training_data = TrainingData()
        self.train_enhanced_model()
        self.emotion_system = EmotionSystem()
        self.personality = PersonalitySystem()
        self.long_term_memory = LongTermMemory()
        self.last_learning_check = datetime.now()  # Modifié ici
        self.learning_interval = timedelta(hours=1)
        self.task_manager = TaskManager()
        self.rl = ReinforcementLearning()
        self.plugin_system = PluginSystem()
        self.external_api = ExternalAPI()
        self.context_manager = ContextManager()
        self.emotion_manager = EmotionManager()
        self.task_manager = PrioritizedTaskManager()
        self.activation_keywords = {"san", "sent", "sand", "son"}  # Mots similaires pour la reconnaissance vocale
        self.humor_detector = HumorDetector()
        self.spell_checker = SpellChecker()
        self.jokes = [
            "Pourquoi les plongeurs plongent-ils toujours en arrière ? Parce que sinon ils tombent dans le bateau !",
            "Que fait une fraise sur un cheval ? Tagada tagada !",
            "Quel est le comble pour un électricien ? Ne pas être au courant !",
            # Ajouter d'autres blagues
        ]
        self.dialogue_manager = DialogueManager()
        self.last_interaction_time = datetime.now()
        self.interaction_timeout = timedelta(minutes=5)
        self.learning_rate = 0.1
        self.max_context_length = 10
        self.min_confidence_threshold = 0.6
        self.conversation_scenario = ConversationScenario()
        self.feedback_system = FeedbackSystem()
        self.response_cache = ResponseCache()
        self.conversation_stats = ConversationStats()
        self.voice_profile_manager = VoiceProfileManager()
        self.pending_voice_identification = {}
        self.memory_assistant = MemoryAssistant()
        self.continuous_listening = False
        self.memory_keywords = {
            'important': ['important', 'crucial', 'essentiel', 'urgent'],
            'routine': ['toujours', 'habitude', 'quotidien', 'routine'],
            'people': ['avec', 'rencontré', 'vu', 'parlé'],
            'location': ['à', 'chez', 'dans', 'au', 'en']
        }
        from memory_manager import MemoryManager
        self.memory_manager = MemoryManager()

    def is_activated(self, text):
        """Vérifie si l'assistant est appelé dans le texte"""
        words = set(text.lower().split())
        return bool(words & self.activation_keywords)

    def process_input(self, user_input):
        try:
            # Correction orthographique et nettoyage
            cleaned_input = self.spell_checker.correct(user_input.strip())
            
            # Vérifier l'activation
            if not self.is_activated(cleaned_input):
                return None

            # Extraire le contexte actuel
            current_context = self.context_manager.get_context()
            
            # Identifier le locuteur si disponible
            speaker = self.voice_profile_manager.get_current_speaker()
            
            # Analyser l'intention et le contexte
            intent = self._analyze_intent(cleaned_input)
            
            # Traiter les commandes spéciales
            if intent.get('type') == 'command':
                return self._handle_command(intent, cleaned_input)
            
            # Générer la réponse
            response = self._generate_response(cleaned_input, current_context, speaker)
            
            # Mémoriser l'interaction
            self._store_interaction(cleaned_input, response, speaker)
            
            # Mettre à jour le contexte
            self.context_manager.update_context(cleaned_input, response)
            
            return response
            
        except Exception as e:
            print(f"Erreur dans process_input: {e}")
            return "Je suis désolé, j'ai rencontré une erreur lors du traitement de votre demande."

    def _analyze_intent(self, text):
        """Analyse l'intention de l'utilisateur"""
        # Détecter les commandes
        if text.startswith(('rappelle', 'ajoute', 'note')):
            return {'type': 'command', 'action': 'task'}
            
        # Détecter les questions
        if any(q in text.lower() for q in ['quoi', 'comment', 'pourquoi', 'quand', 'où']):
            return {'type': 'question'}
            
        return {'type': 'statement'}

    def _handle_command(self, intent, text):
        """Gère les commandes spéciales"""
        if intent.get('action') == 'task':
            return self._handle_task_command(text)
        return None

    def _handle_task_command(self, text):
        """Gère les commandes de tâches et rappels"""
        try:
            # Extraire la date/heure si présente
            time_match = re.search(r"(?:à|dans|pour) ([\w\s]+)", text)
            task_time = None
            if time_match:
                try:
                    task_time = parse_time(time_match.group(1))
                except:
                    pass

            # Extraire la description de la tâche
            task_desc = re.sub(r"rappelle(?:-moi)? |ajoute |note |(?:à|dans|pour) [\w\s]+", "", text).strip()

            # Ajouter la tâche
            if task_time:
                self.task_manager.add_reminder(task_desc, str(task_time))
                return f"D'accord, je vous rappellerai de {task_desc} à {task_time.strftime('%H:%M le %d/%m/%Y')}"
            else:
                self.task_manager.add_task(task_desc)
                return f"J'ai ajouté la tâche : {task_desc}"

        except Exception as e:
            print(f"Erreur dans _handle_task_command: {e}")
            return "Je n'ai pas pu traiter cette commande. Pourriez-vous la reformuler ?"

    def _generate_response(self, input_text, context, speaker=None):
        """Génère une réponse contextuelle"""
        # Chercher dans la mémoire
        memory_response = self.memory_manager.find_relevant_memory(input_text)
        
        if memory_response and memory_response.get('confidence', 0) > self.confidence_threshold:
            return memory_response['content']
            
        # Utiliser le modèle de langage
        response = self.language_model.generate_response(input_text, context)
        
        # Personnaliser pour le locuteur
        if speaker:
            response = self._personalize_response(response, speaker)
            
        return response

    def _store_interaction(self, input_text, response, speaker=None):
        """Stocke l'interaction dans la mémoire"""
        # Calculer l'importance
        importance = self._calculate_importance(input_text)
        
        # Stocker dans la mémoire à long terme
        self.memory_manager.add_memory(
            content=f"User: {input_text}\nAssistant: {response}",
            category="conversation",
            importance=importance,
            speaker=speaker
        )
        
        # Mettre à jour les statistiques d'apprentissage
        self.conversation_stats.log_interaction(
            topic=self.context_manager.get_current_topic(),
            response_time=0.1  # À remplacer par le temps réel
        )

    def _personalize_response(self, response, speaker):
        """Personnalise la réponse en fonction du locuteur"""
        # Obtenir les préférences du locuteur
        speaker_prefs = self.voice_profile_manager.get_profile_info(speaker)
        
        if not speaker_prefs:
            return response
            
        # Adapter le ton et le style
        if speaker_prefs.get('formal', False):
            response = response.replace('tu', 'vous')
            
        # Ajouter des références personnelles
        if 'name' in speaker_prefs:
            response = f"{speaker_prefs['name']}, {response}"
            
        return response

    def _calculate_importance(self, text):
        """Calcule l'importance d'une interaction"""
        factors = {
            'length': min(1.0, len(text) / 100),
            'question': 0.8 if '?' in text else 0.5,
            'command': 0.9 if text.startswith(('rappelle', 'ajoute', 'note')) else 0.5,
            'keywords': sum(1 for word in self.memory_keywords.values() if word in text.lower()) * 0.1
        }
        return min(1.0, sum(factors.values()) / len(factors))

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                self.memory = json.load(f)
        else:
            self.memory = {}
            self.save_memory()

    def save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f)

    def train_basic_model(self):
        # Données d'entraînement basiques
        X = [
            "bonjour", "salut", "hello",
            "comment vas-tu", "ça va",
            "au revoir", "bye", "à plus",
            "quel temps fait-il",
            "quelle heure est-il"
        ]
        y = [
            "greeting", "greeting", "greeting",
            "mood_query", "mood_query",
            "goodbye", "goodbye", "goodbye",
            "weather", "time"
        ]
        self.vectorizer.fit(X)
        X_transformed = self.vectorizer.transform(X)
        self.classifier.fit(X_transformed, y)

    def train_enhanced_model(self):
        X = []
        y = []
        for category, data in self.training_data.categories.items():
            X.extend(data["patterns"])
            y.extend([category] * len(data["patterns"]))
        
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        X_transformed = self.vectorizer.fit_transform(X)
        self.classifier = MultinomialNB()
        self.classifier.fit(X_transformed, y)

    def get_context(self):
        if not self.context_history:
            return ""
        return "\n".join([
            f"User: {msg['user']}\nAssistant: {msg['assistant']}"
            for msg in self.context_history[-5:]
        ])

    def get_response_for_category(self, category, **kwargs):
        # Amélioration des réponses
        if (category not in self.training_data.categories):
            # Utiliser le modèle de langage pour générer une réponse appropriée
            context = self.get_context()
            return self.language_model.generate_response(f"Comment répondre à une question de type {category}?", context)
            
        responses = self.training_data.categories[category]["responses"]
        response = random.choice(responses)
        
        # Personnalisation de la réponse selon l'heure
        current_hour = datetime.now().hour
        if (current_hour < 6):
            response = "En cette nuit tardive, " + response
        elif (current_hour < 12):
            response = "En cette belle matinée, " + response
        elif (current_hour < 18):
            response = "En cet après-midi, " + response
        else:
            response = "En cette soirée, " + response
            
        return response

    def analyze_sentiment(self, text):
        # Amélioration de l'analyse des sentiments
        analysis = TextBlob(text)
        sentiment = analysis.sentiment.polarity
        
        # Ajout d'une analyse plus fine
        emotions = {
            'joie': 0,
            'tristesse': 0,
            'colère': 0,
            'surprise': 0
        }
        
        # Mots-clés pour chaque émotion
        emotion_keywords = {
            'joie': ['content', 'heureux', 'super', 'génial', 'excellent'],
            'tristesse': ['triste', 'déçu', 'malheureux', 'dommage'],
            'colère': ['énervé', 'fâché', 'agacé', 'furieux'],
            'surprise': ['wow', 'incroyable', 'surprenant', 'étonnant']
        }
        
        # Calculer le score pour chaque émotion
        text_lower = text.lower()
        for emotion, keywords in emotion_keywords.items():
            emotions[emotion] = sum(1 for word in keywords if word in text_lower)
            
        return {
            'sentiment': sentiment,
            'emotions': emotions
        }

    def check_for_learning(self):
        now = datetime.now()  # Modifié ici
        if now - self.last_learning_check > self.learning_interval:
            self.auto_learn()
            self.last_learning_check = now

    def auto_learn(self):
        # Analyser les conversations récentes pour l'apprentissage
        recent_interactions = self.conversation_history[-50:]
        new_patterns = {}
        
        for interaction in recent_interactions:
            if 'user' in interaction and 'assistant' in interaction:
                text = interaction['user']
                response = interaction['assistant']
                
                # Identifier les motifs récurrents
                if text.lower() not in self.training_data.categories:
                    similar_responses = [i['assistant'] for i in recent_interactions 
                                      if i.get('user', '').lower() == text.lower()]
                    if len(similar_responses) >= 2:  # Motif récurrent
                        new_patterns[text.lower()] = similar_responses

        # Ajouter les nouveaux motifs au modèle
        if new_patterns:
            self.update_training_data(new_patterns)

    def update_training_data(self, new_patterns):
        for pattern, responses in new_patterns.items():
            category_name = f"learned_{len(self.training_data.categories)}"
            self.training_data.categories[category_name] = {
                "patterns": [pattern],
                "responses": responses
            }
        self.train_enhanced_model()

    def process_voice(self, audio_file):
        """Traitement vocal amélioré"""
        try:
            # Prétraitement de l'audio
            text = self.voice_handler.speech_to_text(audio_file)
            print(f"Texte reconnu: {text}")
            
            if not text.strip():
                return {
                    "text": "Je n'ai pas bien compris, pourriez-vous répéter ?",
                    "audio_file": self.voice_handler.text_to_speech(
                        "Je n'ai pas bien compris, pourriez-vous répéter ?"
                    )
                }
            
            # Traitement de la requête
            response = self.process_input(text)
            
            # Si l'assistant n'est pas appelé
            if response is None:
                return {
                    "text": "",
                    "audio_file": None
                }
            
            # Génération de la réponse vocale
            audio_response = self.voice_handler.text_to_speech(response)
            
            return {
                "text": response,
                "audio_file": audio_response
            }

        except Exception as e:
            error_msg = "Désolé, une erreur s'est produite lors du traitement vocal."
            return {
                "text": error_msg,
                "audio_file": self.voice_handler.text_to_speech(error_msg)
            }

    def get_conversation_history(self):
        return self.conversation_history

    def feedback(self, response_id, positive=True):
        """Permet à l'utilisateur de donner un feedback sur une réponse"""
        reward = 1.0 if positive else -0.1
        self.rl.update_reward(response_id, reward)

    def process_input(self, user_input):
        # Correction orthographique améliorée
        user_input = self.spell_checker.correct(user_input)
        
        # Vérification du timeout
        now = datetime.now()
        if now - self.last_interaction_time > self.interaction_timeout:
            self.context_manager = ContextManager()  # Reset du contexte
        self.last_interaction_time = now
        
        # Vérification de l'activation
        if not self.is_activated(user_input):
            return None
            
        cleaned_input = ' '.join(word for word in user_input.split() 
                               if word not in self.activation_keywords)
                               
        if not cleaned_input.strip():
            return "Oui, je vous écoute ?"

        # Obtention du contexte
        context = self.context_manager.get_context()
        
        # Utilisation du DialogueManager pour gérer la conversation
        response = self.dialogue_manager.manage_dialogue(cleaned_input, context)
        
        if not response:  # Si le DialogueManager ne trouve pas de réponse appropriée
            response = self._process_input_internal(cleaned_input)

        # Mise à jour du contexte et de la mémoire
        self.context_manager.update_context(cleaned_input, response)
        self.update_memory(cleaned_input, response)
        
        return response

    def update_memory(self, user_input, response):
        # Stockage intelligent dans la mémoire
        memory_entry = {
            "user_input": user_input,
            "response": response,
            "timestamp": str(datetime.now()),
            "context": self.context_manager.get_current_topic()
        }
        
        # Calcul de l'importance
        importance = self.calculate_importance(user_input, response)
        
        if importance > self.min_confidence_threshold:
            self.long_term_memory.add_memory(
                f"User: {user_input}\nAssistant: {response}", 
                importance
            )

    def calculate_importance(self, user_input, response):
        # Calcul de l'importance basé sur plusieurs facteurs
        factors = {
            'length': len(user_input) / 100,  # Normalisation
            'sentiment': abs(TextBlob(user_input).sentiment.polarity),
            'complexity': len(set(user_input.split())) / len(user_input.split()),
            'question': 1.0 if '?' in user_input else 0.5,
        }
        
        return sum(factors.values()) / len(factors)

    def ask_speaker_identity(self, audio_file, text):
        """Demande l'identité d'un nouveau locuteur"""
        response = "J'ai détecté une nouvelle voix. Qui est-ce qui parle ?"
        self.pending_voice_identification[audio_file] = text
        return self.voice_handler.text_to_speech(response)

    def add_speaker_identity(self, name, audio_file):
        """Ajoute l'identité d'un locuteur"""
        if self.voice_profile_manager.add_voice_profile(name, audio_file):
            if (audio_file in self.pending_voice_identification):
                text = self.pending_voice_identification[audio_file]
                self.learn_from_conversation(text, name)
                del self.pending_voice_identification[audio_file]
            return f"J'ai enregistré la voix de {name}. Je pourrai maintenant le/la reconnaître."
        return "Désolé, je n'ai pas pu enregistrer cette voix."

    def learn_from_conversation(self, text, speaker=None):
        """Apprentissage à partir des conversations avec identification du locuteur"""
        if speaker:
            # Ajouter les informations du locuteur à la mémoire
            self.long_term_memory.add_memory(f"Conversation avec {speaker}: {text}", 
                                          importance=0.8)
            
            # Mettre à jour le contexte
            self.context_manager.update_context(f"{speaker}: {text}", 
                                             self._generate_response_for_speaker(speaker))
            
            # Analyser le texte pour extraire des informations pertinentes
            analysis = self.analyze_conversation(text, speaker)
            
            # Mettre à jour la base de connaissances du locuteur
            self.update_speaker_knowledge(speaker, analysis)
            
            # Ajuster le modèle de langage pour ce locuteur
            self.adapt_language_model(speaker, text)
            
        else:
            # Apprentissage standard sans identification du locuteur
            importance = self.calculate_importance(text, "")
            self.long_term_memory.add_memory(text, importance)
            
            # Mettre à jour le modèle d'apprentissage
            self.update_training_data({text.lower(): [self._generate_response_for_speaker(None)]})
            
            # Réentraîner le modèle si nécessaire
            if len(self.conversation_history) % 10 == 0:
                self.train_enhanced_model()

    def analyze_conversation(self, text, speaker):
        """Analyse le contenu de la conversation pour en extraire des informations"""
        analysis = {
            'sentiment': self.analyze_sentiment(text),
            'topics': self.context_manager.get_current_topic(),
            'keywords': self.extract_keywords(text),
            'timestamp': str(datetime.now())
        }
        return analysis

    def update_speaker_knowledge(self, speaker, analysis):
        """Met à jour la base de connaissances spécifique au locuteur"""
        if not hasattr(self, 'speaker_knowledge'):
            self.speaker_knowledge = defaultdict(dict)
            
        self.speaker_knowledge[speaker].update({
            'last_interaction': analysis['timestamp'],
            'topics': analysis['topics'],
            'sentiment_history': self.speaker_knowledge[speaker].get('sentiment_history', []) + [analysis['sentiment']]
        })

    def adapt_language_model(self, speaker, text):
        """Adapte le modèle de langage aux particularités du locuteur"""
        if hasattr(self, 'speaker_knowledge') and speaker in self.speaker_knowledge:
            # Utiliser l'historique des interactions pour affiner les réponses
            history = self.speaker_knowledge[speaker]
            
            # Ajuster les paramètres du modèle selon les préférences du locuteur
            if len(history.get('sentiment_history', [])) > 0:
                avg_sentiment = sum(history['sentiment_history']) / len(history['sentiment_history'])
                self.language_model.adjust_parameters(speaker, avg_sentiment)

    def extract_keywords(self, text):
        """Extrait les mots-clés importants du texte"""
        # Utiliser TF-IDF pour identifier les mots importants
        text_vector = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Obtenir les indices des mots les plus importants
        important_words_idx = text_vector.toarray()[0].argsort()[-5:][::-1]
        
        # Retourner les mots-clés
        return [feature_names[idx] for idx in important_words_idx]

    def _generate_response_for_speaker(self, speaker):
        """Génère une réponse adaptée au locuteur"""
        speaker_memories = self.long_term_memory.get_speaker_memories(speaker)
        context = f"Conversation avec {speaker}. "
        if speaker_memories:
            context += f"Précédentes interactions: {speaker_memories[-3:]}"
        return self.language_model.generate_response(context)

    def process_continuous_input(self, text):
        """Traite les entrées du mode d'écoute continue"""
        # Calculer l'importance du souvenir
        importance = self._calculate_memory_importance(text)
        
        # Extraire le contexte
        category = self._categorize_memory(text)
        location = self._extract_location(text)
        people = self._extract_people(text)
        
        # Stocker le souvenir
        self.memory_assistant.store_memory(
            content=text,
            category=category,
            importance=importance,
            location=location,
            people=people,
            context=self.context_manager.get_context()
        )

        # Générer une confirmation
        confirmation = self._generate_memory_confirmation(text, category)
        return confirmation

    def _calculate_memory_importance(self, text):
        """Calcule l'importance d'un souvenir basé sur des mots-clés"""
        importance = 0.5  # Importance par défaut
        text_lower = text.lower()
        
        # Augmenter l'importance pour les mots-clés importants
        for word in self.memory_keywords['important']:
            if word in text_lower:
                importance += 0.1

        # Analyser le sentiment pour ajuster l'importance
        sentiment = self.analyze_sentiment(text)['sentiment']
        importance += abs(sentiment) * 0.2
        
        return min(1.0, importance)

    def _categorize_memory(self, text):
        """Catégorise un souvenir"""
        text_lower = text.lower()
        categories = {
            'rendez-vous': ['rendez-vous', 'rencontre', 'réunion'],
            'tâche': ['faire', 'tâche', 'objectif'],
            'conversation': ['parlé', 'dit', 'discuté'],
            'événement': ['passé', 'arrivé', 'événement']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        return 'général'

    def get_memory_summary(self, period="aujourd'hui"):
        """Génère un résumé des souvenirs"""
        if period == "aujourd'hui":
            summary = self.memory_assistant.get_daily_summary()
            response = "Voici le résumé de votre journée :\n\n"
            
            for category, memories in summary.items():
                response += f"\n{category.title()} :\n"
                for memory in sorted(memories, key=lambda x: x['importance'], reverse=True):
                    response += f"- {memory['content']}\n"
            
            return response
        return "Je ne peux générer un résumé que pour aujourd'hui pour le moment."

    def find_memory(self, query):
        """Recherche dans les souvenirs"""
        memories = self.memory_assistant.find_related_memories(query)
        if memories:
            response = "Voici ce que j'ai trouvé dans vos souvenirs :\n\n"
            for i, memory in enumerate(memories, 1):
                response += f"{i}. {memory}\n"
            return response
        return "Je n'ai pas trouvé de souvenirs correspondant à votre recherche."

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60)
ai = SanAI()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    if 'input' in data:
        response = ai.process_input(data['input'])
        return jsonify({
            "response": response,
            "history": ai.get_conversation_history()
        })
    return jsonify({"error": "No input provided"}), 400

@app.route('/voice', methods=['POST'])
def voice():
    if 'audio' not in request.files:
        return jsonify({
            "success": False,
            "message": "Fichier audio manquant",
            "response": None,
            "audio_path": None
        }), 400
    
    try:
        audio_file = request.files['audio']
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        audio_file.save(temp_file.name)
        
        result = ai.process_voice(temp_file.name)
        os.unlink(temp_file.name)
        
        if not result or not result.get("text"):
            return jsonify({
                "success": False,
                "message": "Aucune parole détectée",
                "response": None,
                "audio_path": None
            })
        
        return jsonify({
            "success": True,
            "response": result["text"],
            "audio_path": result["audio_file"],
            "message": "Traitement réussi"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}",
            "response": None,
            "audio_path": None
        }), 500

@app.route('/audio/<path:filename>')
@app.route('/static/audio/<path:filename>')
def serve_audio(filename):
    if filename.startswith('static/audio/'):
        return send_file(filename, mimetype='audio/mp3')
    return send_file(f"static/audio/{filename}", mimetype='audio/mp3')

@app.route('/mobile')
def mobile():
    """Route pour l'interface mobile"""
    return render_template('mobile.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    ai.voice_handler.start_listening()

@socketio.on('audio_stream')
@socketio.on('continuous_audio')
def handle_continuous_audio(audio_data):
    try:
        # Créer un fichier temporaire pour le flux audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp_file:
            # Sauvegarder les données audio
            tmp_file.write(audio_data)
            tmp_file.flush()
            
            # Traiter l'audio
            result = ai.voice_handler.process_continuous_stream(tmp_file.name)
            
            if result and result.get('text'):
                # Émettre la transcription
                socketio.emit('continuous_transcription', {
                    'text': result['text']
                })
                
                # Traiter la réponse
                response = ai.process_input(result['text'])
                if response:
                    audio_path = ai.voice_handler.text_to_speech(response)
                    socketio.emit('response', {
                        'text': response,
                        'audio_path': audio_path,
                        'success': True
                    })
    
    except Exception as e:
        print(f"Erreur dans le traitement audio continu: {str(e)}")
        socketio.emit('error', {'message': 'Erreur dans le traitement audio continu'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    ai.voice_handler.is_listening = False

if __name__ == '__main__':
    # Retiré allow_unsafe_werkzeug car non supporté par eventlet
    socketio.run(app, host='0.0.0.0', port=5050)