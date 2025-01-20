from flask import Flask, request, jsonify, render_template, send_file
from flask_socketio import SocketIO, emit
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
from collections import defaultdict
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

    def speech_to_text(self, audio_file):
        try:
            # Amélioration du prétraitement audio
            if self.noise_reduction:
                # Appliquer la réduction de bruit
                audio_data = self._reduce_noise(audio_file)
            else:
                audio_data = audio_file

            # Augmenter la précision de la reconnaissance
            result = self.model.transcribe(
                audio_data,
                language="fr",
                task="transcribe",
                temperature=0.2,
                no_speech_threshold=0.6,
                logprob_threshold=None
            )
            
            text = result["text"].strip()
            print(f"Texte reconnu amélioré: {text}")
            return text

        except Exception as e:
            print(f"Erreur dans speech_to_text amélioré: {e}")
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
            print("Réception de données audio...")
            
            if not audio_data:
                print("Données audio vides")
                socketio.emit('error', {'message': 'Audio non détecté'})
                return

            # Créer un fichier temporaire WAV
            temp_path = os.path.join(self.temp_dir, f'temp_{datetime.now().timestamp()}.wav')
            
            try:
                # Decoder les données base64 si nécessaire
                if isinstance(audio_data, str):
                    if ',' in audio_data:
                        audio_data = audio_data.split(',')[1]
                    import base64
                    audio_data = base64.b64decode(audio_data)

                # Écrire le fichier WAV avec les bons paramètres
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_data)

                print(f"Fichier temporaire créé: {temp_path}")
                
                # Vérifier la durée du fichier audio
                with wave.open(temp_path, 'rb') as wf:
                    duration = wf.getnframes() / float(wf.getframerate())
                    print(f"Durée audio: {duration}s")
                    
                    if duration < self.min_audio_length:
                        print("Audio trop court")
                        return

                # Vérification du niveau sonore avant traitement
                with wave.open(temp_path, 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16)
                    audio_array = audio_array.astype(np.float32) / 32768.0
                    
                    # Calcul du niveau sonore
                    rms = np.sqrt(np.mean(np.square(audio_array)))
                    if rms < self.noise_threshold:
                        print("Niveau sonore trop faible")
                        socketio.emit('error', {'message': 'Voix trop faible, parlez plus fort'})
                        return
                
                # Réduction du bruit
                filtered_path = self._reduce_noise(temp_path)
                if filtered_path is None:
                    print("Signal audio rejeté - trop de bruit")
                    socketio.emit('error', {'message': 'Trop de bruit, essayez dans un environnement plus calme'})
                    return
                
                # Traiter l'audio filtré
                text = self.speech_to_text(filtered_path)
                
                if not text.strip():
                    print("Aucun texte détecté")
                    socketio.emit('error', {'message': 'Parole non détectée'})
                    return
                
                print(f"Texte détecté: {text}")
                now = datetime.now()
                
                if now - self.last_transcription_time >= self.min_silence_duration:
                    self.last_transcription_time = now
                    socketio.emit('transcription', {'text': text})
                    
                    response = ai.process_input(text)
                    if response:
                        try:
                            audio_path = self.text_to_speech(response)
                            if audio_path:
                                socketio.emit('response', {
                                    'text': response,
                                    'audio_path': audio_path,
                                    'success': True
                                })
                            else:
                                socketio.emit('response', {
                                    'text': response,
                                    'success': False,
                                    'message': 'Erreur de synthèse vocale'
                                })
                        except Exception as e:
                            print(f"Erreur de synthèse vocale: {e}")
                            socketio.emit('response', {
                                'text': response,
                                'success': False,
                                'message': 'Erreur lors de la génération audio'
                            })

            finally:
                # Nettoyage
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    print(f"Erreur lors du nettoyage: {e}")

        except Exception as e:
            print(f"Erreur dans process_stream: {str(e)}")
            socketio.emit('error', {
                'message': 'Erreur dans le traitement audio',
                'details': str(e)
            })

class LanguageModel:
    def __init__(self):
        self.models = {
            'small': {
                'name': "asi/gpt-fr-cased-small",
                'model': None,
                'tokenizer': None
            },
            'large': {
                'name': "bigscience/bloom-3b",  # Modèle large multilingue
                'model': None,
                'tokenizer': None
            }
        }
        self.current_model = 'small'  # Modèle par défaut
        self.device = self._get_device()
        self._initialize_default_model()
        
    def _get_device(self):
        """Détermine le meilleur device disponible pour le modèle"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _initialize_default_model(self):
        try:
            model_info = self.models[self.current_model]
            model_info['tokenizer'] = AutoTokenizer.from_pretrained(model_info['name'])
            model_info['model'] = AutoModelForCausalLM.from_pretrained(
                model_info['name'],
                device_map=None,  # Désactiver device_map auto
                torch_dtype=torch.float32  # Utiliser float32 au lieu de float16
            ).to(self.device)  # Déplacer explicitement vers le device
        except Exception as e:
            print(f"Erreur lors du chargement du modèle large: {e}")
            # Fallback sur le petit modèle
            self.current_model = 'small'
            model_info = self.models['small']
            model_info['tokenizer'] = GPT2Tokenizer.from_pretrained("gpt2")
            model_info['model'] = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)

    def switch_model(self, model_size):
        """Change le modèle actif (small/large)"""
        if model_size not in self.models:
            return False
            
        if not self.models[model_size]['model']:
            try:
                model_info = self.models[model_size]
                model_info['tokenizer'] = AutoTokenizer.from_pretrained(model_info['name'])
                model_info['model'] = AutoModelForCausalLM.from_pretrained(
                    model_info['name'],
                    device_map=None,
                    torch_dtype=torch.float32
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
            inputs = inputs.to(model.device)  # Déplacer sur le même device que le modèle

            output = model.generate(
                inputs["input_ids"],
                max_length=self.max_tokens,
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
        if context:
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

    def is_activated(self, text):
        """Vérifie si l'assistant est appelé dans le texte"""
        words = set(text.lower().split())
        return bool(words & self.activation_keywords)

    def process_input(self, user_input):
        user_input = user_input.lower()
        
        # Vérifier si l'assistant est appelé
        if not self.is_activated(user_input):
            return None  # Ne pas répondre si l'assistant n'est pas appelé
            
        # Nettoyer l'entrée en retirant le mot d'activation
        cleaned_input = ' '.join(word for word in user_input.split() 
                               if word not in self.activation_keywords)
        
        # Si l'entrée ne contient que le mot d'activation
        if not cleaned_input.strip():
            response = "Oui, je vous écoute ?"
            self.conversation_history.append({
                "user": user_input,
                "assistant": response,
                "timestamp": str(datetime.now())
            })
            return response

        # Continuer avec le traitement normal avec l'entrée nettoyée
        return self._process_input_internal(cleaned_input)

    def _process_input_internal(self, user_input):
        """Méthode interne pour traiter l'entrée une fois activée"""
        self.conversation_history.append({"user": user_input, "timestamp": str(datetime.now())})
        
        # Correction orthographique
        user_input = self.spell_checker.correct(user_input)
        
        # Vérifier si c'est une demande d'humour
        if self.humor_detector.is_humor_request(user_input):
            return random.choice(self.jokes)
        
        # Obtenir le contexte actuel avant l'analyse émotionnelle
        context = self.get_context()
        
        # Vérifier d'abord si c'est une salutation simple
        if user_input in self.training_data.categories["greeting"]["patterns"]:
            response = random.choice(self.training_data.categories["greeting"]["responses"])
            emotional_state = self.emotion_manager.update_emotion(user_input, context)
            response = self.personality.adjust_response(response, emotional_state)
            
            self.conversation_history.append({
                "user": user_input,
                "assistant": response,
                "timestamp": str(datetime.now())
            })
            return response
            
        # Le reste du code existant de process_input...
        # Gestion du mode apprentissage
        if user_input.startswith("apprends que"):
            self.learning_mode = True
            return "D'accord, quelle est la réponse que je dois donner à cette question?"

        if self.learning_mode:
            last_question = self.conversation_history[-2]["user"]
            if last_question.startswith("apprends que"):
                question = last_question[12:].strip()
                answer = user_input
                self.knowledge_base.add_knowledge(question, answer)
                self.learning_mode = False
                return f"Merci! J'ai appris que la réponse à '{question}' est '{answer}'"

        # Recherche dans la base de connaissances
        similar_q, confidence = self.knowledge_base.find_similar_question(user_input)
        if similar_q and confidence > 0.7:
            response = self.knowledge_base.knowledge[similar_q]
            self.conversation_history.append({
                "assistant": response,
                "timestamp": str(datetime.now()),
                "confidence": confidence
            })
            return response

        # Classification et réponse
        input_transformed = self.vectorizer.transform([user_input])
        intent = self.classifier.predict(input_transformed)[0]
        probas = self.classifier.predict_proba(input_transformed)[0]
        max_proba = max(probas)

        # Analyse des émotions et du sentiment
        sentiment_analysis = self.analyze_sentiment(user_input)
        sentiment = sentiment_analysis['sentiment']  # Extraire la valeur numérique du sentiment
        emotional_state = self.emotion_manager.update_emotion(user_input, context)

        # Gestion des tâches
        if "rappelle-moi" in user_input:
            match = re.search(r"rappelle-moi (.*?) (?:dans|à) (.*)", user_input)
            if match:
                message = match.group(1)
                time_str = match.group(2)
                # Conversion du temps en datetime
                try:
                    trigger_time = parse_time(time_str)
                    self.task_manager.add_reminder(message, str(trigger_time))
                    return f"Je vous rappellerai de {message} à {trigger_time.strftime('%H:%M')}"
                except:
                    return "Je n'ai pas compris le format du temps"

        # Gestion de la météo
        if "météo" in user_input or "temps" in user_input:
            match = re.search(r"(?:météo|temps).*?(?:à|a|dans) ([\w\s]+)", user_input)
            if match:
                location = match.group(1)
                weather_data = self.external_api.get_weather(location)
                if weather_data:
                    temp = weather_data['main']['temp']
                    desc = weather_data['weather'][0]['description']
                    return f"À {location}, il fait {temp}°C avec {desc}"

        # Génération de la réponse
        if max_proba > 0.4:
            response = self.get_response_for_category(intent)
        else:
            # Utiliser le modèle de langage pour les réponses inconnues
            context = self.get_context()
            response = self.language_model.generate_response(user_input, context)

        # Ajuster la réponse selon la personnalité et l'état émotionnel
        response = self.personality.adjust_response(response, emotional_state)

        # Mémorisation à long terme
        importance = max(abs(sentiment), max_proba)
        self.long_term_memory.add_memory(f"User: {user_input}\nAssistant: {response}", importance)

        # Apprentissage par renforcement
        self.rl.update_reward(response, 0.5)  # Récompense neutre par défaut

        # Auto-apprentissage périodique
        self.check_for_learning()

        # Mise à jour du gestionnaire de contexte avec la nouvelle interaction
        self.context_manager.update_context(user_input, response)

        # Mise à jour du contexte
        self.context_history.append({
            "user": user_input,
            "assistant": response
        })
        if len(self.context_history) > 10:
            self.context_history.pop(0)

        self.conversation_history.append({"assistant": response, "timestamp": str(datetime.now())})  # Modifié ici
        self.save_memory()
        return response

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
        if category not in self.training_data.categories:
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
def handle_audio_stream(audio_data):
    try:
        ai.voice_handler.process_stream(audio_data)
    except Exception as e:
        print(f"Erreur dans le traitement audio: {str(e)}")
        # Émettre une erreur au client
        socketio.emit('error', {'message': 'Erreur dans le traitement audio'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    ai.voice_handler.is_listening = False

if __name__ == '__main__':
    # Retiré allow_unsafe_werkzeug car non supporté par eventlet
    socketio.run(app, host='0.0.0.0', port=5050)