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
from sentence_transformers import SentenceTransformer
import torch
import random
from transformers import pipeline
from textblob import TextBlob
import numpy as np
from datetime import timedelta
import requests
from datetime import datetime, timedelta
import importlib.util
import glob
from collections import defaultdict
import re
from dateutil.parser import parse
from datetime import datetime, timedelta

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
        self.model = whisper.load_model("base")
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        self.is_listening = False

    def speech_to_text(self, audio_file):
        result = self.model.transcribe(audio_file)
        return result["text"]

    def text_to_speech(self, text, lang='fr'):
        tts = gTTS(text=text, lang=lang)
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        tts.save(temp_file.name)
        return temp_file.name

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

    def process_stream(self, audio_chunk):
        self.audio_queue.put(audio_chunk)

class LanguageModel:
    def __init__(self):
        self.model_name = "flaubert/flaubert-base-cased"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        self.context_window = 5

    def generate_response(self, prompt, context=None):
        if context:
            full_prompt = f"{context}\nQuestion: {prompt}\nRéponse:"
        else:
            full_prompt = f"Question: {prompt}\nRéponse:"
            
        inputs = self.tokenizer(full_prompt, return_tensors="pt", max_length=512, truncation=True)
        output = self.model.generate(
            inputs["input_ids"],
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def get_embedding(self, text):
        return self.embedding_model.encode(text)

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
        if self.language_model:
            query_embedding = self.language_model.get_embedding(question)
            max_sim = -1
            best_question = None
            
            for q, emb in self.embeddings.items():
                sim = cosine_similarity([query_embedding], [emb])[0][0]
                if sim > max_sim and sim >= threshold:
                    max_sim = sim
                    best_question = q
                    
            return best_question, max_sim
        return super().find_similar_question(question, threshold)

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
                'timestamp': str(datetime.datetime.now()),
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

class SanAI:
    def __init__(self):
        self.name = "San"
        self.creation_date = datetime.datetime.now()
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
        self.last_learning_check = datetime.datetime.now()
        self.learning_interval = timedelta(hours=1)
        self.task_manager = TaskManager()
        self.rl = ReinforcementLearning()
        self.plugin_system = PluginSystem()
        self.external_api = ExternalAPI()

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
        if category not in self.training_data.categories:
            return "Je ne sais pas comment répondre à cela."
            
        responses = self.training_data.categories[category]["responses"]
        response = random.choice(responses)
        
        if category == "time":
            current_time = datetime.datetime.now().strftime("%H:%M")
            response = response.format(time=current_time)
            
        return response

    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def process_input(self, user_input):
        user_input = user_input.lower()
        self.conversation_history.append({"user": user_input, "timestamp": str(datetime.datetime.now())})
        
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
                "timestamp": str(datetime.datetime.now()),
                "confidence": confidence
            })
            return response

        # Classification et réponse
        input_transformed = self.vectorizer.transform([user_input])
        intent = self.classifier.predict(input_transformed)[0]
        probas = self.classifier.predict_proba(input_transformed)[0]
        max_proba = max(probas)

        # Analyse des émotions et du sentiment
        emotional_state = self.emotion_system.update_emotion(user_input)
        sentiment = self.analyze_sentiment(user_input)

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

        # Mise à jour du contexte
        self.context_history.append({
            "user": user_input,
            "assistant": response
        })
        if len(self.context_history) > 10:
            self.context_history.pop(0)

        self.conversation_history.append({"assistant": response, "timestamp": str(datetime.datetime.now())})
        self.save_memory()
        return response

    def check_for_learning(self):
        now = datetime.datetime.now()
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
        text = self.voice_handler.speech_to_text(audio_file)
        response = self.process_input(text)
        audio_response = self.voice_handler.text_to_speech(response)
        return {
            "text": response,
            "audio_file": audio_response
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
        return jsonify({"error": "No audio file"}), 400
    
    audio_file = request.files['audio']
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    audio_file.save(temp_file.name)
    
    result = ai.process_voice(temp_file.name)
    os.unlink(temp_file.name)
    
    return jsonify({
        "response": result["text"],
        "audio_path": result["audio_file"]
    })

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_file(filename, mimetype='audio/mp3')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    ai.voice_handler.start_listening()

@socketio.on('audio_stream')
def handle_audio_stream(audio_data):
    ai.voice_handler.process_stream(audio_data)
    
@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    ai.voice_handler.is_listening = False

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
