import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import logging
import os
import json
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Import des modules locaux
from Modules.assistant.memory import MemoryManager
from Modules.assistant.learning_monitor import LearningMonitor
from Modules.assistant.adaptive_learning import AdaptiveLearning, LearningMetrics
from Modules.assistant.response_generator import ResponseGenerator

class SanAI:
    def __init__(self, model_name='gpt2-large'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Configuration avancée du modèle
        config = GPT2Config.from_pretrained(model_name)
        config.attention_layers = 24
        config.gradient_checkpointing = True
        
        self.model = GPT2LMHeadModel.from_pretrained(model_name, config=config).to(self.device)
        self.model.train()
        
        self.conversation_history = []
        self.logger = self._setup_logging()
        self._setup_advanced_training()
        
        self.memory_manager = MemoryManager()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.emotional_state = EmotionalState()
        self.learning_monitor = LearningMonitor()
        self.personality_engine = PersonalityEngine()
        self.adaptive_learning = AdaptiveLearning()
        self.response_generator = ResponseGenerator(self.model, self.tokenizer)
        
    def _setup_logging(self):
        logger = logging.getLogger('SanAI')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('san_ai.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def _setup_advanced_training(self):
        # Ajout de techniques d'apprentissage avancées
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=2e-5,
            steps_per_epoch=100,
            epochs=10
        )
        
    def process_input(self, user_input):
        # Analyse émotionnelle de l'entrée
        emotional_context = self.emotional_state.analyze(user_input)
        personality_context = self.personality_engine.get_context()
        
        # Fusion du contexte émotionnel avec le contexte existant
        context = f"Emotional state: {emotional_context}\n"
        context += f"Personality traits: {personality_context}\n"
        
        # Génération de l'embedding pour la recherche en mémoire
        input_embedding = self.embedding_model.encode([user_input])[0]
        relevant_memories = self.memory_manager.retrieve_similar_memories(input_embedding)
        
        # Construire le contexte avec les souvenirs pertinents
        context += "Relevant context:\n"
        for memory, similarity in relevant_memories:
            context += f"- {memory}\n"
        context += f"\nCurrent input: {user_input}"
        
        inputs = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        
        # Amélioration du traitement avec attention dynamique
        inputs = self.tokenizer.encode(user_input, return_tensors="pt").to(self.device)
        
        # Ajout de l'attention multi-tête améliorée
        attention_weights = self._compute_dynamic_attention(inputs)
        
        outputs = self.model.generate(
            inputs,
            max_length=200,
            num_return_sequences=3,
            do_sample=True,
            top_p=0.92,
            temperature=0.85,
            repetition_penalty=1.2,
            attention_weights=attention_weights
        )
        
        # Sélection de la meilleure réponse
        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        response = self._select_best_response(responses, user_input)
        
        # Stocker la nouvelle interaction en mémoire
        response_embedding = self.embedding_model.encode([response])[0]
        self.memory_manager.store_memory(
            f"User: {user_input}\nAssistant: {response}",
            response_embedding
        )
        
        # Utilisation du nouveau générateur de réponses
        context = {
            "emotion": self.emotional_state.current_emotion,
            "personality": self.personality_engine.get_context(),
            "memories": relevant_memories
        }
        
        response = self.response_generator.generate_structured_response(
            user_input,
            context,
            max_length=200
        )
        
        # Mise à jour des métriques d'apprentissage
        metrics = LearningMetrics(
            loss=outputs.loss.item(),
            accuracy=self._calculate_response_accuracy(response),
            confidence=self._calculate_confidence(outputs),
            diversity=self._calculate_response_diversity([response])
        )
        
        # Ajustement des paramètres d'apprentissage
        new_lr, new_batch = self.adaptive_learning.adjust_parameters(metrics)
        self._update_learning_parameters(new_lr, new_batch)
        
        return response
        
    def _compute_dynamic_attention(self, inputs):
        # Calcul d'attention dynamique basé sur le contexte
        with torch.no_grad():
            hidden_states = self.model.transformer(inputs)[0]
            attention_weights = F.softmax(torch.matmul(hidden_states, hidden_states.transpose(-1, -2)), dim=-1)
        return attention_weights
        
    def _select_best_response(self, responses, query):
        # Sélection de la meilleure réponse basée sur la cohérence
        scores = []
        for response in responses:
            coherence_score = self._calculate_coherence(query, response)
            scores.append(coherence_score)
        return responses[scores.index(max(scores))]
        
    def learn_from_interaction(self, conversation_history):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        inputs = self.tokenizer.encode(conversation_history, return_tensors="pt").to(self.device)
        labels = inputs.clone()
        
        outputs = self.model(inputs, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return loss.item()
        
    def save_model(self, path='model_checkpoint'):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Sauvegarder l'historique des conversations
        with open(os.path.join(path, 'conversation_history.json'), 'w') as f:
            json.dump(self.conversation_history, f)
        
        self.logger.info(f"Modèle sauvegardé dans {path}")
        
    def load_model(self, path='model_checkpoint'):
        if os.path.exists(path):
            self.model = GPT2LMHeadModel.from_pretrained(path).to(self.device)
            self.tokenizer = GPT2Tokenizer.from_pretrained(path)
            
            # Charger l'historique des conversations
            history_path = os.path.join(path, 'conversation_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.conversation_history = json.load(f)
                    
            self.logger.info(f"Modèle chargé depuis {path}")
        else:
            self.logger.error(f"Chemin de sauvegarde {path} non trouvé")

    def _calculate_confidence(self, outputs):
        probs = F.softmax(outputs.logits, dim=-1)
        return torch.max(probs).item()
        
    def _calculate_response_diversity(self, responses):
        if len(responses) < 2:
            return 1.0
        embeddings = self.embedding_model.encode(responses)
        similarities = cosine_similarity(embeddings)
        return 1.0 - np.mean(similarities[np.triu_indices(len(responses), k=1)])

class EmotionalState:
    def __init__(self):
        self.current_emotion = "neutral"
        self.emotion_history = []
        
    def analyze(self, text):
        # Analyse simple des émotions basée sur des mots-clés
        emotions = {
            "joy": ["happy", "great", "excellent"],
            "sadness": ["sad", "sorry", "unfortunate"],
            "concern": ["worried", "concerned", "careful"]
        }
        
        text = text.lower()
        detected_emotions = []
        
        for emotion, keywords in emotions.items():
            if any(keyword in text for keyword in keywords):
                detected_emotions.append(emotion)
                
        self.current_emotion = detected_emotions[0] if detected_emotions else "neutral"
        self.emotion_history.append(self.current_emotion)
        return self.current_emotion

class PersonalityEngine:
    def __init__(self):
        self.traits = {
            "openness": 0.8,
            "conscientiousness": 0.9,
            "extraversion": 0.6,
            "agreeableness": 0.85,
            "neuroticism": 0.3
        }
        
    def get_context(self):
        return " | ".join(f"{trait}: {value}" for trait, value in self.traits.items())
