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
import sqlite3
from datetime import datetime
import pyicloud
import networkx as nx
from typing import Dict, List, Any
from Modules.assistant.cognitive_control import CognitiveControl
from Modules.assistant.memory_consolidation import MemoryConsolidation
from Modules.assistant.working_memory import WorkingMemory
from Modules.assistant.attention_system import AttentionSystem
from Modules.assistant.self_improvement import SelfImprovementModule

# Import des modules locaux
from Modules.assistant.memory import MemoryManager
from Modules.assistant.learning_monitor import LearningMonitor
from Modules.assistant.adaptive_learning import AdaptiveLearning, LearningMetrics
from Modules.assistant.response_generator import ResponseGenerator
from Modules.assistant.nlp_processor import NLPProcessor
from Modules.assistant.image_generator import ImageGenerator
from Modules.assistant.web_researcher import WebResearcher
from Modules.assistant.code_generator import CodeGenerator
from Modules.assistant.mail import ICloudMailAssistant
from Modules.assistant.agenda import AgendaAssistant
import re
from dateutil import parser

import re

class KnowledgeBase:
    def __init__(self, db_path='knowledge.db'):
        self.db_path = db_path
        self.setup_database()
        
    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Création des tables
        c.execute('''CREATE TABLE IF NOT EXISTS facts
                    (id INTEGER PRIMARY KEY, fact TEXT, category TEXT, 
                     confidence FLOAT, timestamp DATETIME)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS relationships
                    (id INTEGER PRIMARY KEY, fact1_id INTEGER, fact2_id INTEGER, 
                     relation_type TEXT, confidence FLOAT)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS concepts
                    (id INTEGER PRIMARY KEY, concept TEXT, description TEXT, 
                     examples TEXT)''')
        
        conn.commit()
        conn.close()
    
    def add_fact(self, fact, category, confidence=1.0):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO facts (fact, category, confidence, timestamp) VALUES (?, ?, ?, ?)",
                 (fact, category, confidence, datetime.now()))
        conn.commit()
        conn.close()
    
    def query_knowledge(self, query, category=None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        if category:
            c.execute("SELECT fact FROM facts WHERE fact LIKE ? AND category = ?",
                     (f"%{query}%", category))
        else:
            c.execute("SELECT fact FROM facts WHERE fact LIKE ?", (f"%{query}%",))
            
        results = c.fetchall()
        conn.close()
        return [r[0] for r in results]
    
    def add_relationship(self, fact1_id, fact2_id, relation_type, confidence=1.0):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO relationships (fact1_id, fact2_id, relation_type, confidence) VALUES (?, ?, ?, ?)",
                 (fact1_id, fact2_id, relation_type, confidence))
        conn.commit()
        conn.close()

class SanAI:
    def __init__(self, model_name='gpt2-large'):
        # Détection automatique du meilleur device disponible
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # Optimisations CUDA
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            print(f"Utilisation de CUDA avec {torch.cuda.device_count()} GPU(s)")
            print(f"GPU principal : {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Utilisation du CPU")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Configuration avancée du modèle
        config = GPT2Config.from_pretrained(model_name)
        config.attention_layers = 24
        config.gradient_checkpointing = True
        
        self.model = GPT2LMHeadModel.from_pretrained(model_name, config=config).to(self.device)
        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
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
        self.meta_learning = MetaLearning()
        self.knowledge_graph = KnowledgeGraph()
        self.reasoning_engine = ReasoningEngine()
        self.knowledge_base = KnowledgeBase()
        self.nlp_processor = NLPProcessor()
        self.image_generator = ImageGenerator()
        self.web_researcher = WebResearcher()
        self.code_generator = CodeGenerator()
        self.icloud_mail = None
        self.icloud_calendar = None
        
        # Nouveaux composants
        self.self_improvement = SelfImprovementModule()
        self.concept_learner = ConceptLearningModule()
        self.context_analyzer = ContextAnalyzer()
        self.feedback_analyzer = FeedbackAnalyzer()
        
        # Nouveaux modules cognitifs
        self.cognitive_architecture = CognitiveArchitecture()
        self.pattern_recognition = PatternRecognitionModule()
        self.causal_reasoning = CausalReasoningEngine()
        self.memory_consolidation = MemoryConsolidation()
        
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
        
    def connect_icloud(self, email, password):
        """Connecte l'assistant aux services iCloud"""
        try:
            self.icloud_mail = ICloudMailAssistant(email, password)
            self.icloud_calendar = AgendaAssistant()
            return True
        except Exception as e:
            self.logger.error(f"Erreur de connexion iCloud: {str(e)}")
            return False

    def process_input(self, user_input):
        # Ajouter au début de la méthode
        if "email" in user_input.lower() or "mail" in user_input.lower():
            if not self.icloud_mail:
                return "Veuillez d'abord me connecter à iCloud."
            # Gestion des emails
            if "envoyer" in user_input.lower():
                return self.icloud_mail.send_email(
                    to_email=self._extract_email(user_input),
                    subject=self._extract_subject(user_input),
                    body=self._extract_body(user_input)
                )
            
        elif "agenda" in user_input.lower() or "calendrier" in user_input.lower():
            if not self.icloud_calendar:
                return "Veuillez d'abord me connecter à iCloud."
            # Gestion du calendrier
            if "ajouter" in user_input.lower():
                event_details = self._extract_event_details(user_input)
                return self.icloud_calendar.add_event(**event_details)
            elif "voir" in user_input.lower():
                return self.icloud_calendar.get_events_for_day(self._extract_date(user_input))

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
        
        # Analyse métacognitive
        meta_context = self.meta_learning.analyze_current_state()
        
        # Raisonnement avancé
        reasoning_result = self.reasoning_engine.analyze(user_input, meta_context)
        
        # Mise à jour du graphe de connaissances
        self.knowledge_graph.update(user_input, reasoning_result)
        
        # Amélioration de la sélection de réponse
        response = self._select_best_response(responses, user_input)
        response = self.reasoning_engine.enhance_response(response, reasoning_result)
        
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
        
        # Auto-amélioration
        self.meta_learning.update(user_input, response, metrics)
        
        # Recherche de connaissances pertinentes
        relevant_knowledge = self.knowledge_base.query_knowledge(user_input)
        
        # Ajout des connaissances au contexte
        context = f"Knowledge context:\n"
        for knowledge in relevant_knowledge:
            context += f"- {knowledge}\n"
            
        # Stockage des nouvelles connaissances
        if response:
            self.knowledge_base.add_fact(
                fact=response,
                category="conversation",
                confidence=self._calculate_confidence(outputs)
            )
        
        # Analyse NLP avancée
        nlp_analysis = self.nlp_processor.analyze_text(user_input)
        
        # Recherche web si nécessaire
        if "recherche" in user_input.lower():
            web_results = self.web_researcher.search(user_input)
            context += f"\nWeb results: {web_results}\n"
            
        # Génération de code si demandé
        if "générer du code" in user_input.lower():
            generated_code = self.code_generator.generate_code(user_input)
            response = f"Voici le code généré :\n```\n{generated_code}\n```"
            
        # Génération d'image si demandé
        if "générer une image" in user_input.lower():
            image = self.image_generator.generate_image(user_input)
            # Sauvegarder l'image et retourner le chemin
        
        # Analyse approfondie du contexte
        context = self.context_analyzer.analyze(user_input, self.conversation_history)
        
        # Auto-amélioration continue
        self.self_improvement.analyze_performance(context)
        
        # Apprentissage de nouveaux concepts
        new_concepts = self.concept_learner.extract_concepts(user_input)
        if new_concepts:
            self.knowledge_base.add_concepts(new_concepts)
        
        # Amélioration basée sur le feedback
        response_quality = self.feedback_analyzer.analyze(response)
        self.adaptive_learning.update_strategy(response_quality)
        
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
            knowledge_score = self.knowledge_graph.evaluate_relevance(response)
            reasoning_score = self.reasoning_engine.evaluate_logic(response)
            
            combined_score = (
                coherence_score * 0.3 +
                knowledge_score * 0.4 +
                reasoning_score * 0.3
            )
            scores.append(combined_score)
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
        
        # Extraction et stockage des connaissances
        extracted_knowledge = self._extract_knowledge(conversation_history)
        for knowledge in extracted_knowledge:
            self.knowledge_base.add_fact(
                fact=knowledge['fact'],
                category=knowledge['category'],
                confidence=knowledge['confidence']
            )
        
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
            if (os.path.exists(history_path)):
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

    def _extract_knowledge(self, text):
        # Analyse du texte pour extraire des connaissances pertinentes
        knowledge = []
        
        # Utilisation du reasoning engine pour l'extraction
        analysis = self.reasoning_engine.analyze(text, {})
        
        if analysis:
            knowledge.append({
                'fact': analysis['main_point'],
                'category': 'learned_concept',
                'confidence': analysis['confidence']
            })
            
        return knowledge

    def _extract_email(self, text):
        """Extract email address from text using simple pattern matching"""
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        matches = re.findall(email_pattern, text)
        return matches[0] if matches else ""

    def _extract_subject(self, text):
        """Extract email subject from text"""
        # Look for subject after "subject:" or "sujet:" (case insensitive)
        subject_pattern = r'(?:subject|sujet)\s*:\s*([^\n]+)'
        match = re.search(subject_pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_body(self, text):
        """Extract email body from text"""
        # Look for message body after "body:", "message:", or "contenu:" (case insensitive)
        body_pattern = r'(?:body|message|contenu)\s*:\s*(.+?)(?:\s*$|\s*subject|\s*to:)'
        match = re.search(body_pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_event_details(self, text):
        """Extract calendar event details from text"""

        # Extract title
        title_pattern = r'(?:title|titre|événement)\s*:\s*([^\n]+)'
        title_match = re.search(title_pattern, text, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""

        # Extract date and time
        date_pattern = r'(?:date|le)\s*:\s*([^\n]+)'
        date_match = re.search(date_pattern, text, re.IGNORECASE)
        try:
            date = parser.parse(date_match.group(1)) if date_match else datetime.now()
        except:
            date = datetime.now()

        # Extract duration (in minutes)
        duration_pattern = r'(?:duration|durée)\s*:\s*(\d+)'
        duration_match = re.search(duration_pattern, text, re.IGNORECASE)
        duration = int(duration_match.group(1)) if duration_match else 60

        # Extract description
        desc_pattern = r'(?:description|détails)\s*:\s*([^\n]+)'
        desc_match = re.search(desc_pattern, text, re.IGNORECASE)
        description = desc_match.group(1).strip() if desc_match else ""

        return {
            "title": title,
            "date": date.strftime("%Y-%m-%d %H:%M"),
            "duration": duration,
            "description": description
        }

    def _extract_date(self, text):
        """Extract date from text using natural language processing"""

        # Look for date patterns in text
        date_pattern = r'(?:le|pour|date)\s*:?\s*([^\n]+)'
        match = re.search(date_pattern, text, re.IGNORECASE)
        if match:
            try:
                date = parser.parse(match.group(1), dayfirst=True)
                return date.strftime("%Y-%m-%d")
            except:
                pass
        return datetime.now().strftime("%Y-%m-%d")

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

class MetaLearning:
    def __init__(self):
        self.performance_history = []
        self.learning_patterns = {}
        self.adaptation_strategies = {}

    def analyze_current_state(self):
        # Analyse des performances et patterns d'apprentissage
        return {
            'performance_trend': self._analyze_performance_trend(),
            'learning_efficiency': self._calculate_learning_efficiency(),
            'adaptation_needs': self._identify_adaptation_needs()
        }

    def update(self, input_data, output_data, metrics):
        self.performance_history.append(metrics)
        self._update_learning_patterns(input_data, output_data)
        self._adjust_strategies()

class ReasoningEngine:
    def __init__(self):
        self.logical_frameworks = []
        self.inference_rules = {}
        
    def analyze(self, input_data, context):
        # Analyse logique approfondie
        logical_structure = self._extract_logical_structure(input_data)
        inferences = self._apply_inference_rules(logical_structure)
        return self._synthesize_reasoning(inferences, context)
    
    def enhance_response(self, response, reasoning):
        # Amélioration de la réponse basée sur le raisonnement
        return self._restructure_response(response, reasoning)

class KnowledgeGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.semantic_index = {}
        
    def update(self, input_data, reasoning):
        # Mise à jour du graphe de connaissances
        new_nodes = self._extract_concepts(input_data)
        self._integrate_knowledge(new_nodes, reasoning)
        self._update_semantic_index()

    def evaluate_relevance(self, response):
        # Évaluation de la pertinence basée sur le graphe de connaissances
        concepts = self._extract_concepts(response)
        return self._calculate_relevance_score(concepts)

class ConceptLearningModule:
    def __init__(self):
        self.concept_patterns = []
        self.learned_concepts = {}
        
    def extract_concepts(self, text):
        # Extraction de nouveaux concepts à partir du texte
        concepts = []
        # Implémentation de l'extraction...
        return concepts

class ContextAnalyzer:
    def analyze(self, input_text, history):
        return {
            'intent': self._detect_intent(input_text),
            'emotion': self._analyze_emotion(input_text),
            'context': self._extract_context(history),
            'knowledge_gaps': self._identify_knowledge_gaps(input_text)
        }

class FeedbackAnalyzer:
    def analyze(self, response):
        # Analyse la qualité de la réponse
        coherence = self._measure_coherence(response)
        relevance = self._measure_relevance(response)
        complexity = self._measure_complexity(response)
        return (coherence + relevance + complexity) / 3

class CognitiveArchitecture:
    def __init__(self):
        self.working_memory = WorkingMemory()
        self.attention_system = AttentionSystem()
        self.cognitive_control = CognitiveControl()
        
    def process_input(self, input_data):
        attention_focus = self.attention_system.focus(input_data)
        working_memory_state = self.working_memory.update(attention_focus)
        return self.cognitive_control.regulate(working_memory_state)

class PatternRecognitionModule:
    def __init__(self):
        self.pattern_database = {}
        self.recognition_threshold = 0.75
        self.learning_rate = 0.1
        
    def identify_patterns(self, data):
        temporal_patterns = self._analyze_temporal_sequences(data)
        semantic_patterns = self._analyze_semantic_structures(data)
        return self._merge_patterns(temporal_patterns, semantic_patterns)

class CausalReasoningEngine:
    def __init__(self):
        self.causal_graph = nx.DiGraph()
        self.inference_rules = {}
        
    def infer_causality(self, events):
        temporal_relations = self._analyze_temporal_order(events)
        correlation_strength = self._measure_correlations(events)
        return self._build_causal_model(temporal_relations, correlation_strength)
