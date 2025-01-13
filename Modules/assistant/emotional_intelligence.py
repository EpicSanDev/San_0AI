import numpy as np
from transformers import pipeline
import json
import os
from typing import Dict, List, Optional
import logging

class EmotionalIntelligence:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.emotion_classifier = pipeline("text-classification", 
                                        model="j-hartmann/emotion-english-distilroberta-base")
        self.emotion_memory = []
        self.empathy_patterns = self._load_empathy_patterns()
        self.emotion_threshold = 0.7
        
    def analyze_emotions(self, text: str) -> Dict[str, float]:
        """Analyse les émotions dans le texte"""
        emotions = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "neutral": 1.0  # Par défaut
        }
        
        # Analyse simple basée sur des mots-clés
        for emotion, patterns in self.empathy_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text.lower():
                    emotions[emotion] += 0.3
                    emotions["neutral"] -= 0.1
                    
        # Normalisation
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
            
        return emotions

    def generate_empathetic_response(self, emotional_insight: Dict[str, float]) -> str:
        """Génère une réponse empathique basée sur l'analyse émotionnelle"""
        dominant_emotion = max(emotional_insight.items(), key=lambda x: x[1])[0]
        
        if emotional_insight[dominant_emotion] < self.emotion_threshold:
            return self.empathy_patterns["neutral"][0]
            
        if dominant_emotion in self.empathy_patterns:
            responses = self.empathy_patterns[dominant_emotion]
            return responses[0] if responses else self.empathy_patterns["neutral"][0]
            
        return self.empathy_patterns["neutral"][0]

    def update_emotional_memory(self, emotion: str, intensity: float):
        """Met à jour la mémoire émotionnelle"""
        self.emotion_memory.append({
            "emotion": emotion,
            "intensity": intensity,
            "timestamp": os.time()
        })
        
        # Garde uniquement les 100 dernières entrées
        if len(self.emotion_memory) > 100:
            self.emotion_memory = self.emotion_memory[-100:]

    def get_emotional_context(self) -> Dict:
        """Retourne le contexte émotionnel actuel"""
        if not self.emotion_memory:
            return {"dominant_emotion": "neutral", "intensity": 0.0}
            
        recent_emotions = self.emotion_memory[-5:]
        emotions_count = {}
        total_intensity = 0
        
        for entry in recent_emotions:
            emotion = entry["emotion"]
            intensity = entry["intensity"]
            emotions_count[emotion] = emotions_count.get(emotion, 0) + 1
            total_intensity += intensity
            
        dominant_emotion = max(emotions_count.items(), key=lambda x: x[1])[0]
        avg_intensity = total_intensity / len(recent_emotions)
        
        return {
            "dominant_emotion": dominant_emotion,
            "intensity": avg_intensity
        }

    def adapt_response_tone(self, base_response: str, emotional_context: Dict) -> str:
        """Adapte le ton de la réponse en fonction du contexte émotionnel"""
        if emotional_context["intensity"] > self.emotion_threshold:
            if emotional_context["dominant_emotion"] in self.empathy_patterns:
                empathetic_prefix = self.empathy_patterns[emotional_context["dominant_emotion"]][0]
                return f"{empathetic_prefix} {base_response}"
        return base_response

    def _load_empathy_patterns(self) -> Dict:
        """Charge les patterns d'empathie depuis un fichier ou utilise les défauts"""
        default_patterns = {
            "joy": ["C'est formidable !", "Je suis ravi(e) pour vous !"],
            "sadness": ["Je comprends que cela soit difficile.", "Je suis désolé(e) d'entendre cela."],
            "anger": ["Je comprends votre frustration.", "C'est normal de ressentir cela."],
            "fear": ["Je suis là pour vous aider.", "Prenons le temps d'examiner cela ensemble."],
            "surprise": ["C'est effectivement inattendu !", "Je comprends votre étonnement."],
            "neutral": ["Je vous écoute.", "Continuez, je suis là."]
        }
        
        try:
            pattern_path = os.path.join(os.path.dirname(__file__), 'empathy_patterns.json')
            if os.path.exists(pattern_path):
                with open(pattern_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return default_patterns
        except Exception as e:
            self.logger.warning(f"Erreur lors du chargement des patterns d'empathie: {e}")
            return default_patterns

    def _analyze_emotional_context(self, text):
        # Analyse contextuelle des émotions
        pass
        
    def _analyze_emotional_subtext(self, text):
        # Analyse du sous-texte émotionnel
        pass
