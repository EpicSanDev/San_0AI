from datetime import datetime
import json
import os

class ConversationLearner:
    def __init__(self):
        self.memory_file = "conversation_memory.json"
        self.memories = self.load_memories()
        
    def load_memories(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return {
            "conversations": [],
            "topics": {},
            "people": {},
            "learned_facts": []
        }

    def learn_from_conversation(self, text):
        # Extraire les informations importantes
        timestamp = str(datetime.now())
        
        # Sauvegarder la conversation
        self.memories["conversations"].append({
            "text": text,
            "timestamp": timestamp,
            "analyzed": False
        })
        
        # Extraire les sujets et les personnes mentionnés
        self._extract_topics(text)
        self._extract_people(text)
        
        # Sauvegarder les modifications
        self._save_memories()
        
    def _extract_topics(self, text):
        # Exemple simple d'extraction de sujets
        common_topics = ["travail", "famille", "sport", "santé", "loisirs"]
        for topic in common_topics:
            if topic.lower() in text.lower():
                if topic not in self.memories["topics"]:
                    self.memories["topics"][topic] = []
                self.memories["topics"][topic].append({
                    "text": text,
                    "timestamp": str(datetime.now())
                })

    def _extract_people(self, text):
        # TODO: Implémenter une détection plus sophistiquée des noms de personnes
        pass

    def _save_memories(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memories, f, indent=2)

    def get_related_memories(self, query):
        # Rechercher dans les souvenirs existants
        related = []
        for conv in self.memories["conversations"]:
            if any(word in conv["text"].lower() for word in query.lower().split()):
                related.append(conv)
        return related
