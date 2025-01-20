import json
import os
from datetime import datetime
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MemoryManager:
    def __init__(self):
        self.memory_file = "enhanced_memory.json"
        self.memories = self.load_memories()
        self.vectorizer = TfidfVectorizer()
        self.memory_vectors = {}
        self.importance_threshold = 0.7
        self.max_memories = 1000
        self.memory_categories = {
            'conversations': [],
            'facts': [],
            'user_preferences': defaultdict(dict),
            'learned_topics': defaultdict(list)
        }
        
    def load_memories(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return {"short_term": [], "long_term": [], "permanent": []}

    def save_memories(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memories, f, indent=2)

    def add_memory(self, content, category='conversations', importance=0.5):
        memory = {
            'content': content,
            'timestamp': str(datetime.now()),
            'importance': importance,
            'category': category,
            'access_count': 0,
            'last_accessed': str(datetime.now())
        }
        
        # Déterminer le stockage basé sur l'importance
        if importance >= self.importance_threshold:
            self.memories['long_term'].append(memory)
        else:
            self.memories['short_term'].append(memory)
            
        # Mise à jour des vecteurs
        self._update_memory_vectors(memory)
        
        # Nettoyage si nécessaire
        self._cleanup_memories()
        
        # Sauvegarder
        self.save_memories()

    def get_relevant_memories(self, query, n=5):
        query_vector = self.vectorizer.transform([query])
        similarities = []
        
        for memory_type in ['permanent', 'long_term', 'short_term']:
            for memory in self.memories[memory_type]:
                if memory['content'] in self.memory_vectors:
                    similarity = cosine_similarity(
                        query_vector,
                        self.memory_vectors[memory['content']]
                    )[0][0]
                    similarities.append((memory, similarity))
        
        # Trier par similarité et importance
        sorted_memories = sorted(
            similarities,
            key=lambda x: (x[1], x[0]['importance']),
            reverse=True
        )
        
        return [m[0] for m in sorted_memories[:n]]

    def update_importance(self, memory_id, new_importance):
        for memory_type in self.memories:
            for memory in self.memories[memory_type]:
                if memory.get('id') == memory_id:
                    memory['importance'] = new_importance
                    memory['last_accessed'] = str(datetime.now())
                    memory['access_count'] += 1
                    self.save_memories()
                    return True
        return False

    def forget_old_memories(self, days_threshold=30):
        current_time = datetime.now()
        for memory_type in ['short_term', 'long_term']:
            self.memories[memory_type] = [
                m for m in self.memories[memory_type]
                if (current_time - datetime.strptime(m['timestamp'], '%Y-%m-%d %H:%M:%S.%f')).days < days_threshold
                or m['importance'] >= self.importance_threshold
            ]
        self.save_memories()

    def _update_memory_vectors(self, memory):
        texts = [m['content'] for m in self.memories['short_term'] + 
                self.memories['long_term'] + self.memories['permanent']]
        if texts:
            self.vectorizer.fit(texts)
            self.memory_vectors = {
                text: self.vectorizer.transform([text]) 
                for text in texts
            }

    def _cleanup_memories(self):
        if len(self.memories['short_term']) > self.max_memories:
            # Garder les plus importants et les plus récents
            self.memories['short_term'].sort(
                key=lambda x: (x['importance'], x['timestamp']),
                reverse=True
            )
            self.memories['short_term'] = self.memories['short_term'][:self.max_memories]

    def add_user_preference(self, user, preference_type, value):
        self.memory_categories['user_preferences'][user][preference_type] = {
            'value': value,
            'timestamp': str(datetime.now())
        }
        self.save_memories()

    def get_user_preferences(self, user):
        return dict(self.memory_categories['user_preferences'].get(user, {}))

    def add_learned_topic(self, topic, content):
        self.memory_categories['learned_topics'][topic].append({
            'content': content,
            'timestamp': str(datetime.now())
        })
        self.save_memories()

    def get_learned_topics(self, topic=None):
        if topic:
            return self.memory_categories['learned_topics'].get(topic, [])
        return dict(self.memory_categories['learned_topics'])
