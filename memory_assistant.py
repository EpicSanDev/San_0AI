import datetime
import json
from collections import defaultdict
import sqlite3
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MemoryAssistant:
    def __init__(self, db_path="memories.db"):
        self.db_path = db_path
        self.vectorizer = TfidfVectorizer()
        self.init_database()
        
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS memories
                    (timestamp TEXT, content TEXT, category TEXT, importance REAL, 
                     location TEXT, people TEXT, context TEXT)''')
        conn.commit()
        conn.close()

    def store_memory(self, content, category="général", importance=0.5, 
                    location=None, people=None, context=None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        c.execute("""INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?)""",
                 (timestamp, content, category, importance, location, people, context))
        conn.commit()
        conn.close()

    def find_related_memories(self, query, limit=5):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        memories = c.execute("SELECT content FROM memories").fetchall()
        conn.close()

        if not memories:
            return []

        # Vectoriser la requête et les souvenirs
        memory_texts = [m[0] for m in memories]
        vectors = self.vectorizer.fit_transform([query] + memory_texts)
        
        # Calculer les similarités
        similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]
        
        # Retourner les souvenirs les plus pertinents
        most_similar_idx = np.argsort(similarities)[-limit:][::-1]
        return [memory_texts[i] for i in most_similar_idx]

    def get_daily_summary(self, date=None):
        if date is None:
            date = datetime.datetime.now().date()

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        start = datetime.datetime.combine(date, datetime.time.min).isoformat()
        end = datetime.datetime.combine(date, datetime.time.max).isoformat()
        
        memories = c.execute("""SELECT content, category, importance 
                              FROM memories 
                              WHERE timestamp BETWEEN ? AND ?
                              ORDER BY importance DESC""", (start, end)).fetchall()
        conn.close()

        summary = defaultdict(list)
        for content, category, importance in memories:
            summary[category].append({
                'content': content,
                'importance': importance
            })
        
        return dict(summary)
