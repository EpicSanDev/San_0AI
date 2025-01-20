import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import networkx as nx

class MemoryManager:
    def __init__(self):
        self.db_path = "enhanced_memories.db"
        self.init_database()
        self.vectorizer = TfidfVectorizer()
        self.memory_graph = nx.Graph()
        self.categories = {
            'daily_routine': ['matin', 'midi', 'soir', 'repas', 'médicaments'],
            'appointments': ['rendez-vous', 'médecin', 'réunion', 'important'],
            'events': ['anniversaire', 'fête', 'sortie', 'visite'],
            'tasks': ['tâche', 'courses', 'ménage', 'urgent'],
            'interactions': ['conversation', 'appel', 'message', 'rencontre'],
            'health': ['symptôme', 'traitement', 'médication', 'douleur'],
            'locations': ['lieu', 'adresse', 'endroit', 'destination']
        }
        
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Table principale des souvenirs
        c.execute('''CREATE TABLE IF NOT EXISTS memories
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp TEXT,
                     content TEXT,
                     category TEXT,
                     importance REAL,
                     urgency REAL,
                     location TEXT,
                     people TEXT,
                     tags TEXT,
                     context TEXT,
                     reminder_date TEXT,
                     last_accessed TEXT)''')
        
        # Table pour les relations entre souvenirs
        c.execute('''CREATE TABLE IF NOT EXISTS memory_relations
                    (memory_id1 INTEGER,
                     memory_id2 INTEGER,
                     relation_type TEXT,
                     strength REAL)''')
        
        conn.commit()
        conn.close()

    def add_memory(self, content, category=None, importance=0.5, urgency=0.0,
                  location=None, people=None, tags=None, context=None, 
                  reminder_date=None):
        """Ajoute un nouveau souvenir avec des métadonnées enrichies"""
        
        # Auto-catégorisation si non spécifiée
        if category is None:
            category = self._auto_categorize(content)
            
        # Conversion des tags en chaîne JSON
        tags_json = json.dumps(tags) if tags else "[]"
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""INSERT INTO memories 
                    (timestamp, content, category, importance, urgency,
                     location, people, tags, context, reminder_date, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                 (datetime.now().isoformat(), content, category, importance, urgency,
                  location, people, tags_json, context, reminder_date, 
                  datetime.now().isoformat()))
        
        memory_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Mise à jour du graphe de relations
        self._update_memory_relations(memory_id, content)
        
        return memory_id

    def _auto_categorize(self, content):
        """Catégorise automatiquement un souvenir basé sur son contenu"""
        content_lower = content.lower()
        for category, keywords in self.categories.items():
            if any(keyword in content_lower for keyword in keywords):
                return category
        return 'general'

    def get_urgent_reminders(self):
        """Récupère les souvenirs urgents et les rappels"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        now = datetime.now().isoformat()
        
        reminders = c.execute("""SELECT id, content, category, importance, urgency
                               FROM memories
                               WHERE reminder_date <= ? AND reminder_date IS NOT NULL
                               ORDER BY urgency DESC, importance DESC""", (now,)).fetchall()
        conn.close()
        return reminders

    def search_memories(self, query, context=None, category=None, 
                       date_range=None, min_importance=0.0):
        """Recherche avancée dans les souvenirs"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        base_query = """SELECT id, content, category, importance, timestamp 
                       FROM memories WHERE 1=1"""
        params = []
        
        if category:
            base_query += " AND category = ?"
            params.append(category)
            
        if date_range:
            start_date, end_date = date_range
            base_query += " AND timestamp BETWEEN ? AND ?"
            params.extend([start_date.isoformat(), end_date.isoformat()])
            
        if min_importance > 0:
            base_query += " AND importance >= ?"
            params.append(min_importance)
            
        memories = c.execute(base_query, params).fetchall()
        conn.close()
        
        if not memories:
            return []
            
        # Recherche sémantique
        texts = [m[1] for m in memories]  # Extraction du contenu
        vectors = self.vectorizer.fit_transform([query] + texts)
        similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]
        
        # Combiner les résultats avec les métadonnées
        results = [(memories[i], similarities[i]) 
                  for i in range(len(memories))]
        
        # Trier par similarité et importance
        results.sort(key=lambda x: (x[1], x[0][3]), reverse=True)
        
        return results

    def update_memory_importance(self, memory_id, importance_delta=0.1):
        """Met à jour l'importance d'un souvenir basé sur son utilisation"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""UPDATE memories 
                    SET importance = MIN(1.0, importance + ?),
                        last_accessed = ?
                    WHERE id = ?""",
                 (importance_delta, datetime.now().isoformat(), memory_id))
        
        conn.commit()
        conn.close()

    def get_daily_summary(self, date=None):
        """Génère un résumé quotidien structuré"""
        if date is None:
            date = datetime.now().date()
            
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        start = datetime.combine(date, datetime.min.time()).isoformat()
        end = datetime.combine(date, datetime.max.time()).isoformat()
        
        memories = c.execute("""SELECT category, content, importance, urgency, 
                                      location, people, tags
                               FROM memories 
                               WHERE timestamp BETWEEN ? AND ?
                               ORDER BY urgency DESC, importance DESC""",
                           (start, end)).fetchall()
        
        conn.close()
        
        summary = defaultdict(list)
        for memory in memories:
            category, content, importance, urgency, location, people, tags = memory
            summary[category].append({
                'content': content,
                'importance': importance,
                'urgency': urgency,
                'location': location,
                'people': people,
                'tags': json.loads(tags)
            })
            
        return dict(summary)

    def export_memories(self, format='json', start_date=None, end_date=None):
        """Exporte les souvenirs dans différents formats"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        query = "SELECT * FROM memories"
        params = []
        
        if start_date and end_date:
            query += " WHERE timestamp BETWEEN ? AND ?"
            params.extend([start_date.isoformat(), end_date.isoformat()])
            
        memories = c.execute(query, params).fetchall()
        conn.close()
        
        if format == 'json':
            return json.dumps(memories, indent=2)
        elif format == 'txt':
            return '\n\n'.join([f"{m[1]}: {m[2]}" for m in memories])
        else:
            raise ValueError(f"Format non supporté: {format}")

    def import_memories(self, data, format='json'):
        """Importe des souvenirs depuis différents formats"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        if format == 'json':
            memories = json.loads(data)
            for memory in memories:
                c.execute("""INSERT INTO memories VALUES 
                           (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", memory)
                
        conn.commit()
        conn.close()

    def _update_memory_relations(self, memory_id, content):
        """Met à jour les relations entre les souvenirs"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Rechercher des souvenirs similaires
        existing_memories = c.execute("""SELECT id, content FROM memories 
                                       WHERE id != ?""", (memory_id,)).fetchall()
        
        if existing_memories:
            texts = [m[1] for m in existing_memories]
            vectors = self.vectorizer.fit_transform([content] + texts)
            similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]
            
            # Créer des relations pour les souvenirs similaires
            for i, similarity in enumerate(similarities):
                if similarity > 0.3:  # Seuil de similarité
                    c.execute("""INSERT INTO memory_relations VALUES (?, ?, ?, ?)""",
                             (memory_id, existing_memories[i][0], 
                              'similar', float(similarity)))
        
        conn.commit()
        conn.close()
