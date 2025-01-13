import sqlite3
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MemoryManager:
    def __init__(self, db_path='memory.db'):
        self.db_path = db_path
        self.setup_database()
        
    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS memories
                    (id INTEGER PRIMARY KEY,
                     content TEXT,
                     embedding BLOB,
                     timestamp DATETIME,
                     importance FLOAT)''')
        conn.commit()
        conn.close()
    
    def store_memory(self, content, embedding, importance=0.5):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        embedding_bytes = np.ndarray.tobytes(embedding)
        c.execute("INSERT INTO memories VALUES (NULL, ?, ?, ?, ?)",
                 (content, embedding_bytes, datetime.now(), importance))
        conn.commit()
        conn.close()
    
    def retrieve_similar_memories(self, query_embedding, limit=5):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        memories = c.execute("SELECT content, embedding, importance FROM memories").fetchall()
        
        if not memories:
            return []
            
        similarities = []
        for content, emb_bytes, importance in memories:
            emb = np.frombuffer(emb_bytes).reshape(-1, 1)
            similarity = cosine_similarity(query_embedding.reshape(1, -1), emb.reshape(1, -1))[0][0]
            similarities.append((content, similarity * importance))
            
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:limit]
