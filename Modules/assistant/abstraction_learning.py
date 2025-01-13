from typing import List, Dict
from sklearn.cluster import DBSCAN
import numpy as np
import torch
import networkx as nx
from transformers import TransformerEncoder

class AbstractionLearning:
    def __init__(self):
        self.abstraction_levels = {}
        self.pattern_database = {}
        self.clustering_model = DBSCAN(eps=0.3, min_samples=2)
        self.encoder = TransformerEncoder(
            num_layers=6,
            d_model=512,
            nhead=8,
            dim_feedforward=2048
        )
        self.concept_hierarchy = ConceptHierarchy()
        
    def extract_abstract_concepts(self, text: str) -> List[Dict]:
        concrete_elements = self._identify_concrete_elements(text)
        patterns = self._detect_patterns(concrete_elements)
        abstractions = self._generate_abstractions(patterns)
        return self._validate_abstractions(abstractions)
        
    def _identify_concrete_elements(self, text: str) -> List[str]:
        # Analyse sémantique améliorée
        doc = self.nlp(text)
        elements = []
        
        for chunk in doc.noun_chunks:
            elements.append({
                'text': chunk.text,
                'root': chunk.root.text,
                'deps': [child.dep_ for child in chunk.root.children],
                'embedding': self.get_embedding(chunk.text)
            })
            
        return elements
        
    def _detect_patterns(self, elements: List[Dict]) -> List[Dict]:
        # Détection de motifs avec apprentissage profond
        embeddings = torch.stack([elem['embedding'] for elem in elements])
        
        with torch.no_grad():
            encoded = self.encoder(embeddings)
            patterns = self.clustering_model.fit_predict(encoded.numpy())
            
        return self._analyze_clusters(elements, patterns)
        
    def _generate_abstractions(self, patterns: List[Dict]) -> List[Dict]:
        abstractions = []
        
        for pattern in patterns:
            # Agrège les éléments du pattern
            elements = pattern['elements']
            embeddings = [elem['embedding'] for elem in elements]
            
            # Calcule le centroïde du cluster comme représentation abstraite
            centroid = torch.mean(torch.stack(embeddings), dim=0)
            
            # Identifie les caractéristiques communes
            common_deps = set.intersection(*[set(elem['deps']) for elem in elements])
            common_roots = self._find_common_roots([elem['root'] for elem in elements])
            
            # Génère une abstraction
            abstraction = {
                'name': f"concept_{len(abstractions)}",
                'elements': elements,
                'embedding': centroid,
                'common_deps': list(common_deps),
                'common_roots': common_roots,
                'level': self._determine_abstraction_level(elements),
                'confidence': self._calculate_confidence(elements, centroid)
            }
            
            # Ajoute au graphe de concepts
            self.concept_hierarchy.add_concept(abstraction)
            abstractions.append(abstraction)
        
        return abstractions
        
    def _validate_abstractions(self, abstractions: List[Dict]) -> List[Dict]:
        """Valide et filtre les abstractions selon des critères de qualité."""
        validated = []
        for abstraction in abstractions:
            # Vérifie la cohérence interne
            if self._check_internal_coherence(abstraction):
                # Vérifie la distance avec les concepts existants
                if self._check_novelty(abstraction):
                    # Vérifie le niveau de confiance minimum
                    if abstraction['confidence'] > 0.7:
                        validated.append(abstraction)
        return validated

    def get_embedding(self, text: str) -> torch.Tensor:
        """Génère un embedding contextuel pour un texte donné."""
        # Tokenize et encode le texte
        tokens = self.tokenizer(text, return_tensors="pt", padding=True)
        
        # Génère l'embedding avec le transformer
        with torch.no_grad():
            outputs = self.encoder(tokens["input_ids"], 
                                attention_mask=tokens["attention_mask"])
            # Utilise la moyenne des dernières couches cachées
            embedding = torch.mean(outputs.last_hidden_state, dim=1)
            
        return embedding.squeeze()

    def _check_internal_coherence(self, abstraction: Dict) -> bool:
        """Vérifie la cohérence interne d'une abstraction."""
        elements = abstraction['elements']
        centroid = abstraction['embedding']
        
        # Calcule les distances au centroid
        distances = [torch.dist(elem['embedding'], centroid) for elem in elements]
        avg_distance = sum(distances) / len(distances)
        
        # Vérifie la dispersion
        return avg_distance < 0.5 and max(distances) < 0.8

    def _check_novelty(self, abstraction: Dict) -> bool:
        """Vérifie que l'abstraction n'est pas trop similaire aux concepts existants."""
        if not self.pattern_database:
            return True
            
        min_distance = float('inf')
        for pattern in self.pattern_database.values():
            distance = torch.dist(abstraction['embedding'], pattern['embedding'])
            min_distance = min(min_distance, distance)
            
        return min_distance > 0.3

class ConceptHierarchy:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concept_embeddings = {}
