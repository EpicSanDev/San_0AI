import os
import numpy as np
import pickle
from datetime import datetime
import librosa
from sklearn.mixture import GaussianMixture
from pathlib import Path

class VoiceProfileManager:
    def __init__(self):
        self.profiles_dir = "voice_profiles"
        self.profiles = {}
        self.models = {}
        self.similarity_threshold = 0.75
        self.min_duration = 1.0  # Durée minimale en secondes
        self.sample_rate = 16000
        self.feature_params = {
            'n_mfcc': 20,
            'n_mels': 40,
            'n_components': 16  # Nombre de composants pour GMM
        }
        
        # Créer le dossier des profils s'il n'existe pas
        if not os.path.exists(self.profiles_dir):
            os.makedirs(self.profiles_dir)
            
        self.load_profiles()

    def load_profiles(self):
        """Charge tous les profils vocaux existants"""
        try:
            for profile_file in Path(self.profiles_dir).glob("*.pkl"):
                name = profile_file.stem
                with open(profile_file, 'rb') as f:
                    self.profiles[name] = pickle.load(f)
                    if 'model' in self.profiles[name]:
                        self.models[name] = self.profiles[name]['model']
        except Exception as e:
            print(f"Erreur lors du chargement des profils: {e}")

    def add_voice_profile(self, name, audio_file):
        """Ajoute un nouveau profil vocal"""
        try:
            # Vérifier si le fichier audio existe
            if not os.path.exists(audio_file):
                raise FileNotFoundError("Fichier audio non trouvé")
                
            # Extraire les caractéristiques vocales
            features = self._extract_voice_features(audio_file)
            if features is None:
                return False
                
            # Créer/mettre à jour le modèle GMM
            model = GaussianMixture(
                n_components=self.feature_params['n_components'],
                covariance_type='full',
                random_state=42
            )
            model.fit(features)
            
            # Sauvegarder le profil
            profile_data = {
                'name': name,
                'created': str(datetime.now()),
                'updated': str(datetime.now()),
                'features': features,
                'model': model,
                'audio_samples': [audio_file]
            }
            
            self.profiles[name] = profile_data
            self.models[name] = model
            
            # Sauvegarder dans un fichier
            profile_path = os.path.join(self.profiles_dir, f"{name}.pkl")
            with open(profile_path, 'wb') as f:
                pickle.dump(profile_data, f)
                
            return True
            
        except Exception as e:
            print(f"Erreur lors de l'ajout du profil vocal: {e}")
            return False

    def identify_speaker(self, audio_file):
        """Identifie le locuteur à partir d'un échantillon audio"""
        try:
            if not self.models:
                return None
                
            features = self._extract_voice_features(audio_file)
            if features is None:
                return None
                
            # Calculer les scores pour chaque modèle
            scores = {}
            for name, model in self.models.items():
                score = np.mean(model.score_samples(features))
                scores[name] = score
                
            # Trouver le meilleur score
            best_speaker = max(scores.items(), key=lambda x: x[1])
            
            # Vérifier si le score est suffisant
            if best_speaker[1] > self.similarity_threshold:
                return best_speaker[0]
                
            return None
            
        except Exception as e:
            print(f"Erreur lors de l'identification du locuteur: {e}")
            return None

    def update_profile(self, name, audio_file):
        """Met à jour un profil vocal existant"""
        try:
            if name not in self.profiles:
                return False
                
            # Extraire les nouvelles caractéristiques
            new_features = self._extract_voice_features(audio_file)
            if new_features is None:
                return False
                
            # Combiner avec les caractéristiques existantes
            existing_features = self.profiles[name]['features']
            combined_features = np.vstack([existing_features, new_features])
            
            # Mettre à jour le modèle
            model = GaussianMixture(
                n_components=self.feature_params['n_components'],
                covariance_type='full',
                random_state=42
            )
            model.fit(combined_features)
            
            # Mettre à jour le profil
            self.profiles[name].update({
                'updated': str(datetime.now()),
                'features': combined_features,
                'model': model
            })
            self.profiles[name]['audio_samples'].append(audio_file)
            
            # Sauvegarder les modifications
            profile_path = os.path.join(self.profiles_dir, f"{name}.pkl")
            with open(profile_path, 'wb') as f:
                pickle.dump(self.profiles[name], f)
                
            self.models[name] = model
            return True
            
        except Exception as e:
            print(f"Erreur lors de la mise à jour du profil: {e}")
            return False

    def _extract_voice_features(self, audio_file):
        """Extrait les caractéristiques vocales d'un fichier audio"""
        try:
            # Charger l'audio
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Vérifier la durée
            if librosa.get_duration(y=audio, sr=sr) < self.min_duration:
                return None
                
            # Extraire les MFCC
            mfccs = librosa.feature.mfcc(
                y=audio, 
                sr=sr,
                n_mfcc=self.feature_params['n_mfcc']
            )
            
            # Extraire les mel-spectrogrammes
            mel_spect = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=self.feature_params['n_mels']
            )
            
            # Combiner les caractéristiques
            features = np.vstack([
                mfccs,
                librosa.power_to_db(mel_spect)
            ])
            
            # Normaliser
            features = (features - np.mean(features)) / np.std(features)
            
            return features.T
            
        except Exception as e:
            print(f"Erreur lors de l'extraction des caractéristiques: {e}")
            return None

    def remove_profile(self, name):
        """Supprime un profil vocal"""
        try:
            if name in self.profiles:
                profile_path = os.path.join(self.profiles_dir, f"{name}.pkl")
                if os.path.exists(profile_path):
                    os.remove(profile_path)
                del self.profiles[name]
                if name in self.models:
                    del self.models[name]
                return True
            return False
        except Exception as e:
            print(f"Erreur lors de la suppression du profil: {e}")
            return False

    def get_profile_info(self, name):
        """Récupère les informations d'un profil"""
        if name in self.profiles:
            profile = self.profiles[name]
            return {
                'name': profile['name'],
                'created': profile['created'],
                'updated': profile['updated'],
                'samples_count': len(profile['audio_samples'])
            }
        return None

    def list_profiles(self):
        """Liste tous les profils vocaux disponibles"""
        return list(self.profiles.keys())
