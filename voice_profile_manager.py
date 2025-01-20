import numpy as np
import pickle
import os
from datetime import datetime
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture

class VoiceProfileManager:
    def __init__(self):
        self.profiles_dir = "voice_profiles"
        self.profiles = {}
        self.voice_models = {}
        self.unknown_counter = 0
        
        if not os.path.exists(self.profiles_dir):
            os.makedirs(self.profiles_dir)
            
        self.load_profiles()

    def load_profiles(self):
        """Charge les profils vocaux existants"""
        for file in os.listdir(self.profiles_dir):
            if file.endswith('.pkl'):
                name = file.replace('.pkl', '')
                with open(os.path.join(self.profiles_dir, file), 'rb') as f:
                    self.voice_models[name] = pickle.load(f)
                self.profiles[name] = {
                    'created': datetime.now(),
                    'samples': []
                }

    def extract_voice_features(self, audio_data, sample_rate):
        """Extrait les caractéristiques de la voix"""
        # Conversion en mono si nécessaire
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Extraction des caractéristiques (MFCC simplifié)
        frame_length = int(sample_rate * 0.025)  # 25ms frames
        features = []
        for i in range(0, len(audio_data) - frame_length, frame_length):
            frame = audio_data[i:i + frame_length]
            features.append([
                np.mean(frame),
                np.std(frame),
                np.max(np.abs(frame)),
                np.mean(np.abs(np.fft.fft(frame)))
            ])
        return np.array(features)

    def identify_speaker(self, audio_file):
        """Identifie le locuteur à partir d'un fichier audio"""
        try:
            sample_rate, audio_data = wavfile.read(audio_file)
            features = self.extract_voice_features(audio_data, sample_rate)
            
            if not self.voice_models:
                return None
                
            scores = {}
            for name, model in self.voice_models.items():
                try:
                    score = model.score(features)
                    scores[name] = score
                except:
                    continue
                    
            if scores:
                best_match = max(scores.items(), key=lambda x: x[1])
                if best_match[1] > -100:  # Seuil de confiance
                    return best_match[0]
                    
            return None
            
        except Exception as e:
            print(f"Erreur d'identification: {e}")
            return None

    def add_voice_profile(self, name, audio_file):
        """Ajoute un nouveau profil vocal"""
        try:
            sample_rate, audio_data = wavfile.read(audio_file)
            features = self.extract_voice_features(audio_data, sample_rate)
            
            # Création du modèle GMM
            model = GaussianMixture(n_components=4, random_state=0)
            model.fit(features)
            
            # Sauvegarde du modèle
            self.voice_models[name] = model
            with open(os.path.join(self.profiles_dir, f"{name}.pkl"), 'wb') as f:
                pickle.dump(model, f)
                
            self.profiles[name] = {
                'created': datetime.now(),
                'samples': [audio_file]
            }
            
            return True
            
        except Exception as e:
            print(f"Erreur création profil: {e}")
            return False

    def update_profile(self, name, audio_file):
        """Met à jour un profil vocal existant"""
        if name in self.profiles:
            self.profiles[name]['samples'].append(audio_file)
            # Réentraîner le modèle avec tous les échantillons
            return self.add_voice_profile(name, audio_file)
        return False
