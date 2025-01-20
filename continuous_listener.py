import pyaudio
import wave
import threading
import time
import speech_recognition as sr
import numpy as np
from datetime import datetime
import os
from voice_profile_manager import VoiceProfileManager
from scipy.signal import butter, filtfilt
from collections import deque

class ContinuousListener:
    def __init__(self, ai_instance):
        self.ai = ai_instance
        self.is_listening = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.recording_thread = None
        self.processing_thread = None
        self.conversation_dir = "conversations"
        
        if not os.path.exists(self.conversation_dir):
            os.makedirs(self.conversation_dir)
        
        self.voice_profile_manager = VoiceProfileManager()
        self.current_speaker = None
        self.unknown_voice_counter = 0

        # Paramètres améliorés pour la détection de la parole
        self.energy_threshold = 1500  # Seuil d'énergie plus sensible
        self.silence_threshold = 0.1  # Durée de silence en secondes
        self.min_phrase_duration = 0.5  # Durée minimale d'une phrase
        self.max_phrase_duration = 10.0  # Durée maximale d'une phrase
        self.pause_threshold = 0.8  # Durée de pause entre les phrases
        
        # Buffer circulaire pour l'analyse en temps réel
        self.audio_buffer = deque(maxlen=int(self.RATE * 2))  # Buffer de 2 secondes
        
        # Filtres audio
        self.noise_reduction_strength = 0.1
        self.frequency_range = (80, 3400)  # Plage de fréquences de la voix humaine
        
        # État de la conversation
        self.is_speaking = False
        self.current_conversation = []
        self.last_speech_time = None
        self.speech_buffer = []
        
        # Paramètres de qualité audio
        self.min_snr = 10  # Rapport signal/bruit minimum
        self.quality_threshold = 0.6  # Seuil de qualité minimum

    def start_listening(self):
        self.is_listening = True
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.recording_thread.start()
        self.processing_thread.start()

    def stop_listening(self):
        self.is_listening = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def _record_audio(self):
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        while self.is_listening:
            try:
                data = self.stream.read(self.CHUNK)
                self.frames.append(data)
            except Exception as e:
                print(f"Erreur d'enregistrement: {e}")
                break

    def _create_filters(self):
        """Crée les filtres audio"""
        nyquist = self.RATE / 2
        low = self.frequency_range[0] / nyquist
        high = self.frequency_range[1] / nyquist
        b, a = butter(4, [low, high], btype='band')
        return b, a

    def _process_audio(self):
        b, a = self._create_filters()
        speech_start_time = None
        is_speaking = False
        
        while self.is_listening:
            if len(self.frames) >= self.CHUNK * 2:  # Au moins 2 chunks pour l'analyse
                # Convertir les frames en numpy array
                audio_chunk = np.frombuffer(b''.join(self.frames[:2]), dtype=np.int16)
                
                # Appliquer le filtre passe-bande
                filtered_chunk = filtfilt(b, a, audio_chunk)
                
                # Calcul de l'énergie
                energy = np.mean(np.abs(filtered_chunk))
                
                # Détection de la parole
                if energy > self.energy_threshold and not is_speaking:
                    is_speaking = True
                    speech_start_time = time.time()
                    self.speech_buffer = []
                elif energy < self.energy_threshold and is_speaking:
                    # Vérifier la durée du silence
                    silence_duration = time.time() - (speech_start_time or time.time())
                    if silence_duration > self.silence_threshold:
                        is_speaking = False
                        self._process_speech_segment()
                
                # Ajouter l'audio au buffer si on parle
                if is_speaking:
                    self.speech_buffer.extend(self.frames[:2])
                    
                    # Vérifier la durée maximale
                    if time.time() - speech_start_time > self.max_phrase_duration:
                        is_speaking = False
                        self._process_speech_segment()
                
                # Retirer les frames traités
                self.frames = self.frames[2:]
            
            time.sleep(0.01)

    def _process_speech_segment(self):
        """Traite un segment de parole complet"""
        if not self.speech_buffer:
            return
            
        # Créer un fichier temporaire
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.conversation_dir}/segment_{timestamp}.wav"
        
        # Sauvegarder l'audio
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.speech_buffer))
        
        # Vérifier la qualité
        if self._check_audio_quality(filename):
            self._analyze_conversation(filename)
        else:
            os.remove(filename)

    def _check_audio_quality(self, audio_file):
        """Vérifie la qualité de l'enregistrement"""
        try:
            with wave.open(audio_file, 'rb') as wf:
                # Lire l'audio
                audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                
                # Calculer le SNR
                signal_power = np.mean(np.abs(audio_data)**2)
                noise_power = np.mean(np.abs(audio_data[:1000])**2)  # Utiliser le début comme estimation du bruit
                if noise_power == 0:
                    snr = float('inf')
                else:
                    snr = 10 * np.log10(signal_power/noise_power)
                
                # Vérifier la durée
                duration = len(audio_data) / self.RATE
                if duration < self.min_phrase_duration:
                    return False
                
                # Vérifier le niveau sonore
                if np.max(np.abs(audio_data)) < 100:
                    return False
                
                return snr >= self.min_snr
                
        except Exception as e:
            print(f"Erreur lors de la vérification de la qualité: {e}")
            return False

    def _analyze_conversation(self, audio_file):
        """Version améliorée de l'analyse de conversation"""
        try:
            # Identifier le locuteur
            speaker = self.voice_profile_manager.identify_speaker(audio_file)
            transcription = self._transcribe_audio(audio_file)
            
            if not transcription:
                return
                
            # Mettre à jour le contexte de la conversation
            if speaker:
                self._update_conversation_context(speaker, transcription)
            else:
                # Nouvelle voix détectée
                self._handle_unknown_speaker(audio_file, transcription)
            
            # Sauvegarder la transcription
            self._save_transcription(audio_file, speaker, transcription)
            
        except Exception as e:
            print(f"Erreur dans l'analyse de la conversation: {e}")

    def _transcribe_audio(self, audio_file):
        """Transcription améliorée avec vérification de la qualité"""
        recognizer = sr.Recognizer()
        text = None
        
        try:
            with sr.AudioFile(audio_file) as source:
                # Ajuster pour le bruit ambiant
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.record(source)
                
                # Essayer plusieurs moteurs de reconnaissance
                engines = [
                    ('google', lambda: recognizer.recognize_google(audio, language='fr-FR')),
                    ('sphinx', lambda: recognizer.recognize_sphinx(audio, language='fr-fr'))
                ]
                
                for engine_name, engine_func in engines:
                    try:
                        text = engine_func()
                        if text and len(text.split()) >= 2:  # Au moins deux mots
                            print(f"Transcription réussie avec {engine_name}: {text}")
                            break
                    except:
                        continue
                        
            return text
            
        except Exception as e:
            print(f"Erreur de transcription: {e}")
            return None

    def _update_conversation_context(self, speaker, text):
        """Met à jour le contexte de la conversation"""
        now = datetime.now()
        
        # Vérifier si c'est une nouvelle conversation
        if not self.current_conversation or \
           (now - self.last_speech_time).seconds > 30:  # Nouvelle conversation après 30s de silence
            self.current_conversation = []
        
        self.current_conversation.append({
            'speaker': speaker,
            'text': text,
            'timestamp': now
        })
        
        self.last_speech_time = now
        self.ai.learn_from_conversation(text, speaker)

    def _handle_unknown_speaker(self, audio_file, text):
        """Gère la détection d'un nouveau locuteur"""
        self.unknown_voice_counter += 1
        self.ai.ask_speaker_identity(audio_file, text)
