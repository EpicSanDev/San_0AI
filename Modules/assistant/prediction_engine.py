import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

class PredictionEngine:
    def __init__(self):
        self.user_behavior_model = self._build_prediction_model()
        self.pattern_memory = []
        self.scaler = StandardScaler()
        
    def predict_user_needs(self, current_input):
        behavior_pattern = self._extract_behavior_pattern(current_input)
        scaled_pattern = self.scaler.transform([behavior_pattern])
        prediction = self.user_behavior_model.predict(scaled_pattern)
        return self._interpret_prediction(prediction)
        
    def generate_suggestions(self, predictions):
        relevant_suggestions = self._filter_relevant_suggestions(predictions)
        prioritized_suggestions = self._prioritize_suggestions(relevant_suggestions)
        return self._format_suggestions(prioritized_suggestions)
        
    def _build_prediction_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _extract_behavior_pattern(self, current_input):
        # Extract features from user input
        features = []
        features.append(len(current_input))
        features.append(current_input.count(' '))
        features.append(len(current_input.split('.')))
        features.append(1 if '?' in current_input else 0)
        features.append(1 if '!' in current_input else 0)
        
        # Pad to ensure 10 features
        while len(features) < 10:
            features.append(0)
            
        return features

    def _interpret_prediction(self, prediction):
        categories = ['question', 'command', 'statement', 'request', 'other']
        pred_index = np.argmax(prediction[0])
        confidence = prediction[0][pred_index]
        
        return {
            'category': categories[pred_index],
            'confidence': float(confidence),
            'raw_prediction': prediction.tolist()
        }

    def _filter_relevant_suggestions(self, predictions):
        # Filter predictions with confidence > 0.5
        return [pred for pred in predictions 
                if pred.get('confidence', 0) > 0.5]

    def _prioritize_suggestions(self, suggestions):
        # Sort by confidence score
        return sorted(suggestions, 
                     key=lambda x: x.get('confidence', 0), 
                     reverse=True)

    def _format_suggestions(self, suggestions):
        formatted = []
        for i, sugg in enumerate(suggestions):
            formatted.append({
                'id': i + 1,
                'category': sugg['category'],
                'confidence': f"{sugg['confidence']:.2%}",
                'recommendation': self._get_recommendation(sugg['category'])
            })
        return formatted

    def _get_recommendation(self, category):
        recommendations = {
            'question': "Consider providing a detailed answer",
            'command': "Execute the requested action",
            'statement': "Acknowledge and provide relevant information",
            'request': "Process the request and confirm action",
            'other': "Analyze context for appropriate response"
        }
        return recommendations.get(category, "Provide general response")

