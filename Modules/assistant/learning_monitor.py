import numpy as np
from datetime import datetime

class LearningMonitor:
    def __init__(self):
        self.performance_history = []
        self.learning_rate_history = []
        self.adaptation_threshold = 0.1
        
    def track_performance(self, loss, accuracy, learning_rate):
        self.performance_history.append({
            'timestamp': datetime.now(),
            'loss': loss,
            'accuracy': accuracy
        })
        self.learning_rate_history.append(learning_rate)
        
    def should_adapt_learning(self):
        if len(self.performance_history) < 10:
            return False
            
        recent_losses = [p['loss'] for p in self.performance_history[-10:]]
        loss_trend = np.gradient(recent_losses)
        
        if np.mean(loss_trend) > self.adaptation_threshold:
            return True
        return False
        
    def get_optimal_learning_rate(self):
        if not self.performance_history:
            return 1e-5
            
        recent_performance = self.performance_history[-10:]
        best_loss_idx = np.argmin([p['loss'] for p in recent_performance])
        
        return self.learning_rate_history[-10:][best_loss_idx]

    def generate_learning_report(self):
        if not self.performance_history:
            return "Insufficient data for learning report"
            
        avg_loss = np.mean([p['loss'] for p in self.performance_history])
        avg_accuracy = np.mean([p['accuracy'] for p in self.performance_history])
        
        return {
            "average_loss": avg_loss,
            "average_accuracy": avg_accuracy,
            "learning_stability": self._calculate_stability(),
            "improvement_rate": self._calculate_improvement_rate()
        }
        
    def _calculate_stability(self):
        if len(self.performance_history) < 2:
            return 1.0
            
        losses = [p['loss'] for p in self.performance_history]
        return 1.0 / (np.std(losses) + 1e-6)
        
    def _calculate_improvement_rate(self):
        if len(self.performance_history) < 2:
            return 0.0
            
        recent_losses = [p['loss'] for p in self.performance_history[-10:]]
        return (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
