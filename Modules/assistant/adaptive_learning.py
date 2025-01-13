import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class LearningMetrics:
    loss: float
    accuracy: float
    confidence: float
    diversity: float

class AdaptiveLearning:
    def __init__(self):
        self.learning_rates = [1e-5, 3e-5, 1e-4]
        self.batch_sizes = [16, 32, 64]
        self.current_metrics = None
        
    def adjust_parameters(self, metrics: LearningMetrics):
        if not self.current_metrics:
            self.current_metrics = metrics
            return self.learning_rates[0], self.batch_sizes[0]
            
        if metrics.loss < self.current_metrics.loss:
            # Amélioration : augmenter le learning rate
            lr_index = min(self.learning_rates.index(self.current_lr) + 1,
                         len(self.learning_rates) - 1)
            return self.learning_rates[lr_index], self.batch_sizes[0]
            
        # Si la performance se dégrade
        if metrics.loss > self.current_metrics.loss * 1.1:
            # Réduire le learning rate et augmenter la taille du batch
            lr_index = max(0, self.learning_rates.index(self.current_lr) - 1)
            batch_index = min(len(self.batch_sizes) - 1,
                            self.batch_sizes.index(self.current_batch) + 1)
            return self.learning_rates[lr_index], self.batch_sizes[batch_index]
            
        return self.current_lr, self.current_batch
