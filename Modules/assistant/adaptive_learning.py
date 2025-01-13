import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import logging

@dataclass
class LearningMetrics:
    loss: float
    accuracy: float
    confidence: float
    diversity: float
    gradient_norm: float
    validation_loss: float
    epoch_time: float

class AdaptiveLearning:
    def __init__(self):
        self.learning_rates = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
        self.batch_sizes = [8, 16, 32, 64, 128]
        self.warmup_steps = 1000
        self.history: List[LearningMetrics] = []
        self.patience = 5
        self.min_delta = 0.001
        self.learning_strategies = {
            'default': self._default_strategy,
            'aggressive': self._aggressive_strategy,
            'conservative': self._conservative_strategy,
        }
        self.current_strategy = 'default'
        
    def adjust_parameters(self, metrics: LearningMetrics) -> Tuple[float, int]:
        self.history.append(metrics)
        
        # Période de warmup
        if len(self.history) < self.warmup_steps:
            return self._warmup_strategy()
            
        # Détection de plateau
        if self._is_plateau():
            return self._plateau_strategy()
            
        # Détection d'overfitting
        if self._is_overfitting():
            return self._overfitting_strategy()
            
        # Ajustement basé sur le gradient
        if metrics.gradient_norm > 10.0:
            return self._high_gradient_strategy()
            
        return self._default_strategy()
        
    def _warmup_strategy(self) -> Tuple[float, int]:
        progress = len(self.history) / self.warmup_steps
        lr_index = min(int(progress * len(self.learning_rates)), len(self.learning_rates) - 1)
        return self.learning_rates[lr_index], self.batch_sizes[0]
        
    def _is_plateau(self) -> bool:
        if len(self.history) < self.patience:
            return False
        recent_losses = [m.loss for m in self.history[-self.patience:]]
        return max(recent_losses) - min(recent_losses) < self.min_delta
        
    def _is_overfitting(self) -> bool:
        if len(self.history) < 2:
            return False
        return (self.history[-1].validation_loss > self.history[-2].validation_loss and
                self.history[-1].loss < self.history[-2].loss)
                
    def update_strategy(self, performance_score):
        if performance_score < 0.5:
            self.current_strategy = 'aggressive'
        elif performance_score > 0.8:
            self.current_strategy = 'conservative'
        else:
            self.current_strategy = 'default'
            
    def _aggressive_strategy(self) -> Tuple[float, int]:
        return max(self.learning_rates), min(self.batch_sizes)
        
    def _conservative_strategy(self) -> Tuple[float, int]:
        return min(self.learning_rates), max(self.batch_sizes)
