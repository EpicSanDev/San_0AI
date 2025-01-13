from typing import Dict, Any
import numpy as np

class WorkingMemory:
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.contents: Dict[str, Any] = {}
        self.activation_levels: Dict[str, float] = {}
        
    def update(self, new_input: Any) -> Dict[str, Any]:
        # Mise à jour de la mémoire de travail
        self._update_activation_levels()
        self._add_new_content(new_input)
        self._remove_inactive_content()
        return self.contents
        
    def _update_activation_levels(self):
        decay_rate = 0.1
        for key in self.activation_levels:
            self.activation_levels[key] *= (1 - decay_rate)
