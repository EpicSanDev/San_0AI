from typing import List, Tuple
import torch.nn.functional as F

class ResponseGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.response_templates = {
            "question": "Je pense que {answer}, mais je vous invite à explorer davantage.",
            "statement": "En effet, {answer}. De plus, {additional_info}",
            "command": "Je vais {answer}. Souhaitez-vous que je fasse autre chose ?"
        }
        
    def generate_structured_response(self, 
                                  input_text: str,
                                  context: dict,
                                  max_length: int = 200) -> str:
        input_type = self._classify_input(input_text)
        template = self.response_templates[input_type]
        
        # Génération de base
        response_parts = self._generate_response_parts(input_text, context)
        
        # Application du template
        formatted_response = template.format(**response_parts)
        
        # Post-traitement pour cohérence
        return self._post_process_response(formatted_response, context)
        
    def _classify_input(self, text: str) -> str:
        if "?" in text:
            return "question"
        if any(cmd in text.lower() for cmd in ["fais", "peux-tu", "pourrais-tu"]):
            return "command"
        return "statement"
        
    def _generate_response_parts(self, 
                               text: str, 
                               context: dict) -> dict:
        # Génération avec attention au contexte
        encoded = self.tokenizer.encode(text, return_tensors="pt")
        outputs = self.model.generate(
            encoded,
            max_length=100,
            num_return_sequences=3,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        
        responses = [self.tokenizer.decode(output) for output in outputs]
        return {
            "answer": responses[0],
            "additional_info": responses[1]
        }
        
    def _post_process_response(self, 
                             response: str, 
                             context: dict) -> str:
        # Ajustement en fonction du contexte émotionnel
        if context.get("emotion") == "joy":
            response = response.replace(".", "! ")
        elif context.get("emotion") == "sadness":
            response = response.replace("!", "... ")
            
        return response.strip()
