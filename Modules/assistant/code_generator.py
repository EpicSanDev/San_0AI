from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class CodeGenerator:
    def __init__(self):
        self.model_name = "Salesforce/codegen-350M-multi"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
    def generate_code(self, prompt, max_length=200):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
    def complete_code(self, partial_code):
        prompt = f"Complete this code:\n{partial_code}\n"
        return self.generate_code(prompt)
