import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
from nltk.tokenize import word_tokenize
import nltk
import logging
import warnings

class NLPProcessor:
    def __init__(self):
        self.nlp = spacy.load('fr_core_news_lg')
        try:
            self.t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')
        except ImportError:
            warnings.warn(
                "SentencePiece library not found. Using basic tokenizer instead. "
                "Install sentencepiece for full NLP functionality."
            )
            # Use a simpler tokenizer as fallback
            self.t5_tokenizer = None
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        nltk.download('punkt')
        
    def analyze_text(self, text):
        if self.t5_tokenizer is None:
            # Basic fallback tokenization
            return {
                'tokens': text.split(),
                'length': len(text),
                'limited_mode': True
            }
            
        # Full analysis mode with T5 tokenizer
        tokens = self.t5_tokenizer.tokenize(text)
        return {
            'tokens': tokens,
            'length': len(text), 
            'token_count': len(tokens),
            'limited_mode': False
        }
        
    def _extract_entities(self, doc):
        return [(ent.text, ent.label_) for ent in doc.ents]
        
    def _generate_summary(self, text):
        inputs = self.t5_tokenizer("summarize: " + text, return_tensors="pt", max_length=512)
        summary_ids = self.t5_model.generate(inputs["input_ids"], max_length=150)
        return self.t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
