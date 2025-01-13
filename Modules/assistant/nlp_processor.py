import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import word_tokenize
import nltk

class NLPProcessor:
    def __init__(self):
        self.nlp = spacy.load('fr_core_news_lg')
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        nltk.download('punkt')
        
    def analyze_text(self, text):
        doc = self.nlp(text)
        return {
            'entities': self._extract_entities(doc),
            'summary': self._generate_summary(text),
            'sentiment': self._analyze_sentiment(doc),
            'key_phrases': self._extract_key_phrases(doc)
        }
        
    def _extract_entities(self, doc):
        return [(ent.text, ent.label_) for ent in doc.ents]
        
    def _generate_summary(self, text):
        inputs = self.t5_tokenizer("summarize: " + text, return_tensors="pt", max_length=512)
        summary_ids = self.t5_model.generate(inputs["input_ids"], max_length=150)
        return self.t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
