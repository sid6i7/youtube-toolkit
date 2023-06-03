from config import *
from transformers import pipeline
from nltk.tokenize import sent_tokenize

class Summarizer:
    """
        Responsible for summarization
    """
    def __init__(self) -> None:
        self.model = pipeline("summarization", model=SUMMARIZER_TRANSFORMERS_MODEL, tokenizer=SUMMARIZER_TRANSFORMERS_MODEL)
    
    def generate_summary(self, text, ratio=3):
        """Generates summary of a list of sentences

        Args:
            sentences (List of Strings): List of sentences segmented from a paragraph
            ratio (int): Length of sentence gets reduced by this ratio when creating summary

        Returns:
            String: Summarized paragraph
        """
        sentences = sent_tokenize(text)
        summary = ''
        para = ''
        for sentence in sentences:    
            if len(para + " " + sentence) < 1024:
                para += " " + sentence
                para = para.strip()
            else:
                summary += self.model(para, min_length=0, max_length=len(para)//ratio)[0]['summary_text']
                para = ''

        return summary