import spacy
import nltk

class SentimentAnalyzer:
    def __init__(self, text) -> None:
        self.nlp = spacy.load('en_core_web_sm')
        self.text = text
        self.doc = self.nlp(text)
    
    def get_readability(self):
        return self.doc._.flesch_kincaid_grade_level

    def get_emotional_tone(self):
        sentences = nltk.sent_tokenize(self.text)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
            synsets = []
            for word, pos in pos_tags:
                synset = nltk.wn.synsets(word, pos=pos)
                if synset:
                    synsets.append(synset[0])
            if synsets:
                sentiment_score = nltk.sentiment.util.extract_unigram_feats(sentence, synsets)[0]['sentiment']
                return sentiment_score
    
    def get_call_to_action(self):
        calls = []
        for ent in self.doc.ents:
            if ent.label_ == 'ORG' or ent.label_ == 'PRODUCT':
                calls.append("Call to action:", ent.text, "is available now!")
        return calls
    
    def get_most_frequent_words(self):
        words = nltk.word_tokenize(self.text)
        freq = nltk.FreqDist(words)
        
        return freq.most_common(10)
    
    def generate_report(self):
        return {
            'readability': self.get_readability(),
            'emotional_tone': self.get_emotional_tone(),
            'call_to_action': self.get_call_to_action(),
            'most_frequent_words': self.get_most_frequent_words()
        }
    
    def generate_wordcloud(self):
        pass
    
    