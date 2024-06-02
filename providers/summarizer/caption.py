from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import re
from urllib.parse import urlparse, parse_qs
import nltk
nltk.download('punkt', halt_on_error=False)
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import torch
from nltk import sent_tokenize

class Caption:
    """
        Responsible for obtaining and parsing caption of a YouTube video
    """
    def __init__(self) -> None:
        self.transcriber = YouTubeTranscriptApi()
        self.transcriptFormatter = TextFormatter()
        self.lemmatizer = WordNetLemmatizer()
        model, example_texts, languages, punct, apply_te = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_te')
        self.sent_seg_model = apply_te
    
    def get_video_id(self, url):
        parsed_url = urlparse(url)
        
        # Check if it's a shortened YouTube URL
        if parsed_url.netloc == 'youtu.be':
            video_id = parsed_url.path[1:]  # Remove the leading slash
            return video_id
        
        query_params = parse_qs(parsed_url.query)
        
        if 'v' in query_params:
            return query_params['v'][0]
        else:
            # If the 'v' parameter is not present, check if it's in the path
            path_parts = parsed_url.path.split('/')
            if len(path_parts) > 1:
                return path_parts[1]
        
        # If no video ID found
        return None
        
    def get_caption(self, link, preprocess=False):
        """Fetches transcripts of youtube videos using video ID

        Args:
            videoId (String): YouTube video ID to get a transcript of
            preprocess (bool, optional): Whether to pre-process the caption. Defaults to False.

        Returns:
            String: caption of the YouTube video
        """
        
        caption = ''
        try:
            videoId = self.get_video_id(link)
            # captions in all available languages
            transcripts = self.transcriber.list_transcripts(video_id=videoId)
            try:
                # getting english if available
                engTranscripts = transcripts.find_transcript(['en'])
                caption = self.transcriptFormatter.format_transcript(engTranscripts[0].fetch())
            except:
                # translating caption to english if not available
                for transcript in transcripts:
                    caption = self.transcriptFormatter.format_transcript(transcript.translate('en').fetch())
        except:
            print('some error occured')
            
        if preprocess:
            caption = self.preprocess_caption(caption)
            
        return self.sent_segmentation(caption)
        
    def sent_segmentation(self, caption):
        """Adds punctuations to raw captions and performs sentence segmentation

        Args:
            caption (String): Raw youtube caption

        Returns:
            _type_: List of sentences
        """
        
        caption = self.sent_seg_model(caption, lan='en')
        # return sent_tokenize(caption)
        return caption

    def preprocess_caption(self, caption):
        """Applies some basic NLP text pre-processing

        Args:
            caption (String): raw text

        Returns:
            String: processed text
        """
        # Remove timestamps in format [00:00:00]
        caption = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', caption)
        # Remove speaker names in format SPEAKER:
        caption = re.sub(r'[A-Z]+:', '', caption)
        # Remove sound effects in format (SOUND EFFECT)
        caption = re.sub(r'\([^)]*\)', '', caption)
        # Remove music lyrics in format [LYRICS]
        caption = re.sub(r'\[[^]]*\]', '', caption)
        
        # Tokenization
        tokens = nltk.word_tokenize(caption)
        # Removing stop-words
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
        # Stemming
        stemmed_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
        caption = " ".join(stemmed_tokens)
        # Removing irrelevant content
        doc = self.nlp(caption)
        filtered_tokens = []
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space or token.text.lower() in ['like', 'repeated']:
                continue
            filtered_tokens.append(token.text)
        caption = ' '.join(filtered_tokens)

        return caption