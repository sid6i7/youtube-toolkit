from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

class Caption:
    def __init__(self) -> None:
        self.transcriber = YouTubeTranscriptApi()
        self.transcriptFormatter = TextFormatter()
        
    def get_caption(self, videoId):
        """
            Fetches transcripts of youtube videos using video ID
            Input Parameters:
                - videoIds: list of video IDs to get a transcript of
            
            Returns: caption of the youtube video
        """
        caption = ''
        try:
            transcripts = self.transcriber.list_transcripts(video_id=videoId)
            try:
                engTranscripts = transcripts.find_transcript(['en'])
                caption = self.transcriptFormatter.format_transcript(engTranscripts[0].fetch())
            except:
                for transcript in transcripts:
                    caption = self.transcriptFormatter.format_transcript(transcript.translate('en').fetch())
        except:
            print('some error occured')

        return caption
        
        