# YouTube Toolkit

A Python toolkit for working with YouTube videos: fetch captions, summarize them, run sentiment / readability analysis, and chat with a video using a local LLM and RAG.

## What it does

Three providers, all driven by a video URL:

### 1. Caption fetcher
Pulls the transcript for any YouTube video (using `youtube-transcript-api`), translates non-English transcripts to English, and runs Silero TE for punctuation restoration and sentence segmentation on raw captions.

### 2. Summarizer
Feeds the segmented transcript into a fine-tuned BART model (`philschmid/bart-large-cnn-samsum`) chunk by chunk to produce a readable summary. The chunk size and compression ratio are configurable.

### 3. Chat with a video (RAG)
Splits the transcript with LangChain's `RecursiveCharacterTextSplitter`, embeds the chunks using HuggingFace's `instructor-xl`, indexes them in a local Chroma vector store, and answers questions about the video using a locally-loaded Vicuna 7B (`TheBloke/vicuna-7B-1.1-HF`) wrapped in a LangChain `RetrievalQA` chain.

### 4. Sentiment / readability report
A separate analyzer over arbitrary text that returns Flesch-Kincaid readability, emotional tone, detected calls-to-action via spaCy NER, and the top frequent words.

## Stack

- **Captions:** `youtube-transcript-api`, Silero TE (`snakers4/silero-models`)
- **Summarization:** HuggingFace Transformers, BART
- **RAG:** LangChain, Chroma, `HuggingFaceInstructEmbeddings`, Vicuna 7B
- **NLP analysis:** NLTK, spaCy

## Project layout

```
config.py                            model names, paths, Chroma settings
providers/summarizer/caption.py      caption fetch, translation, sentence segmentation
providers/summarizer/summary.py      BART-based summarization
providers/chat/chat.py               RAG: chunk, embed, Chroma, Vicuna QA
providers/sentiment/sentiment.py     readability, tone, CTA, word freq
```

## Running it yourself

```bash
git clone https://github.com/sid6i7/youtube-toolkit
cd youtube-toolkit
pip install -r requirements.txt
```

Then import the provider you need:

```python
from providers.summarizer.caption import Caption
from providers.summarizer.summary import Summarizer

caption = Caption().get_caption("https://www.youtube.com/watch?v=...")
summary = Summarizer().generate_summary(caption)
```

## Note

The chat provider loads Vicuna 7B locally, which is GPU-friendly. CPU inference works but is slow.
