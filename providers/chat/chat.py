import os
from providers.summarizer.caption import *
from providers.summarizer.summary import *
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from config import *

class Chat:
    def __init__(self):
        self.captioner = Caption()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.llm = self.load_model()
        self.qa = None
        self.db = None

    def load_model(self):
        model_id = "TheBloke/vicuna-7B-1.1-HF"
        tokenizer = LlamaTokenizer.from_pretrained(model_id)

        model = LlamaForCausalLM.from_pretrained(model_id,
                                                #   load_in_8bit=True, # set these options if your GPU supports them!
                                                #   device_map=1#'auto',
                                                #   torch_dtype=torch.float16,
                                                #   low_cpu_mem_usage=True
                                                  )

        pipe = pipeline(
            "text-generation",
            model=model, 
            tokenizer=tokenizer, 
            max_length=2048,
            temperature=0,
            top_p=0.95,
            repetition_penalty=1.15
        )

        local_llm = HuggingFacePipeline(pipeline=pipe)

        return local_llm

    def process_video(self, video_link):
        caption = self.captioner.get_caption(video_link)
        print(f'Got caption of length: {len(caption)}')

        texts = self.text_splitter.split_text(caption)
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                        model_kwargs={"device": 'cpu'})
        print('Got embeddings')

        self.db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=PERSIST_DIRECTORY,
            client_settings=CHROMA_SETTINGS
        )

        print('Loaded model')

    def ask_question(self, query):
        if self.qa is None:
            self.qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.db.as_retriever(),
                return_source_documents=True
            )

        print('Asking question')
        res = self.qa(query)
        return res['result']

    def clean_up(self):
        if self.db is not None:
            self.db.persist()
            self.db = None