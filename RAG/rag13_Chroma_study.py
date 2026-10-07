# Chroma 공부하기 

import os
from langchain_cummunity.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import TextSplitter
from langchain_chroma import Chroma
from glob import glob

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url='https://monogpt.kr/api/monorouter/v1'

path = './_data/rag_data'
txt_files = glob(os.path.join(path, '*.txt'))



