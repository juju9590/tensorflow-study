# 10-1 copy

# LCEL = Langchain Expression Language
# chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url='https://monogpt.kr/api/monorouter/v1'

prompt = "삼성전자의 창업주는 누구인가요?"

# from langchain_openai import OpenAIEmbeddings

# embedding = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     # model="text-embedding-3-large",
#     api_key=api_key,   
#     base_url=base_url,     
# )

# pip install langchain_huggingface
# pip install sentence-transformers
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3", #한국일본중국에서 강력한놈
    model_kwargs={
        "device": "cpu",
        # "device": "cuda", # AssertionError: Torch not compiled with CUDA enabled
        # "local_files_only" : True,
    }
)

vector = embedding.embed_query(prompt)
# print(vector)
print("============================================")
print("임베딩 벡터의 차원: ", len(vector)) # 임베딩 벡터의 차원:  1024

# model="text-embedding-3-small", 임베딩 벡터의 차원: 1536 ==> output_dim이 1536개란 뜻이다.
# model="text-embedding-3-large" 임베딩 벡터의 차원:  3072

