# 8-2 copy

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

from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,   
    base_url=base_url,  
    dimensions=10,   
)

# [0.6533203125, -0.389404296875, 0.01232147216796875, 0.31787109375, 0.5654296875]
# ============================================
# 임베딩 벡터의 차원:  5

vector = embedding.embed_query(prompt)
print(vector)
print("============================================")
print("임베딩 벡터의 차원: ", len(vector)) # 1536 ==> output_dim이 1536개란 뜻이다.

# 코사인 유사도(Cosine Similarity)는 두 벡터 사이의 각도를 이용해 방향의 유사함을 측정하는 방식
# 범위: -1에서 1 사이의 값을 가집니다.