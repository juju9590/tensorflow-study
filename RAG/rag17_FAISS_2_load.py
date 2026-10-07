# 17-1 copy

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
# Chroma : 코사인 유사도 기반 (통상적으로)

# pip install faiss-cpu
import faiss 
# faiss : 유클리드 거리 기반 (통상적으로)
# 검색에 강력
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url='https://monogpt.kr/api/monorouter/v1'


# 임베딩 준비
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,   
    base_url=base_url,  
)

DB_path ='./_db/faiss_17/'

# db.save_local(
#     folder_path=DB_path,
#     index_name='faiss_index17'
# )

db = FAISS.load_local(
    folder_path=DB_path,
    index_name='faiss_index17',
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

print("============================================")
# 문서 저장소 ID 확인
print("문서 저장소 ID 확인: ", db.index_to_docstore_id)
# 저장된 결과 확인
print("저장된 결과 확인: ",db.docstore._dict)
# 유사도 검색
aaa  = db.similarity_search("삼성전자 창업주에 대해 알려줘", k=2)
print(aaa)

