#데이터불러오기 - 잘라서 - 임베팅 - 저장하기
# 11-1 copy

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

#01. (문서를)데이터를 불러온다
path = './_data/rag_data/'
loader1= TextLoader(path + 'samsung_outlook.txt', encoding='utf-8',)
loader2= TextLoader(path + 'nvidia_outlook.txt', encoding='utf-8',)

#02. 데이터를 자른다
Text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100, 
    separators=["\n\n", "\n", " ", ""],  #통상 디폴트
)

# 불러온 데이터를 #02 기준으로 잘라라
split_doc1 = loader1.load_and_split(Text_splitter) #청크 300, 오버랩 100
split_doc2 = loader2.load_and_split(Text_splitter) #청크 300, 오버랩 100

# 문서 갯수 확인
# print(split_doc1)
# print(split_doc2)
# print(len(split_doc1), len(split_doc2)) # 9 9
# # 청킹하는 이유는 ? 벡터DB 만들기 위해 

# 임베딩 준비
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,   
    base_url=base_url,  
    # dimensions=10,   #1536
)

###################################
'''
# 여기부터 faiss 
# 1) faiss 자제 제공 

faiss_index = faiss.IndexFlat(len(embeddings.embed_query("hello world")))
# faiss_index = faiss.IndexFlat2(1536) # 차원의 수를 안다면...
print("FAISS 인덱스 초기화 준비 완료")

# FAISS 벡터 저장소의 벡터 차원 수 (임베딩 차원 수)
print(faiss_index.d) #1536
# d = dimensions 차원...

faiss_db = FAISS(
    embedding_function=embeddings,
    index = faiss_index,
    docstore=InMemoryDocstore(), # 메모리 기반으로 작업하겠다.. 껏다켜면 사라진다.
    index_to_docstore_id={},
)

# 저장된 문서의 갯수 확인
print(faiss_db.index.ntotal) # 0 
# 준비 완료
'''

###################################
# 2) langchain에서 제공하는 FAISS 에서 제공
# 파이스 DB
DB_path ='./_db/faiss_17/'

db = FAISS.from_documents(
    documents=split_doc1 + split_doc2,
    embedding=embeddings,
)

db.save_local(
    folder_path=DB_path,
    index_name='faiss_index17'
)

# 크로마 DB와 비교
# DB_path ='./_db/Chroma_12/'
# vector_store = Chroma.from_documents(
#     documents=texts,
#     embedding=embeddings,
#     persist_directory=DB_path, #지속가능한 디렉토리
#     collection_name='chroma12',
# )


