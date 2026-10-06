#11-1 copy

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url='https://monogpt.kr/api/monorouter/v1'

# #01. (문서를)데이터를 불러온다
# path = './_data/rag_data/'
# loader1= TextLoader(path + 'samsung_outlook.txt', encoding='utf-8',)
# loader2= TextLoader(path + 'nvidia_outlook.txt', encoding='utf-8',)

# #02. 데이터를 자른다
# Text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=300,
#     chunk_overlap=100, # 데이터를 300으로 자를때 앞의 100은 중복하여 뒤의 300을 자른다 (중복)
#     separators=["\n\n", "\n", " ", ""],  #통상 디폴트
# )

# # 불러온 데이터를 #02 기준으로 잘라라
# split_doc1 = loader1.load_and_split(Text_splitter) #청크 300, 오버랩 100
# split_doc2 = loader2.load_and_split(Text_splitter) #청크 300, 오버랩 100

# # 문서 갯수 확인
# print(split_doc1)
# print(split_doc2)
# print(len(split_doc1), len(split_doc2)) # 9 9

# 임베딩 준비
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,   
    base_url=base_url,  
    # dimensions=10,   
)

# 불러오기
DB_path ='./_db/Chroma_11/'
db = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_path,  # 지속가능한 디렉토리
    collection_name='chroma11', # 저장된 DB불러올때 설정한 이름으로 불러온다.
)

# 저장된 데이터 확인
print("=========================")
print(db.get())
print("=========================")
aaa = db.similarity_search("삼성전자 사업전망에 대해 알려줘", k=2) 
# aaa = db.similarity_search("엔비디아 사업전망에 대해 알려줘", k=2) 
# 유사도 검색
# 요청한 문장과 비슷한 2개 골라줘 # 디폴트 = 4
print(aaa)








