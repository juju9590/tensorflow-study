# 데이터 불러오기 > 청킹하기  > 임베딩 하기 > 저장하기 

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url='https://monogpt.kr/api/monorouter/v1'

"""
from glob import glob

# 폴더에서 텍스트 파일 목록 가져오기 
path = './_data/rag_data/'
txt_files = glob(os.path.join(path, '*.txt'))
# print(txt_files)
# ['./_data/rag_data\\2026_AI_for_All.txt', './_data/rag_data\\nvidia_outlook.txt', './_data/rag_data\\samsung_outlook.txt']
# 리스트 =>  이터레이터 => for 문 

#01. (문서를)데이터를 불러온다
data = []
for text_file in txt_files:
    loader = TextLoader(text_file, encoding='utf-8')
    data = data + loader.load()

# print(data[0])
# print(len(data)) #3
# print(data[2].page_content)

char_count = [len(doc.page_content) for doc in data]
# print(char_count) #[8158, 2049, 1898] 각 텍스트 문서의 문장길이

#02. 데이터를 자른다 (청킹)
Text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50, # 데이터를 300으로 자를때 앞의 100은 중복하여 뒤의 300을 자른다 (중복)
    separators=["\n\n", "\n", " ", ""],  #통상 디폴트
    # \n\n 문단바꿈
    # \n 줄바꿈
    # " " 띄어쓰기
)
# 청킹 단위로 잘라주기 3가지 방법 있음
# 1. RecursiveCharacterTextSplitter
# 2. CharacterTextSplitter
# 3. TextSplitter

texts = Text_splitter.split_documents(data)
# print("생성된 텍스트 청크 수 : ", len(texts)) # 56
# 벡터 DB가 56개 생길예정
# print("각 청크의 길이 : ", list(len(text.page_content) for text in texts))
# [259, 154, 150, ..., 68, 187, 249]

# [[ 길이 단위 청킹 - RecursiveCharacterTextSplitter ]]
# ㄴ청크의 길이가  300이 되지 않는 이유는 ???

# 사용자는 각 청크의 최대 길이를 지정할 수 있습니다. 
# 예를 들어, 길이가 10,000인 텍스트에 대해 각 청크의 최대 길이를 500으로 설정하면, 
# 해당 텍스트는 길이 500을 초과하지 않는 여러 개의 청크로 분할됩니다. 
# 이 분할기는 ["\n\n", "\n", " ", ""] 총 4개의 문자를 기준으로 사용합니다. 
# 동작 방식은 다음과 같습니다.

# 먼저 \n\n(두 번의 줄바꿈)을 기준으로 텍스트를 나눕니다.
# 나눈 청크가 여전히 원하는 길이보다 크다면, \n(한 번의 줄바꿈)을 기준으로 다시 나눕니다.
# 그래도 지정된 크기를 초과한다면, " "(공백)을 사용하여 나누는 작업을 반복합니다.

# 56개의 청크 중에 0번째 내용 확인
# print("두번째 청크의 내용 : ",  texts[0]) #page_content(내용), metadata(경로)
# 첫번째 청크의 내용 :  
# page_content='1. 한국의 AI for All 프로젝트
# 한국 정부는 생성형 인공지능을 일부 전문가나 기업만 사용하는 기술이 아니라 
# 일반 국민이 일상적으로 활용할 수 있는 디지털 기반 기술로 확산하는 정책을 추진하고
# 있다. 이러한 방향을 대표하는 사업이 'AI for All' 프로젝트이다.' 
# metadata={'source': './_data/rag_data\\2026_AI_for_All.txt'}

# print("두번째 청크의 내용 : ",  texts[1].page_content) # 내용만
# print("두번째 청크의 길이 : ",  len(texts[1].page_content)) #154
# print("첫번째 청크의 내용 : ",  texts[1]) 
"""

#03. 임베딩 준비 
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,   
    base_url=base_url,  
)

# sample_text = '삼성전자 창업주는?'
# vector = embeddings.embed_query(sample_text)
# # print(vector)
# # print(len(vector)) # 1536 차원


DB_path ='./_db/Chroma_12/'

# 저장하기 
# vector_store = Chroma.from_documents(
#     documents=texts,
#     embedding=embeddings,
#     persist_directory=DB_path, #지속가능한 디렉토리
#     collection_name='chroma12',
# )

# 불러오기
vector_store = Chroma(
    embedding_function=embeddings, 
    # 어떤 임베딩을 사용할껀지 있기때문에 임베딩 준비는 남겨야 한다.
    persist_directory=DB_path, 
    collection_name='chroma12',
)

print(f"벡터 DB에 저장된 문서 개수: {vector_store._collection.count()}")
# 벡터 DB에 저장된 문서 개수: 56


query = "삼성전자는 어떤 기업인가요?"
result = vector_store.similarity_search(query)
print(f"검색 결과의 길이: {len(result)}") 
# 검색 결과의 길이: 4 >>> K 디폴트가 4
# aaa = db.similarity_search("엔비디아 사업전망에 대해 알려줘", k=2) 


############# 검색기 Retrievers = vector_store 검색

retrievers = vector_store.as_retriever(search_kwargs={"k":3}) #검색된 관련 문서 수 : 3
print(retrievers)
# tags=['Chroma', 'OpenAIEmbeddings'] vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x000001212752EB50> search_kwargs={'k': 3}
aaa = retrievers.invoke(query)
print(f"검색된 관련 문서 수 : {len(aaa)}") # 검색된 관련 문서 수 : 3
print(f"첫번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}")
# 첫번째 관련 문서 내용 미리보기 : 삼성전자 사업 전망
# 삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사
print(f"두번째 관련 문서 내용 미리보기 : {aaa[1].page_content[:50]}")
# 두번째 관련 문서 내용 미리보기 : 삼성전자의 중장기 전망은 인공지능 메모리의 경쟁력, 첨단 공정의 수율 개선, 파운드리 고객
print(f"세번째 관련 문서 내용 미리보기 : {aaa[2].page_content[:50]}")
# 세번째 관련 문서 내용 미리보기 : 스마트폰 사업은 프리미엄 제품의 교체 수요와 중저가 제품의 판매량이 함께 영향을 미친다.
# print(f"네번째 관련 문서 내용 미리보기 : {aaa[3].page_content[:50]}")
# IndexError: list index out of range
