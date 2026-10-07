# 12-4 copy

import os

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url='https://monogpt.kr/api/monorouter/v1'

#03. 임베딩 준비 
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,   
    base_url=base_url,  
)

DB_path ='./_db/Chroma_12/'

# 불러오기
vector_store = Chroma(
    embedding_function=embeddings, 
    persist_directory=DB_path, 
    collection_name='chroma12',
)

print(f"벡터 DB에 저장된 문서 개수: {vector_store._collection.count()}")
# 벡터 DB에 저장된 문서 개수: 56

query = "게임용 GPU 시장에 대해 설명해주세요?"
result = vector_store.similarity_search(query)
print(f"검색 결과의 길이: {len(result)}") #검색 결과의 길이: 4


### 검색기 Retrievers = vector_store 검색

retrievers = vector_store.as_retriever(search_kwargs={"k":3}) #검색된 관련 문서 수 : 3
print(retrievers)
aaa = retrievers.invoke(query)
print(f"검색된 관련 문서 수 : {len(aaa)}") # 검색된 관련 문서 수 : 3
print(f"첫번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}")
# print(f"두번째 관련 문서 내용 미리보기 : {aaa[1].page_content[:50]}")
# print(f"세번째 관련 문서 내용 미리보기 : {aaa[2].page_content[:50]}")

### 모델 연결
## 
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model= 'gpt-5-nano',
    temperature=0, # 있는그대로.. temperature=1 창의적으로
    max_tokens=1000, #디폴트 제한 없음
    api_key=api_key,
    base_url=base_url,
)
# 모델이 워낙 완벽하기 때문에 #3의 컴파일, 훈련 과정은 생략 

#4. 예측
# response = model.invoke('이번 프로젝트에 참여하는 컨소시엄 기업은?') # invoke = predict와 같은 역할
# print('model 답변 : ', response.content)


query_with_context = f"""
{aaa[0].page_content}\n\n
위 내용에 근거하여 다음 질문에 답변하세요. \n\n{query}
"""
response = model.invoke(query_with_context)
print("model의 응답 :", response.content)
