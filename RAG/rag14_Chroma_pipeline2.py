# 13-1 copy

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

query = "엔비디아의 창업자는 누구인가요?"
# result = vector_store.similarity_search(query)
# print(f"검색 결과의 길이: {len(result)}") #검색 결과의 길이: 4


### 검색기 Retrievers = vector_store 검색
retriever = vector_store.as_retriever(search_kwargs={"k":2}) #검색된 관련 문서 수 : 3
print(retriever)
# aaa = retriever.invoke(query)
# print(f"검색된 관련 문서 수 : {len(aaa)}") # 검색된 관련 문서 수 : 3
# print(f"첫번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}")
# print(f"두번째 관련 문서 내용 미리보기 : {aaa[1].page_content[:50]}")
# print(f"세번째 관련 문서 내용 미리보기 : {aaa[2].page_content[:50]}")

### 모델 연결
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model= 'gpt-5.6-terra', #luna < terra < sol
    temperature=0, # 있는그대로.. temperature=1 창의적으로
    max_tokens=1000, #디폴트 제한 없음
    api_key=api_key,
    base_url=base_url,
)
# 모델이 워낙 완벽하기 때문에 #3의 컴파일, 훈련 과정은 생략 

#4. 예측
# response = model.invoke('이번 프로젝트에 참여하는 컨소시엄 기업은?') # invoke = predict와 같은 역할
# print('model 답변 : ', response.content)


# query_with_context = f"""
# {aaa[0].page_content}\n\n
# 위 내용에 근거하여 다음 질문에 답변하세요. \n\n{query}
# """
# response = model.invoke(query_with_context)
# print("model의 응답 :", response.content)

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain #많이 씀
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 컨텍스트 관련 정보가 없다면,
"주어진 정보로는 답변할 수 없습니다." 라고 말씀해 주세요

컨텍스트 : {context}
질문 : {input}
답변 : 
""")

# 체인 만들기 (둘은 "세트"다)
docu_chain = create_stuff_documents_chain(model, prompt) # ===> prompt | model
rag_chain = create_retrieval_chain(retriever, docu_chain) # ===> 검색 | docu_chain

# 체인 실행
query = '엔비디아의 창업자는 누구인가요?'
response = rag_chain.invoke({"input" : query})

print(response)
print("keys : ", response.keys()) #dict_keys(['input', 'context', 'answer'])

print("==========================================================")

print("input : ", response['input'])
# input :  엔비디아의 창업자는 누구인가요?
print("context : ", response['context'][0].page_content)
# context :  엔비디아 사업 전망
# 엔비디아는 그래픽처리장치와 이를 활용하는 소프트웨어 및 시스템을 공급하는 기술기업이다. 
# 과거에는 게임용 그래픽카드로 널리 알려졌지만, 현재 사업 전망을 설명할 때는 인공지능 
# 데이터센터용 가속기와 소프트웨어 생태계를 함께 살펴봐야 한다. 대규모 인공지능 모델을 개발하고 운영하는 기업이 늘면서 고성능 연산 장비에 대한 수요가 성장 동력으로 주목받고 있다.
print("answer :" , response['answer'])
# answer : 주어진 정보로는 답변할 수 없습니다.


