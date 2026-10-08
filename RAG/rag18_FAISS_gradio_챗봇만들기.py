# 17-1 copy

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
# Chroma : 코사인 유사도 기반 (통상적으로)

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain #많이 씀
from langchain_classic.chains import create_retrieval_chain

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

db = FAISS.load_local(
    folder_path=DB_path,
    index_name='faiss_index17',
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

# 검색기 (Retriever : 벡터DB 땡겨오기)
retriever = db.as_retriever(search_kwarge={"k":2})

# 모델 연결
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model= 'gpt-5.6-terra', 
    temperature=0, 
    max_tokens=1000, 
    api_key=api_key,
    base_url=base_url,
)

prompt = ChatPromptTemplate.from_template("""
"당신은 [챗봇 개발 전문가]의 역할을 맡게 될 것입니다.

 "고양이 애호가 독자를 위해 재미있는 이야기를 포함하고, 질문으로 끝내고, 
토리라는 고양이를 키우고 있으며, 방문자에게 고양이 다운 이모티콘과 행동을
대화로 풀어주세요"

"고양이 토리는 사람들과 대화시 유머러스 하면서 짧고 간결하게 대답합니다. 
또한 친근함이 느껴질 수 있게 말어미에 '냥' 또는 '옹'을 붙여서 말만 들어도 
토리는 고양이라고 알 수 있도록 이야기 합니다."

" 항상 욕설은 삼가해주세요"
" 대답은 300자 넘지 않게 하세요"

컨텍스트 : {context}
질문 : {input}
답변 : 
""")

# 체인 만들기 (둘은 "세트"다)
docu_chain = create_stuff_documents_chain(model, prompt) # ===> prompt | model
rag_chain = create_retrieval_chain(retriever, docu_chain) # ===> 검색 | docu_chain

### Gradio 챗봇 
import gradio as gr

def answer_invoke(message, history) : 
    response = rag_chain.invoke({"input" : message})
    return response['answer']

### Gradio  인터페이스 만들자
demo =gr.ChatInterface(fn=answer_invoke, title='tori_bot')

### Gradio 실행
demo.launch() 

exit()

print("============================================")
# 문서 저장소 ID 확인
print("문서 저장소 ID 확인: ", db.index_to_docstore_id)
# 저장된 결과 확인
print("저장된 결과 확인: ",db.docstore._dict)
# 유사도 검색
aaa  = db.similarity_search("삼성전자 창업주에 대해 알려줘", k=2)
print(aaa)

