import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

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
    chunk_overlap=100, # 데이터를 300으로 자를때 앞의 100은 중복하여 뒤의 300을 자른다 (중복)
    separators=["\n\n", "\n", " ", ""],  #통상 디폴트
)

# 불러온 데이터를 #02 기준으로 잘라라
split_doc1 = loader1.load_and_split(Text_splitter) #청크 300, 오버랩 100
split_doc2 = loader2.load_and_split(Text_splitter) #청크 300, 오버랩 100

# 문서 갯수 확인
'''
[Document(metadata={'source': './_data/rag_data/nvidia_outlook.txt'}, 
page_content='엔비디아 사업 전망\n\n엔비디아는 그래픽처리장치와 이를 활용하는 
소프트웨어 및 ~~~ 늘면서 고성능 연산 장비에 대한 수요가 성장 동력으로 주목받고 있다.'), 
...
Document(metadata={'source': './_data/rag_data/nvidia_outlook.txt'}, 
page_content='게임용 GPU, 전문 시각화, 자동차 분야는 데이터센터 외의 사업 기반을
제공한다. ~~아니다.')]

'''
print(split_doc1)
print(split_doc2)
print(len(split_doc1), len(split_doc2)) # 9 9
# 청킹하는 이유는 ? 벡터DB 만들기 위해 

# 임베딩 준비
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,   
    base_url=base_url,  
    # dimensions=10,   
)

DB_path ='./_db/Chroma_11/'
db = Chroma.from_documents(
    documents=split_doc1 + split_doc2,
    embedding=embeddings,
    persist_directory=DB_path, #지속가능한 디렉토리
    collection_name='chroma11',
)

print("Chroma 문서저장 끝")







