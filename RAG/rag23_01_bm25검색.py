#pip install rank_bm25

from langchain_core.documents import Document

texts = [
    Document(page_content="삼성전자는 반도체를 생산하는 기업이다."),
    Document(page_content="삼성전자는 갤럭시 스마트폰을 판매한다."),
    Document(page_content="SK하이닉스는 메모리 반도체를 생산한다."),
    Document(page_content="엔비디아는 인공지능 연산용 GPU를 설계한다."),
    Document(page_content="애플은 아이폰을 개발하는 기업이다."),
    Document(page_content="테슬라는 전기자동차를 제조한다."),
    Document(page_content="삼성전자는 AI 반도체 기술을 연구한다."),
    Document(page_content="반도체는 컴퓨터의 핵심 부품이다."),
]

query = "삼성전자의 반도체 사업"

###### BM25 검색 ########
# BM25(best Match 25)는  키워드 기반 검색 
# ㄴ 유사도 검사와 다르게 단어 빈도 및 어절 단위로 찾아 줌... 
# 유사도 속도는 느림, 일치 및 빈도수만 찾아주기 때문에 상대적으로 속도 빠름
# 이건 대부분 앙상블로 사용

from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 3  # 문장 3개 뽑아준다

bm25_docs = bm25_retriever.invoke(query)

print("=============== BM25 검색 결과 ====================")

for doc in bm25_docs:                 # bm25_docs는 리스트
    print(doc.page_content)



