from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url='https://monogpt.kr/api/monorouter/v1'


llm = ChatOpenAI(
    model_name='gpt-5.6-terra', 
    temperature=0,   
    max_tokens=1000,               
    api_key=api_key, 
    base_url=base_url, 
)

# Q -> (R) -> M -> A
# 중요 키워드 : node, edge, state, reducer
# reducer : 리스트 덮어쓰지 않고 연결해 주는 기능

from typing import Annotated, TypedDict  
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages

##### 1. State 정의 #####
class State(TypedDict):
    messages : Annotated[list, add_messages]  # Annotated 반드시 써주면 된다(문법)
    aaa : str
    bbb : str
    ccc : int 


# 위에꺼는 한마디로 요거다.
# state ={
#     'messages' : [],
#     'aaa' : 'aaa',
#     'bbb' : 'bbb'
#     'ccc' : 123
# }

##### 2. Node 정의 #####
def chatbot(state : State) : # State = 타입힌트 
    return{"messages" : [llm.invoke(state["messages"])]}

##### 3. 그래프 구성 #####
graph_01 = StateGraph(State)  # 랭그래프를 준비한다

graph_01.add_node("chatbot", chatbot)  # 엮을 노드를 준비한것 뿐


##### 4. 엣지 추가 #####
graph_01.add_edge(START, "chatbot") # 시작해서 챗봇을 돌린다

graph_01.add_edge("chatbot", END)

##### 5. 그래프 컴파일 #####
graph = graph_01.compile()

##### 6. 그래프 실행 #####
query = '삼성전자의 주가를 예측해줘'

response = graph.invoke({
    "messages" : [("user", query)]
})

# print(response)
# print(response["messages"][0])
#content='삼성전자의 주가를 예측해줘' additional_kwargs={} response_metadata={} id='92a86bef-454f-4dc6-b019-ddf9ce9f4b4c'
# print(response["messages"][1])
# content='삼성전자(보통주 005930) 주가를 **정확히 예측하는 것은 ... **조건부 시나리오**는 다음처럼 볼 수 있습니다.
# print(response["messages"][1].content)
# | 시나리오 | 예상 주가 방향(현 주가 대비) | 핵심 조건 |
# 원하시면 현재 보유단가와 투자 기간(단기 3개월 / 중기 1년 / 장기 3년 이상)을 기준으로 **매수·보유·매도 시나리오**까지 구체적으로 정리해드릴 수 있습니다.
print(response["messages"][0].type) # human
print(response["messages"][1].type) # ai












