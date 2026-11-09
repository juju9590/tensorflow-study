# Tool calling : 외부 기능을 가져와서 사용하기!!!


from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url='https://monogpt.kr/api/monorouter/v1'

# 모델 준비
llm = ChatOpenAI(
    model_name='gpt-5.6-terra', 
    temperature=0,   
    max_tokens=2000,               
    api_key=api_key, 
    base_url=base_url, 
)

##### 0. TOOL 정의 #####
# pip install langchain-tavily
from langchain_tavily import TavilySearch

# 웹 검색 도구 생성
web_search = TavilySearch(
    max_results=5  # 검색결과 3가지 정도 가져오겠다
)

# result = web_search.invoke({
#     "query" : "삼성전자의 어제 종가 알려줘"
# })

# print(result)
# print(result['results'][-1]['content'])

tools = [web_search] # 2개이상 검색엔진들어갈 수 있어 리스트로 묶는다

llm_with_tools = llm.bind_tools(tools)

from typing import Annotated,TypedDict  # state에 필요
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages #reducer
from langgraph.prebuilt import ToolNode, tools_condition

##### 1. State 정의 #####
class State(TypedDict):
    messages : Annotated[list, add_messages]

##### 2. 노드 정의 #####
def chatbot(state : State):
    return {"messages":[llm_with_tools.invoke(state['messages'])]}

##### 3. 그래프 구성, 노드 추가 #####
graph_builder = StateGraph(State)  #상속받은 State  그래프 준비

graph_builder.add_node("chatbot", chatbot)

# tool_node = ToolNode(tools)  # 노드를 하나 만들어줌
# graph_builder.add_node("tools", tool_node)

graph_builder.add_node("tools", ToolNode(tools)) #한줄로

#  그래프 모델 그림
# start - chatbot  - end
#        ㄴ tools  

##### 4. 엣지 추가 ######
# graph_builder.add_edge(START, "chatbot") 
# graph_builder.add_edge("chatbot", "tools") 
# graph_builder.add_edge("tools", END) 
# 요거는 "start - chatbot - tools  - end" 로 진행되는 엣지

graph_builder.add_edge(START, "chatbot") 


# Tool calling 도구 (키워드)
# add_conditional_edges
# tools_condition

# graph_builder.add_edge("chatbot", "tools") 
graph_builder.add_conditional_edges(        #조건부 엣지
    "chatbot", 
    tools_condition # 조건문 : 조건에 맞는 값으로 분기
    # tool_calls 가 있으면 'tools'반환
    # tool_calls 가 없으면 END 반환 (tools_condition에 END 포함)
    # 요건 chatbot model이 판단
)  

graph_builder.add_edge("tools", "chatbot") 

##### 5. 컴파일 #####
graph = graph_builder.compile()

##### 6. 그래프 실행 #####
query = '에코프로 오늘 주가 알려줘'

response = graph.invoke({
    "messages" : [("user", query)]
}) 

##### 7. 결과 출력 #####
print("AI :", response['messages'][-1].content) #[-1] 가장 마지막 메시지

# 질문 : 삼성전자 오늘 주가 알려줘
# AI : 삼성전자(005930)는 **오늘 장중 시세를 이 환경에서 신뢰성 있게 
# 실시간 확인하기 어렵습니다.**  

# 실시간 현재가는 [네이버페이 증권 삼성전자](https://finance.naver.com/item/main.naver?code=005930)에서 확인하실 수 있습니다. 
# 장중에는 가격이 계속 변동되며, 장 마감 후에는 종가로 확정됩니다.


# 질문 : 에코프로 오늘 주가 알려줘
# AI : 에코프로(086520)는 **90,400원**으로 확인됩니다.  
# 전일 대비 **+9,100원 (+11.19%)**이며, 장중 고가도 90,400원입니다. 
# 거래량은 약 **106.8만 주**입니다.  

# *검색 시점 기준이며 실시간 시세는 변동될 수 있습니다.*  
# [매일경제 시세 보기](https://stock.mk.co.kr/price/home/KR7086520004)






