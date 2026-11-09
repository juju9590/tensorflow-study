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
# print("AI :", response['messages'][-1].content) #[-1] 가장 마지막 메시지

##### 그래프 이미지 저장 #####
path = './_save/image/'

png_data = graph.get_graph().draw_mermaid_png()


# 파일 입출력
with open(path + "lg07_ToolNode.png", 'wb') as file: 
    file.write(png_data)

# with open(path + "lg07_ToolNode.png", 'wb') as f: # as 뒤 변수명
#     f.write(png_data)

# with문과 동일
# file = open(path + "lg07_ToolNode1.png", 'wb') # 파일이 만들어짐
# file.write(png_data)
# file.close()



