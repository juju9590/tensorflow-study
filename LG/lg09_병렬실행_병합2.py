# 08-2 카피

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
    max_results=5  # 검색결과 5가지 정도 가져오겠다
)

tools = [web_search] # 2개이상 검색엔진들어갈 수 있어 리스트로 묶는다

llm_with_tools = llm.bind_tools(tools)

from typing import Annotated,TypedDict  # state에 필요
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages #reducer
from langgraph.prebuilt import ToolNode, tools_condition

# (메모리 추가 첫번째)
from langgraph.checkpoint.memory import InMemorySaver

##### 1. State 정의 #####
class State(TypedDict):
    messages : Annotated[list, add_messages]
    search_result : str
    analysis_result : str

##### 2. 노드 정의 #####
# 검색 에이전트
def search_agent(state : State) :
    print("검색 에이전트 실행")
    question = state["messages"][-1].content

    result = web_search.invoke({
        "query" : question
    })
    return {"search_result" :  str(result)}

def analysis_agent(state : State):
    print("분석 에이전트 실행")
    question = state["messages"][-1].content

    response = llm.invoke(
        f"다음 질문을 기존 지식을 바탕으로 분석해줘.\n {question}"
    )
    return {"analysis_result" : response.content}


def merge_agent(state : State):
    result =(
        state["search_result"] +"\n\n" + state["analysis_result"]
    )
    return {"message" : [("assistant", result)]}

##### 3. 그래프 구성 #####

builder = StateGraph(State)

builder.add_node("search", search_agent)
builder.add_node("analysis", analysis_agent)
builder.add_node("merge", merge_agent)

##### 병렬 실행 #####
builder.add_edge(START, "search")
builder.add_edge(START, "analysis")

##### 병합 #####
builder.add_edge("search", "merge")
builder.add_edge("analysis", "merge")

builder.add_edge("merge", END)

# 컴파일
graph = builder.compile()

# 그래프 실행
query = '엔비다아 미래 전망은?'

response = graph.invoke(
    {"messages" : [("user", query)]}
) 
print("Assistant :" , response["messages"][-1].content)

##### 그래프 이미지 그리기 #####

import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO 

png_data = graph.get_graph().draw_mermaid_png()

img = Image.open(BytesIO(png_data))

# 파일 입출력
# with open(path + "lg07_ToolNode.png", 'wb') as file: 
#     file.write(png_data)

##### 이미지 출력 #####

plt.figure(figsize=(8,6))
plt.imshow(img)
plt.axis('Off')
plt.show()
