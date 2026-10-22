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
    messages : Annotated[list, add_messages] # add_messages=> Reducer -> 계속 이어 붙인다.
    search_result : str
    analysis_result : str

##### 2. 노드 정의 #####
# 검색 에이전트
def search_agent(state : State) : #State 라는 딕셔너리(TypedDict)를 받아들여라, 타입힌트 : 강제성은 없지만 이걸 써달라고 요구하는거
    print("검색 에이전트 실행")
    question = state["messages"][-1].content #딕셔너리(TypedDict)의 메세지의 가장 마지막을 가져와라

    result = web_search.invoke({  #웹 검색 도구(TavilySearch)
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


def merge_agent(state : State):  ### 결과 통합 에이전트
    # result =(
    #     state["search_result"] +"\n\n" + state["analysis_result"]
    # )
    # return {"messages" : [("assistant", result)]}

##### 머지에 프롬프트화

    print("결과 통합 에이전트 실행")

    prompt = f"""
    다음 두 결과를 종합하여 최종 답변을 작성해 주세요.

    [웹 검색 결과]
    {state['search_result']}  ## 딕셔너리의 키값

    [AI 분석 결과]
    {state['analysis_result']}

    검색 결과를 우선적으로 참고하고
    최신 정보와 기존 분석을 구분해서 설명해줘.

    """

    response = llm.invoke(prompt)

    return {"messages" : [response]}




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
