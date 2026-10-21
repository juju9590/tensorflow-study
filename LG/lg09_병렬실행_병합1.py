from typing import TypedDict
from langgraph.graph import StateGraph, START, END

#1. State 정으; 
class State(TypedDict):
    question : str
    search_result : str
    analysis_result : str
    final_result :str

#2. 검색노드 
def search_agent(state : State):
    print("검색 에이전트 실행")
    return {"search_result" : "엔비디아는 AI GPU 시장의 주요 기업이다."}


#3. 분석 노드
def analysis_agent(state : State):
    print("분석 에이전트 실행")
    return {"analysis_result" : "AI 반도체  수요 증가가 성장 요인이다."}

#4. 결과 통합(merge) 노드
def merge_agent(state : State):
    result =(
        state["search_result"] + "\n" + state["analysis_result"]
    )
    return {"final_result" : result}


#5. 그래프 구성
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

# 실행
response = graph.invoke({
    "question" : "엔비디아 전망"
})

print(response["final_result"])

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

plt.figure(figsize=(6,4))
plt.imshow(img)
plt.axis('Off')
plt.show()





