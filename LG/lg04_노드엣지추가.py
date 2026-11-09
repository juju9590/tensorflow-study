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
from langchain_core.messages import AIMessage, HumanMessage

##### 1. State 정의 #####
class State(TypedDict):
    messages : Annotated[list, add_messages]  # Annotated 반드시 써주면 된다(문법)
    aaa : str
    bbb : str
    ccc : int 


##### 2. Node 정의 #####
def prepare(state : State):
    return{"messages" : 
           [AIMessage(content="질문을 접수했습니다.")]
           }

def chatbot(state : State) : # State = 타입힌트 
    return{"messages" : [llm.invoke(state["messages"])]}

def finish(state : State):
    return{"messages" : [AIMessage(content="답변 처리가 완료되었습니다.")]}


##### 3. 그래프 구성 #####
graph_01 = StateGraph(State)  # 랭그래프를 준비한다

graph_01.add_node("prepare", prepare) 
graph_01.add_node("chatbot", chatbot)  
graph_01.add_node("finish", finish)  

##### 4. 엣지 추가 #####
graph_01.add_edge(START, "prepare") 
graph_01.add_edge("prepare", "chatbot") 
graph_01.add_edge("chatbot", "finish") 
graph_01.add_edge("finish", END)

##### 5. 그래프 컴파일 #####
graph = graph_01.compile()

##### 6. 그래프 실행 #####
query = '삼성전자의 주가를 예측해줘'

response = graph.invoke({
    "messages" : [("user", query)]
}) 

# 리스트 = 이터레이터 = For문 = enumerate 까지 연상해야 한다

##### 7. 결과 출력 #####
for i, msg in enumerate(response['messages']):
    # enumerate : 인덱스 순서와 메세지가 순차적으로 나오게 하라
    print(f"[{i}] {msg.type} : {msg.content}")



# [0] human : 삼성전자의 주가를 예측해줘
# [1] ai : 질문을 접수했습니다.
# [2] ai : 삼성전자 주가를 **정확한 숫자로 예측하는 것은 불가능**하며, 아래는 투자 권유가 아닌 시나리오 분석입니다. 또한 저는 실시간 주가·실적·뉴스를 직접 확인할 수 없으므로, 현재 가격 기준으로 판단하는 것이 필요합니다.

# ### 주가에 가장 큰 영향을 줄 변수
# 1. **메모리 반도체 업황**
#    - DRAM·NAND 가격 상승과 재고 정상화는 실적과 주가에 긍정적입니다.
#    - 반대로 공급 과잉, PC·스마트폰 수요 둔화가 나타나면 하방 압력이 커질 수 있습니다.

# 2. **HBM·AI 반도체 경쟁력**
#    - 엔비디아 등 AI 생태계에 HBM 공급 확대가 확인되면 강한 재평가 요인이 될 수 있습니다.
#    - HBM 인증·수율·고객사 확보가 기대에 못 미치면 주가 변동성이 커질 수 있습니다.

# 3. **파운드리 사업의 수익성**
#    - 2nm 등 첨단 공정 수율 개선, 대형 고객 확보는 중장기 상승 요인입니다.
#    - 반대로 TSMC와의 격차 확대 및 적자 지속은 할인 요인입니다.

# 4. **스마트폰·가전 실적**
#    - 갤럭시 프리미엄 판매, AI폰 교체 수요, 원가 절감 여부가 중요합니다.
#    - 다만 삼성전자 전체 주가의 핵심은 여전히 반도체 사업입니다.

# 5. **환율·금리·미국 기술주 흐름**
#    - 원화 약세는 수출기업인 삼성전자 실적에 대체로 우호적일 수 있습니다.
#    - 미국 금리 상승, 글로벌 기술주 조정, 미·중 규제 강화는 단기 리스크입니다.

# ---

# ## 시나리오별 전망

# | 시나리오 | 전제 | 주가 방향 |
# |---|---|---|
# | 낙관적 | HBM 공급 확대, 메모리 가격 강세 지속, 파운드리 수율 개선 | 중장기 상승 가능성 확대 |
# | 기준 | 메모리 업황은 회복하지만 HBM·파운드리 성과는 기대 수준 | 실적 발표에 따른 박스권·완만한 상승 |
# | 비관적 | 메모리 가격 하락 전환, HBM 경쟁력 우려, 글로벌 경기 둔화 | 조정 가능성 확대 |

# ### 개인적으로 볼 핵심 체크포인트
# - 분기 실적에서 **반도체(DS) 영업이익**이 계속 개선되는지  
# - HBM 매출 및 주요 고객사 공급 관련 공식 발표  
# - DRAM/NAND 계약가격 추세  
# - 파운드리 수율·수주 관련 발표  
# - 자사주 소각, 배당 확대 등 주주환원 정책  

# ### 결론
# 삼성전자는 단기적으로는 반도체 업황과 HBM 뉴스에 크게 흔들릴 수 있지만, 중장기적으로는 **AI 메모리 시장에서의 경쟁력 회복 여부**가 가장 중요한 변수입니다.  
# 따라서 “얼마까지 오른다”보다, **HBM 성과와 반도체 영업이익의 추세가 실제로 개선되는지**를 확인하면서 분할 접근하는 방식이 상대적으로 합리적입니다.
# [3] ai : 답변 처리가 완료되었습니다..












