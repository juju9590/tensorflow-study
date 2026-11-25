from kingkeras import get_llm
from langchain_core.tools import tool
from langchain.agents import create_agent

@tool  #데코레이터 : 함수를 불러오는 함수
def add(a:int, b:int) -> int:
    """ 두 정수를 더합니다 """
    return a + b

@tool
def multiply(a:int, b:int) -> int: # 타입힌트 
    ''' 두 정수를 곱합니다 '''
    return a * b

llm = get_llm(max_tokens=500,)

agent = create_agent(
    model = llm, 
    tools =  [add, multiply],
    system_prompt="계산이 필요하면 도구를 사용하세요."
    # 에이전트한테 역할과 행동지침을 설정
)

result = agent.invoke({
    "messages": [
        {"role":"user","content": "33과  7을 곱한 후  4를 더해줘."}
    ]
})

# print(result)

print(" ====================================== ")
for i, msg in enumerate(result["messages"]):
    print(f"{i+1}번째 메시지")
    print('타입 :', type(msg).__name__)
    print('내용 :', msg.content)

print(" ====================================== ")
print("최종답변 :", result["messages"][-1].content)
