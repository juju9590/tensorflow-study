from kingkeras import get_llm
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

@tool
def add(a:int, b:int) -> int:
    """ 두 정수를 더합니다 """
    return a + b

@tool
def subtract(a:int, b:int) -> int:
    """ 두 정수를 뺍니다 (첫번째정수-두번째정수) """
    return a - b

@tool
def multiply(a:int, b:int) -> int:
    ''' 두 정수를 곱합니다 '''
    return a * b

@tool
def divide(a:int, b:int) -> float:
    ''' 두 정수를 나눕니다 (첫번째정수/두번째정수) '''
    return a / b

llm = get_llm(max_tokens=500,)
tools =  [add, multiply, subtract, divide]

agent = create_agent(
    model = llm, 
    tools = tools,
    system_prompt=(
        "계산이 필요하면 도구를 사용하세요."
        "도구 실행에 실패하면 오류를 설명하세요."
    )
)

qeustions = "10을 0으로 나눠줘" 

try: 
    result = agent.invoke(
        {
        "messages": [
            {"role":"user","content": qeustions}
        ]
        },
        config={"recursion_limit" : 5}
    )

    print(" ====================================== ")
    for i, msg in enumerate(result["messages"]):
        print(f"{i+1}번째 메시지")
        print('타입 :', type(msg).__name__)
        # print('내용 :', msg.content)

        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                print("도구 이름 :", call['name'])
                print("전달 인자 :", call['args'])

    print(" ====================================== ")
    print("최종답변 :", result["messages"][-1].content)
except ValueError as e:
    print("계산 오류 :", e)

except Exception as e:
    print("Agent 실행 오류 :", type(e).__name__, e)

# 트라이 전
# ZeroDivisionError: division by zero


# 트라이 후 
# 1번째 메시지
# 타입 : HumanMessage
# 2번째 메시지
# 타입 : AIMessage
#  ====================================== 
# 최종답변 : <think>
# Okay, the user is asking to divide 10 by 0. Let me think about this. Division by zero is undefined in mathematics. The functions provided include a divide function that takes two integers. But if I call divide with a=10 and b=0, it would technically be trying to perform 10/0, which isn't allowed. The function might return an error or handle it somehow. Since the user's request is invalid, I should explain that dividing by zero is impossible and mention it's undefined. I don't need to call any function here because it's not a valid operation. So I'll just state the error reason directly.
# </think>

# 10을 0으로 나눌 수 없습니다. 
# 0으로 나누는 것은 수학적으로 정의되지 않은 операция입니다.
