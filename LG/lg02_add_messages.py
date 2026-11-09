# 클래스와 함수의 차이
# 함수는 기능만, 괄호안의 값을 받는다.
# 클래슨 기능 + 여러가지, 상속(모든기능 + 알파)받는다

# 오늘의 키워드 : Node, Edge, state, Reducer
 
from typing import Annotated, TypedDict    #파이썬에서 제공하는 클랫
from langgraph.graph import add_messages

class MyDate(TypedDict):
    messages : Annotated[list, add_messages]   
    # add_messages = Reducer(리듀서) : 메시지를 이어 붙인다
    #딕셔너리 형태 정의 ('a' : '1')

from langchain_core.messages import AIMessage, HumanMessage

msg1 = [HumanMessage(content='안녕', id='1')]
msg2 = [AIMessage(content='방가', id='2')]

results = add_messages(msg1, msg2)
print(results)

# [HumanMessage(content='안녕', additional_kwargs={}, response_metadata={}, id='1'), 
# AIMessage(content='방가', additional_kwargs={}, response_metadata={}, id='2', 
# tool_calls=[], invalid_tool_calls=[])]


#######################################################

msg3 = [HumanMessage(content='안녕', id='3')]
msg4 = [AIMessage(content='방가', id='3')]

results2 = add_messages(msg3, msg4)
print(results2)

# ID가 동일하면 덮어 쓴다 (통상적으로 동일한 아이디는 없다)
# [AIMessage(content='방가', additional_kwargs={}, response_metadata={}, id='3', 
# tool_calls=[], invalid_tool_calls=[])]




