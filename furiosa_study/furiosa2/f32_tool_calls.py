'''
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 \
    --devices npu:5 \
    --host 127.0.0.1 \
    --port 8000 \
    --enable-overlap-scheduling \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

'''


from kingkeras import get_llm
from langchain_core.tools import tool

#1. 도구 정의
@tool 
def add(aaa: int, bbb: int) -> int:  #  type hint , -> int : return 값에 대한 type hint
    "두 정수를 더합니다"
    return aaa + bbb

print(add.name) # add, 함수명
print(add.description) # add 함수의 주석, '두 정수를 더합니다'
print(add.args) # {'aaa': {'title': 'Aaa', 'type': 'integer'}, 'bbb': {'title': 'Bbb', 'type': 'integer'}}

#2. llm 생성
llm = get_llm(max_tokens=10000)

#3. 도구 등록
llm_with_tools = llm.bind_tools([add])

#4. 질문 
response = llm_with_tools.invoke(
    "237341과 31321을 add로 더해줘"
)

#5. 결과확인
for call in response.tool_calls :  #response.tool_calls= 리스트 
    print('tool 이름 :', call['name'])
    print('매개 변수 :', call['args'])
    print('호출 ID :', call['id'])

# tool 이름 : add
# 매개 변수 : {'aaa': 237341, 'bbb': 31321}
# 호출 ID : call-789816700969459dbb3cab4cdbab00a2


# Gpu에서 하는걸 npu에서 할 수 있다



