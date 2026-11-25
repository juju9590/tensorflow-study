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

#2. llm 생성
llm = get_llm(max_tokens=10000)

#3. 도구 등록
llm_with_tools = llm.bind_tools([add])

#4. 질문 
response = llm_with_tools.invoke(
    "237341과 31321을 add로 더해줘"
)

for call in response.tool_calls:
    result = add.invoke(call['args'])
    print("계산 결과:", result) #계산 결과: 268662







