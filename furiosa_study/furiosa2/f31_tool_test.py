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
print('일반응답:', response.content) #일반응답: <think> Okay, the user ...
print("Tool Calls:", response.tool_calls) # 리스트 형태 
# Tool Calls: [{'name': 'add', 'args': {'aaa': 237341, 'bbb': 31321}, 
# 'id': 'call-333711efa4954890b7725e7bd46f0a57', 'type': 'tool_call'}]
print("메타 데이터:", response.response_metadata)
# 메타 데이터: {'token_usage': {'completion_tokens': 133, 'prompt_tokens': 176, 'total_tokens': 309, 
# 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 
# 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': None, 
# 'cached_tokens': 175}}, 'model_provider': 'openai', 'model_name': 'furiosa-ai/Qwen3-8B-FP8', 
# 'system_fingerprint': None, 'id': 'chat-22eaf244d7c146908172ca99a0687629', 'finish_reason': 'tool_calls', 
# 'logprobs': None}





