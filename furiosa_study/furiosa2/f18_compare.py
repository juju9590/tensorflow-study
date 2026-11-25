from kingkeras import get_llm
import time

llm = get_llm(max_tokens=2000,)

prompt = [
    'npu란?',
    'npu와 gpu의 차이를 5가지 설명해줘',
    'npu의 구조와 장점, 활용 분야를 자세히 설명해줘',
]

for prompt in prompt:
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start

    input_tokens =response.usage_metadata["input_tokens"]
    outpue_tokens = response.usage_metadata["output_tokens"]

    print("질문 :", prompt)
    print("입력 토큰 :", input_tokens)
    print("출력 토큰 :", outpue_tokens)
    print("응답 시간 :", round(elapsed, 2), '초')

# 질문 : npu란?
# 입력 토큰 : 12
# 출력 토큰 : 862
# 응답 시간 : 9.89 초

# 질문 : npu와 gpu의 차이를 5가지 설명해줘
# 입력 토큰 : 23
# 출력 토큰 : 1364
# 응답 시간 : 15.8 초

# 질문 : npu의 구조와 장점, 활용 분야를 자세히 설명해줘
# 입력 토큰 : 27
# 출력 토큰 : 2000
# 응답 시간 : 23.6 초

