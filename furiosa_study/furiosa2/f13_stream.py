from kingkeras import get_llm

llm = get_llm(max_tokens=500)

prompt = '인공지능의 발전 과정을 설명해줘'

for chunk in llm.stream(prompt):
# invoke 아니면 stream

    # print(chunk.content) #chunk 단위로 출력
    print(chunk.content, end="", flush=True)

# | `end=""` | 출력 후 줄을 바꾸지 않고 이어 붙임 |
# | `flush=True` | 출력 버퍼에 기다리게 두지 않고 바로 출력 |