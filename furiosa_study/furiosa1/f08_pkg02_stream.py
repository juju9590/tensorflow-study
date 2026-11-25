from furiosa1.a_kingkeras import get_llm 
# 내가 만든 파이썬 패키지는 같은 폴더에 있어야 한다.

llm = get_llm()

for chunk in llm.stream("npu와 gpu의 차이점을 설명해줘"):
    print(chunk.content, end="", flush=True)