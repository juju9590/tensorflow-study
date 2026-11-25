from kingkeras import get_llm

llm = get_llm(max_tokens=1000, temperature=None, top_p=None)
# get_llm(max_tokens=None, temperature=None, top_p=None): ==> 모델기준 디폴트

promft ="npu와 gpu의 차이를 설명해줘"
response = llm.invoke(promft)

print(response.content)



