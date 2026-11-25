# copy f05 

# from openai import OpenAI
from langchain_openai import ChatOpenAI

# client = OpenAI(
#     base_url = "http://127.0.0.1:8000/v1",    #퓨리오사 서버에 접속을 했고
#     api_key="EMPTY",
# )

# # 서버에 로드된 모델 확인
# model = client.models.list().data[0].id


llm = ChatOpenAI(
    base_url = "http://127.0.0.1:8000/v1",    #퓨리오사 서버에 접속을 했고
    api_key="EMPTY",
    max_tokens=1000,
    temperature=0,
    top_p=0.9, 
)

# # 추론
# streaming = client.chat.completions.create(
#     model = model,
#     messages=[
#         {
#             "role" : "user",
#             "content" : "npu와 gpu의 차이점을 설명해줘"
#         }
#     ],
#     max_tokens=1000,
#     stream=True,
#     # stream=False: 10초 동안 기다렸다가 완성된 답변을 한 번에 출력
#     # stream=True: 첫 토큰이 준비되면 바로 보여주고, 이후 생성되는 내용도 순차적으로 출력. 전체 완료는 약 10초
# )

# for chunk in streaming:
#     if chunk.choices:
#         content = chunk.choices[0].delta.content
#         if content:
#             print(content, end="", flush=True)

for chunk in llm.stream("npu와 gpu의 차이점을 설명해줘"):
    print(chunk.content, end="", flush=True)








