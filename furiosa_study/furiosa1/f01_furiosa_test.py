'''
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8
furiosa-llm serve furiosa-ai/gpt-oss-20b
furiosa-llm serve furiosa-ai/Llama-3.1-8B-Instruct (x)
furiosa-llm serve furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic 
furiosa-llm serve furiosa-ai/Qwen3.5-0.8B (x) 
furiosa-llm serve furiosa-ai/Qwen3-4B-FP8 

furiosa-llm serve furiosa-ai/Qwen3-4B-FP8 
    --device npu:5 \
    --host 0.0.0.0 \
    --port 8000


# 질문이 들어올 때마다 모델을 다시 읽으면 오래 걸리니까 계속 올려놓는 거임
'''

from openai import OpenAI

client = OpenAI(
    base_url = "http://127.0.0.1:8000/v1",    
    api_key="EMPTY",

)
# 원래 리퀘스트/리스판스를 써야 하나, 코드 간결을 위해 오픈ai로 설정

# 서버에 로드된 모델 확인
models = client.models.list()
model_id = models.data[0].id

print("모델 :", model_id)

#추론
reponse = client.chat.completions.create(
    model = model_id,
    messages=[
        {
            "role" : "user",
            # "content" : "npu와 gpu의 차이점을 설명해줘"
            "content" : "26년 11월 유행하는 캐릭터 3위만 이야기 해줘?"
        }
    ],
    max_tokens=1500
)

print(reponse.choices[0].message.content)



# 허깅페이스에서 캐쉬로 댕겨온 모델로 작업한거.. 아직 컴파일 안됨 