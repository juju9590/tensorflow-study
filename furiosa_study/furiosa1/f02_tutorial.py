from furiosa_llm import LLM, SamplingParams #퓨리오사 제공
# pip install furiosa-llm 

with LLM("furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic") as llm:
    # 가중치의 갯수가 8B(8빌리언)만큼 있다. FP8=8bit=1byte ==> 8B는 8GB와 같다(8B는 ‘개수’, 8GB는 ‘용량)

    # 8B: 파라미터(가중치)가 약 80억 개
    # FP8: 값 하나를 8비트(1바이트)로 표현
    # 가중치를 모두 FP8로 저장하면 가중치만 약 8GB
    # 실제 추론에는 KV 캐시·중간 계산용 메모리가 추가로 필요

    pramas = SamplingParams(max_tokens=1000, top_p=0.3, top_k=100)
    message = [{"role": "user","content":"오늘 엔비디아 주가가 어떻게 돼?"}]
    prompt = llm.tokenizer.apply_chat_template(message, tokenize=False)

    response = llm.generate([prompt], pramas)
print(response[0].outputs[0].text)