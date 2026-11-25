
import time
from langchain_openai import ChatOpenAI

'''
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 \
    --devices npu:5 \
    --host 127.0.0.1 \
    --port 8000 \
    --enable-overlap-scheduling \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

'''

def get_llm(max_tokens=None, temperature=None, top_p=None):

    llm = ChatOpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return llm

def run_once(prompt, max_tokens=1000,):
    llm = get_llm(max_tokens)

    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start

    usage = response.usage_metadata or {} #프롬프트가 없다면 빈 딕셔너리 들어감

    return{
        "elapsed" : elapsed,
        # "input_tokens" : usage['input_tokens'],
        "input_tokens" : usage.get("input_tokens",0), #input_tokens 있으면 input_tokens, 없으면 0
        "output_tokens" : usage.get("output_tokens",0),
        "text" : response.content,
    }

import urllib.request
def get_cache_metrics():
    url = 'http://127.0.0.1:8000/metrics'

    with urllib.request.urlopen(url, timeout=5) as response:
        data = response.read().decode('utf-8')
    hits =0
    queries =0
    kv_usage = 0

    for line in data.splitlines():
        if line.startswith("#"):
            continue
        if "furiosa_llm_prefix_cache_hits{" in line:
            hits += float(line.split()[-1])

        elif "furiosa_llm_prefix_cache_queries{" in line:
            queries += float(line.split()[-1])

        elif "furiosa_llm_kv_cache_usage_percent{" in line:
            kv_usage = float(line.split()[-1])*100

    hit_rate = hits / queries * 100 if queries > 0 else 0

    print("\n====== furiosa NPU Cache 상태 =======")
    print(f'Prefix Cache 조회 횟수 : {queries:.0f}') # float 0까지
    print(f'Prefix Cache 적중 횟수 : {hits:.0f}')
    print(f'Prefix Cache 적중률 : {hit_rate:.2f}') # 소수 둘째자리 까지
    print(f'kv Cache 사용률 : {kv_usage:.2f}')

    return {
        "queries" : queries,
        "hits" : hits, # 몇번 맞췄니?
        "hit_rate" : hit_rate,
        "kv_usage" : kv_usage,
    }


# get_cache_metrics()

# ====== furiosa NPU Cache 상태 =======
# Prefix Cache 조회 횟수 : 7289
# Prefix Cache 적중 횟수 : 0
# Prefix Cache 적중률 : 0.00
# kv Cache 사용률 : 1.09



