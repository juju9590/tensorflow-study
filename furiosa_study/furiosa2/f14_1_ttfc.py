# tps : Tokens Per Second
# ttfc : Time To First Chunk
# TTFT : Time to First Token

from kingkeras import get_llm
import time

llm = get_llm()

prompt = "npu의 장점을 자세히 설명해줘"

start = time.perf_counter()
first_time = None

for chunk in llm.stream(prompt):

    if chunk.content: #청크 콘텐트가 있으면 
        if first_time is None: #퍼스트 타임이 none 이면
            first_time = time.perf_counter()-start
        print(chunk.content, end="", flush=True)

print("============================================")    
print("첫 컨텐츠 응답시간 :", round(first_time,2), "초")




