# tps : Tokens Per Second
# ttfc : Time To First Chunk - 첫 청크가 나오기 직전 시간 
# TTFT : Time to First Token - 토큰이 나가기 직전 시간

from kingkeras import get_llm
import time

llm = get_llm(max_tokens=1000)

prompt = "npu의 장점을 자세히 설명해줘"

start = time.perf_counter()

chunk = next(iter(llm.stream(prompt)))
# 1번째 청크의 넥스트가 나온다

first_time = time.perf_counter() - start

print("============================================")    
print("첫 컨텐츠 응답시간 :", round(first_time,2), "초")
# 첫 컨텐츠 응답시간 : 0.03 초






