from kingkeras import get_llm

llm = get_llm(max_tokens=500)

response = llm.invoke('npu란 무엇인가?')

print(response.content)
print("==============================")
print(response.usage_metadata) # 토큰양 조절 필요할 때 사용

 


# max_tokens = None 일때

# - **NVIDIA**: GPU(예: TENSOR CORE)를 AI 연산에 활용.  
# - **AMD**: 빅토리아 시리즈 등. 
# ### NPU vs GPU:
# - **GPU**는 일반적인 그래픽 처리와 병렬 연산에 적합하지만, NPU는 **AI 전용 최적화**가 더 강합니다.  
# - **NPU**는 특정 AI 작업에 더 효율적이지만, **GPU**는 다양한 작업에 유연하게 활용 가능합니다.  

# NPU는 AI 기술 발전에 따라 점점 더 중요해지고 있으며, 다양한 분야에서 실시간 AI 처리를 가능하게 합니다.
# ==============================
# {'input_tokens': 15, 'output_tokens': 725, 'total_tokens': 740, 
# 'input_token_details': {'audio': 0, 'cache_read': 14}, 
# 'output_token_details': {'audio': 0, 'reasoning': 0}}


# max_tokens = 500 일때
# 1. **AI 최적화**:  
#    - NPU는 신경망의 연산(예: 행렬 연산, 활성화 함수, 정규화)을 특화하여, AI 모델의 처리 속도를 높이고 에너지 소비를 줄입니다.
#    - 특히 **추론**(AI 모델이 입력 데이터를 처리하여 결과를 예측하는 과정)에 강점이 있습니다.

# 2. **GPU와의 차이**:  
#    - GPU는 일반적으로 병렬 처리를 위한 일반적인 그래픽 처리를 위한 하드웨어로, AI 작업에도 사용되지만, NPU
# ==============================
# {'input_tokens': 15, 'output_tokens': 500, 'total_tokens': 515, 
# 'input_token_details': {'audio': 0, 'cache_read': 14}, 
# 'output_token_details': {'audio': 0, 'reasoning': 0}}
