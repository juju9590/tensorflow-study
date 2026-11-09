# 임베딩을 위한 환경 사전준비


import torch

print("PyTorch 버전:", torch.__version__)
print("CUDA 사용 가능:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("CPU 환경")


# 1. PC 환경 확인  
# 항목	                확인 결과	          상태
# Conda 환경	        mini	             활성화
# Python	           3.11.16        	    호환
# GPU	               NVIDIA RTX 3060	    인식
# GPU 메모리	        12GB	             사용 가능
# NVIDIA 드라이버	    616.92	              인식
# 드라이버 CUDA 지원	 13.4	               확인
# PyTorch	            미설치	             설치 필요  
# 

# 1. GPU용 PyTorch 설치
# python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128  
# CUDA 12.8을 지원하는 PyTorch를 설치. PyTorch 공식 안내에서 제공하는 버전 조합 기준 

# 2. 설치 후 GPU 확인
# PyTorch 버전: 2.8.0+cu128
# CUDA 사용 가능: True
# GPU: NVIDIA GeForce RTX 3060

# 3. 다음에 사용할 임베딩 모델

# python -m pip install sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "dragonkue/snowflake-arctic-embed-l-v2.0-ko",
    device="cuda"
)

print("모델 로드 완료")
print("임베딩 차원:", model.get_sentence_embedding_dimension())