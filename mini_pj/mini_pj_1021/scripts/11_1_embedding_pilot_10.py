
# ============================================================
# 1. 라이브러리
# ============================================================
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# 2. GPU 확인
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

print("PyTorch 버전:", torch.__version__)
print("CUDA 사용 가능:", torch.cuda.is_available())
print("사용 장치:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. 임베딩 모델 로드
# ============================================================
model_name = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"

model = SentenceTransformer(
    model_name,
    device=device
)

print("모델 로드 완료")
print("임베딩 차원:", model.get_embedding_dimension())


# ============================================================
# 4. CSV 파일 불러오기
# ============================================================
csv_path = Path(
    r"C:\mini_pj\data\processed\embedding_pilot_1000\embedding_sample_1000.csv"
)

df = pd.read_csv(csv_path, encoding="utf-8-sig")

print("\n전체 데이터:", df.shape)
print("컬럼명:", df.columns.tolist())


# ============================================================
# 5. 테스트 데이터 10개 추출
# ============================================================
required_cols = ["chunk_id", "chunk_index", "text"]

# 필수 컬럼 확인
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"컬럼이 없습니다: {col}")

# 빈 텍스트 제거
df = df.dropna(subset=["text"]).copy()
df = df[df["text"].astype(str).str.strip() != ""]

# 10개 선택
df_test = df.head(10).copy()

texts = df_test["text"].astype(str).tolist()

print("테스트 청크 개수:", len(texts))


# ============================================================
# 6. 임베딩 실행
# ============================================================
embeddings = model.encode(
    texts,
    batch_size=10,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  ### 임베딩 벡터의 길이를 1로 정규화, 코사인 유사도 비교에 활용하기 편리
)


# ============================================================
# 7. 결과 확인
# ============================================================
print("\n========== 임베딩 결과 ==========")

print("임베딩 shape:", embeddings.shape)
print("첫 번째 chunk_id:", df_test.iloc[0]["chunk_id"])
print("첫 번째 청크:", texts[0][:200])
print("첫 번째 벡터 일부:", embeddings[0][:10])


# ============================================================
# 8. 임베딩 결과 저장
# ============================================================
output_dir = csv_path.parent
output_dir.mkdir(parents=True, exist_ok=True)

# 8-1. 벡터 저장
npy_path = output_dir / "embedding_pilot_10.npy"
np.save(npy_path, embeddings)

# 8-2. 원본 청크 정보 저장
csv_output = output_dir / "embedding_pilot_10.csv"

df_test.to_csv(
    csv_output,
    index=False,
    encoding="utf-8-sig"
)

print("\n========== 저장 완료 ==========")
print("벡터 파일:", npy_path)
print("청크 정보 파일:", csv_output)

# 두 파일은 동일한 행 순서로 저장돼. 
# CSV는 원문과 식별 정보를 보존하고, 
# NPY는 실제 임베딩 벡터를 저장하는 역할이야.
