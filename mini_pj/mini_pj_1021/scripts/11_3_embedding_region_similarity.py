
# ============================================================
# 1. 라이브러리
# ============================================================
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 2. 경로 설정
# ============================================================
base_dir = Path(
    r"C:\mini_pj\data\processed\embedding_pilot_1000"
)

csv_path = base_dir / "embedding_sample_1000.csv"


# ============================================================
# 3. GPU 확인
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

print("사용 장치:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 4. CSV 불러오기
# ============================================================
df = pd.read_csv(csv_path, encoding="utf-8-sig")

required_cols = [
    "assembly", "date", "source_id",
    "chunk_id", "chunk_index", "text"
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"필수 컬럼 누락: {col}")

df = df.dropna(subset=["assembly", "text"]).copy()
df = df[df["text"].astype(str).str.strip() != ""]

print("\n전체 데이터:", df.shape)
print("전체 지역 수:", df["assembly"].nunique())


# ============================================================
# 5. 지역별 청크 3개 추출
# ============================================================
df_sample = (
    df.groupby("assembly", group_keys=False)
    .sample(n=3, random_state=42)
    .sort_values(["assembly", "date", "chunk_id"])
    .reset_index(drop=True)
)

print("\n지역별 추출 결과:")
print(df_sample["assembly"].value_counts().sort_index())

print("\n선택된 청크:", len(df_sample))

# 원본 14개 컬럼 유지
sample_path = base_dir / "embedding_region_51.csv"

df_sample.to_csv(
    sample_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 6. 임베딩 모델 로드
# ============================================================
model_name = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"

model = SentenceTransformer(
    model_name,
    device=device
)

print("\n모델 로드 완료")
print("임베딩 차원:", model.get_embedding_dimension())


# ============================================================
# 7. 임베딩 실행
# ============================================================
texts = df_sample["text"].astype(str).tolist()

embeddings = model.encode(
    texts,
    batch_size=8,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("\n임베딩 shape:", embeddings.shape)


# ============================================================
# 8. 코사인 유사도 계산
# ============================================================
sim_matrix = cosine_similarity(embeddings)

print("유사도 행렬 shape:", sim_matrix.shape)


# ============================================================
# 9. 서로 다른 지역 간 유사도 비교
# ============================================================
results = []

for i in range(len(df_sample)):
    for j in range(i + 1, len(df_sample)):

        row1 = df_sample.iloc[i]
        row2 = df_sample.iloc[j]

        # 같은 지역은 비교 대상에서 제외
        if row1["assembly"] == row2["assembly"]:
            continue

        similarity = float(sim_matrix[i, j])

        results.append({
            "similarity": similarity,

            "assembly_1": row1["assembly"],
            "assembly_2": row2["assembly"],

            "date_1": row1["date"],
            "date_2": row2["date"],

            "source_id_1": row1["source_id"],
            "source_id_2": row2["source_id"],

            "chunk_id_1": row1["chunk_id"],
            "chunk_id_2": row2["chunk_id"],

            "chunk_index_1": row1["chunk_index"],
            "chunk_index_2": row2["chunk_index"],

            "text_1": row1["text"],
            "text_2": row2["text"]
        })


# ============================================================
# 10. 유사도 순위 정렬
# ============================================================
df_result = pd.DataFrame(results)

df_result = df_result.sort_values(
    "similarity",
    ascending=False
).reset_index(drop=True)

print("\n지역 간 비교 쌍:", len(df_result))


# ============================================================
# 11. 유사도 TOP 20 출력
# ============================================================
top_n = 20

print("\n========== 지역 간 유사도 TOP 20 ==========")

for i, row in df_result.head(top_n).iterrows():

    print(f"\n[{i+1}위] 유사도: {row['similarity']:.4f}")

    print("지역 1:", row["assembly_1"])
    print("날짜 1:", row["date_1"])
    print("청크 ID 1:", row["chunk_id_1"])
    print("원문 1:", str(row["text_1"])[:300])

    print("-" * 50)

    print("지역 2:", row["assembly_2"])
    print("날짜 2:", row["date_2"])
    print("청크 ID 2:", row["chunk_id_2"])
    print("원문 2:", str(row["text_2"])[:300])

    print("=" * 70)


# ============================================================
# 12. 결과 저장
# ============================================================
# 임베딩 벡터 저장
np.save(
    base_dir / "embedding_region_51.npy",
    embeddings
)

# 지역 간 유사도 TOP 20 저장
output_path = base_dir / "embedding_region_similarity_top20.csv"

df_result.head(top_n).to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n========== 저장 완료 ==========")
print("표본 파일:", sample_path)
print("임베딩 파일: embedding_region_51.npy")
print("유사도 결과:", output_path)
