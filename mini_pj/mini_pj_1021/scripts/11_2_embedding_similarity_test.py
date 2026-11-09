
# ============================================================
# 1. 라이브러리
# ============================================================
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 2. 파일 경로
# ============================================================
base_dir = Path(
    r"C:\mini_pj\data\processed\embedding_pilot_1000"
)

npy_path = base_dir / "embedding_pilot_10.npy"
csv_path = base_dir / "embedding_pilot_10.csv"


# ============================================================
# 3. 임베딩 및 메타데이터 불러오기
# ============================================================
embeddings = np.load(npy_path)
df = pd.read_csv(csv_path, encoding="utf-8-sig")

print("임베딩 shape:", embeddings.shape)
print("데이터 shape:", df.shape)

# 데이터 검증
if len(df) != len(embeddings):
    raise ValueError("청크 개수와 임베딩 개수가 다릅니다.")

required_cols = [
    "date", "assembly", "source_id",
    "chunk_id", "chunk_index", "text"
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"필수 컬럼 누락: {col}")


# ============================================================
# 4. 코사인 유사도 계산
# ============================================================
sim_matrix = cosine_similarity(embeddings)

print("\n코사인 유사도 행렬 shape:", sim_matrix.shape)
print(np.round(sim_matrix, 3))


# ============================================================
# 5. 청크 간 유사도 비교
# ============================================================
results = []

for i in range(len(df)):
    for j in range(i + 1, len(df)):

        similarity = float(sim_matrix[i, j])

        results.append({
            "chunk_id_1": df.iloc[i]["chunk_id"],
            "chunk_id_2": df.iloc[j]["chunk_id"],

            "date_1": df.iloc[i]["date"],
            "date_2": df.iloc[j]["date"],

            "assembly_1": df.iloc[i]["assembly"],
            "assembly_2": df.iloc[j]["assembly"],

            "source_id_1": df.iloc[i]["source_id"],
            "source_id_2": df.iloc[j]["source_id"],

            "chunk_index_1": df.iloc[i]["chunk_index"],
            "chunk_index_2": df.iloc[j]["chunk_index"],

            "similarity": similarity,

            "text_1": df.iloc[i]["text"],
            "text_2": df.iloc[j]["text"]
        })


# ============================================================
# 6. 유사도 높은 순서로 정렬
# ============================================================
df_result = pd.DataFrame(results)

df_result = df_result.sort_values(
    by="similarity",
    ascending=False
).reset_index(drop=True)

print("\n전체 비교 쌍:", len(df_result))


# ============================================================
# 7. 상위 10개 결과 출력
# ============================================================
top_n = 10

print("\n========== 유사도 TOP 10 ==========")

for i, row in df_result.head(top_n).iterrows():

    print(f"\n[{i + 1}위] 유사도: {row['similarity']:.4f}")

    print("청크 1:", row["chunk_id_1"])
    print("지역 1:", row["assembly_1"])
    print("날짜 1:", row["date_1"])
    print("원문 1:", str(row["text_1"])[:300])

    print("-" * 50)

    print("청크 2:", row["chunk_id_2"])
    print("지역 2:", row["assembly_2"])
    print("날짜 2:", row["date_2"])
    print("원문 2:", str(row["text_2"])[:300])

    print("=" * 70)


# ============================================================
# 8. 유사도 결과 저장
# ============================================================
output_path = base_dir / "embedding_similarity_top10.csv"

df_result.head(top_n).to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n결과 저장 완료:", output_path)



# 임베딩 shape: (10, 1024)
# 데이터 shape: (10, 14)

# 코사인 유사도 행렬 shape: (10, 10)
# [[1.    0.481 0.365 0.334 0.338 0.432 0.38  0.21  0.304 0.333]
#  [0.481 1.    0.423 0.391 0.369 0.424 0.37  0.325 0.275 0.389]
#  [0.365 0.423 1.    0.494 0.458 0.458 0.48  0.344 0.259 0.385]
#  [0.334 0.391 0.494 1.    0.41  0.347 0.423 0.453 0.293 0.413]
#  [0.338 0.369 0.458 0.41  1.    0.377 0.459 0.39  0.295 0.349]
#  [0.432 0.424 0.458 0.347 0.377 1.    0.384 0.328 0.366 0.376]
#  [0.38  0.37  0.48  0.423 0.459 0.384 1.    0.395 0.365 0.377]
#  [0.21  0.325 0.344 0.453 0.39  0.328 0.395 1.    0.146 0.262]
#  [0.304 0.275 0.259 0.293 0.295 0.366 0.365 0.146 1.    0.31 ]
#  [0.333 0.389 0.385 0.413 0.349 0.376 0.377 0.262 0.31  1.   ]]

# 전체 비교 쌍: 45
# ========== 유사도 TOP 10 ==========

# [1위] 유사도: 0.4942
# 청크 1: 2025_강원특별자치도의회_550_47
# 지역 1: 강원특별자치도의회
# 날짜 1: 20251124
# 원문 1: 이상으로 2026년도 예산안 제안설명을 마치겠습니다.
# 감사합니다.
# ○위원장
# 이영욱
# 김진규 공보담당관님 수고하셨습니다.
# 다음 전문위원의 검토보고는 효율적인 회의진행을 위하여 기배부해 드린 유인물로 갈음하고자 합니다.
# 위원님 여러분의 양해를 부탁드립니다.
# 속기사께서는 전문위원의 검토사항을 회의록에 기재해 주시기 바랍니다.
# ㆍ2026년도 강원특별자치도교육비특별회계 예산안 및 2026년도 강원특별자치도교육청 기금운용계획안 검토보고서
# 자료 요구하실 위원님 계십니까?
# 최재민 위원님.