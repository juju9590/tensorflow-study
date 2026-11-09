# 원본 51개 청크 불러오기
# 최소 전처리 및 embedding_text 생성 (우선 정규표현식 기반의 최소 전처리부터 시작)
# 전처리 전후 텍스트 비교 출력
# 정제 텍스트 임베딩 및 코사인 유사도 계산
# 원본 대비 유사도·순위 변화 분석
# 결과 CSV 저장


# ============================================================
# 1. 라이브러리
# ============================================================
from pathlib import Path
import re
import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 2. 파일 경로
# ============================================================
base_dir = Path(
    r"C:\mini_pj\data\processed\embedding_pilot_1000"
)

csv_path = base_dir / "embedding_region_51.csv"
original_npy = base_dir / "embedding_region_51.npy"


# ============================================================
# 3. CSV 불러오기
# ============================================================
df = pd.read_csv(csv_path, encoding="utf-8-sig")

required_cols = [
    "assembly", "date", "source_id",
    "chunk_id", "chunk_index", "text"
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"필수 컬럼 누락: {col}")

print("전체 데이터:", df.shape)
print("지역 수:", df["assembly"].nunique())


# ============================================================
# 4. 임베딩 전처리 함수
# ============================================================
# 정책 내용과 회의 결과를 보존하기 위해
# 명확한 회의 진행 문구만 제한적으로 제거

REMOVE_PATTERNS = [
    r"위원님 여러분의 양해를 부탁드립니다\.?",
    r"다음 질의하실 위원님 계십니까\??",
    r"다음 질의하실 위원님 계시면 질의해 주시기 바랍니다\.?",
    r"이상으로 제안설명을 마치겠습니다\.?",
    r"이상으로 보고를 마치겠습니다\.?",
    r"수고하셨습니다\.?",
    r"감사합니다\.?",
]


def clean_embedding_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # 1. 줄바꿈 통일
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 2. 독립적인 발언 구분기호 제거
    # ○위원장처럼 붙어 있는 경우도 기호만 제거
    text = re.sub(r"○", " ", text)

    # 3. 반복적인 회의 진행 문구 제거
    for pattern in REMOVE_PATTERNS:
        text = re.sub(pattern, " ", text)

    # 4. 공백 및 줄바꿈 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


# ============================================================
# 5. 전처리 적용
# ============================================================
# 원본 text는 유지
df["embedding_text"] = df["text"].apply(
    clean_embedding_text
)

# 전처리 결과가 비어 있으면 원본 사용
empty_mask = df["embedding_text"].str.strip() == ""

df.loc[empty_mask, "embedding_text"] = (
    df.loc[empty_mask, "text"].fillna("")
)

print("\n전처리 완료")
print("전처리 후 데이터:", df.shape)
print("원본 대체 건수:", empty_mask.sum())


# ============================================================
# 6. 전처리 전후 텍스트 확인
# ============================================================
print("\n========== 전처리 전후 비교 ==========")

for i in range(min(3, len(df))):

    print(f"\n[{i+1}번 청크]")
    print("chunk_id:", df.iloc[i]["chunk_id"])

    print("\n[원본]")
    print(str(df.iloc[i]["text"])[:500])

    print("\n[전처리 후]")
    print(str(df.iloc[i]["embedding_text"])[:500])

    print("=" * 70)


# ============================================================
# 7. 전처리 결과 CSV 저장
# ============================================================
clean_csv_path = (
    base_dir / "embedding_region_51_cleaned.csv"
)

df.to_csv(
    clean_csv_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n전처리 CSV 저장:", clean_csv_path)


# ============================================================
# 8. GPU 확인 및 임베딩 모델 로드
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

print("\n사용 장치:", device)

model_name = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"

model = SentenceTransformer(
    model_name,
    device=device
)

print("모델 로드 완료")
print("임베딩 차원:", model.get_embedding_dimension())


# ============================================================
# 9. 전처리 후 임베딩 실행
# ============================================================
texts_clean = df["embedding_text"].astype(str).tolist()

embeddings_clean = model.encode(
    texts_clean,
    batch_size=8,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("\n전처리 후 임베딩 shape:", embeddings_clean.shape)

# 임베딩 저장
clean_npy_path = (
    base_dir / "embedding_region_51_cleaned.npy"
)

np.save(clean_npy_path, embeddings_clean)


# ============================================================
# 10. 기존 임베딩 불러오기
# ============================================================
embeddings_original = np.load(original_npy)

if embeddings_original.shape != embeddings_clean.shape:
    raise ValueError("전처리 전후 임베딩 shape가 다릅니다.")

print("원본 임베딩 shape:", embeddings_original.shape)


# ============================================================
# 11. 전처리 전후 코사인 유사도 계산
# ============================================================
sim_original = cosine_similarity(embeddings_original)
sim_clean = cosine_similarity(embeddings_clean)


# ============================================================
# 12. 지역 간 유사도 비교
# ============================================================
results = []

for i in range(len(df)):
    for j in range(i + 1, len(df)):

        row1 = df.iloc[i]
        row2 = df.iloc[j]

        # 서로 다른 지역만 비교
        if row1["assembly"] == row2["assembly"]:
            continue

        before = float(sim_original[i, j])
        after = float(sim_clean[i, j])

        results.append({
            "assembly_1": row1["assembly"],
            "assembly_2": row2["assembly"],

            "date_1": row1["date"],
            "date_2": row2["date"],

            "source_id_1": row1["source_id"],
            "source_id_2": row2["source_id"],

            "chunk_id_1": row1["chunk_id"],
            "chunk_id_2": row2["chunk_id"],

            "similarity_before": before,
            "similarity_after": after,
            "similarity_change": after - before,

            "text_1": row1["text"],
            "text_2": row2["text"],

            "embedding_text_1": row1["embedding_text"],
            "embedding_text_2": row2["embedding_text"]
        })


# ============================================================
# 13. 전처리 전후 순위 비교
# ============================================================
df_result = pd.DataFrame(results)

# 원본 유사도 순위
df_result["rank_before"] = (
    df_result["similarity_before"]
    .rank(method="first", ascending=False)
    .astype(int)
)

# 전처리 후 유사도 순위
df_result["rank_after"] = (
    df_result["similarity_after"]
    .rank(method="first", ascending=False)
    .astype(int)
)

# 양수이면 순위 상승
df_result["rank_change"] = (
    df_result["rank_before"] -
    df_result["rank_after"]
)

df_result = df_result.sort_values(
    "similarity_after",
    ascending=False
).reset_index(drop=True)

print("\n전체 지역 간 비교:", len(df_result))


# ============================================================
# 14. 전처리 후 TOP 20 출력
# ============================================================
print("\n========== 전처리 후 TOP 20 ==========")

for i, row in df_result.head(20).iterrows():

    print(f"\n[{i+1}위]")

    print("지역:", row["assembly_1"],
          "↔", row["assembly_2"])

    print("전처리 전:", round(row["similarity_before"], 4))
    print("전처리 후:", round(row["similarity_after"], 4))

    print("유사도 변화:",
          round(row["similarity_change"], 4))

    print("기존 순위:", row["rank_before"])
    print("변경 순위:", row["rank_after"])
    print("순위 변화:", row["rank_change"])


# ============================================================
# 15. 결과 저장
# ============================================================
# 전체 1,224쌍 저장
all_path = (
    base_dir / "embedding_comparison_all.csv"
)

df_result.to_csv(
    all_path,
    index=False,
    encoding="utf-8-sig"
)

# TOP 20 저장
top20_path = (
    base_dir / "embedding_comparison_top20.csv"
)

df_result.head(20).to_csv(
    top20_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n========== 저장 완료 ==========")
print("정제 CSV:", clean_csv_path)
print("정제 임베딩:", clean_npy_path)
print("전체 비교:", all_path)
print("TOP 20:", top20_path)

# 전체 데이터: (51, 14)
# 지역 수: 17

# 전처리 완료
# 전처리 후 데이터: (51, 15)
# 원본 대체 건수: 0

# 전처리 후 임베딩 shape: (51, 1024)
# 원본 임베딩 shape: (51, 1024)

# 전체 지역 간 비교: 1224
