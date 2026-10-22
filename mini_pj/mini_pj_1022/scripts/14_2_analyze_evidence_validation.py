from pathlib import Path
import pandas as pd

# =====================================================
# 1. 경로 설정
# =====================================================

base_dir = Path(r"C:\mini_pj\data\processed\validation_samples")

evidence_dir = (
    base_dir / "evidence_validation"
    / "2024" / "강원특별자치도의회"
)

type_dir = (
    base_dir / "agenda_matching"
    / "2024" / "강원특별자치도의회"
)

output_dir = evidence_dir / "analysis"
output_dir.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. 검증 결과 불러오기
# =====================================================

evidence_df = pd.read_csv(
    evidence_dir / "evidence_validation_all.csv",
    dtype={"record_index": str},
    keep_default_na=False
)

type_df = pd.read_csv(
    type_dir / "meeting_type_summary.csv",
    dtype={"record_index": str},
    keep_default_na=False
)

# =====================================================
# 3. 회의 유형 연결
# =====================================================

df = evidence_df.merge(
    type_df[
        ["record_index", "meeting_type",
         "meeting_title", "audit_target"]
    ],
    on="record_index",
    how="left",
    validate="many_to_one",
    suffixes=("", "_original")
)

df["meeting_type"] = df["meeting_type"].fillna("유형 미확인")

# =====================================================
# 4. 회의 유형별 일치 현황
# =====================================================

summary = pd.crosstab(
    df["meeting_type"],
    df["validation_status"]
)

summary["total"] = summary.sum(axis=1)

print("\n===== 회의 유형별 원문 일치 현황 =====")
print(summary.to_string())

# 원문 일치 상태별 비율
ratio = (
    summary.drop(columns="total")
    .div(summary["total"], axis=0) * 100
).round(2)

print("\n===== 회의 유형별 원문 일치율 (%) =====")
print(ratio.to_string())

# =====================================================
# 5. 미일치 청크 247개 분석
# =====================================================

unmatched = df[
    df["validation_status"] == "UNMATCHED"
].copy()

print("\n===== 미일치 청크 분석 =====")
print("전체 미일치:", len(unmatched))

print("\n[회의 유형별 미일치]")
print(unmatched["meeting_type"].value_counts())

# 회의록별 미일치 건수
unmatched_by_record = (
    unmatched.groupby(
        ["record_index", "date", "meeting_type"]
    )
    .size()
    .reset_index(name="unmatched_count")
    .sort_values("unmatched_count", ascending=False)
)

print("\n[미일치 상위 20개 회의록]")
print(unmatched_by_record.head(20).to_string(index=False))

# =====================================================
# 6. 미일치 청크 내용 확인
# =====================================================

print("\n===== 미일치 청크 샘플 10개 =====")

for _, row in unmatched.head(10).iterrows():

    print("\n---------------------------")
    print("청크:", row["chunk_id"])
    print("날짜:", row["date"])
    print("회의 유형:", row["meeting_type"])
    print("청크 길이:", len(str(row["text"])))
    print("내용:", str(row["text"])[:250])

# =====================================================
# 7. 결과 저장
# =====================================================

summary.to_csv(
    output_dir / "evidence_by_meeting_type.csv",
    encoding="utf-8-sig"
)

ratio.to_csv(
    output_dir / "evidence_ratio_by_type.csv",
    encoding="utf-8-sig"
)

unmatched.to_csv(
    output_dir / "evidence_unmatched_247.csv",
    index=False,
    encoding="utf-8-sig"
)

unmatched_by_record.to_csv(
    output_dir / "unmatched_by_record.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n분석 완료!")
print("결과 저장:", output_dir)