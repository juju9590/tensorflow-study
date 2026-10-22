from pathlib import Path
import pandas as pd

base_dir = Path(
    r"C:\mini_pj\data\processed\validation_samples"
    r"\agenda_matching\2024\강원특별자치도의회"
)

# 1. 검증 결과 불러오기
match_df = pd.read_csv(
    base_dir / "improved" / "agenda_match_improved.csv",
    dtype={"record_index": str}
)

type_df = pd.read_csv(
    base_dir / "meeting_type_summary.csv",
    dtype={"record_index": str}
)

# 2. 회의록 유형 연결
merged = match_df.merge(
    type_df[
        ["record_index", "meeting_type", "meeting_title",
         "audit_target"]
    ],
    on="record_index",
    how="left",
    validate="many_to_one"
)

# 3. 회의 유형별 매칭 결과
summary = pd.crosstab(
    merged["meeting_type"].fillna("유형 미확인"),
    merged["match_status"]
)

summary["total"] = summary.sum(axis=1)

print("\n===== 회의 유형별 안건 매칭 결과 =====")
print(summary.to_string())

# 4. 유형별 UNASSIGNED 비율
if "UNASSIGNED" in summary.columns:
    summary["unassigned_ratio"] = (
        summary["UNASSIGNED"] / summary["total"] * 100
    ).round(2)

    print("\n===== 회의 유형별 미배정 현황 =====")
    print(
        summary[["UNASSIGNED", "total", "unassigned_ratio"]]
        .to_string()
    )

# 5. 저장
summary.to_csv(
    base_dir / "matching_by_meeting_type.csv",
    encoding="utf-8-sig"
)

merged.to_csv(
    base_dir / "matching_with_meeting_type.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n분석 완료!")