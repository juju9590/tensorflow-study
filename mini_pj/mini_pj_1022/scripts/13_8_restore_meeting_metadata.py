
from pathlib import Path
from bs4 import BeautifulSoup
from collections import Counter
import pandas as pd
import json

# =====================================================
# 1. 경로 설정
# =====================================================

base_dir = Path(r"C:\mini_pj\data")

year = "2024"
region = "강원특별자치도의회"

json_path = (
    base_dir / "org2"
    / f"{year}_광역 회의록"
    / f"{year}_{region}.json"
)

validation_dir = (
    base_dir / "processed"
    / "validation_samples"
    / "agenda_matching" / year / region
)

match_path = (
    validation_dir / "improved"
    / "agenda_match_improved.csv"
)

output_dir = validation_dir / "restored"
output_dir.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. 원본 데이터 불러오기
# =====================================================

with open(json_path, "r", encoding="utf-8") as f:
    records = json.load(f)

df = pd.read_csv(
    match_path,
    dtype=str,
    keep_default_na=False
)

print("원본 회의록:", len(records))
print("전체 청크:", len(df))

# =====================================================
# 3. 원본 HTML 메타데이터 추출
# =====================================================

def extract_metadata(record):

    soup = BeautifulSoup(
        record.get("MINTS_HTML", ""),
        "html.parser"
    )

    session = soup.select_one("p.session-line")
    title = soup.select_one("#minutes-title-line")
    inspection = soup.select_one("#inspection-line")

    session_text = (
        session.get_text(" ", strip=True)
        if session else ""
    )

    meeting_title = (
        title.get_text(" ", strip=True)
        if title else ""
    )

    audit_target = (
        inspection.get_text(" ", strip=True)
        if inspection else ""
    )

    if audit_target.startswith("피감사기관:"):
        audit_target = audit_target.split(":", 1)[1].strip()

    agenda_items = soup.select("strong.item-in-contents")

    if "행정사무감사" in session_text:
        meeting_type = "행정사무감사"

    elif agenda_items:
        meeting_type = "일반 안건 회의"

    else:
        meeting_type = "기타 / 추가 검증"

    return {
        "meeting_type": meeting_type,
        "meeting_title_original": meeting_title,
        "session_title": session_text,
        "audit_target": audit_target,
        "agenda_count": len(agenda_items)
    }


# 280개 회의록만 파싱
metadata_cache = {
    i: extract_metadata(record)
    for i, record in enumerate(records)
}

# =====================================================
# 4. 청크별 메타데이터 복구
# =====================================================

results = []
stats = Counter()

for _, row in df.iterrows():

    status = row["match_status"]
    candidate = row["agenda_candidate"]

    try:
        idx = int(row["record_index"])
    except (ValueError, TypeError):
        idx = -1

    # 원본 인덱스 유효성 확인
    if not (0 <= idx < len(records)):
        metadata = {
            "meeting_type": "원본 확인 불가",
            "meeting_title_original": "",
            "session_title": "",
            "audit_target": "",
            "agenda_count": 0
        }
        restore_status = "INVALID_SOURCE"
        agenda_title = ""

    else:
        record = records[idx]

        # 원본 날짜·의회 대조
        source_ok = (
            str(row.get("date", "")).strip()
            == str(record.get("MTG_DE", "")).strip()
            and
            str(row.get("assembly", "")).strip()
            == str(record.get("RASMBLY_NM", "")).strip()
        )

        if not source_ok:
            metadata = {
                "meeting_type": "원본 확인 불가",
                "meeting_title_original": "",
                "session_title": "",
                "audit_target": "",
                "agenda_count": 0
            }
            agenda_title = ""
            restore_status = "SOURCE_MISMATCH"

        else:
            metadata = metadata_cache[idx]
            meeting_type = metadata["meeting_type"]

            agenda_title = ""

            if meeting_type == "행정사무감사":
                if metadata["audit_target"]:
                    restore_status = "AUDIT_METADATA_RESTORED"
                else:
                    restore_status = "AUDIT_TARGET_MISSING"

            elif meeting_type == "일반 안건 회의":

                if (
                    status == "SINGLE_CANDIDATE"
                    and candidate
                    and candidate != "안건 미배정"
                ):
                    agenda_title = candidate
                    restore_status = "AGENDA_CANDIDATE"

                else:
                    restore_status = "AGENDA_REVIEW"

            else:
                restore_status = "OTHER_REVIEW"

    stats[restore_status] += 1

    results.append({
        "sample_file": row.get("sample_file", ""),
        "date": row.get("date", ""),
        "assembly": row.get("assembly", ""),
        "record_index": row.get("record_index", ""),
        "source_id": row.get("source_id", ""),
        "chunk_id": row.get("chunk_id", ""),
        "chunk_index": row.get("chunk_index", ""),
        "meeting_type": metadata["meeting_type"],
        "meeting_title_original": metadata["meeting_title_original"],
        "session_title": metadata["session_title"],
        "agenda_title": agenda_title,
        "audit_target": metadata["audit_target"],
        "agenda_count": metadata["agenda_count"],
        "match_status": status,
        "restore_status": restore_status,
        "text": row.get("text", "")
    })

# =====================================================
# 5. 결과 저장
# =====================================================

result_df = pd.DataFrame(results)

result_df.to_csv(
    output_dir / "metadata_restore_candidates.csv",
    index=False,
    encoding="utf-8-sig"
)

# 사람 검토 대상
review_df = result_df[
    ~result_df["restore_status"].isin([
        "AGENDA_CANDIDATE",
        "AUDIT_METADATA_RESTORED"
    ])
]

review_df.to_csv(
    output_dir / "metadata_restore_review.csv",
    index=False,
    encoding="utf-8-sig"
)

# 복구 상태별 집계
summary_df = pd.DataFrame([
    {
        "restore_status": status,
        "count": count,
        "percentage": round(
            count / len(result_df) * 100, 2
        )
    }
    for status, count in sorted(stats.items())
])

summary_df.to_csv(
    output_dir / "metadata_restore_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# 6. 최종 결과 출력
# =====================================================

print("\n===== 메타데이터 복구 후보 생성 결과 =====")
print(summary_df.to_string(index=False))

print("\n===== 회의 유형별 현황 =====")
print(result_df["meeting_type"].value_counts())

print("\n안건명 후보 연결:",
      (result_df["agenda_title"] != "").sum())

print("감사 대상 기관 확인:",
      (
          (result_df["meeting_type"] == "행정사무감사")
          & (result_df["audit_target"] != "")
      ).sum())

print("\n결과 저장 위치:", output_dir)
print("메타데이터 복구 후보 생성 완료!")


# 회의 유형	복구 내용	처리 방식
# 일반 회의	agenda_title	SINGLE_CANDIDATE만 후보 연결
# 행정사무감사	meeting_type, audit_target	원본 HTML에서 직접 추출
# 기타 회의	meeting_type	확인 필요로 보류
# 복수 안건	agenda_title	공란 유지, 검토 대상
# 낮은 일치·미매칭	agenda_title	공란 유지, 검토 대상