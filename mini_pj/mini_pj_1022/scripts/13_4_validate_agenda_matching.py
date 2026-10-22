# 실제 데이터를 수정하지 않는 검증 전용 코드
# 2024년 강원특별자치도의회 전체 배치 CSV와 원본 JSON을 연결해 검사


from pathlib import Path
from bs4 import BeautifulSoup
from collections import Counter
import pandas as pd
import json
import re

# =====================================================
# 1. 경로 설정
# =====================================================

base_dir = Path(r"C:\mini_pj\data")

region = "강원특별자치도의회"
year = "2024"

embedding_dir = (
    base_dir / "processed"
    / "embeddings_by_region" / year / region
)

output_dir = (
    base_dir / "processed"
    / "validation_samples"
    / "agenda_matching" / year / region
)

output_dir.mkdir(parents=True, exist_ok=True)

# 원본 JSON 경로 직접 지정
json_path = (
    base_dir
    / "org2"
    / f"{year}_광역 회의록"
    / f"{year}_{region}.json"
)

if not json_path.exists():
    raise FileNotFoundError(
        f"원본 JSON을 찾을 수 없습니다: {json_path}"
    )

print("원본 JSON:", json_path)

csv_files = sorted(
    embedding_dir.glob("*metadata.csv")
)

if not csv_files:
    raise FileNotFoundError("메타데이터 CSV가 없습니다.")

print("원본 JSON:", json_path)
print("배치 CSV:", len(csv_files), "개")

# =====================================================
# 2. 원본 JSON 불러오기
# =====================================================

with open(json_path, "r", encoding="utf-8") as f:
    records = json.load(f)

print("원본 회의록:", len(records), "개")

# =====================================================
# 3. 텍스트 정규화
# =====================================================

def normalize(text):
    text = str(text or "")
    text = text.replace("\\n", " ")
    return re.sub(r"\s+", "", text)

# =====================================================
# 4. 회의록별 안건·발언 추출
# =====================================================

def extract_agenda_speeches(record):

    soup = BeautifulSoup(
        record.get("MINTS_HTML", ""),
        "html.parser"
    )

    current_agenda = "안건 미배정"
    speeches = []

    elements = soup.select(
        "strong.item-in-contents, "
        "div.contents-block.speaker-block"
    )

    for element in elements:

        if element.name == "strong":
            current_agenda = element.get_text(
                " ", strip=True
            )

        elif element.name == "div":
            speech = normalize(
                element.get_text(" ", strip=True)
            )

            if speech:
                speeches.append({
                    "agenda": current_agenda,
                    "text": speech
                })

    return speeches

# 2024년 강원 전체 회의록을 1회만 파싱
agenda_cache = {
    i: extract_agenda_speeches(record)
    for i, record in enumerate(records)
}

# =====================================================
# 5. 청크 앞·중간·끝 문구 추출
# =====================================================

def get_anchors(text, size=55):

    text = normalize(text)

    if len(text) < size:
        return [text] if text else []

    positions = [
        0,
        max(0, len(text)//2 - size//2),
        len(text) - size
    ]

    return list(dict.fromkeys(
        text[p:p+size] for p in positions
    ))

# =====================================================
# 6. 청크와 안건 매칭
# =====================================================

def match_agenda(text, speeches):

    anchors = get_anchors(text)

    if not anchors or not speeches:
        return "UNMATCHED", "", 0

    matched_agendas = []
    matched_anchor_count = 0

    for anchor in anchors:

        candidates = {
            speech["agenda"]
            for speech in speeches
            if anchor in speech["text"]
        }

        if candidates:
            matched_anchor_count += 1
            matched_agendas.extend(candidates)

    unique_agendas = list(dict.fromkeys(matched_agendas))

    if len(unique_agendas) > 1:
        status = "MULTI_AGENDA"

    elif len(unique_agendas) == 1:
        if matched_anchor_count >= 2:
            status = "SINGLE_CANDIDATE"
        else:
            status = "REVIEW_LOW_MATCH"

    else:
        status = "UNMATCHED"

    return (
        status,
        " | ".join(unique_agendas),
        matched_anchor_count
    )

# =====================================================
# 7. 전체 배치 CSV 검사
# =====================================================

results = []
stats = Counter()
total = 0

for file_no, file in enumerate(csv_files, 1):

    file_count = 0

    # 대용량 CSV 대비 분할 읽기
    for df in pd.read_csv(
        file,
        dtype=str,
        keep_default_na=False,
        chunksize=2000
    ):

        required = {"record_index", "chunk_id", "text"}

        if not required.issubset(df.columns):
            raise ValueError(
                f"필수 컬럼 없음: {file.name}"
            )

        for _, row in df.iterrows():

            total += 1
            file_count += 1

            record_index = str(
                row["record_index"]
            ).strip()

            try:
                idx = int(record_index)
            except ValueError:
                idx = -1

            # 원본 리스트 인덱스 확인
            if 0 <= idx < len(records):

                record = records[idx]

                date_match = (
                    str(row.get("date", "")).strip()
                    == str(record.get("MTG_DE", "")).strip()
                )

                assembly_match = (
                    str(row.get("assembly", "")).strip()
                    == str(record.get("RASMBLY_NM", "")).strip()
                )

                if date_match and assembly_match:
                    status, agenda, anchors = match_agenda(
                        row["text"],
                        agenda_cache[idx]
                    )
                else:
                    status, agenda, anchors = (
                        "SOURCE_MISMATCH", "", 0
                    )

            else:
                status, agenda, anchors = (
                    "INVALID_INDEX", "", 0
                )

            stats[status] += 1

            results.append({
                "sample_file": file.name,
                "date": row.get("date", ""),
                "assembly": row.get("assembly", ""),
                "record_index": record_index,
                "source_id": row.get("source_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "chunk_index": row.get("chunk_index", ""),
                "match_status": status,
                "agenda_candidate": agenda,
                "matched_anchors": anchors,
                "text": row["text"]
            })

    print(
        f"[{file_no}/{len(csv_files)}] "
        f"{file.name}: {file_count:,}개 검사"
    )

# =====================================================
# 8. 결과 저장
# =====================================================

result_df = pd.DataFrame(results)

result_df.to_csv(
    output_dir / "agenda_match_results.csv",
    index=False,
    encoding="utf-8-sig"
)

# 단일 안건 자동 매칭 확정 전 검토 필요 항목
review_df = result_df[
    result_df["match_status"] != "SINGLE_CANDIDATE"
]

review_df.to_csv(
    output_dir / "agenda_match_review.csv",
    index=False,
    encoding="utf-8-sig"
)

summary_df = pd.DataFrame(
    [
        {
            "status": status,
            "count": count,
            "percentage": round(
                count / total * 100, 2
            ) if total else 0
        }
        for status, count in sorted(stats.items())
    ]
)

summary_df.to_csv(
    output_dir / "agenda_match_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# 9. 결과 출력
# =====================================================

print("\n===== 안건명 매칭 검증 결과 =====")
print("전체 청크:", f"{total:,}")
print(summary_df.to_string(index=False))

print("\n결과 저장:", output_dir)
print("검증 완료!")


# 3. 결과를 해석하는 방법
# 상태	의미	후속 처리
# SINGLE_CANDIDATE	앞·중간·끝 중 2개 이상이 한 안건에 일치	자동 복구 후보
# MULTI_AGENDA	둘 이상의 안건에서 문구 발견	안건 혼합 여부 검토
# REVIEW_LOW_MATCH	한 부분만 일치	추가 확인
# UNMATCHED	일치 문구 없음	원본·정제 차이 조사
# SOURCE_MISMATCH	날짜 또는 의회명이 원본과 불일치	원본 연결 확인
# INVALID_INDEX	원본 레코드 번호를 찾지 못함	인덱스 확인