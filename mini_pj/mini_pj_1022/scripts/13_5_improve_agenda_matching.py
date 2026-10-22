
from pathlib import Path
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
import pandas as pd
import json
import re

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

embedding_dir = (
    base_dir / "processed"
    / "embeddings_by_region" / year / region
)

validation_dir = (
    base_dir / "processed"
    / "validation_samples"
    / "agenda_matching" / year / region
)

output_dir = validation_dir / "improved"
output_dir.mkdir(parents=True, exist_ok=True)

csv_files = sorted(embedding_dir.glob("*metadata.csv"))

if not json_path.exists():
    raise FileNotFoundError(json_path)

if not csv_files:
    raise FileNotFoundError(embedding_dir)

print("원본 JSON:", json_path)
print("CSV 파일:", len(csv_files))

# =====================================================
# 2. JSON 로드
# =====================================================

with open(json_path, "r", encoding="utf-8") as f:
    records = json.load(f)

print("원본 회의록:", len(records))

# =====================================================
# 3. 텍스트 정규화
# =====================================================

def normalize(text):
    text = str(text or "")
    text = text.replace("\\n", " ")
    return re.sub(r"\s+", "", text)


# =====================================================
# 4. 안건별 발언 통합
# =====================================================

def extract_agenda_groups(record):

    soup = BeautifulSoup(
        record.get("MINTS_HTML", ""),
        "html.parser"
    )

    groups = defaultdict(list)
    current_agenda = "안건 미배정"

    elements = soup.select(
        "strong.item-in-contents, "
        "div.contents-block.speaker-block"
    )

    for element in elements:

        if element.name == "strong":
            current_agenda = element.get_text(" ", strip=True)

        elif element.name == "div":

            speech = normalize(
                element.get_text(" ", strip=True)
            )

            if speech:
                groups[current_agenda].append(speech)

    # 발언 간 경계를 보존하기 위해 구분자 사용
    # 원문에 없는 접합 문구가 일치하는 문제 방지
    return {
        agenda: "\u241E".join(speeches)
        for agenda, speeches in groups.items()
    }


print("안건별 텍스트 구성 중...")

agenda_cache = {
    i: extract_agenda_groups(record)
    for i, record in enumerate(records)
}

# =====================================================
# 5. 청크의 여러 위치에서 검사 문구 추출
# =====================================================

def get_anchors(text, size=45, count=7):

    text = normalize(text)

    if not text:
        return []

    if len(text) <= size:
        return [text]

    max_start = len(text) - size

    positions = [
        round(max_start * i / (count - 1))
        for i in range(count)
    ]

    return list(dict.fromkeys(
        text[p:p + size] for p in positions
    ))

# =====================================================
# 6. 개선된 안건 매칭
# =====================================================

def match_agenda(text, agenda_groups):

    anchors = get_anchors(text)

    if not anchors or not agenda_groups:
        return "UNMATCHED", "", 0, 0

    votes = Counter()
    matched_count = 0

    for anchor in anchors:

        matched_agendas = [
            agenda
            for agenda, full_text in agenda_groups.items()
            if anchor in full_text
        ]

        if matched_agendas:
            matched_count += 1

            for agenda in matched_agendas:
                votes[agenda] += 1

    if not votes:
        status = "UNMATCHED"
        candidate = ""

    elif len(votes) > 1:
        status = "MULTI_AGENDA"
        candidate = " | ".join(votes.keys())

    else:
        candidate = next(iter(votes))

        if candidate == "안건 미배정":
            status = "UNASSIGNED"

        elif matched_count >= 2:
            status = "SINGLE_CANDIDATE"

        else:
            status = "REVIEW_LOW_MATCH"

    return (
        status,
        candidate,
        matched_count,
        len(anchors)
    )

# =====================================================
# 7. 전체 임베딩 CSV 검사
# =====================================================

results = []
stats = Counter()
total = 0

for file_no, file in enumerate(csv_files, 1):

    file_count = 0

    for df in pd.read_csv(
        file,
        dtype=str,
        keep_default_na=False,
        chunksize=2000
    ):

        required = {
            "record_index", "chunk_id", "text",
            "date", "assembly"
        }

        if not required.issubset(df.columns):
            raise ValueError(
                f"필수 컬럼 누락: {file.name}"
            )

        for _, row in df.iterrows():

            total += 1
            file_count += 1

            try:
                idx = int(row["record_index"])
            except (ValueError, TypeError):
                idx = -1

            status = "INVALID_INDEX"
            candidate = ""
            matched = 0
            anchor_total = 0

            if 0 <= idx < len(records):

                record = records[idx]

                source_ok = (
                    str(row["date"]).strip()
                    == str(record.get("MTG_DE", "")).strip()
                    and
                    str(row["assembly"]).strip()
                    == str(record.get("RASMBLY_NM", "")).strip()
                )

                if source_ok:
                    (
                        status, candidate,
                        matched, anchor_total
                    ) = match_agenda(
                        row["text"],
                        agenda_cache[idx]
                    )
                else:
                    status = "SOURCE_MISMATCH"

            stats[status] += 1

            results.append({
                "sample_file": file.name,
                "date": row["date"],
                "assembly": row["assembly"],
                "record_index": row["record_index"],
                "source_id": row.get("source_id", ""),
                "chunk_id": row["chunk_id"],
                "chunk_index": row.get("chunk_index", ""),
                "match_status": status,
                "agenda_candidate": candidate,
                "matched_anchors": matched,
                "total_anchors": anchor_total,
                "match_ratio": round(
                    matched / anchor_total, 4
                ) if anchor_total else 0,
                "text": row["text"]
            })

    print(
        f"[{file_no}/{len(csv_files)}] "
        f"{file.name}: {file_count:,}개"
    )

# =====================================================
# 8. 결과 저장
# =====================================================

result_df = pd.DataFrame(results)

result_df.to_csv(
    output_dir / "agenda_match_improved.csv",
    index=False,
    encoding="utf-8-sig"
)

review_df = result_df[
    result_df["match_status"] != "SINGLE_CANDIDATE"
]

review_df.to_csv(
    output_dir / "agenda_match_review.csv",
    index=False,
    encoding="utf-8-sig"
)

summary_df = pd.DataFrame([
    {
        "status": status,
        "count": count,
        "percentage": round(count / total * 100, 2)
        if total else 0
    }
    for status, count in sorted(stats.items())
])

summary_df.to_csv(
    output_dir / "agenda_match_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# 9. 기존 13_4 결과와 비교
# =====================================================

old_summary_path = (
    validation_dir / "agenda_match_summary.csv"
)

print("\n===== 개선된 안건명 매칭 결과 =====")
print("전체 청크:", f"{total:,}")
print(summary_df.to_string(index=False))

if old_summary_path.exists():

    old_df = pd.read_csv(old_summary_path)

    comparison = pd.merge(
        old_df[["status", "count"]].rename(
            columns={"count": "before"}
        ),
        summary_df[["status", "count"]].rename(
            columns={"count": "after"}
        ),
        on="status",
        how="outer"
    ).fillna(0)

    comparison["before"] = comparison["before"].astype(int)
    comparison["after"] = comparison["after"].astype(int)

    comparison["difference"] = (
        comparison["after"] - comparison["before"]
    )

    comparison.to_csv(
        output_dir / "agenda_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== 13_4 vs 13_5 비교 =====")
    print(comparison.to_string(index=False))

else:
    print("\n이전 검증 결과가 없어 비교를 생략합니다.")

print("\n검증 완료!")
print("결과 저장:", output_dir)

# 항목	기존 13_4	개선 13_5
# 매칭 대상	개별 발언 블록	안건별 발언 통합 텍스트
# 청크 비교	앞·중간·끝 3곳	최대 7개 지점
# 발언 연결	발언 블록마다 별도 검색	같은 안건의 발언을 순서대로 결합
# 안건 미배정	일반 후보에 포함	별도 분류
# 결과 비교	단독 결과	13_4 대비 비교
# 원본 데이터	읽기 전용	읽기 전용