# 구분	내용
# 입력	embeddings_by_region의 전체 batch_XXXX_metadata.csv
# 검증 컬럼	text
# 검사 대상	UI 문구, HTML 잔여물, 출석자 명단, 회의록 머리말, 반복 안내 문구
# 처리 방법	자동 분류 및 검토 필요 여부 표시
# 원본 수정	없음
# 재임베딩	없음
# 출력	지역별 검사 결과 + 전체 통계 + 의심 청크


from pathlib import Path
from collections import Counter
import pandas as pd
import re
import json
import csv

# =====================================================
# 1. 경로 설정
# =====================================================

BASE_DIR = Path(r"C:\mini_pj\data\processed")

INPUT_DIR = BASE_DIR / "embeddings_by_region"
OUTPUT_DIR = BASE_DIR / "noise_validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 이미 완료된 배치 건너뛰기
CHECKPOINT = OUTPUT_DIR / "completed_batches.json"

BATCH_SIZE = 5000

files = sorted(INPUT_DIR.rglob("batch_*_metadata.csv"))

print("전체 검증 CSV:", len(files))

if not files:
    raise FileNotFoundError("검증 대상 CSV가 없습니다.")

# =====================================================
# 2. 불필요 텍스트 검사 패턴
# =====================================================

PATTERNS = {
    "UI_NOISE": [
        r"의원\s*프로필",
        r"창\s*닫기",
        r"학력\s*및\s*경력사항",
        r"<학력사항>",
        r"<경력사항>",
        r"(?i)\bx\s*close\b",
        r"(?m)^\s*(로그인|회원가입|전체메뉴|통합검색)\s*$",
    ],

    "ATTENDANCE": [
        r"○\s*출석\s*(위원|의원|공무원|감사위원)",
        r"○\s*피감사기관\s*참석자",
        r"○\s*(기록|속기)\s*[:：]?",
    ],

    "HEADER_AGENDA": [
        r"의\s*사\s*일\s*정\s*\(",
        r"부의된\s*안건",
        r"(?m)^\s*제\d+회.*회의록\s*$",
        r"(?m)^\s*강원특별자치도의회사무처\s*$",
    ],

    "HTML_RESIDUE": [
        r"</?[a-zA-Z][^>]*>",
        r"&(?:nbsp|amp|lt|gt|quot|#\d+|#x[0-9a-fA-F]+);",
    ],
}

COMPILED = {
    category: [re.compile(p) for p in patterns]
    for category, patterns in PATTERNS.items()
}

# =====================================================
# 3. 청크별 분류 함수
# =====================================================

def classify_text(text):

    text = str(text or "")
    found = {}

    for category, patterns in COMPILED.items():
        matches = []

        for pattern in patterns:
            matches.extend(
                m.group(0) for m in pattern.finditer(text)
            )

        if matches:
            found[category] = list(dict.fromkeys(matches))[:10]

    if not text.strip():
        return "REVIEW", found, "EMPTY_TEXT"

    categories = list(found.keys())

    if not categories:
        return "POLICY_TEXT", found, "NO_PATTERN"

    # 여러 종류의 잡음이 발견되거나
    # 긴 청크 안에 일부 패턴만 발견된 경우
    if len(categories) > 1 or len(text.strip()) >= 200:
        return "MIXED", found, "CHECK_PARTIAL_CLEANING"

    category = categories[0]

    if category == "HTML_RESIDUE":
        return "REVIEW", found, "HTML_REVIEW"

    return category, found, "REVIEW_BEFORE_REMOVAL"

# =====================================================
# 4. 체크포인트 로드
# =====================================================

if CHECKPOINT.exists():
    completed = set(json.loads(
        CHECKPOINT.read_text(encoding="utf-8")
    ))
else:
    completed = set()

# =====================================================
# 5. 지역별 전체 배치 검사
# =====================================================

for file_no, file in enumerate(files, 1):

    relative = file.relative_to(INPUT_DIR)
    batch_key = relative.as_posix()

    if batch_key in completed:
        print(f"[건너뜀] {batch_key}")
        continue

    # 원본 구조와 동일하게 출력 폴더 생성
    save_dir = OUTPUT_DIR / relative.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    output_file = save_dir / (
        file.stem + "_noise_check.csv"
    )

    temp_file = output_file.with_suffix(".tmp")

    stats = Counter()
    total = 0
    flagged = 0

    with temp_file.open(
        "w", encoding="utf-8-sig", newline=""
    ) as out:

        writer = csv.writer(out)

        writer.writerow([
            "batch_file",
            "row_index",
            "date",
            "assembly",
            "source_id",
            "record_index",
            "chunk_id",
            "noise_type",
            "noise_categories",
            "detected_patterns",
            "review_reason",
            "text_length",
            "text",
        ])

        reader = pd.read_csv(
            file,
            dtype=str,
            keep_default_na=False,
            chunksize=BATCH_SIZE,
        )

        row_offset = 0

        for df in reader:

            required = {
                "date", "assembly", "source_id",
                "record_index", "chunk_id", "text"
            }

            if not required.issubset(df.columns):
                raise ValueError(
                    f"필수 컬럼 없음: {file}"
                )

            for local_index, row in enumerate(
                df.itertuples(index=False, name=None)
            ):
                # 컬럼명 기준으로 값 연결
                record = dict(zip(df.columns, row))
                text = record["text"]

                noise_type, found, reason = classify_text(text)

                stats[noise_type] += 1
                total += 1

                # 정상 청크는 검사 건수만 집계
                # 의심 청크만 CSV에 저장
                if noise_type == "POLICY_TEXT":
                    continue

                flagged += 1

                writer.writerow([
                    batch_key,
                    row_offset + local_index,
                    record["date"],
                    record["assembly"],
                    record["source_id"],
                    record["record_index"],
                    record["chunk_id"],
                    noise_type,
                    "|".join(found.keys()),
                    json.dumps(found, ensure_ascii=False),
                    reason,
                    len(text),
                    text,
                ])

            row_offset += len(df)

    # 배치가 정상 종료된 후 결과 파일 확정
    temp_file.replace(output_file)

    # 배치별 통계 저장
    summary_path = save_dir / (
        file.stem + "_noise_summary.json"
    )

    summary = {
        "batch_file": batch_key,
        "total": total,
        "flagged": flagged,
        "counts": dict(stats),
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    completed.add(batch_key)

    CHECKPOINT.write_text(
        json.dumps(
            sorted(completed),
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        f"[{file_no}/{len(files)}] {batch_key} "
        f"| 전체 {total:,} "
        f"| 의심 {flagged:,}"
    )

# =====================================================
# 6. 전체 결과 집계
# =====================================================

summaries = []

for summary_file in OUTPUT_DIR.rglob("*_noise_summary.json"):
    summary = json.loads(
        summary_file.read_text(encoding="utf-8")
    )

    relative = Path(summary["batch_file"])
    year = relative.parts[0]
    region = relative.parts[1]

    row = {
        "year": year,
        "region": region,
        "batch_file": summary["batch_file"],
        "total": summary["total"],
        "flagged": summary["flagged"],
    }

    row.update(summary["counts"])
    summaries.append(row)

summary_df = pd.DataFrame(summaries).fillna(0)

summary_df.to_csv(
    OUTPUT_DIR / "noise_summary_all.csv",
    index=False,
    encoding="utf-8-sig"
)

region_df = (
    summary_df
    .drop(columns=["batch_file"])
    .groupby(["year", "region"], as_index=False)
    .sum(numeric_only=True)
)

region_df["flagged_ratio"] = (
    region_df["flagged"] / region_df["total"] * 100
).round(2)

region_df.to_csv(
    OUTPUT_DIR / "noise_summary_by_region.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 전체 검증 결과 =====")

print("검증 청크:", f"{summary_df['total'].sum():,}")
print("의심 청크:", f"{summary_df['flagged'].sum():,}")

print("\n===== 유형별 집계 =====")

for col in [
    "POLICY_TEXT",
    "UI_NOISE",
    "ATTENDANCE",
    "HEADER_AGENDA",
    "MIXED",
    "REVIEW",
]:
    count = int(summary_df[col].sum()) if col in summary_df else 0
    print(f"{col}: {count:,}")

print("\n===== 지역별 의심 비율 상위 20개 =====")

print(
    region_df.sort_values(
        "flagged_ratio", ascending=False
    ).head(20).to_string(index=False)
)

print("\n검증 완료!")
print("결과:", OUTPUT_DIR)
