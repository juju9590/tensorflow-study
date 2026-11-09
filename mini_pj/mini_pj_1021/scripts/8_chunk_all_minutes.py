# 지역별 순차 저장 방식으로 청킹 코드 작성
# 1. 지역별 CSV 1개 불러오기
# 2. 회의록을 1,000자 단위로 청킹
# 3. 청크마다 날짜·의회명·출처 ID 유지
# 4. 청크를 결과 CSV에 즉시 저장
# 5. 지역별 처리 건수와 소요 시간 출력
# 6. 다음 지역으로 이동


import csv
import os
import sys
import time
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# pip install langchain_text_splitters

# ==================================================
# 1. 경로 설정
# ==================================================

INPUT_DIR = Path("C:/tensorflow_study/mini_pj/data/processed/by_region")
OUTPUT_DIR = Path("C:/tensorflow_study/mini_pj/data/processed/chunked_by_region")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 지역별 CSV 목록
csv_files = sorted(INPUT_DIR.glob("*.csv"))

print("전체 지역 파일:", len(csv_files), flush=True)


# ==================================================
# 2. 청킹 설정
# ==================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# 긴 회의록도 CSV에서 읽을 수 있도록 설정
limit = sys.maxsize

while True:
    try:
        csv.field_size_limit(limit)
        break
    except OverflowError:
        limit //= 10


# ==================================================
# 3. 지역별 청킹 및 저장
# ==================================================

start_time = time.time()
total_minutes = 0
total_chunks = 0

for region_num, input_path in enumerate(csv_files, start=1):

    region_start = time.time()

    output_path = OUTPUT_DIR / (
        input_path.stem + "_chunked.csv"
    )

    # 작업 중인 임시 파일
    temp_path = OUTPUT_DIR / (
        input_path.stem + "_chunked.partial.csv"
    )

    print(f"\n[{region_num}/{len(csv_files)}] {input_path.stem}", flush=True)

    # 이미 완료된 지역은 건너뛰기
    if output_path.exists():
        print("이미 완료된 지역 → 건너뜀", flush=True)
        continue

    if temp_path.exists():
        print("이전 미완료 파일 발견 → 처음부터 다시 처리", flush=True)

    region_minutes = 0
    region_chunks = 0

    try:
        # 원본 CSV 읽기
        with open(
            input_path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as infile:

            reader = csv.DictReader(infile)

            # 원본 11개 컬럼 중 utterance 제외
            metadata_columns = [
                col for col in reader.fieldnames
                if col != "utterance"
            ]

            output_columns = (
                metadata_columns
                + ["chunk_id", "chunk_index", "text"]
            )

            # 청킹 CSV 순차 저장
            with open(
                temp_path,
                "w",
                encoding="utf-8-sig",
                newline=""
            ) as outfile:

                writer = csv.DictWriter(
                    outfile,
                    fieldnames=output_columns
                )

                writer.writeheader()

                for row in reader:

                    text = row.get("utterance", "").strip()

                    if not text:
                        continue

                    # 회의록 1건 청킹
                    chunks = text_splitter.split_text(text)

                    for i, chunk in enumerate(chunks, start=1):

                        data = {
                            col: row[col]
                            for col in metadata_columns
                        }

                        data["chunk_id"] = f"{row['source_id']}_{i}"
                        data["chunk_index"] = i
                        data["text"] = chunk

                        # 생성 즉시 저장
                        writer.writerow(data)

                        region_chunks += 1

                    region_minutes += 1

                    # 25건마다 진행률 출력
                    if region_minutes % 25 == 0:
                        elapsed = time.time() - region_start

                        print(
                            f"진행: {region_minutes}건 / "
                            f"청크: {region_chunks:,}개 / "
                            f"{elapsed:.1f}초",
                            flush=True
                        )

        # 모든 회의록 처리 성공 시 완료 파일로 변경
        os.replace(temp_path, output_path)

        total_minutes += region_minutes
        total_chunks += region_chunks

        region_elapsed = time.time() - region_start

        print(f"완료: {input_path.stem}", flush=True)
        print(f"회의록: {region_minutes}건", flush=True)
        print(f"청크: {region_chunks:,}개", flush=True)
        print(f"소요 시간: {region_elapsed:.1f}초", flush=True)
        print(f"저장: {output_path}", flush=True)

    except Exception as e:
        print(f"오류 발생: {input_path.stem}", flush=True)
        print(f"오류 내용: {e}", flush=True)
        print(f"미완료 파일: {temp_path}", flush=True)

        # 오류가 발생한 지역에서 중단
        raise


# ==================================================
# 4. 전체 결과
# ==================================================

total_elapsed = time.time() - start_time

print("\n===== 전체 청킹 작업 종료 =====")
print("이번 실행에서 처리한 회의록:", total_minutes)
print("이번 실행에서 생성한 청크:", f"{total_chunks:,}")
print("이번 실행 소요 시간:", round(total_elapsed / 60, 2), "분")

completed_files = list(OUTPUT_DIR.glob("*_chunked.csv"))

print("완료된 지역 파일:", len(completed_files))
print("전체 대상 지역 파일:", len(csv_files))

# ===== 전체 청킹 작업 종료 =====
# 이번 실행에서 처리한 회의록: 6007
# 이번 실행에서 생성한 청크: 402,683
# 이번 실행 소요 시간: 0.72 분
# 완료된 지역 파일: 17
# 전체 대상 지역 파일: 17


# 청킹 결과 평가
# 검증 항목	결과	판단
# 원본 회의록	6,007건	전체 처리
# 생성 청크	402,683개	정상 생성
# 지역별 CSV	17개	전체 완료
# 청킹 처리 시간	0.72분	양호
