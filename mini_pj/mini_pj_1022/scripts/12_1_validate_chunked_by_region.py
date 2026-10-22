 # 항목	검증 내용
# 파일 수	2024년 17개, 2025년 17개
# 청크 수	2024년 281,628개, 2025년 402,683개
# 컬럼 구조	34개 CSV의 컬럼 일치 여부
# 필수 메타데이터	date, assembly, source_id, chunk_id, chunk_index, text
# 청크 ID	전체 684,311개 중 중복 여부
# 원본 연결	chunk_id와 source_id 연결 관계
# 청크 길이	1~1,000자
# 연도·지역	파일명과 데이터 내부 정보 일치 여부


from pathlib import Path
import pandas as pd

# ============================================================
# 1. 경로 설정
# ============================================================
base_dir = Path(
    r"C:.\mini_pj_1021\data\processed\chunked_by_region"
)

# 실제 경로가 다르면 위 경로만 수정

year_folders = {
    2024: base_dir / "2024_광역회의록",
    2025: base_dir / "2025_광역회의록"
}

expected_counts = {
    2024: 281628,
    2025: 402683
}

required_cols = [
    "date",
    "assembly",
    "source_id",
    "chunk_id",
    "chunk_index",
    "text"
]

# ============================================================
# 2. 검증 준비
# ============================================================
seen_ids = set()
reference_cols = None

year_counts = {2024: 0, 2025: 0}
total_rows = 0
errors = []
results = []

# ============================================================
# 3. 지역별 파일 검증
# ============================================================
for year, folder in year_folders.items():

    print(f"\n===== {year}년 검증 =====")

    if not folder.exists():
        errors.append(f"폴더 없음: {folder}")
        continue

    files = sorted(folder.glob("*_chunked.csv"))

    print("지역 파일 수:", len(files))

    if len(files) != 17:
        errors.append(
            f"{year}년 파일 수 오류: {len(files)}개"
        )

    for file_path in files:

        file_rows = 0
        file_errors = 0

        # 파일명에서 지역 이름 추출
        region_name = file_path.stem.replace(
            f"{year}_", ""
        ).replace("_chunked", "")

        try:
            reader = pd.read_csv(
                file_path,
                chunksize=10000,
                dtype={
                    "date": "string",
                    "source_id": "string",
                    "chunk_id": "string",
                    "assembly": "string",
                    "text": "string"
                },
                encoding="utf-8-sig"
            )

            for batch in reader:

                # ① 컬럼 구조 검증
                cols = batch.columns.tolist()

                if reference_cols is None:
                    reference_cols = cols

                if set(cols) != set(reference_cols):
                    errors.append(
                        f"컬럼 불일치: {file_path.name}"
                    )
                    file_errors += 1
                    break

                # ② 필수 컬럼 검증
                missing_cols = set(required_cols) - set(cols)

                if missing_cols:
                    errors.append(
                        f"필수 컬럼 누락: {file_path.name}"
                    )
                    file_errors += 1
                    break

                # ③ 필수값 결측
                if batch[required_cols].isna().any().any():
                    errors.append(
                        f"필수값 결측: {file_path.name}"
                    )
                    file_errors += 1

                # ④ 빈 청크 및 길이
                lengths = batch["text"].str.len()

                if batch["text"].str.strip().eq("").any():
                    errors.append(
                        f"빈 청크: {file_path.name}"
                    )
                    file_errors += 1

                if lengths.gt(1000).any():
                    errors.append(
                        f"1000자 초과: {file_path.name}"
                    )
                    file_errors += 1

                # ⑤ 연도 확인
                if not batch["source_id"].str.startswith(
                    f"{year}_", na=False
                ).all():
                    errors.append(
                        f"source_id 연도 오류: {file_path.name}"
                    )
                    file_errors += 1

                if not batch["date"].str.startswith(
                    str(year), na=False
                ).all():
                    errors.append(
                        f"date 연도 오류: {file_path.name}"
                    )
                    file_errors += 1

                # ⑥ 지역 확인
                if not batch["assembly"].eq(
                    region_name
                ).all():
                    errors.append(
                        f"assembly 지역 불일치: {file_path.name}"
                    )
                    file_errors += 1

                # ⑦ chunk_id 중복 검증
                ids = batch["chunk_id"].dropna().tolist()

                if len(ids) != len(set(ids)):
                    errors.append(
                        f"배치 내부 chunk_id 중복: {file_path.name}"
                    )
                    file_errors += 1

                if not seen_ids.isdisjoint(ids):
                    errors.append(
                        f"전체 chunk_id 중복: {file_path.name}"
                    )
                    file_errors += 1

                seen_ids.update(ids)

                # ⑧ chunk_id와 source_id 관계
                expected_ids = (
                    batch["source_id"]
                    + "_"
                    + batch["chunk_index"].astype("Int64").astype(str)
                )

                if not batch["chunk_id"].eq(
                    expected_ids
                ).all():
                    errors.append(
                        f"chunk_id 연결 오류: {file_path.name}"
                    )
                    file_errors += 1

                file_rows += len(batch)

        except Exception as e:
            errors.append(
                f"읽기 오류: {file_path.name} / {e}"
            )
            file_errors += 1

        year_counts[year] += file_rows
        total_rows += file_rows

        results.append({
            "year": year,
            "region": region_name,
            "rows": file_rows,
            "errors": file_errors
        })

        print(
            f"{file_path.stem}: "
            f"{file_rows:,}개 / 오류 {file_errors}건"
        )

# ============================================================
# 4. 전체 건수 검증
# ============================================================
for year, expected in expected_counts.items():

    actual = year_counts[year]

    if actual != expected:
        errors.append(
            f"{year}년 청크 수 불일치: "
            f"{actual:,} / 예상 {expected:,}"
        )

expected_total = sum(expected_counts.values())

if total_rows != expected_total:
    errors.append(
        f"전체 청크 수 불일치: "
        f"{total_rows:,} / 예상 {expected_total:,}"
    )

# ============================================================
# 5. 결과 출력
# ============================================================
print("\n===== 전체 검증 결과 =====")

print("2024년 청크:", f"{year_counts[2024]:,}")
print("2025년 청크:", f"{year_counts[2025]:,}")
print("전체 청크:", f"{total_rows:,}")
print("고유 chunk_id:", f"{len(seen_ids):,}")
print("검증 오류:", len(errors))

if errors:
    print("\n===== 오류 목록 =====")

    for error in errors[:30]:
        print("-", error)

    print("\n최종 판단: 검토 필요")

else:
    print("\n최종 판단: 전체 구조 검증 통과")
    print("다음 단계: 지역별 배치 임베딩 진행")



# ===== 2024년 검증 =====
# 지역 파일 수: 17
# 2024_강원특별자치도의회_chunked: 15,661개 / 오류 0건
# 2024_경기도의회_chunked: 33,851개 / 오류 0건
# 2024_경상남도의회_chunked: 17,882개 / 오류 0건
# 2024_경상북도의회_chunked: 10,861개 / 오류 0건
# 2024_광주광역시의회_chunked: 9,537개 / 오류 0건
# 2024_대구광역시의회_chunked: 10,323개 / 오류 0건
# 2024_대전광역시의회_chunked: 8,254개 / 오류 0건
# 2024_부산광역시의회_chunked: 21,937개 / 오류 0건
# 2024_서울특별시의회_chunked: 30,349개 / 오류 0건
# 2024_세종특별자치시의회_chunked: 14,795개 / 오류 0건
# 2024_울산광역시의회_chunked: 7,634개 / 오류 0건
# 2024_인천광역시의회_chunked: 10,195개 / 오류 0건
# 2024_전라남도의회_chunked: 27,293개 / 오류 0건
# 2024_전북특별자치도의회_chunked: 11,773개 / 오류 0건
# 2024_제주특별자치도의회_chunked: 17,780개 / 오류 0건
# 2024_충청남도의회_chunked: 20,835개 / 오류 0건
# 2024_충청북도의회_chunked: 12,668개 / 오류 0건

# ===== 2025년 검증 =====
# 지역 파일 수: 17
# 2025_강원특별자치도의회_chunked: 35,649개 / 오류 0건
# 2025_경기도의회_chunked: 23,798개 / 오류 0건
# 2025_경상남도의회_chunked: 17,792개 / 오류 0건
# 2025_경상북도의회_chunked: 23,761개 / 오류 0건
# 2025_광주광역시의회_chunked: 8,357개 / 오류 0건
# 2025_대구광역시의회_chunked: 16,412개 / 오류 0건
# 2025_대전광역시의회_chunked: 8,568개 / 오류 0건
# 2025_부산광역시의회_chunked: 17,603개 / 오류 0건
# 2025_서울특별시의회_chunked: 61,760개 / 오류 0건
# 2025_세종특별자치시의회_chunked: 12,631개 / 오류 0건
# 2025_울산광역시의회_chunked: 15,823개 / 오류 0건
# 2025_인천광역시의회_chunked: 26,034개 / 오류 0건
# 2025_전라남도의회_chunked: 43,577개 / 오류 0건
# 2025_전북특별자치도의회_chunked: 14,569개 / 오류 0건
# 2025_제주특별자치도의회_chunked: 26,375개 / 오류 0건
# 2025_충청남도의회_chunked: 38,087개 / 오류 0건
# 2025_충청북도의회_chunked: 11,887개 / 오류 0건

# ===== 전체 검증 결과 =====
# 2024년 청크: 281,628
# 2025년 청크: 402,683
# 전체 청크: 684,311
# 고유 chunk_id: 684,311
# 검증 오류: 0

# 최종 판단: 전체 구조 검증 통과
# 다음 단계: 지역별 배치 임베딩 진행
