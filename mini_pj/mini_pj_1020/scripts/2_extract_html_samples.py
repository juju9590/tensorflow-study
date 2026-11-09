# 제이슨구조파악 이후 HTML샘플(3개)로 구분가능한 추출코드 파악


import json
from pathlib import Path

# 1. 원본 JSON 폴더 경로
input_dir = Path("C:/tensorflow_study/mini_pj/data/org2/2025_광역 회의록")

# 2. HTML 샘플 저장 폴더
output_dir = Path("C:/tensorflow_study/mini_pj/data/html_samples")
output_dir.mkdir(parents=True, exist_ok=True)

# 3. 모든 JSON 파일 가져오기
json_files = sorted(input_dir.glob("*.json"))

# 지역별 추출할 샘플 수
num_samples = 3

total_samples = 0

print(f"총 {len(json_files)}개 JSON 파일 검사 시작\n")

# 4. 지역별 JSON 파일 반복
for file_path in json_files:

    region = file_path.stem  # 확장자 .json 제거

    print("=" * 60)
    print(f"[지역] {region}")

    try:
        # JSON 로드
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 빈 JSON 처리
        if not data:
            print("데이터 없음 → 건너뜀")
            continue

        # 지역별 저장 폴더 생성
        region_dir = output_dir / region
        region_dir.mkdir(parents=True, exist_ok=True)

        # 샘플 3개 추출
        count = 0

        for i, record in enumerate(data[:num_samples], start=1):

            html = record.get("MINTS_HTML", "")

            # HTML이 비어 있으면 건너뜀
            if not html or not html.strip():
                print(f"샘플 {i}: HTML 없음")
                continue

            # HTML 저장
            output_path = region_dir / f"sample_{i}.html"

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

            count += 1
            total_samples += 1

            print(f"[저장 완료] {output_path.name}")

        print(f"추출 완료: {count}개")

    except Exception as e:
        print(f"[오류] {region}: {e}")

print("\n" + "=" * 60)
print(f"전체 HTML 샘플 추출 완료: {total_samples}개")


##################################################
# 코드 결과
##################################################
# 총 18개 JSON 파일 검사 시작

# ============================================================
# [지역] 2025_강원특별자치도의회
# [저장 완료] sample_1.html
# [저장 완료] sample_2.html
# [저장 완료] sample_3.html
# 추출 완료: 3개
# ============================================================
# [지역] 2025_경기도의회
# [저장 완료] sample_1.html
# [저장 완료] sample_2.html
# [저장 완료] sample_3.html
# 추출 완료: 3개
# ============================================================
# [지역] 2025_경상남도의회
# [저장 완료] sample_1.html
# [저장 완료] sample_2.html
# [저장 완료] sample_3.html
# 추출 완료: 3개
# ============================================================
# [지역] 2025_경상북도의회
# [저장 완료] sample_1.html
# [저장 완료] sample_2.html
# [저장 완료] sample_3.html
# 추출 완료: 3개
# ============================================================
# [지역] 2025_광주광역시의회
# [저장 완료] sample_1.html
# [저장 완료] sample_2.html
# [저장 완료] sample_3.html
# 추출 완료: 3개
# ============================================================
# [지역] 2025_대구광역시의회
# [저장 완료] sample_1.html
# [저장 완료] sample_2.html
# [저장 완료] sample_3.html
# 추출 완료: 3개
# ============================================================
# [지역] 2025_대전광역시의회
# [저장 완료] sample_1.html
# [저장 완료] sample_2.html
# [저장 완료] sample_3.html
# 추출 완료: 3개
# ============================================================
# [지역] 2025_부산광역시의회
# [저장 완료] sample_1.html
# [저장 완료] sample_2.html
# [저장 완료] sample_3.html
# 추출 완료: 3개
# ============================================================
# [지역] 2025_서울특별시의회
# [저장 완료] sample_1.html
# [저장 완료] sample_2.html
# [저장 완료] sample_3.html
# 추출 완료: 3개
# ============================================================
# [지역] 2025_세종특별자치시의회
# [저장 완료] sample_1.html
# [저장 완료] sample_2.html
# [저장 완료] sample_3.html
# 추출 완료: 3개
# ============================================================
# [지역] 2025_울산광역시의회
# [저장 완료] sample_1.html
# [저장 완료] sample_2.html
# [저장 완료] sample_3.html
# 추출 완료: 3개
# ============================================================
# [지역] 2025_인천광역시의회 (1)
# [저장 완료] sample_1.html
# [저장 완료] sample_2.html
# [저장 완료] sample_3.html
# 추출 완료: 3개
# ============================================================
# [지역] 2025_전남광주통합특별시의회
# 데이터 없음 → 건너뜀
# ============================================================
# [지역] 2025_전라남도의회
# [저장 완료] sample_1.html
# [저장 완료] sample_2.html
# [저장 완료] sample_3.html
# 추출 완료: 3개
# ============================================================
# [지역] 2025_전북특별자치도의회
# [저장 완료] sample_1.html
# [저장 완료] sample_2.html
# [저장 완료] sample_3.html
# 추출 완료: 3개
# ============================================================
# [지역] 2025_제주특별자치도의회
# [저장 완료] sample_1.html
# [저장 완료] sample_2.html
# [저장 완료] sample_3.html
# 추출 완료: 3개
# ============================================================
# [지역] 2025_충청남도의회
# [저장 완료] sample_1.html
# [저장 완료] sample_2.html
# [저장 완료] sample_3.html
# 추출 완료: 3개
# ============================================================
# [지역] 2025_충청북도의회
# [저장 완료] sample_1.html
# [저장 완료] sample_2.html
# [저장 완료] sample_3.html
# 추출 완료: 3개

# ============================================================
# 전체 HTML 샘플 추출 완료: 51개
##################################################





# 하나만 (코드)
# import json
# from pathlib import Path

# # 1. 원본 JSON 파일 경로
# file_path = "C:/mini_pj/data/org2/2025_광역 회의록/2025_강원특별자치도의회.json"

# # 2. HTML 샘플 저장 폴더
# output_dir = Path(r"C:\mini_pj\data\html_samples")
# output_dir.mkdir(parents=True, exist_ok=True)

# # 3. JSON 로드
# with open(file_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 4. 샘플 3개 추출
# for i, record in enumerate(data[:3], start=1):

#     html = record.get("MINTS_HTML", "")

#     # HTML 저장
#     output_path = output_dir / f"강원특별자치도의회_sample_{i}.html"

#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(html)

#     print(f"[저장 완료] {output_path}")

# print("HTML 샘플 추출 완료!")