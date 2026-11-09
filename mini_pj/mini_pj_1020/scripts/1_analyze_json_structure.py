# 1. 원본 JSON 구조 확인 코드


import json
from pathlib import Path

# 1. 원본 JSON 폴더 경로
# input_dir = Path("C:/mini_pj/data/org2/2025_광역 회의록") #학원
input_dir = Path("C:/tensorflow_study/mini_pj/data/org2/2025_광역 회의록") #집

# data\org2\2025_광역 회의록

# 2. 폴더 내 모든 JSON 파일 가져오기
json_files = sorted(input_dir.glob("*.json"))

print(f"총 {len(json_files)}개 JSON 파일 검사 시작\n")

# 3. 지역별 JSON 구조 확인
for file_path in json_files:

    print("=" * 60)
    print(f"파일명: {file_path.name}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print("데이터 타입:", type(data).__name__)

        if isinstance(data, dict):

            print("최상위 키:", list(data.keys()))

            for key, value in data.items():
                print(f"  {key}: {type(value).__name__}")

        elif isinstance(data, list):

            print("전체 데이터 개수:", len(data))

            if data:
                print("첫 번째 데이터 타입:", type(data[0]).__name__)

                if isinstance(data[0], dict):

                    print("첫 번째 데이터의 키:", list(data[0].keys()))

                    for key, value in data[0].items():
                        print(f"  {key}: {type(value).__name__}")

        else:
            print("확인되지 않은 JSON 구조입니다.")

    except Exception as e:
        print(f"[오류] {file_path.name}: {e}")

print("\n전체 JSON 구조 검사 완료!")


###############################################
# 코드 결과
###############################################

# 총 18개 JSON 파일 검사 시작

# ============================================================
# 파일명: 2025_강원특별자치도의회.json
# 데이터 타입: list
# 전체 데이터 개수: 624
# 첫 번째 데이터 타입: dict
# 첫 번째 데이터의 키: ['MTG_DE', 'RASMBLY_NM', 'MINTS_HTML', 'MTGNM', 'MINTS_NM']
#   MTG_DE: str
#   RASMBLY_NM: str
#   MINTS_HTML: str
#   MTGNM: str
#   MINTS_NM: str
# ============================================================
# 파일명: 2025_경기도의회.json
# 데이터 타입: list
# 전체 데이터 개수: 274
# 첫 번째 데이터 타입: dict
# 첫 번째 데이터의 키: ['MTG_DE', 'RASMBLY_NM', 'MINTS_HTML', 'MTGNM', 'MINTS_NM']
#   MTG_DE: str
#   RASMBLY_NM: str
#   MINTS_HTML: str
#   MTGNM: str
#   MINTS_NM: str
# ============================================================
# 파일명: 2025_경상남도의회.json
# 데이터 타입: list
# 전체 데이터 개수: 241
# 첫 번째 데이터 타입: dict
# 첫 번째 데이터의 키: ['MTG_DE', 'RASMBLY_NM', 'MINTS_HTML', 'MTGNM', 'MINTS_NM']
#   MTG_DE: str
#   RASMBLY_NM: str
#   MINTS_HTML: str
#   MTGNM: str
#   MINTS_NM: str
# ============================================================
# 파일명: 2025_경상북도의회.json
# 데이터 타입: list
# 전체 데이터 개수: 404
# 첫 번째 데이터 타입: dict
# 첫 번째 데이터의 키: ['MTG_DE', 'RASMBLY_NM', 'MINTS_HTML', 'MTGNM', 'MINTS_NM']
#   MTG_DE: str
#   RASMBLY_NM: str
#   MINTS_HTML: str
#   MTGNM: str
#   MINTS_NM: str
# ============================================================
# 파일명: 2025_광주광역시의회.json
# 데이터 타입: list
# 전체 데이터 개수: 163
# 첫 번째 데이터 타입: dict
# 첫 번째 데이터의 키: ['MTG_DE', 'RASMBLY_NM', 'MINTS_HTML', 'MTGNM', 'MINTS_NM']
#   MTG_DE: str
#   RASMBLY_NM: str
#   MINTS_HTML: str
#   MTGNM: str
#   MINTS_NM: str
# ============================================================
# 파일명: 2025_대구광역시의회.json
# 데이터 타입: list
# 전체 데이터 개수: 285
# 첫 번째 데이터 타입: dict
# 첫 번째 데이터의 키: ['MTG_DE', 'RASMBLY_NM', 'MINTS_HTML', 'MTGNM', 'MINTS_NM']
#   MTG_DE: str
#   RASMBLY_NM: str
#   MINTS_HTML: str
#   MTGNM: str
#   MINTS_NM: str
# ============================================================
# 파일명: 2025_대전광역시의회.json
# 데이터 타입: list
# 전체 데이터 개수: 159
# 첫 번째 데이터 타입: dict
# 첫 번째 데이터의 키: ['MTG_DE', 'RASMBLY_NM', 'MINTS_HTML', 'MTGNM', 'MINTS_NM']
#   MTG_DE: str
#   RASMBLY_NM: str
#   MINTS_HTML: str
#   MTGNM: str
#   MINTS_NM: str
# ============================================================
# 파일명: 2025_부산광역시의회.json
# 데이터 타입: list
# 전체 데이터 개수: 238
# 첫 번째 데이터 타입: dict
# 첫 번째 데이터의 키: ['MTG_DE', 'RASMBLY_NM', 'MINTS_HTML', 'MTGNM', 'MINTS_NM']
#   MTG_DE: str
#   RASMBLY_NM: str
#   MINTS_HTML: str
#   MTGNM: str
#   MINTS_NM: str
# ============================================================
# 파일명: 2025_서울특별시의회.json
# 데이터 타입: list
# 전체 데이터 개수: 715
# 첫 번째 데이터 타입: dict
# 첫 번째 데이터의 키: ['MTG_DE', 'RASMBLY_NM', 'MINTS_HTML', 'MTGNM', 'MINTS_NM']
#   MTG_DE: str
#   RASMBLY_NM: str
#   MINTS_HTML: str
#   MTGNM: str
#   MINTS_NM: str
# ============================================================
# 파일명: 2025_세종특별자치시의회.json
# 데이터 타입: list
# 전체 데이터 개수: 172
# 첫 번째 데이터 타입: dict
# 첫 번째 데이터의 키: ['MTG_DE', 'RASMBLY_NM', 'MINTS_HTML', 'MTGNM', 'MINTS_NM']
#   MTG_DE: str
#   RASMBLY_NM: str
#   MINTS_HTML: str
#   MTGNM: str
#   MINTS_NM: str
# ============================================================
# 파일명: 2025_울산광역시의회.json
# 데이터 타입: list
# 전체 데이터 개수: 358
# 첫 번째 데이터 타입: dict
# 첫 번째 데이터의 키: ['MTG_DE', 'RASMBLY_NM', 'MINTS_HTML', 'MTGNM', 'MINTS_NM']
#   MTG_DE: str
#   RASMBLY_NM: str
#   MINTS_HTML: str
#   MTGNM: str
#   MINTS_NM: str
# ============================================================
# 파일명: 2025_인천광역시의회 (1).json
# 데이터 타입: list
# 전체 데이터 개수: 455
# 첫 번째 데이터 타입: dict
# 첫 번째 데이터의 키: ['MTG_DE', 'RASMBLY_NM', 'MINTS_HTML', 'MTGNM', 'MINTS_NM']
#   MTG_DE: str
#   RASMBLY_NM: str
#   MINTS_HTML: str
#   MTGNM: str
#   MINTS_NM: str
# ============================================================
# 파일명: 2025_전남광주통합특별시의회.json
# 데이터 타입: list
# 전체 데이터 개수: 0
# ============================================================
# 파일명: 2025_전라남도의회.json
# 데이터 타입: list
# 전체 데이터 개수: 653
# 첫 번째 데이터 타입: dict
# 첫 번째 데이터의 키: ['MTG_DE', 'RASMBLY_NM', 'MINTS_HTML', 'MTGNM', 'MINTS_NM']
#   MTG_DE: str
#   RASMBLY_NM: str
#   MINTS_HTML: str
#   MTGNM: str
#   MINTS_NM: str
# ============================================================
# 파일명: 2025_전북특별자치도의회.json
# 데이터 타입: list
# 전체 데이터 개수: 285
# 첫 번째 데이터 타입: dict
# 첫 번째 데이터의 키: ['MTG_DE', 'RASMBLY_NM', 'MINTS_HTML', 'MTGNM', 'MINTS_NM']
#   MTG_DE: str
#   RASMBLY_NM: str
#   MINTS_HTML: str
#   MTGNM: str
#   MINTS_NM: str
# ============================================================
# 파일명: 2025_제주특별자치도의회.json
# 데이터 타입: list
# 전체 데이터 개수: 398
# 첫 번째 데이터 타입: dict
# 첫 번째 데이터의 키: ['MTG_DE', 'RASMBLY_NM', 'MINTS_HTML', 'MTGNM', 'MINTS_NM']
#   MTG_DE: str
#   RASMBLY_NM: str
#   MINTS_HTML: str
#   MTGNM: str
#   MINTS_NM: str
# ============================================================
# 파일명: 2025_충청남도의회.json
# 데이터 타입: list
# 전체 데이터 개수: 423
# 첫 번째 데이터 타입: dict
# 첫 번째 데이터의 키: ['MTG_DE', 'RASMBLY_NM', 'MINTS_HTML', 'MTGNM', 'MINTS_NM']
#   MTG_DE: str
#   RASMBLY_NM: str
#   MINTS_HTML: str
#   MTGNM: str
#   MINTS_NM: str
# ============================================================
# 파일명: 2025_충청북도의회.json
# 데이터 타입: list
# 전체 데이터 개수: 160
# 첫 번째 데이터 타입: dict
# 첫 번째 데이터의 키: ['MTG_DE', 'RASMBLY_NM', 'MINTS_HTML', 'MTGNM', 'MINTS_NM']
#   MTG_DE: str
#   RASMBLY_NM: str
#   MINTS_HTML: str
#   MTGNM: str
#   MINTS_NM: str

# 전체 JSON 구조 검사 완료!

###############################################




# # 하나의 제이슨 분석 코드
# import json

# file_path = "C:/mini_pj/data/org2/2025_광역 회의록/2025_강원특별자치도의회.json"

# # C:\TensorFlow_Study\mini_pj\data\org2\2025_광역 회의록\2025_강원특별자치도의회.json

# with open(file_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# print("데이터 타입:", type(data))

# if isinstance(data, dict):
#     print("최상위 키:", list(data.keys()))

#     for key, value in data.items():
#         print(f"{key}: {type(value).__name__}")

# elif isinstance(data, list):
#     print("전체 데이터 개수:", len(data))

#     if data:
#         print("첫 번째 데이터 타입:", type(data[0]))

#         if isinstance(data[0], dict):
#             print("첫 번째 데이터의 키:", list(data[0].keys()))

#             for key, value in data[0].items():
#                 print(f"{key}: {type(value).__name__}")


