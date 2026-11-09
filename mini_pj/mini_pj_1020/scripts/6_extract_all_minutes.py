#⑥ 2025년 전체 회의록의 최소 필수 전처리


import json
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

# 1. 원본 JSON 폴더
input_dir = Path("C:/tensorflow_study/mini_pj/data/org2/2025_광역 회의록")

# 2. 결과 저장 폴더
output_dir = Path("C:/tensorflow_study/mini_pj/data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# 3. JSON 파일 가져오기
json_files = sorted(input_dir.glob("*.json"))

# 결과를 저장할 리스트
results = []

# 처리 현황
total_records = 0
skipped_records = 0

print("전체 JSON 파일:", len(json_files))


# 4. 지역별 JSON 처리
for file_path in json_files:

    print("\n지역:", file_path.stem)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 빈 파일 처리
    if not data:
        print("데이터 없음 → 건너뜀")
        continue

    # 회의록별 반복
    for i, record in enumerate(data):

        total_records += 1

        # 5. 메타데이터 추출
        date = record.get("MTG_DE")
        assembly = record.get("RASMBLY_NM")
        meeting_title = record.get("MINTS_NM")

        html = record.get("MINTS_HTML") or ""

        # HTML이 없으면 건너뛰기
        if not html.strip():
            skipped_records += 1
            continue

        # 6. HTML 본문 추출
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text("\n", strip=True)

        # 특수 공백 제거
        text = text.replace("\xa0", " ")

        if not text.strip():
            skipped_records += 1
            continue

        # 7. 데이터 구성
        row = {
            "date": date,
            "assembly": assembly,
            "agenda_title": None,
            "speaker_role": None,
            "speaker_name": None,
            "speaker_type": None,
            "utterance": text,
            "meeting_title": meeting_title,
            "source_id": f"{file_path.stem}_{i}",
            "source_file": file_path.name,
            "record_index": i
        }

        results.append(row)


# 8. 데이터프레임 변환
df = pd.DataFrame(results)

# 9. CSV 저장
output_path = output_dir / "2025_all_minutes_baseline.csv"

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

# 10. 결과 확인
print("\n===== 전체 전처리 결과 =====")
print("원본 레코드 수:", total_records)
print("본문 없는 레코드:", skipped_records)
print("저장된 데이터:", len(df))
print("데이터 크기:", df.shape)
print("저장 경로:", output_path)


# ===== 전체 전처리 결과 =====
# 원본 레코드 수: 6007
# 본문 없는 레코드: 0
# 저장된 데이터: 6007
# 데이터 크기: (6007, 11)
# 저장 경로: C:\tensorflow_study\mini_pj\data\processed\2025_all_minutes_baseline.csv
