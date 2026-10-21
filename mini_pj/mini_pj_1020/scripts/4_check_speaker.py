#3_3


from pathlib import Path
from bs4 import BeautifulSoup

# 1. HTML 샘플 파일 경로
file_path = Path(
    "C:/tensorflow_study/mini_pj/data/html_samples/"
    "2025_강원특별자치도의회/sample_1.html"
)

# 2. HTML 파일 읽기
with open(file_path, "r", encoding="utf-8") as f:
    html = f.read()

# 3. HTML 분석
soup = BeautifulSoup(html, "html.parser")

# 4. 불필요한 태그 제거
for tag in soup(["script", "style"]):
    tag.decompose()

# 5. HTML에서 텍스트 추출
text = soup.get_text("\n", strip=True)

# 6. 홈페이지 불필요 문구 제거
remove_text = "강원특별자치도의회 의원프로필"

if remove_text in text:
    text = text.split(remove_text)[0]

# 7. 결과 확인
# print("정리 후 글자 수:", len(text))

# print("\n===== 마지막 500자 =====")
# print(text[-500:])


# 3_2 Json 메타데이터 추출

import json

from pathlib import Path
from bs4 import BeautifulSoup

# 1. HTML 샘플 파일 경로
file_path = Path(
    "C:/tensorflow_study/mini_pj/data/html_samples/"
    "2025_강원특별자치도의회/sample_1.html"
)

# 2. HTML 파일 읽기
with open(file_path, "r", encoding="utf-8") as f:
    html = f.read()

# 3. HTML 분석
soup = BeautifulSoup(html, "html.parser")

# 4. 불필요한 태그 제거
for tag in soup(["script", "style"]):
    tag.decompose()

# 5. HTML에서 텍스트 추출
text = soup.get_text("\n", strip=True)

# 6. 홈페이지 불필요 문구 제거
remove_text = "강원특별자치도의회 의원프로필"

if remove_text in text:
    text = text.split(remove_text)[0]

# 7. 결과 확인
# print("정리 후 글자 수:", len(text))

# print("\n===== 마지막 500자 =====")
# print(text[-500:])


##### 제이슨 메타 데이터 추출

# 1. 강원도 원본 JSON 경로
json_path = "C:/tensorflow_study/mini_pj/data/org2/2025_광역 회의록/2025_강원특별자치도의회.json"

# 2. JSON 파일 읽기
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3. 첫 번째 회의록 선택
record = data[0]

# 4. 메타데이터 추출
date = record.get("MTG_DE")
assembly = record.get("RASMBLY_NM")
meeting_title = record.get("MINTS_NM")

# 5. 출력
# print("회의 날짜:", date)
# print("의회명:", assembly)
# print("회의록명:", meeting_title)

#####################################################
# 메타데이터 + 본문 통합
# # ③-5. 최종 데이터 구조 만들기

result = {
    "date": date,
    "assembly": assembly,
    "agenda_title": None,
    "speaker_role": None,
    "speaker_name": None,
    "speaker_type": None,
    "utterance": text,
    "meeting_title": meeting_title,
    "source_id": "2025_강원특별자치도의회_0"
}

# 결과 확인
print("\n===== 최종 데이터 구조 =====")

for key, value in result.items():

    if key == "utterance":
        print(f"{key}: {len(value)}자")

    else:
        print(f"{key}: {value}")


        # ④-1. 발언자 구분 후보 확인

lines = text.splitlines()

speaker_candidates = []

for line in lines:
    if line.startswith("○"):
        speaker_candidates.append(line)

print("\n===== 발언자 후보 =====")

for candidate in speaker_candidates:
    print(candidate)

# print("\n후보 개수:", len(speaker_candidates))

# ④-2. ○ 단독 구분자의 앞뒤 내용 확인

lines = text.splitlines()

for i, line in enumerate(lines):

    if line.strip() == "○":

        print("\n===== ○ 단독 발견 =====")

        start = max(0, i - 3)
        end = min(len(lines), i + 5)

        for j in range(start, end):
            print(f"{j}: {lines[j]}")

# ④-5. 발언자 정보 통합 및 순서 유지

lines = text.splitlines()

speakers = []

for i, line in enumerate(lines):

    line = line.strip()

    # ○으로 시작하지 않으면 건너뛰기
    if not line.startswith("○"):
        continue

    # 유형 ③ : ○ 단독
    if line == "○":

        if i + 2 >= len(lines):
            continue

        name = lines[i + 1].strip()
        role = lines[i + 2].strip()

        # 확인된 의원 발언 형태만 추출
        if role != "의원":
            continue

    # 유형 ①, ② : ○직책 또는 ○직책 이름
    else:

        speaker_text = line[1:].strip()

        # 발언자가 아닌 항목 제외
        if speaker_text.startswith((
            "출석", "청가", "의회사무처", "기록"
        )):
            continue

        parts = speaker_text.split()

        if len(parts) >= 2:
            role = " ".join(parts[:-1])
            name = parts[-1]

        else:
            if i + 1 >= len(lines):
                continue

            role = speaker_text
            name = lines[i + 1].strip()

    # 발언 순서대로 저장
    speakers.append({
        "line_index": i,
        "speaker_name": name,
        "speaker_role": role
    })


# 결과 확인
for n, speaker in enumerate(speakers, start=1):

    print(
        n,
        speaker["speaker_name"],
        speaker["speaker_role"],
        speaker["line_index"]
    )

# print("\n발언 구간 후보 개수:", len(speakers))

# ④-6. 발언자별 본문 분리 테스트

utterances = []

for i in range(len(speakers) - 1):

    current = speakers[i]
    next_speaker = speakers[i + 1]

    # 현재 발언 시작 위치
    start = current["line_index"]

    # 다음 발언 시작 위치
    end = next_speaker["line_index"]

    # 발언 원문 추출 (발언자 표시 포함)
    speech_text = "\n".join(lines[start:end]).strip()

    utterances.append({
        "speaker_name": current["speaker_name"],
        "speaker_role": current["speaker_role"],
        "utterance": speech_text,
        "start_line": start,
        "end_line": end
    })


# 추출 결과 확인
# print("분리된 발언 구간:", len(utterances))

# for i, item in enumerate(utterances, start=1):

#     print(f"\n===== 발언 {i} =====")
#     print("발언자:", item["speaker_name"])
#     print("직책:", item["speaker_role"])
#     print("시작 위치:", item["start_line"])
#     print("종료 위치:", item["end_line"])
#     print("본문 앞부분:", item["utterance"][:150])

# ④-7. 마지막 발언 종료 위치 확인

lines = text.splitlines()

start = speakers[-1]["line_index"]

print("마지막 발언 시작 위치:", start)

for i in range(start, len(lines)):

    # 공백을 제거하고 출석의원 포함 여부 확인
    line = lines[i].replace(" ", "")

    if "출석의원" in line:

        print("출석의원 명단 시작 위치:", i)

        print("\n===== 앞뒤 내용 =====")

        for j in range(max(start, i-10), min(len(lines), i+5)):
            print(j, lines[j])

        break


# ④-8. 마지막 발언까지 포함한 본문 분리

utterances = []

# 이번 샘플에서 확인한 부속자료 시작 위치
last_end = 332

for i in range(len(speakers)):

    current = speakers[i]

    start = current["line_index"]

    # 다음 발언자가 있는 경우
    if i < len(speakers) - 1:
        end = speakers[i + 1]["line_index"]

    # 마지막 발언자인 경우
    else:
        end = last_end

    speech_text = "\n".join(lines[start:end]).strip()

    utterances.append({
        "speaker_name": current["speaker_name"],
        "speaker_role": current["speaker_role"],
        "utterance": speech_text,
        "start_line": start,
        "end_line": end
    })


print("분리된 발언 구간:", len(utterances))

print("\n===== 마지막 발언 =====")
print(utterances[-1])

# ④-9. 발언자 표시 제거

for item in utterances:

    speech_lines = item["utterance"].splitlines()

    # 발언자 표시가 ○ 단독인 경우
    if speech_lines[0].strip() == "○":
        clean_lines = speech_lines[3:]

    # 발언자 표시가 ○직책 형태인 경우
    elif speech_lines[0].strip() == "○의장":
        clean_lines = speech_lines[2:]

    # 그 밖의 ○직책 이름 형태
    else:
        clean_lines = speech_lines[1:]

    item["utterance"] = "\n".join(clean_lines).strip()


# 결과 확인
for i, item in enumerate(utterances, start=1):

    print(f"\n===== 발언 {i} =====")
    print("발언자:", item["speaker_name"])
    print("직책:", item["speaker_role"])
    print("본문:", item["utterance"][:100])


# ④-10. 발언별 최종 데이터 구조 생성

final_data = []

for i, item in enumerate(utterances, start=1):

    row = {
        "date": date,
        "assembly": assembly,
        "agenda_title": None,
        "speaker_role": item["speaker_role"],
        "speaker_name": item["speaker_name"],
        "speaker_type": None,
        "utterance": item["utterance"],
        "meeting_title": meeting_title,
        "source_id": f"2025_강원특별자치도의회_0_{i}",
        "start_line": item["start_line"],
        "end_line": item["end_line"]
    }

    final_data.append(row)


# 결과 확인
print("최종 발언 데이터 개수:", len(final_data))

for row in final_data[:3]:
    print("\n", row["speaker_name"], row["speaker_role"])
    print("날짜:", row["date"])
    print("본문:", row["utterance"][:100])
    print("출처:", row["source_id"])

# ④-11. 발언 내용 특수 공백 정리

for row in final_data:

    row["utterance"] = row["utterance"].replace("\xa0", " ")


# 결과 확인
print("최종 데이터 개수:", len(final_data))

print("\n===== 마지막 발언 =====")
print("발언자:", final_data[-1]["speaker_name"])
print("본문:", final_data[-1]["utterance"][:200])

# ④-12. 최종 데이터 CSV 저장

import pandas as pd
from pathlib import Path

# 1. 저장 폴더
output_dir = Path("C:/tensorflow_study/mini_pj/data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. 리스트를 데이터프레임으로 변환
df = pd.DataFrame(final_data)

# 3. CSV 저장 경로
output_path = output_dir / "2025_강원특별자치도의회_sample_1.csv"

# 4. CSV 저장
df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

# 5. 결과 확인
print("\n===== CSV 저장 완료 =====")
print("저장 경로:", output_path)
print("데이터 크기:", df.shape)
print("컬럼명:", df.columns.tolist())