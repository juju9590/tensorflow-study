# 샘플 추출을 위한 개선 초안


import os
import glob
import json
import random
import re

from bs4 import BeautifulSoup
import _constant as constant


# =====================================================
# 1. 긴 HTML 텍스트 축약
# =====================================================

def shorten_html(tag, max_text=120):

    soup = BeautifulSoup(str(tag), "html.parser")

    for node in list(soup.find_all(string=True)):

        text = node.strip()

        if len(text) > max_text:
            node.replace_with(text[:max_text] + "...")

    return str(soup)


# =====================================================
# 2. HTML 구조별 샘플 추출
# =====================================================

def extract_structure_samples(html_content):

    soup = BeautifulSoup(html_content, "html.parser")

    # 불필요한 태그 제거
    for tag in soup([
        "script", "style", "head", "noscript", "form"
    ]):
        tag.decompose()

    result = {
        "committee": [],
        "agenda": [],
        "speaker": []
    }

    # ---------------------------------------------
    # ① 위원회 후보
    # ---------------------------------------------

    for tag in soup.find_all(["h1", "h2", "h3", "h4", "div"]):

        text = tag.get_text(" ", strip=True)

        # 회의록 제목 관련 후보
        if (
            0 < len(text) < 100
            and re.search(r"위원회.*회의록|본회의.*회의록", text)
        ):
            result["committee"].append(
                shorten_html(tag)
            )

        if len(result["committee"]) >= 5:
            break

    # ---------------------------------------------
    # ② 안건 후보
    # ---------------------------------------------

    agenda_selectors = [
        "#agenda-block",
        "#item-block",
        "strong[id^='item']",
        "[id*='agenda']",
        "[class*='agenda']"
    ]

    seen_agenda = set()

    for selector in agenda_selectors:

        for tag in soup.select(selector):

            html = shorten_html(tag)

            if html in seen_agenda:
                continue

            seen_agenda.add(html)
            result["agenda"].append(html)

            if len(result["agenda"]) >= 5:
                break

        if len(result["agenda"]) >= 5:
            break

    # 구조적 후보가 없으면 텍스트 기반 후보 탐색
    if not result["agenda"]:

        for tag in soup.find_all(
            ["strong", "h3", "h4", "li", "p"]
        ):

            text = tag.get_text(" ", strip=True)

            if re.match(
                r"^(?:의사일정\s*)?제\s*\d+\s*항\b|^\d+\s*[.．]\s*\S",
                text
            ):
                result["agenda"].append(
                    shorten_html(tag)
                )

            if len(result["agenda"]) >= 5:
                break

    # ---------------------------------------------
    # ③ 발언 블록 후보
    # ---------------------------------------------

    speaker_selectors = [
        ".speaker-block",
        "spk",
        "div[class^='sManPV']",
        "div.remark",
        "[class*='speech']"
    ]

    speaker_tags = []

    for selector in speaker_selectors:

        tags = soup.select(selector)

        if tags:
            speaker_tags = tags
            break

    # 서로 다른 발언자 구조 우선 수집
    seen_speakers = set()

    for tag in speaker_tags:

        speaker_label = tag.find(
            ["strong", "span"],
            recursive=True
        )

        if speaker_label is None:
            continue

        speaker_name = speaker_label.get_text(
            " ", strip=True
        )

        # 같은 발언자 중복 방지
        if speaker_name in seen_speakers:
            continue

        seen_speakers.add(speaker_name)

        result["speaker"].append(
            shorten_html(tag)
        )

        if len(result["speaker"]) >= 5:
            break

    return result


# =====================================================
# 3. 지역별 샘플 파일 생성
# =====================================================

def extract_samples_from_folder(
    input_dir="./input_data",
    output_dir="./sample_outputs",
    num_records=3
):

    os.makedirs(output_dir, exist_ok=True)

    json_files = glob.glob(
        os.path.join(input_dir, "*.json")
    )

    if not json_files:
        print("JSON 파일이 없습니다.")
        return

    for input_file in json_files:

        base_name = os.path.splitext(
            os.path.basename(input_file)
        )[0]

        try:

            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            records = (
                data if isinstance(data, list)
                else [data]
            )

            # 앞뒤 5개 제외
            middle_records = (
                records[5:-5]
                if len(records) > 10
                else records
            )

            valid_records = [
                record
                for record in middle_records
                if isinstance(record, dict)
                and record.get("MINTS_HTML")
            ]

            random.seed(42)

            selected_records = random.sample(
                valid_records,
                min(num_records, len(valid_records))
            )

            samples = []

            for record in selected_records:

                structure = extract_structure_samples(
                    record["MINTS_HTML"]
                )

                samples.append(structure)

            # JSON 형식으로 저장
            output_file = os.path.join(
                output_dir,
                f"sample_{base_name}.json"
            )

            with open(
                output_file, "w", encoding="utf-8"
            ) as f:

                json.dump(
                    samples,
                    f,
                    ensure_ascii=False,
                    indent=4
                )

            print(f"[성공] {base_name}")

        except Exception as e:

            print(f"[오류] {base_name}: {e}")


# =====================================================
# 4. 실행
# =====================================================

if __name__ == "__main__":

    extract_samples_from_folder(
        input_dir=constant.org_dir,
        output_dir=constant.sample_dir,
        num_records=constant.records_num
    )
