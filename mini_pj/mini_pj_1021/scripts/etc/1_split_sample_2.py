import json
import os
import glob
import random
import re
from bs4 import BeautifulSoup
import _constant as constant


def shorten_html(tag, max_text=80):
    """HTML 태그 구조를 유지하면서 긴 텍스트만 줄이기"""

    soup = BeautifulSoup(str(tag), "html.parser")

    for node in list(soup.find_all(string=True)):
        text = node.strip()

        if len(text) > max_text:
            node.replace_with(text[:max_text] + "...")

    return str(soup)


def extract_structure_samples(html_content):
    """위원회, 안건, 발언자 후보를 각각 추출"""

    soup = BeautifulSoup(html_content, "html.parser")

    # 불필요한 태그 제거
    for tag in soup(["script", "style", "head", "noscript", "form"]):
        tag.decompose()

    result = {
        "committee": [],
        "agenda": [],
        "speaker": []
    }

    # 1. 위원회 관련 후보
    for tag in soup.find_all(["h1", "h2", "h3", "p", "div"]):
        text = tag.get_text(" ", strip=True)

        # 너무 긴 부모 태그 제외
        if 0 < len(text) < 120 and re.search(r"위원회|회의록", text):
            result["committee"].append(shorten_html(tag))

            if len(result["committee"]) >= 5:
                break

    # 2. 안건 관련 후보
    agenda_selectors = [
        "#agenda-block",
        "#item-block",
        ".agenda",
        ".item",
        "[id*='agenda']",
        "[class*='agenda']"
    ]

    for selector in agenda_selectors:
        tags = soup.select(selector)

        for tag in tags[:2]:
            result["agenda"].append(shorten_html(tag))

        if result["agenda"]:
            break

    # 3. 발언자 관련 후보
    speaker_selectors = [
    ".speaker-block",
    "spk",
    "[class*='sManPV']",
    "[class*='sMan']",
    "[class*='speaker']",
    "[class*='speech']",
    "[class*='remark']"
]

    for selector in speaker_selectors:
        tags = soup.select(selector)

        for tag in tags[:5]:
            result["speaker"].append(shorten_html(tag))

        if result["speaker"]:
            break

    return result


def extract_samples_from_folder(
    input_dir="./input_data",
    output_dir="./sample_outputs",
    num_records=2
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

            records = data if isinstance(data, list) else [data]

            middle_records = (
                records[5:-5] if len(records) > 10
                else records
            )

            # HTML이 존재하는 레코드만 선택
            valid_records = [
                record for record in middle_records
                if isinstance(record, dict)
                and record.get("MINTS_HTML")
            ]

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

            output_file = os.path.join(
                output_dir,
                f"sample_{base_name}.txt"  #파일을 .txt 나 .json
            )

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    samples,
                    f,
                    ensure_ascii=False,
                    indent=4
                )

            print(f"[성공] {base_name}")

        except Exception as e:
            print(f"[오류] {base_name}: {e}")


if __name__ == "__main__":

    extract_samples_from_folder(
        input_dir=constant.org_dir,
        output_dir=constant.sample_dir,
        num_records=constant.records_num
    )