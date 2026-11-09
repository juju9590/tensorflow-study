# 제이슨파일 원본에서 지역별  선택자 후보 탐색기 (최종 HTML 추출기가 아니라)

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from bs4 import BeautifulSoup, Tag


# ============================================================
# 1. 경로 및 설정
# ============================================================

input_dir = Path(
    r"C:\mini_pj\data\org2\2025_광역 회의록"
)

output_path = Path(
    r"C:\mini_pj\data\html_rules_candidates.json"
)

# 지역별 분석할 HTML 샘플 수
SAMPLE_COUNT = 5

# 후보별 최대 출력 개수
TOP_K = 5


# ============================================================
# 2. 텍스트 정리
# ============================================================

def clean_text(text):
    return " ".join(str(text).split())


# ============================================================
# 3. 지역별 샘플 선택
# ============================================================

def select_samples(data, sample_count=5):

    valid_records = [
        record
        for record in data
        if isinstance(record, dict)
        and isinstance(record.get("MINTS_HTML"), str)
        and record["MINTS_HTML"].strip()
    ]

    total = len(valid_records)

    if total <= sample_count:
        return valid_records

    # 전체 데이터에서 균등하게 선택
    positions = [
        round(i * (total - 1) / (sample_count - 1))
        for i in range(sample_count)
    ]

    return [valid_records[i] for i in sorted(set(positions))]


# ============================================================
# 4. HTML 선택자 생성
# ============================================================

def make_selectors(tag):

    selectors = set()

    if not isinstance(tag, Tag):
        return selectors

    tag_name = tag.name

    if not tag_name:
        return selectors

    # ID가 있는 경우
    tag_id = tag.get("id")

    if tag_id:

        # item1, item2 -> [id^="item"]
        match = re.match(r"^([a-zA-Z_-]+)\d+$", tag_id)

        if match:
            prefix = match.group(1)
            selectors.add(
                f'{tag_name}[id^="{prefix}"]'
            )
        else:
            selectors.add(f"#{tag_id}")

    # class가 있는 경우
    classes = tag.get("class", [])

    for cls in classes:

        if re.match(r"^[a-zA-Z_-][\w-]*$", cls):
            selectors.add(f"{tag_name}.{cls}")

    # 클래스 조합
    if len(classes) >= 2:
        selectors.add(
            tag_name + "." + ".".join(classes[:2])
        )

    # 특수 태그
    if tag_name in ["spk", "speaker"]:
        selectors.add(tag_name)

    return selectors


# ============================================================
# 5. HTML 항목별 특징 탐색
# ============================================================

def detect_fields(tag):

    text = clean_text(tag.get_text(" ", strip=True))

    if not text:
        return []

    classes = " ".join(tag.get("class", []))
    tag_id = tag.get("id", "")

    cls = classes.lower()
    tid = tag_id.lower()

    fields = []

    # -----------------------------------------
    # 안건 제목
    # -----------------------------------------

    agenda_pattern = bool(
        re.match(r"^\d+[.)]\s*.+", text)
    )

    agenda_class = any(
        keyword in cls or keyword in tid
        for keyword in [
            "agenda",
            "agnd",
            "item",
            "bill"
        ]
    )

    if (
        (agenda_pattern and len(text) < 300)
        or (agenda_class and len(text) < 300)
    ):
        fields.append("agenda_title")

    # -----------------------------------------
    # 발언자 표시
    # -----------------------------------------

    speaker_pattern = bool(
        re.match(r"^[○◯●]\s*.+", text)
    )

    speaker_class = any(
        keyword in cls
        for keyword in [
            "speaker",
            "spkr",
            "sman",
            "remark",
            "member"
        ]
    )

    # 긴 본문 전체가 발언자로 잡히는 것을 제한
    if (
        speaker_pattern and len(text) < 100
    ) or (
        speaker_class and len(text) < 1500
    ):
        fields.append("speaker_role")

    # -----------------------------------------
    # 발언자 이름
    # -----------------------------------------

    name_class = any(
        keyword in cls
        for keyword in [
            "profile",
            "speaker",
            "member",
            "spkr"
        ]
    )

    if (
        name_class and len(text) < 100
    ):
        fields.append("speaker_name")

    # -----------------------------------------
    # 발언자 유형
    # -----------------------------------------

    if (
        "member-speech" in cls
        or tag.has_attr("data-member_code")
    ):
        fields.append("speaker_type")

    # -----------------------------------------
    # 발언 내용
    # -----------------------------------------

    utterance_class = any(
        keyword in cls
        for keyword in [
            "speaker-block",
            "speech",
            "remark",
            "sman",
            "content",
            "text"
        ]
    )

    if utterance_class and len(text) >= 30:
        fields.append("utterance")

    return fields


# ============================================================
# 6. HTML 샘플 분석
# ============================================================

def analyze_html(html_text):

    soup = BeautifulSoup(
        html_text,
        "lxml"
    )

    candidates = defaultdict(Counter)

    for tag in soup.find_all(True):

        fields = detect_fields(tag)

        if not fields:
            continue

        selectors = make_selectors(tag)

        for field in fields:

            for selector in selectors:
                candidates[field][selector] += 1

    # 제주처럼 별도 발언 블록이 없는 구조의 후보
    for tag in soup.find_all("strong"):

        text = clean_text(
            tag.get_text(" ", strip=True)
        )

        if re.match(r"^[○◯●]", text):

            parent = tag.parent

            if parent and parent.get("id"):
                selector = (
                    f"#{parent['id']} > strong"
                )
            else:
                selector = "strong"

            candidates["speaker_role"][selector] += 1

            if tag.find("a"):
                candidates["speaker_name"][
                    selector + " > a"
                ] += 1

    # 제주 안건 구조: anguna 번호 뒤 strong
    for anchor in soup.find_all("a", id=True):

        if re.match(
            r"^anguna\d+$",
            anchor.get("id", "")
        ):

            next_tag = anchor.find_next_sibling()

            if next_tag and next_tag.name == "strong":

                selector = (
                    'a[id^="anguna"] + strong'
                )

                candidates["agenda_title"][selector] += 1

    return candidates


# ============================================================
# 7. 여러 샘플 결과 통합
# ============================================================

def merge_candidates(sample_results):

    merged = defaultdict(Counter)

    coverage = defaultdict(
        lambda: defaultdict(int)
    )

    for result in sample_results:

        for field, selectors in result.items():

            for selector, count in selectors.items():

                merged[field][selector] += count

                coverage[field][selector] += 1

    final = {}

    target_fields = [
        "agenda_title",
        "speaker_role",
        "speaker_name",
        "speaker_type",
        "utterance"
    ]

    for field in target_fields:

        ranked = sorted(
            merged[field].items(),
            key=lambda x: (
                coverage[field][x[0]],
                x[1]
            ),
            reverse=True
        )

        candidate_list = []

        for selector, count in ranked[:TOP_K]:

            candidate_list.append({
                "selector": selector,
                "matched_count": count,
                "sample_coverage": coverage[field][selector]
            })

        final[field] = {
            "recommended_selector": (
                candidate_list[0]["selector"]
                if candidate_list else None
            ),
            "candidates": candidate_list,
            "needs_review": True
        }

    return final


# ============================================================
# 8. 원본 JSON 파일 분석
# ============================================================

def analyze_json_file(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8-sig"
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):
        return {
            "status": "invalid_json_structure"
        }

    if not data:
        return {
            "status": "empty",
            "record_count": 0
        }

    samples = select_samples(
        data,
        SAMPLE_COUNT
    )

    sample_results = []

    for record in samples:

        html_text = record.get(
            "MINTS_HTML", ""
        )

        result = analyze_html(html_text)

        sample_results.append(result)

    html_candidates = merge_candidates(
        sample_results
    )

    # JSON에서 직접 추출 가능한 공통 항목
    rules = {
        "status": "candidate_only",
        "record_count": len(data),
        "sample_count": len(samples),

        "date": {
            "source": "json",
            "key": "MTG_DE"
        },

        "assembly": {
            "source": "json",
            "key": "RASMBLY_NM"
        },

        **html_candidates
    }

    return rules


# ============================================================
# 9. 18개 지역 전체 실행
# ============================================================

def main():

    files = sorted(
        input_dir.glob("*.json")
    )

    if not files:
        print("JSON 파일이 없습니다.")
        print("경로:", input_dir)
        return

    all_rules = {}

    print("=" * 60)
    print("지역별 HTML 구분자 후보 자동 탐색 시작")
    print("=" * 60)

    for file_path in files:

        region = file_path.stem

        print(f"\n[분석 시작] {region}")

        try:

            rules = analyze_json_file(
                file_path
            )

            all_rules[region] = rules

            if rules.get("status") == "empty":

                print("  데이터 없음 - 제외")
                continue

            if rules.get("status") != "candidate_only":

                print("  JSON 구조 확인 필요")
                continue

            for field in [
                "agenda_title",
                "speaker_role",
                "speaker_name",
                "speaker_type",
                "utterance"
            ]:

                selector = rules[field][
                    "recommended_selector"
                ]

                print(
                    f"  {field}: {selector}"
                )

        except Exception as e:

            print(f"  [오류] {e}")

            all_rules[region] = {
                "status": "error",
                "message": str(e)
            }

    # -----------------------------------------
    # 최종 JSON 저장
    # -----------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_rules,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\n" + "=" * 60)
    print("HTML 구분자 후보 추출 완료!")
    print("분석 파일:", len(files))
    print("저장 위치:", output_path)
    print("=" * 60)


# ============================================================
# 10. 실행
# ============================================================

if __name__ == "__main__":
    main()
