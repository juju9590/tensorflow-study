
import os
import glob
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
import _constant as constant
from ollama import chat


# ============================================================
# 1. 설정
# ============================================================

FIELDS = [
    "committee",
    "agenda",
    "speaker_block",
    "speaker",
]

CSS_SCHEMA = {
    "type": "object",
    "properties": {
        "committee": {
            "type": ["string", "null"]
        },
        "agenda": {
            "type": ["string", "null"]
        },
        "speaker_block": {
            "type": ["string", "null"]
        },
        "speaker": {
            "type": ["string", "null"]
        }
    },
    "required": FIELDS,
    "additionalProperties": False
}

# 스키마는 LLM이 결과를 반드시 지정된 4개 필드로 출력하도록 제한하는 데 사용
# Ollama의 공식 API는 JSON Schema를 전달하는 구조화된 출력 지원

# ============================================================
# 2. HTML 샘플 준비
# ============================================================

def prepare_sample(sample_file):
    """
    기존 샘플 추출 코드가 생성한 JSON 파일을 읽고
    LLM 분석에 필요한 HTML 구조만 정리한다.
    """

    with open(sample_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if isinstance(samples, dict):
        samples = [samples]

    sections = {
        "committee": [],
        "agenda": [],
        "speaker": []
    }

    for sample in samples:

        for section in sections:

            html_list = sample.get(section, [])

            if isinstance(html_list, str):
                html_list = [html_list]

            for html in html_list:

                if not isinstance(html, str):
                    continue

                soup = BeautifulSoup(html, "html.parser")

                # 필요 없는 태그 제거
                for tag in soup([
                    "script", "style", "noscript", "form"
                ]):
                    tag.decompose()

                # HTML 구조 자체는 유지
                sections[section].append(str(soup))

    # LLM에 전달할 샘플 개수 제한
    committee_html = sections["committee"][:2]
    agenda_html = sections["agenda"][:2]

    # 경기도의회 구조가 존재하는지 확인
    has_sman = any(
        re.search(r'class="sManPV\d+"', html)
        for html in sections["speaker"]
    )

    if has_sman:
        # 서로 다른 발언자 샘플 추출
        speaker_html = []
        seen_classes = set()

        for html in sections["speaker"]:

            soup = BeautifulSoup(html, "html.parser")

            tag = soup.find(
                class_=re.compile(r"sManPV\d+")
            )

            if tag is None:
                continue

            class_name = next(
                (
                    cls for cls in tag.get("class", [])
                    if re.fullmatch(r"sManPV\d+", cls)
                ),
                None
            )

            if class_name is None or class_name in seen_classes:
                continue

            seen_classes.add(class_name)
            speaker_html.append(html)

            if len(speaker_html) >= 5:
                break

    else:
        # 강원·울산 등 다른 HTML 구조
        speaker_html = sections["speaker"][:3]

    # CSS 검증에 사용할 HTML
    validation_html = "\n".join(
        committee_html + agenda_html + speaker_html
    )

    # LLM 프롬프트용 HTML
    prompt_html = f"""
[COMMITTEE HTML]
{chr(10).join(committee_html)}

[AGENDA HTML]
{chr(10).join(agenda_html)}

[SPEAKER HTML]
{chr(10).join(speaker_html)}
"""


    print("\n====== 경기도 HTML 샘플 ======")

    print("committee:", len(sections["committee"]))
    print("agenda:", len(sections["agenda"]))
    print("speaker:", len(sections["speaker"]))

    # 안건 샘플 존재 여부 확인
    has_agenda = len(sections["agenda"]) > 0

    return prompt_html, validation_html, has_agenda


# ============================================================
# 3. LLM 시스템 프롬프트
# ============================================================
SYSTEM_PROMPT = """
You are a deterministic CSS selector extractor.

TASK:
Find CSS selectors in the provided HTML.

You are NOT a document summarizer.

Return exactly these four keys:

{
    "committee": null,
    "agenda": null,
    "speaker_block": null,
    "speaker": null
}

DEFINITIONS:
committee: HTML element containing the committee title.
agenda: HTML elements containing individual agenda items.
speaker_block: HTML element containing a speaker and speech.
speaker: HTML element containing the speaker label.

STRICT RULES:
- Output only JSON.
- Every value must be a CSS selector string or null.
- Never extract meeting contents, names, dates or budgets.
- Never invent selectors.
- Prefer specific IDs and classes.
- Never use generic selectors such as div, p or span alone.
- If information is unavailable, return null.
- For agenda, prefer selectors scoped to the agenda container.
- For speaker, prefer direct child selectors scoped to speaker_block.
- Never use speaker speech paragraphs as agenda selectors.
- If AGENDA HTML is empty, return null for agenda.
- Speaker class names may contain dynamic numeric IDs.
- Generalize dynamic speaker classes using CSS attribute selectors.
- Example: .sManPV10062 -> [class^='sManPV']
- NEVER generate invalid wildcard selectors such as .sManPV*.
- NEVER append * directly to a CSS class name.
- Use CSS attribute selectors for dynamic class names.

VALID:
div[class^='sManPV']
div[class^='sManPV'] > p > span.bold

INVALID:
.sManPV*
.sManPV* span.bold
"""

# ============================================================
# 4. CSS 선택자 검증
# ============================================================

def normalize_selector(selector):

    if not isinstance(selector, str):
        return selector

    # .sManPV* -> div[class^='sManPV']
    selector = re.sub(
        r"\.sManPV\*",
        "div[class^='sManPV']",
        selector
    )

    return selector

def validate_selectors(html, selectors):
    """
    LLM이 생성한 CSS 선택자가
    실제 HTML에서 요소를 찾는지 검증한다.
    """

    soup = BeautifulSoup(html, "html.parser")

    validation_results = {}
    validated_rules = {}

    for field in FIELDS:

        selector = selectors.get(field)

        if not isinstance(selector, str) or not selector.strip():

            
            validated_rules[field] = None

            validation_results[field] = {
                "selector": None,
                "count": 0,
                "valid": False
            }

            continue

        selector = selector.strip()

        # CSS 선택자 문법 보정
        selector = normalize_selector(selector)

        try:
            elements = soup.select(selector)

            count = len(elements)
            valid = count > 0

            validation_results[field] = {
                "selector": selector,
                "count": count,
                "valid": valid,
                "examples": [
                    tag.get_text(" ", strip=True)[:80]
                    for tag in elements[:3]
                ]
            }

            # 실제 HTML에서 검색되지 않으면 null
            validated_rules[field] = selector if valid else None

        except Exception as e:

            validated_rules[field] = None

            validation_results[field] = {
                "selector": selector,
                "count": 0,
                "valid": False,
                "error": str(e)
            }

    return validated_rules, validation_results


# ============================================================
# 5. 로컬 LLM 분석
# ============================================================

def analyze_samples_with_local_llm(
    input_dir=constant.sample_dir,
    output_dir=constant.rules_output_dir,
    model_name=constant.make_model_name
):

    client = constant.client

    os.makedirs(output_dir, exist_ok=True)

    # 새로운 JSON 샘플 검색 (전체)
    sample_files = glob.glob(
        os.path.join(input_dir, "**", "*.txt"),
        recursive=True
    )

    # # 테스트: 강원특별자치도의회만 분석
    # sample_files = [
    #     file for file in sample_files
    #     if "2025_강원특별자치도의회" in os.path.basename(file)
    # ]

        # 테스트: 경기도의회만 분석
    sample_files = [
        file for file in sample_files
        if "2025_경기도의회" in os.path.basename(file)
    ]


    if not sample_files:
        print("분석할 JSON 샘플이 없습니다.")
        return

    print(f"총 {len(sample_files)}개 파일 분석 시작")

    for sample_file in sample_files:

        base_name = Path(sample_file).stem

        # sample_2025_강원특별자치도의회 -> 2025_강원특별자치도의회
        region_name = (
            base_name.removeprefix("sample_")
        )

        output_rule_file = os.path.join(
            output_dir,
            f"rule_{region_name}.json"
        )

        output_validation_file = os.path.join(
            output_dir,
            f"validation_{region_name}.json"
        )

        print(f"\n[{region_name}] LLM 분석 요청 중...")

        try:

            # ----------------------------------------
            # 1. 샘플 준비
            # ----------------------------------------

            prompt_html, validation_html, sections_have_agenda = prepare_sample(
                sample_file
            )

            # ----------------------------------------
            # 2. 사용자 프롬프트
            # ----------------------------------------

            user_prompt = f"""
Extract CSS selectors from the HTML below.

Return exactly these four fields:
committee, agenda, speaker_block, speaker.

IMPORTANT:
- Extract HTML CSS selectors only.
- Do not summarize meeting contents.
- Do not extract budget amounts.
- Use null when no selector is found.

HTML:
{prompt_html}
"""

            # ----------------------------------------
            # 3. Ollama API 호출 부분 교체
            # ----------------------------------------

            response = chat(
                model=model_name,                  # Qwen2.5-Coder-7B 사용
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                format=CSS_SCHEMA,                #JSON 필드 구조 제한
                options={
                    "temperature": 0,             #출력의 무작위성 최소화
                    "num_ctx": 8192,              #입력 컨텍스트 최대 8,192토큰
                    "num_predict": 400            #출력 토큰 최대 400개
                }
            )            

            # ----------------------------------------
            # 4. LLM 결과 파싱
            # ----------------------------------------
            # ----------------------------------------
            # 4. LLM 결과 파싱
            # ----------------------------------------

            # Ollama chat() 응답에서 내용 추출
            llm_result = response.message.content

            if not llm_result:
                raise ValueError("LLM 응답이 비어 있습니다.")

            # JSON 문자열을 Python 딕셔너리로 변환
            rule_json = json.loads(llm_result)

            # 샘플에 안건 HTML이 없다면 agenda는 null 처리
            if not sections_have_agenda:
                rule_json["agenda"] = None

            print("\n====== Qwen2.5-Coder 결과 ======")
            print(json.dumps(rule_json, ensure_ascii=False, indent=4))

            # JSON 객체인지 확인
            if not isinstance(rule_json, dict):
                raise ValueError("LLM 결과가 JSON 객체가 아닙니다.")

            # 필수 필드 정의
            required_fields = [
                "committee",
                "agenda",
                "speaker_block",
                "speaker"
            ]

            # 필수 필드 누락 확인
            missing_fields = [
                field for field in required_fields
                if field not in rule_json
            ]

            if missing_fields:
                raise ValueError(
                    f"LLM 응답에 필수 필드가 없습니다: {missing_fields}"
                )

            # 예상하지 못한 추가 필드 확인
            unexpected_fields = [
                field for field in rule_json
                if field not in required_fields
            ]

            if unexpected_fields:
                raise ValueError(
                    f"예상하지 못한 필드: {unexpected_fields}"
                )

            # 각 필드의 값이 CSS 문자열 또는 None인지 검사
            for field in required_fields:

                value = rule_json[field]

                if value is not None and not isinstance(value, str):
                    raise ValueError(
                        f"{field}의 값은 문자열 또는 None이어야 합니다."
                    )


            print("\n====== LLM 원본 결과 ======")
            print(json.dumps(rule_json, ensure_ascii=False, indent=4))

            print("\n====== HTML 샘플 확인 ======")
            print("샘플 길이:", len(validation_html))
            print("위원회 태그 존재:", "minutes-title-line" in validation_html)
            print("안건 태그 존재:", "agenda-block" in validation_html)
            print("발언 태그 존재:", "speaker-block" in validation_html)

            if not isinstance(rule_json, dict):
                raise ValueError("LLM 결과가 JSON 객체가 아닙니다.")

            # ----------------------------------------
            # 5. CSS 선택자 검증
            # ----------------------------------------

            validated_rules, validation_results = (
                validate_selectors(
                    validation_html,
                    rule_json
                )
            )

            # ----------------------------------------
            # 6. 최종 규칙 JSON
            # ----------------------------------------

            final_rule = {
                region_name: validated_rules
            }

            with open(
                output_rule_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    final_rule,
                    f,
                    ensure_ascii=False,
                    indent=4
                )

            # ----------------------------------------
            # 7. 검증 결과 별도 저장
            # ----------------------------------------

            with open(
                output_validation_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    validation_results,
                    f,
                    ensure_ascii=False,
                    indent=4
                )

            # ----------------------------------------
            # 8. 콘솔 출력
            # ----------------------------------------

            print("[분석 완료]")

            for field, result in validation_results.items():

                status = (
                    "성공"
                    if result["valid"]
                    else "확인 필요"
                )

                print(
                    f"  {field:15} "
                    f"{status:10} "
                    f"{result['selector']}"
                )

            print(f"저장: {output_rule_file}")

        except Exception as e:

            print(
                f"[오류] {region_name} 분석 실패: {e}"
            )

    print("\n모든 지역의 분석이 완료되었습니다.")


# ============================================================
# 6. 실행
# ============================================================

if __name__ == "__main__":

    analyze_samples_with_local_llm()



