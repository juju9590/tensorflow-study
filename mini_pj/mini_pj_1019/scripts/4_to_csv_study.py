# 폴더 생성, 파일 경로 결합 등에 사용하는 모듈
import os

# 특정 패턴에 해당하는 파일 목록을 찾는 모듈
import glob

# JSON 파일을 Python의 리스트·딕셔너리로 읽는 모듈
import json

# 정규표현식으로 특정 문자열을 찾거나 바꾸는 모듈
import re

# HTML을 분석하고 태그·본문을 추출하는 도구
from bs4 import BeautifulSoup

# 추출 결과를 표 형태로 정리하고 저장하는 도구
import pandas as pd

# 프로젝트의 경로 설정 등이 들어 있는 별도 Python 파일
# _constant.py 안에 org_dir, text_dir, all_rule_file이 정의되어 있어야 함
import _constant as constant


class DynamicCouncilPreprocessor:
    """
    의회별 HTML 분석 규칙을 적용해
    안건·발언자·발언 내용을 추출하는 클래스
    """

    def __init__(self, min_char_len: int = 30):
        """
        객체 생성 시 실행되는 초기 설정.

        min_char_len:
            정제한 발언을 저장하기 위한 최소 글자 수.
            기본값은 30자.
        """

        # 최소 발언 길이를 객체 내부에 저장
        self.min_char_len = min_char_len

        # 괄호 안의 의사 진행·행동 표현을 제거하기 위한 패턴
        #
        # \(       : 여는 괄호 (
        # [^)]*    : 닫는 괄호가 아닌 문자가 0개 이상
        # (?:...)  : 아래 단어 중 하나
        # \)       : 닫는 괄호 )
        #
        # 제거 예:
        # (10시 개의), (인사), (일동기립), (박수)
        self.formality_patterns = [
            r"\([^)]*(?:개식|폐식|개의|산회|인사|기립|묵념|박수|마이크)[^)]*\)",
        ]

        # 여러 패턴을 |로 연결
        # |는 정규표현식에서 '또는'을 의미
        #
        # re.compile:
        # 정규표현식을 미리 준비해 반복 사용할 수 있게 함
        #
        # re.MULTILINE:
        # ^와 $가 문자열 전체뿐 아니라 각 줄의 시작·끝에도 적용되게 함
        # 현재 패턴에는 ^와 $가 없어 직접적인 영향은 없음
        self.combined_regex = re.compile(
            "|".join(self.formality_patterns),
            re.MULTILINE
        )

    def fix_fragmented_spacing(self, text: str) -> str:
        """
        공백으로 나뉜 한 글자 토큰들을 이어 붙이는 함수.

        주의:
            실제 띄어쓰기 복원 모델은 아니며,
            한 글자 단어도 합쳐질 수 있는 규칙 기반 처리.
        """

        # 쉼표·마침표·물음표·느낌표 앞의 공백 제거
        #
        # 예:
        # "맞습니다 ." → "맞습니다."
        #
        # ([,.?!])는 구두점을 그룹으로 저장
        # \1은 그 그룹에서 찾은 구두점을 다시 넣는다는 뜻
        text = re.sub(r"\s+([,.?!])", r"\1", text)

        # 공백·줄바꿈 등을 기준으로 문자열을 나눔
        #
        # 예:
        # "의 원 여러분" → ["의", "원", "여러분"]
        tokens = text.split()

        # 최종적으로 정리한 토큰을 담을 리스트
        fixed_tokens = []

        # 연속된 한 글자 토큰을 임시로 붙여 둘 문자열
        temp_word = ""

        # 토큰을 앞에서부터 하나씩 확인
        for token in tokens:

            # 길이 판단을 위해 한글·영문·숫자만 남김
            # 실제 출력에서는 아래 pure_text가 아니라 원래 token을 사용
            pure_text = re.sub(
                r"[^가-힣A-Za-z0-9]",
                "",
                token
            )

            # 문자·숫자 기준 길이가 1 이하라면 임시 문자열에 누적
            #
            # 예:
            # "의" → "의"
            # "원" → "의원"
            #
            # 구두점만 있는 토큰도 길이 0이므로 이 조건에 들어감
            if len(pure_text) <= 1:
                temp_word += token

            else:
                # 두 글자 이상 토큰을 만났을 때
                # 앞에서 누적한 문자열이 있으면 먼저 결과에 추가
                if temp_word:
                    fixed_tokens.append(temp_word)

                    # 다음 누적을 위해 초기화
                    temp_word = ""

                # 현재 두 글자 이상 토큰을 결과에 추가
                fixed_tokens.append(token)

        # 마지막까지 남아 있는 한 글자 토큰 묶음을 결과에 추가
        if temp_word:
            fixed_tokens.append(temp_word)

        # 결과 토큰들을 공백 하나로 연결
        return " ".join(fixed_tokens)

    def clean_text(self, text: str) -> str:
        """
        추출한 발언의 의사 진행 표현과 공백을 정리하는 함수.
        """

        # 설정된 괄호 표현을 공백으로 바꿈
        cleaned = self.combined_regex.sub(" ", text)

        # 연속된 한 글자 토큰을 이어 붙임
        cleaned = self.fix_fragmented_spacing(cleaned)

        # 여러 공백·줄바꿈·탭을 공백 하나로 통일
        # 문자열 앞뒤 공백도 제거
        return re.sub(r"\s+", " ", cleaned).strip()

    def parse_speaker(self, raw_speaker: str) -> tuple:
        """
        발언자 표기를 직책·이름·유형으로 나누는 함수.

        반환:
            (직책, 이름, 발언자 유형)

        예:
            "○위원장 홍길동"
            → ("위원장", "홍길동", "의회")

        주의:
            현재 코드는 '직책 이름' 순서를 가정함.
        """

        # 문자열 시작 부분의 발언 표시 기호·공백 제거
        #
        # ^             : 문자열 시작
        # [○□■\-\s]+    : 해당 기호 또는 공백이 1개 이상
        text = re.sub(
            r"^[○□■\-\s]+",
            "",
            raw_speaker
        ).strip()

        # 공백 기준으로 분리
        tokens = text.split()

        if len(tokens) >= 2:
            # 첫 토큰은 직책으로 처리
            role = tokens[0]

            # 나머지 토큰을 붙여 이름으로 처리
            #
            # 예:
            # ["위원장", "최", "선", "국"]
            # → 직책 "위원장", 이름 "최선국"
            name = "".join(tokens[1:])

        elif len(tokens) == 1:
            # 토큰이 하나뿐이면 직책은 미상으로 처리
            role = "미상"

            # 유일한 토큰을 이름으로 사용
            name = tokens[0]

        else:
            # 아무 정보도 없으면 기본값 반환
            return "미상", "미상", "기타"

        # 직책에 아래 표현이 포함되면 의회 측으로 분류
        # 그 외는 집행부로 분류
        #
        # 주의:
        # 외부 전문가·참고인 등도 현재 규칙에서는 집행부가 될 수 있음
        speaker_type = (
            "의회"
            if any(
                role_word in role
                for role_word in [
                    "의원",
                    "의장",
                    "부의장",
                    "위원장",
                    "부위원장",
                    "위원"
                ]
            )
            else "집행부"
        )

        return role, name, speaker_type

    def process_record(
        self,
        record: dict,
        rule: dict,
        filename_info: str
    ) -> list:
        """
        회의록 한 건에서 발언들을 추출하는 함수.

        record:
            회의록 한 건의 JSON 객체.

        rule:
            해당 파일에 적용할 HTML 선택 규칙.

        filename_info:
            경고 메시지에 표시할 파일명.

        반환:
            발언별 딕셔너리가 들어 있는 리스트.
        """

        # 현재 회의록에서 추출한 발언을 저장할 리스트
        results = []

        # JSON 객체에서 HTML 본문 가져오기
        # 해당 키가 없으면 빈 문자열 사용
        html_content = record.get("MINTS_HTML", "")

        # 본문이 없으면 빈 결과 반환
        if not html_content:
            return results

        # HTML 문자열을 분석 가능한 객체로 변환
        soup = BeautifulSoup(html_content, "html.parser")

        # 지정한 태그 이름에 해당하는 요소를 제거
        #
        # 주의:
        # soup([...])는 CSS 선택자가 아니라 '태그 이름' 검색임.
        # 따라서 "ul.skipUl", "#topFunction", "#mySidenav2"는
        # 의도한 class·id 선택자로 동작하지 않음.
        #
        # table은 실제 표 내용까지 함께 제거함.
        for tag in soup([
            "script",
            "style",
            "form",
            "table",
            "ul.skipUl",
            "#topFunction",
            "#mySidenav2"
        ]):
            tag.decompose()

        # 규칙에서 안건 선택자를 읽음
        # 설정이 없으면 모든 div를 후보로 사용
        agenda_selector = rule.get("agenda", "div")

        # 규칙에서 발언자 선택자를 읽음
        # 설정이 없으면 span.spkrbld 사용
        speaker_selector = rule.get(
            "speaker",
            "span.spkrbld"
        )

        # 규칙에서 발언 본문 선택자를 읽음
        # 설정이 없으면 p 사용
        utterance_selector = rule.get("utterance", "p")

        try:
            # CSS 선택자로 발언자 요소들을 찾음
            speaker_tags = soup.select(speaker_selector)

            # CSS 선택자로 안건 요소들을 찾음
            agenda_tags = soup.select(agenda_selector)

        except Exception:
            # 선택자 오류 등이 발생하면 현재 회의록을 건너뜀
            return results

        # 설정된 규칙으로 발언자를 찾지 못한 경우
        if len(speaker_tags) == 0:
            print(
                f"  [경고] '{filename_info}'에서 "
                f"발언자({speaker_selector})를 찾지 못해 "
                f"기본값(span.spkrbld)으로 재시도합니다."
            )

            # 기본 발언자 선택자로 변경
            speaker_selector = "span.spkrbld"

            try:
                speaker_tags = soup.select(speaker_selector)

            except Exception:
                # 재시도 중 오류가 나면 추가 처리 없이 아래로 이동
                pass

            # 기본 선택자로도 찾지 못하면 현재 회의록 건너뜀
            if len(speaker_tags) == 0:
                print(
                    f"  [실패] 기본값으로도 "
                    f"'{filename_info}'의 발언자를 찾을 수 없습니다. "
                    f"건너뜁니다."
                )

                return results

        # 찾은 발언자 요소를 문서 순서대로 처리
        for spk_tag in speaker_tags:

            # 발언자 요소 내부 텍스트 추출
            # strip=True는 각 텍스트 조각 앞뒤 공백을 제거
            raw_speaker = spk_tag.get_text(strip=True)

            # 직책·이름·발언자 유형으로 분리
            role, name, speaker_type = self.parse_speaker(
                raw_speaker
            )

            # 안건을 못 찾았을 때 사용할 기본값
            agenda_title = "일반안건"

            # 현재 발언자보다 앞에 있는 요소들을
            # 가까운 것부터 역방향으로 탐색
            for prev in spk_tag.find_all_previous():

                # 앞에서 찾은 안건 후보에 해당하면
                if prev in agenda_tags:

                    # 해당 요소의 텍스트를 안건명으로 사용
                    # 최대 50자까지만 저장
                    agenda_title = prev.get_text(
                        strip=True
                    )[:50]

                    # 가장 가까운 후보 하나를 찾았으므로 종료
                    break

            # 정제된 발언 내용을 담을 변수
            utterance_cleaned = ""

            # 본문 선택자가 지정되어 있고 p가 아닌 경우에만 실행
            #
            # 현재 로직에서는 기본값 p일 때
            # 이 분기를 사용하지 않고 아래 부모 요소 방식으로 처리
            if utterance_selector and utterance_selector != "p":

                # 현재 발언자 다음에서 지정한 '태그 이름'을 찾음
                #
                # 주의:
                # find_next("div")는 태그 검색이지만,
                # find_next("div.content")는 CSS 선택자 검색이 아님.
                # 또한 다음 발언자의 영역까지 넘어갈 가능성이 있음.
                next_content = spk_tag.find_next(
                    utterance_selector
                )

                if next_content:
                    # 찾은 요소의 본문을 추출하고 정리
                    utterance_cleaned = self.clean_text(
                        next_content.get_text(strip=True)
                    )

            # 본문이 없거나 10자 미만이면 대체 방식 사용
            if not utterance_cleaned or len(utterance_cleaned) < 10:

                # 현재 발언자를 감싸는 가장 가까운 div를 찾음
                # div가 없으면 바로 위 부모 요소 사용
                parent_block = (
                    spk_tag.find_parent("div")
                    or spk_tag.parent
                )

                # 부모 본문에 발언자 이름이 중복 포함되지 않도록
                # 발언자 태그 자체를 HTML 트리에서 제거
                #
                # 주의:
                # 원본 파일은 바꾸지 않지만 현재 soup 객체는 변경됨.
                spk_tag.decompose()

                # 부모 요소 전체의 텍스트 추출
                #
                # 부모 범위가 크면 다른 발언까지 포함될 수 있음
                utterance_raw = (
                    parent_block.get_text(strip=True)
                    if parent_block
                    else ""
                )

                # 추출한 텍스트 정리
                utterance_cleaned = self.clean_text(
                    utterance_raw
                )

            # 최소 길이 이상인 발언만 저장
            #
            # 기본값 30자 미만의 짧은 답변·의결 문구는 제외될 수 있음
            if len(utterance_cleaned) >= self.min_char_len:

                results.append({
                    # 원본 날짜 그대로 저장
                    "date": record.get("MTG_DE", ""),

                    # 의회명 저장
                    "assembly": record.get("RASMBLY_NM", ""),

                    # 앞에서 찾은 안건명 저장
                    "agenda_title": agenda_title,

                    # 발언자 직책 저장
                    "speaker_role": role,

                    # 발언자 이름 저장
                    "speaker_name": name,

                    # 의회·집행부 등 유형 저장
                    "speaker_type": speaker_type,

                    # 정제한 발언 본문 저장
                    "utterance": utterance_cleaned
                })

        # 현재 회의록에서 추출한 모든 발언 반환
        return results


# 이 파일을 직접 실행할 때만 아래 작업 수행
# 다른 파일에서 import하면 실행되지 않음
if __name__ == "__main__":

    # 원본 JSON 폴더 경로
    input_dir = constant.org_dir

    # 정제 결과를 저장할 폴더 경로
    output_dir = constant.text_dir

    # 의회별 HTML 분석 규칙 JSON 파일 경로
    rules_file = constant.all_rule_file

    # 결과 폴더가 없으면 생성
    # 이미 있어도 오류를 발생시키지 않음
    os.makedirs(output_dir, exist_ok=True)

    # 의회별 분석 규칙을 딕셔너리로 읽음
    with open(rules_file, "r", encoding="utf-8") as f:
        parsing_rules = json.load(f)

    # 원본 폴더 바로 아래의 JSON 파일 목록 가져오기
    # 하위 폴더까지 검색하지는 않음
    json_files = glob.glob(
        os.path.join(input_dir, "*.json")
    )

    # 최소 발언 길이 30자로 전처리 객체 생성
    preprocessor = DynamicCouncilPreprocessor(
        min_char_len=30
    )

    # JSON 파일을 하나씩 처리
    for input_file in json_files:

        # 전체 경로에서 파일명만 추출
        # 예: C:/data/2025_인천.json → 2025_인천.json
        base_name = os.path.basename(input_file)

        # 확장자를 제거해 규칙 검색용 키 생성
        # 예: 2025_인천.json → 2025_인천
        file_key = os.path.splitext(base_name)[0]

        # 파일명에 해당하는 규칙을 가져옴
        # 없으면 기본 규칙 사용
        matched_rule = parsing_rules.get(
            file_key,
            {
                "agenda": "div",
                "speaker": "span.spkrbld",
                "utterance": "p"
            }
        )

        print(f"[{base_name}] 전처리 시작...")

        try:
            # 원본 JSON 파일 읽기
            # 회의록 객체들이 담긴 리스트라고 가정
            with open(
                input_file,
                "r",
                encoding="utf-8"
            ) as f:
                raw_data = json.load(f)

            # 현재 파일의 전체 발언을 담을 리스트
            all_utterances = []

            # 회의록 한 건씩 처리
            for record in raw_data:

                # 해당 회의록에서 발언 추출
                extracted = preprocessor.process_record(
                    record,
                    matched_rule,
                    base_name
                )

                # 추출된 발언들을 전체 리스트에 추가
                # append와 달리 리스트 내부 원소를 각각 추가
                all_utterances.extend(extracted)

            # 발언 딕셔너리 목록을 표로 변환
            # 발언 한 건이 한 행이 됨
            df_clean = pd.DataFrame(all_utterances)

            # 추출된 행이 있을 때만 병합·저장
            if not df_clean.empty:

                # 현재 행과 바로 이전 행의 이름·유형 비교
                #
                # 이름 또는 유형이 달라지면 True
                # 둘 다 같으면 False
                #
                # shift()는 값을 한 행 아래로 이동시켜
                # 현재 행에서 이전 행 값과 비교하게 함
                condition = (
                    (
                        df_clean["speaker_name"]
                        != df_clean["speaker_name"].shift()
                    )
                    |
                    (
                        df_clean["speaker_type"]
                        != df_clean["speaker_type"].shift()
                    )
                )

                # True를 만날 때마다 그룹 번호를 1씩 증가
                #
                # 예:
                # condition: True, False, True, False
                # group_id:     1,     1,    2,     2
                df_clean["group_id"] = condition.cumsum()

                # 같은 그룹의 연속 발언을 한 행으로 합침
                df_merged = (
                    df_clean
                    .groupby("group_id")
                    .agg({
                        # 그룹 첫 행의 날짜 사용
                        "date": "first",

                        # 그룹 첫 행의 의회명 사용
                        "assembly": "first",

                        # 그룹 첫 행의 안건명 사용
                        "agenda_title": "first",

                        # 그룹 첫 행의 직책 사용
                        "speaker_role": "first",

                        # 그룹 첫 행의 이름 사용
                        "speaker_name": "first",

                        # 그룹 첫 행의 유형 사용
                        "speaker_type": "first",

                        # 그룹 내 발언들을 공백으로 연결
                        "utterance": lambda x: " ".join(x)
                    })

                    # group_id 인덱스를 없애고
                    # 0부터 시작하는 새 인덱스 생성
                    .reset_index(drop=True)
                )

                # 저장할 파일 경로 생성
                output_path = os.path.join(
                    output_dir,
                    f"cleaned_{file_key}.txt"
                )

                # 탭으로 구분한 표 형태로 저장
                #
                # 확장자는 txt지만 실제 내용은 TSV 형식
                # sep="\t"       : 컬럼 구분자를 탭으로 지정
                # index=False    : DataFrame 행 번호는 저장하지 않음
                # utf-8-sig      : Excel에서 한글 인식에 도움
                df_merged.to_csv(
                    output_path,
                    sep="\t",
                    index=False,
                    encoding="utf-8-sig"
                )

                print(
                    f"  -> 완료: {len(df_merged)}건 추출 "
                    f"(연속 발언 병합됨) ({output_path})"
                )

            else:
                # 발언자 추출 실패 또는 길이 조건 등으로
                # 저장할 데이터가 없었던 경우
                print("  -> 추출된 유효 데이터 없음")

        except Exception as e:
            # 현재 파일 처리 중 오류가 발생하면 출력
            # 이후 다음 파일 처리를 계속함
            #
            # 현재 파일은 일부 처리됐더라도
            # 오류가 저장 이전에 발생하면 결과 파일이 생성되지 않음
            print(
                f"  -> 파일 처리 중 치명적 오류 발생: {e}"
            )