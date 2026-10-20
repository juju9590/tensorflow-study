import os
import glob
import json
import re
from bs4 import BeautifulSoup
import pandas as pd
import _constant as constant

class DynamicCouncilPreprocessor:
    def __init__(self, min_char_len: int = 30):
        self.min_char_len = min_char_len
        self.formality_patterns = [
            r"\([^\)]*(?:개식|폐식|개의|산회|인사|기립|묵념|박수|마이크)[^\)]*\)",
        ]
        self.combined_regex = re.compile("|".join(self.formality_patterns), re.MULTILINE)

    def fix_fragmented_spacing(self, text: str) -> str:
        """1글자씩 쪼개진 텍스트('의 원 여 러 분')를 데이터 손실 없이 병합합니다."""
        text = re.sub(r'\s+([,.\?!])', r'\1', text) # 구두점 앞 공백 제거
        tokens = text.split()
        fixed_tokens = []
        temp_word = ""
        
        for token in tokens:
            pure_text = re.sub(r'[^가-힣A-Za-z0-9]', '', token)
            if len(pure_text) <= 1:
                temp_word += token
            else:
                if temp_word:
                    fixed_tokens.append(temp_word)
                    temp_word = ""
                fixed_tokens.append(token)
                
        if temp_word:
            fixed_tokens.append(temp_word)
            
        return " ".join(fixed_tokens)

    def clean_text(self, text: str) -> str:
        cleaned = self.combined_regex.sub(" ", text)
        cleaned = self.fix_fragmented_spacing(cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def parse_speaker(self, raw_speaker: str) -> tuple:
        text = re.sub(r"^[○□■\-\s]+", "", raw_speaker).strip()
        tokens = text.split()
        
        if len(tokens) >= 2:
            role, name = tokens[0], "".join(tokens[1:])
        elif len(tokens) == 1:
            role, name = "미상", tokens[0]
        else:
            return "미상", "미상", "기타"

        speaker_type = "의회" if any(r in role for r in ["의원", "의장", "부의장", "위원장", "부위원장", "위원"]) else "집행부"
        return role, name, speaker_type

    def process_record(self, record: dict, rule: dict, filename_info: str) -> list:
        results = []
        html_content = record.get("MINTS_HTML", "")
        if not html_content:
            return results

        soup = BeautifulSoup(html_content, "html.parser")

        for tag in soup(["script", "style", "form", "table", "ul.skipUl", "#topFunction", "#mySidenav2"]):
            tag.decompose()

        agenda_selector = rule.get("agenda", "div")
        speaker_selector = rule.get("speaker", "span.spkrbld") #"span.spkrbld")
        utterance_selector = rule.get("utterance", "p")

        try:
            speaker_tags = soup.select(speaker_selector)
            agenda_tags = soup.select(agenda_selector)
        except Exception:
            return results

        if len(speaker_tags) == 0:
            print(f"  [경고] '{filename_info}'에서 발언자({speaker_selector})를 찾지 못해 기본값(span.spkrbld)으로 재시도합니다.")
            speaker_selector = "span.spkrbld"
            try:
                speaker_tags = soup.select(speaker_selector)
            except Exception:
                pass
            
            if len(speaker_tags) == 0:
                print(f"  [실패] 기본값으로도 '{filename_info}'의 발언자를 찾을 수 없습니다. 건너뜁니다.")
                return results

        for spk_tag in speaker_tags:
            raw_speaker = spk_tag.get_text(strip=True)
            # 오류가 발생했던 부분 수정 완료
            role, name, speaker_type = self.parse_speaker(raw_speaker)

            agenda_title = "일반안건"
            for prev in spk_tag.find_all_previous():
                if prev in agenda_tags:
                    agenda_title = prev.get_text(strip=True)[:50]
                    break

            utterance_cleaned = ""
            
            if utterance_selector and utterance_selector != "p":
                next_content = spk_tag.find_next(utterance_selector)
                if next_content:
                    utterance_cleaned = self.clean_text(next_content.get_text(strip=True))

            if not utterance_cleaned or len(utterance_cleaned) < 10:
                parent_block = spk_tag.find_parent("div") or spk_tag.parent
                spk_tag.decompose() 
                utterance_raw = parent_block.get_text(strip=True) if parent_block else ""
                utterance_cleaned = self.clean_text(utterance_raw)

            if len(utterance_cleaned) >= self.min_char_len:
                results.append({
                    "date": record.get("MTG_DE", ""),
                    "assembly": record.get("RASMBLY_NM", ""),
                    "agenda_title": agenda_title,
                    "speaker_role": role,
                    "speaker_name": name,
                    "speaker_type": speaker_type,
                    "utterance": utterance_cleaned
                })
        return results

if __name__ == "__main__":
    input_dir = constant.org_dir
    output_dir = constant.text_dir
    rules_file = constant.all_rule_file

    os.makedirs(output_dir, exist_ok=True)

    with open(rules_file, "r", encoding="utf-8") as f:
        parsing_rules = json.load(f)

    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    preprocessor = DynamicCouncilPreprocessor(min_char_len=30)

    for input_file in json_files:
        base_name = os.path.basename(input_file)
        file_key = os.path.splitext(base_name)[0]
        
        matched_rule = parsing_rules.get(file_key, {"agenda": "div", "speaker": "span.spkrbld", "utterance": "p"})

        print(f"[{base_name}] 전처리 시작...")
        
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                
            all_utterances = []
            for record in raw_data:
                all_utterances.extend(preprocessor.process_record(record, matched_rule, base_name))

            df_clean = pd.DataFrame(all_utterances)
            
            if not df_clean.empty:
                # 동일 화자(speaker_name, speaker_type 동일) 연속 발언 병합
                condition = (df_clean['speaker_name'] != df_clean['speaker_name'].shift()) | \
                            (df_clean['speaker_type'] != df_clean['speaker_type'].shift())
                df_clean['group_id'] = condition.cumsum()

                df_merged = df_clean.groupby('group_id').agg({
                    'date': 'first',
                    'assembly': 'first',
                    'agenda_title': 'first',
                    'speaker_role': 'first',
                    'speaker_name': 'first',
                    'speaker_type': 'first',
                    'utterance': lambda x: ' '.join(x)
                }).reset_index(drop=True)

                output_path = os.path.join(output_dir, f"cleaned_{file_key}.txt")
                df_merged.to_csv(output_path, sep="\t", index=False, encoding="utf-8-sig")
                print(f"  -> 완료: {len(df_merged)}건 추출 (연속 발언 병합됨) ({output_path})")
            else:
                print(f"  -> 추출된 유효 데이터 없음")
        except Exception as e:
             print(f"  -> 파일 처리 중 치명적 오류 발생: {e}")