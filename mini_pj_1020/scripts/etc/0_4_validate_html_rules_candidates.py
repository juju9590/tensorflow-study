"""Validate discovered CSS candidates against real council records (diagnostic only).

VS Code F5 runs with default Windows paths. Optional CLI arguments override them.
Requires: pip install beautifulsoup4 lxml
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup


DEFAULT_INPUT = Path(r"C:\mini_pj\data\org2\2025_광역 회의록")
DEFAULT_RULES = Path(r"C:\mini_pj\data\html_rules_candidates_v2.json")
DEFAULT_OUTPUT = Path(r"C:\mini_pj\data\html_rules_validation.json")
FIELDS = ["agenda_title", "speaker_role", "speaker_name", "speaker_type", "utterance"]
SPEAKER_MARKER = re.compile(r"^[○◯●]\s*")
AGENDA_MARKER = re.compile(r"^(\d{1,3}[.)]|[가-하][.)]|[oㅇ]\s*5분|의사일정\s*제\d+항)")


def normalize(value):
    return " ".join(str(value).split())


def snippet(node, maximum=170):
    text = normalize(node.get_text(" ", strip=True))
    return text[:maximum]


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as fp:
        return json.load(fp)


def sample_rows(records, requested=5):
    indexed = [
        (i, row) for i, row in enumerate(records)
        if isinstance(row, dict) and isinstance(row.get("MINTS_HTML"), str)
        and row["MINTS_HTML"].strip()
    ]
    if len(indexed) <= requested:
        return indexed
    indices = sorted({round(k * (len(indexed) - 1) / (requested - 1)) for k in range(requested)}) if requested > 1 else [0]
    return [indexed[i] for i in indices]


def find_css(soup, selector):
    if not selector:
        return [], None
    try:
        return soup.select(selector), None
    except Exception as exc:
        return [], f"CSS 선택자 오류: {str(exc)[:120]}"


def speaker_details(node):
    """Show a *tentative* interpretation; do not treat as verified metadata."""
    header = node if node.name in {"strong", "spk", "span", "a"} else node.find("strong")
    if header is None:
        header = node.find(class_=re.compile(r"spkrbld|member|speaker", re.I)) or node
    text = normalize(header.get_text(" ", strip=True))
    text = SPEAKER_MARKER.sub("", text)
    profile = header.select_one("a.member_profile, a.profile_view, a[href*='profile'], a[title*='프로필']")
    name = normalize(profile.get_text(" ", strip=True)) if profile else None
    role = None
    if name and name in text:
        role = normalize(text.replace(name, "", 1))
    else:
        # Handles both "위원장 홍길동" and "홍길동 위원" as hints only.
        components = text.split()
        if len(components) >= 2:
            if re.fullmatch(r"[가-힣]{2,4}", components[-1]):
                name = components[-1]
                role = " ".join(components[:-1])
            elif re.fullmatch(r"[가-힣]{2,4}", components[0]):
                name = components[0]
                role = " ".join(components[1:])
    classes = node.get("class", [])
    member_code = node.get("data-member_code")
    if "member-speech" in classes and member_code:
        speaker_type = "의원(후보)"
    else:
        speaker_type = None
    return {"speaker_role": role, "speaker_name": name, "speaker_type": speaker_type,
            "speaker_marker_text": text[:90]}


def extract_preview(node):
    """Only reliable for a self-contained speaker block, otherwise flag review."""
    clone = BeautifulSoup(str(node), "lxml")
    root = clone.body.find() if clone.body else clone.find()
    if root is None:
        return ""
    first_strong = root.find("strong")
    if first_strong:
        first_strong.decompose()
    return snippet(root, 240)


def review_record(record, rules, index):
    soup = BeautifulSoup(record["MINTS_HTML"], "lxml")
    results = {}
    warnings = []
    fields = rules.get("fields", rules)
    for field in FIELDS:
        config = fields.get(field, {})
        selector = config.get("recommended_selector")
        matches, error = find_css(soup, selector)
        samples = [snippet(node) for node in matches[:3]]
        result = {"selector": selector, "matched_elements": len(matches), "examples": samples}
        if error:
            result["error"] = error
            warnings.append(f"{field}: {error}")
        elif selector is None:
            warnings.append(f"{field}: 후보 선택자 없음")
        elif not matches:
            warnings.append(f"{field}: 선택자에 맞는 요소 없음")
        if field == "agenda_title":
            result["agenda_like_count"] = sum(bool(AGENDA_MARKER.match(snippet(n))) for n in matches)
            if len(matches) > 0 and result["agenda_like_count"] == 0:
                warnings.append("agenda_title: 번호가 있는 안건 형식을 찾지 못함 (다른 형식일 수 있음)")
            # The linked list is a diagnostic preview, not full agenda assignment.
            result["agenda_preview"] = list(dict.fromkeys(snippet(n, 170) for n in matches[:12]))
        if field == "speaker_role":
            result["speaker_preview"] = [speaker_details(n) for n in matches[:3]]
        if field == "speaker_type":
            result["type_verified"] = False
            if matches:
                warnings.append("speaker_type: 선택자는 발견됐지만 의원·공무원 유형 분류는 추가 검증 필요")
        if field == "utterance":
            sequential = bool(config.get("requires_sequential_dom_scan"))
            result["requires_sequential_dom_scan"] = sequential
            if sequential:
                result["speech_preview"] = []
                warnings.append("utterance: 전체 본문 경계 탐색 방식으로, 발언별 추출 미검증")
            else:
                result["speech_preview"] = [extract_preview(n) for n in matches[:3]]
                if matches and any(len(extract_preview(n)) == 0 for n in matches[:3]):
                    warnings.append("utterance: 추출된 본문이 비어 있음")
        results[field] = result
    return {"source_record_index": index,
            "date": record.get("MTG_DE"), "assembly": record.get("RASMBLY_NM"),
            "fields": results, "warnings": list(dict.fromkeys(warnings))}


def main():
    ap = argparse.ArgumentParser(description="HTML 선택자 후보를 원본 회의록으로 검증")
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--sample-per-region", type=int, default=5)
    args = ap.parse_args()
    if not args.input_dir.is_dir():
        ap.error(f"원본 폴더를 찾지 못했습니다: {args.input_dir}")
    if not args.rules.is_file():
        ap.error(f"후보 규칙 JSON이 없습니다: {args.rules}")
    if args.sample_per_region < 1:
        ap.error("샘플 개수는 1 이상이어야 합니다")
    candidates = load_json(args.rules)
    regions = candidates.get("regions", candidates)
    report = {"_meta": {"purpose": "candidate_diagnostics_not_accuracy", "validated": False,
                        "warning": "matched_elements는 정답 건수가 아니며 표본을 눈으로 확인해야 합니다."},
              "regions": {}}
    for region, config in regions.items():
        filepath = args.input_dir / f"{region}.json"
        if config.get("status") == "empty":
            report["regions"][region] = {"status": "empty", "records": []}
            print(f"[제외] {region}: 빈 JSON")
            continue
        if not filepath.exists():
            report["regions"][region] = {"status": "source_missing", "records": []}
            print(f"[없음] {filepath}")
            continue
        try:
            source = load_json(filepath)
            if not isinstance(source, list):
                raise ValueError("최상위 JSON이 리스트가 아닙니다")
            reviewed = [review_record(row, config, index) for index, row in sample_rows(source, args.sample_per_region)]
            totals = {key: sum(r["fields"][key]["matched_elements"] for r in reviewed) for key in FIELDS}
            review_count = sum(bool(r["warnings"]) for r in reviewed)
            report["regions"][region] = {"status": "diagnostics_only", "sample_count": len(reviewed),
                                          "match_counts": totals, "samples_with_warnings": review_count,
                                          "records": reviewed}
            print(f"[검사] {region} | 표본 {len(reviewed)} | 검토 경고 {review_count} | 후보 선택자 요소 {totals}")
        except Exception as exc:
            report["regions"][region] = {"status": "error", "message": str(exc)}
            print(f"[오류] {region}: {exc}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print(f"\n진단 보고서 저장 완료: {args.output}")


if __name__ == "__main__":
    main()
