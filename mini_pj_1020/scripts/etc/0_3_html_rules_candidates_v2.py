"""Discover *candidate* extraction selectors from council JSON HTML.

Usage (Windows PowerShell):
    python html_rules_candidates_v2.py --input-dir "C:/mini_pj/data/org2/2025_광역 회의록" --output "C:/mini_pj/data/html_rules_candidates_v2.json"

Requires: pip install lxml
The output is an auditable suggestion, not a validated production parser.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from lxml import html
except ImportError as exc:
    raise SystemExit("lxml이 필요합니다: pip install lxml") from exc


FIELDS = ("agenda_title", "speaker_role", "speaker_name", "speaker_type", "utterance")
AGENDAS = re.compile(r"^\s*(?:\d{1,3}[.．)]|[가-하][.．)]|의사일정\s*제\d+항)")
SPEAKER = re.compile(r"^\s*[○◯●]\s*[^\n]{1,50}")
HEADING = re.compile(r"(?:안건|의사일정|회의록|개회식|보고의\s*건|조례안|결의안|건의안|규칙안|심사)")
NAME = re.compile(r"^[가-힣]{2,4}$")
ID_STABLE = re.compile(r"^[a-zA-Z][\w-]{1,45}$")
SPECIAL_CLASS = re.compile(
    r"(speaker|speech|member|profile|remark|spkr|sman|item|agenda|agnd|bill|text|content|con93|label|pv|em1|bold)",
    re.IGNORECASE,
)
ROLE_CLASS = re.compile(r"(speaker|spkr|member|sman|remark|labelq|labela)", re.IGNORECASE)
AGENDA_CLASS = re.compile(r"(agenda|agnd|item|bill|em1)", re.IGNORECASE)
UTTERANCE_CLASS = re.compile(r"(speaker|speech|remark|text|content|con93|sman)", re.IGNORECASE)
VOID_OR_BAD = {"html", "head", "body", "script", "style", "link", "meta", "br", "hr"}


def clean(value: str) -> str:
    return " ".join(value.split())


def css_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def element_selector(node) -> list[str]:
    """Selector suggestions derived from an observed element, not a region list."""
    tag = node.tag
    if not isinstance(tag, str) or tag in VOID_OR_BAD:
        return []
    output = []
    node_id = node.get("id", "")
    if ID_STABLE.fullmatch(node_id) and not re.search(r"\d{3,}", node_id) and not re.search(r"\d+$", node_id):
        output.append(f'#{node_id}')
    classes = node.get("class", "").split()
    meaningful = [x for x in classes if ID_STABLE.fullmatch(x) and SPECIAL_CLASS.search(x)]
    if meaningful:
        output.extend(f"{tag}.{x}" for x in meaningful[:3])
        if len(meaningful) >= 2:
            output.append(f"{tag}.{meaningful[0]}.{meaningful[1]}")
    for attr in ("data-member_code", "data-con_idx", "data-item"):
        if node.get(attr) is not None:
            output.append(f"{tag}[{attr}]")
    if re.fullmatch(r"(?i)sman\w*", classes[0] if classes else ""):
        output.append(f'{tag}[class^="sMan"]')
    if node_id and re.match(r"^(item|anguna|angun|agnd)\d+", node_id, re.I):
        prefix = re.match(r"^[a-zA-Z]+", node_id).group()
        output.append(f'{tag}[id^="{prefix}"]')
    if tag in {"spk", "speaker"}:
        output.append(tag)
    if tag == "a" and node.getparent() is not None and node.getparent().tag == "strong" and SPEAKER.match(clean(node.getparent().text_content())[:90]):
        output.append("strong > a")
        grandparent = node.getparent().getparent()
        if grandparent is not None and grandparent.get("id"):
            output.append(f'#{grandparent.get("id")} > strong > a')
    if any(re.match(r"(?i)^spkrmbr\d+", x) for x in classes):
        output.append(f'{tag}[class^="spkrMBR"]')
    if any(re.match(r"(?i)^sman", x) for x in classes):
        output.append(f'{tag}[class^="sMan"]')
    if tag in {"strong", "span"} and SPEAKER.match(clean(node.text_content())[:90]):
        output.append(tag)
    previous = node.getprevious()
    if tag == "strong" and previous is not None and previous.tag == "a" and re.match(r"^anguna\d+$", previous.get("id", "")):
        output.append('a[id^="anguna"] + strong')
    if tag in {"a", "span"} and "labelQColor" in classes:
        output.append('a.labelQColor, a.labelAColor')
    if tag in {"a", "span"} and "labelAColor" in classes:
        output.append('a.labelQColor, a.labelAColor')
    # Parent-anchored selector is useful when common classes occur in headers too.
    parent = node.getparent()
    if parent is not None and parent.get("id") and ID_STABLE.fullmatch(parent.get("id")):
        output.extend(f'#{parent.get("id")} > {s}' for s in output if not s.startswith("#"))
    return list(dict.fromkeys(output))


def select_sample_records(records: list, k: int) -> list[tuple[int, dict]]:
    valid = [(i, r) for i, r in enumerate(records) if isinstance(r, dict) and isinstance(r.get("MINTS_HTML"), str) and r["MINTS_HTML"].strip()]
    if len(valid) <= k:
        return valid
    # First/middle/end rather than first three: better coverage of layout changes.
    positions = sorted({round(j * (len(valid) - 1) / (k - 1)) for j in range(k)}) if k > 1 else [0]
    return [valid[j] for j in positions]


def nearest_speaker_text(node) -> str:
    if node.tag == "strong":
        return clean(node.text_content())[:90]
    for child in node.xpath("./strong[1] | ./p[1] | ./span[1]"):
        value = clean(child.text_content())
        if SPEAKER.match(value):
            return value[:90]
    return clean(node.text_content())[:90]


def signals(node) -> dict[str, float]:
    cls = node.get("class", "")
    tag = node.tag.lower() if isinstance(node.tag, str) else ""
    text = clean(node.text_content())
    leading = nearest_speaker_text(node)
    node_id = node.get("id", "")
    profile = bool(node.xpath('.//a[contains(@class,"profile") or contains(@href,"profile")]'))
    words = len(text)
    scores = {field: 0.0 for field in FIELDS}
    if AGENDAS.match(text) and words < 350:
        scores["agenda_title"] += 4
    if HEADING.search(text[:140]) and words < 300:
        scores["agenda_title"] += 1.4
    if AGENDA_CLASS.search(cls) or re.match(r"^(item|anguna|agnd)\d+", node_id, re.I):
        scores["agenda_title"] += 5
    if tag == "strong" and AGENDAS.match(text):
        scores["agenda_title"] += 2
    if SPEAKER.match(leading) and (words < 1500 or ROLE_CLASS.search(cls)):
        scores["speaker_role"] += 5
        scores["speaker_name"] += 2
        scores["speaker_type"] += 1
        if words > 70:
            scores["utterance"] += 4
    if ROLE_CLASS.search(cls) and words < 1500:
        scores["speaker_role"] += 3
        scores["speaker_name"] += 2
        scores["speaker_type"] += 2
    if tag == "spk":
        scores["speaker_role"] += 6
        scores["speaker_name"] += 3
    if profile:
        scores["speaker_name"] += 2
        scores["speaker_type"] += 1
    if "member-speech" in cls or node.get("data-member_code") is not None:
        scores["speaker_type"] += 5
    if tag == "a" and ("labelQColor" in cls or "labelAColor" in cls):
        scores["speaker_role"] += 10
        scores["speaker_name"] += 10
        scores["speaker_type"] += 3
    if tag == "div" and "con93" in cls:
        scores["utterance"] += 10
    if tag == "div" and "text" in cls and words > 35:
        scores["utterance"] += 3
    if tag == "span" and "spkrbld" in cls:
        scores["speaker_role"] += 10
        scores["speaker_name"] += 5
    if tag == "a" and "profile_view" in cls:
        scores["speaker_name"] += 8
    if tag == "a" and node.getparent() is not None and node.getparent().tag == "strong" and SPEAKER.match(clean(node.getparent().text_content())[:90]):
        scores["speaker_name"] += 10
    if tag == "span" and "member_name" in cls:
        scores["speaker_name"] += 7
    if tag == "strong" and "item-in-contents" in cls:
        scores["agenda_title"] += 12
    if tag == "strong" and re.match(r"item\d+$", node_id):
        scores["agenda_title"] += 9
    if tag == "spk":
        scores["speaker_role"] += 8
    if "profile" in cls and NAME.fullmatch(text):
        scores["speaker_name"] += 8
    if UTTERANCE_CLASS.search(cls):
        scores["utterance"] += 3
        if ROLE_CLASS.search(cls) and len(text) > 30:
            scores["utterance"] += 2
    if words > 100 and len(node) < 35:
        scores["utterance"] += 2
    if words > 3000:
        scores["utterance"] -= 5
        scores["agenda_title"] -= 5
    if tag in {"a", "strong", "span"} and words < 70:
        scores["utterance"] -= 5
    if words > 700:
        scores["agenda_title"] -= 5
    if "item" in cls and tag == "div" and words > 100:
        scores["agenda_title"] -= 7
    return scores


def analyze_region(records: list, sample_count: int) -> dict:
    chosen = select_sample_records(records, sample_count)
    buckets = defaultdict(lambda: defaultdict(lambda: {"scores": [], "docs": set(), "matches": [], "examples": []}))
    stats = []
    for record_index, record in chosen:
        raw = record["MINTS_HTML"]
        try:
            tree = html.fromstring(raw)
        except (ValueError, TypeError) as exc:
            stats.append({"record_index": record_index, "error": str(exc)})
            continue
        nodes = tree.xpath("//*")
        stats.append({"record_index": record_index, "html_chars": len(raw), "nodes": len(nodes)})
        for node in nodes:
            if not isinstance(node.tag, str):
                continue
            score_map = signals(node)
            for field, score in score_map.items():
                if score < 4.5:
                    continue
                for selector in element_selector(node):
                    bucket = buckets[field][selector]
                    bucket["scores"].append(score)
                    bucket["docs"].add(record_index)
                    bucket["matches"].append(node)
                    if len(bucket["examples"]) < 2:
                        bucket["examples"].append(clean(node.text_content())[:110])
    result = {}
    usable = len([s for s in stats if "error" not in s])
    for field in FIELDS:
        candidates = []
        for selector, val in buckets[field].items():
            matches = len(val["scores"])
            support = len(val["docs"])
            mean_signal = sum(val["scores"]) / matches
            # Reward cross-record consistency, but do not claim correctness.
            ranking = round(mean_signal + min(matches, 35) / 35 * 2.5 + support / max(usable, 1) * 3, 2)
            if selector.endswith('.contents-block') or selector.endswith('.content-wrapper'):
                ranking -= 3
            if selector in {'li.font-bold', 'li.item_line', 'div.item', 'div.lineht', 'span.bold', 'strong'}:
                ranking -= 5
            if field == "agenda_title" and ('item-in-contents' in selector or 'id^="item"' in selector or 'anguna' in selector):
                ranking += 5
            if field == "speaker_role" and ('speaker-block' in selector or selector == 'spk' or 'labelQColor' in selector or 'spkrbld' in selector):
                ranking += 4
            if field == "speaker_name" and ('profile' in selector or 'member_name' in selector or 'labelQColor' in selector):
                ranking += 4
            if field == "speaker_type" and ('member-speech' in selector or 'data-member_code' in selector):
                ranking += 4
            if field == "utterance" and ('speaker-block' in selector or 'con93' in selector or 'remark' in selector or selector == 'div.text'):
                ranking += 4
            if field in {"speaker_role", "speaker_name"} and selector.startswith('#canvas') and 'strong' not in selector:
                ranking -= 4
            candidates.append({
                "selector": selector,
                "rank_score": ranking,
                "matched_elements": matches,
                "sample_coverage": f"{support}/{usable}",
                "examples": val["examples"],
            })
        candidates.sort(key=lambda x: (x["rank_score"], x["matched_elements"]), reverse=True)
        result[field] = {"recommended_selector": candidates[0]["selector"] if candidates else None, "candidates": candidates[:8], "needs_review": True}
    # Flat-format transcripts (e.g. strong ○role name between raw text nodes)
    # cannot be segmented using CSS alone. Advertise a boundary algorithm.
    if result["speaker_role"]["recommended_selector"] in {"strong", "#canvas > strong"}:
        if not result["utterance"]["recommended_selector"]:
            result["utterance"]["recommended_selector"] = "#canvas"
            result["utterance"]["candidates"] = [{"selector": "#canvas", "rank_score": None, "matched_elements": None, "sample_coverage": f"{usable}/{usable}", "examples": ["발언자 strong 사이의 텍스트를 순서대로 수집해야 함"]}]
        result["utterance"]["suggested_extractor"] = "collect_text_until_next_speaker_or_agenda"
        result["utterance"]["requires_sequential_dom_scan"] = True
    # Find structures where siblings may be needed to reconstruct speech.
    speaker_css = result["speaker_role"]["recommended_selector"]
    speech_css = result["utterance"]["recommended_selector"]
    if result["utterance"].get("requires_sequential_dom_scan"):
        pass
    elif speaker_css and (not speech_css or speaker_css == speech_css):
        result["utterance"]["suggested_extractor"] = "collect_text_until_next_speaker_or_agenda"
    else:
        result["utterance"]["suggested_extractor"] = "remove_speaker_header_then_text"
    result["speaker_name"]["suggested_extractor"] = "profile_link_then_role_name_split"
    result["speaker_role"]["suggested_extractor"] = "speaker_marker_role_name_split"
    result["speaker_type"]["suggested_extractor"] = "member_class_or_code_then_role_classification"
    result["agenda_title"]["suggested_extractor"] = "ordered_agenda_heading_list"
    return {"sampled_records": stats, "fields": result}


def main() -> None:
    parser = argparse.ArgumentParser(description="원본 JSON의 HTML에서 지역별 선택자 후보 자동 탐색")
    parser.add_argument(
    "--input-dir",
    type=Path,
    default=Path(r"C:\mini_pj\data\org2\2025_광역 회의록"),
    help="원본 JSON 폴더"
    )
    parser.add_argument(
    "--output",
    type=Path,
    default=Path(r"C:\mini_pj\data\html_rules_candidates_v2.json")
    )
    parser.add_argument("--sample-per-region", type=int, default=3)
    args = parser.parse_args()
    if args.sample_per_region < 1:
        parser.error("--sample-per-region은 1 이상이어야 합니다")
    files = sorted(args.input_dir.glob("*.json"))
    if not files:
        parser.error(f"JSON 파일이 없습니다: {args.input_dir}")
    output = {"_meta": {"kind": "automatically_discovered_candidates", "validated": False, "sample_policy": "first_middle_last_evenly_spaced", "warning": "선택자 후보일 뿐입니다. 7개 항목 추출 정확성은 반드시 확인하세요."}, "regions": {}}
    for filepath in files:
        try:
            with filepath.open("r", encoding="utf-8-sig") as fp:
                rows = json.load(fp)
            if not isinstance(rows, list):
                raise ValueError("최상위 JSON은 list여야 합니다")
            if not rows:
                output["regions"][filepath.stem] = {"status": "empty", "record_count": 0}
                print(f"[비어 있음] {filepath.name}")
                continue
            analysis = analyze_region(rows, args.sample_per_region)
            output["regions"][filepath.stem] = {
                "status": "candidate_only", "record_count": len(rows),
                "date": {"source": "json", "key": "MTG_DE"},
                "assembly": {"source": "json", "key": "RASMBLY_NM"},
                **analysis,
            }
            fields = analysis["fields"]
            print(f"[후보 생성] {filepath.name} ({len(rows)}건)")
            for field in FIELDS:
                print(f"  {field:14} : {fields[field]['recommended_selector'] or '(후보 없음)'}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[오류] {filepath.name}: {exc}")
            output["regions"][filepath.stem] = {"status": "error", "error": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)
    print(f"\n완료: {args.output} / {len(files)}개 파일")


if __name__ == "__main__":
    main()
