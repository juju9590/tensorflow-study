"""Extract 7 fields from Gangwon/Ulsan/Chungnam/Chungbuk meeting minutes.

First-stage, auditable parser. F5 works on Windows with paths below.
Requires: pip install lxml
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from lxml import etree, html


DEFAULT_INPUT = Path(r"C:\mini_pj\data\org2\2025_광역 회의록")
DEFAULT_OUTPUT = Path(r"C:\mini_pj\data\extracted_group1")
REGIONS = (
    "2025_강원특별자치도의회",
    "2025_울산광역시의회",
    "2025_충청남도의회",
    "2025_충청북도의회",
)
COLUMNS = ("date", "assembly", "agenda_title", "speaker_role", "speaker_name", "speaker_type", "utterance")

PRIMARY_AGENDA = re.compile(r"^\s*\d{1,3}\s*[.．)]\s*")
SUB_AGENDA = re.compile(r"^\s*[가-하]\s*[.．)]\s*")
NAME_PATTERN = re.compile(r"^[가-힣]{2,4}$")
SPACED_NAME = re.compile(r"^[가-힣](?:\s+[가-힣]){1,3}$")
CIVIL_ROLE = re.compile(r"(담당관|국장|과장|실장|사무처장|전문위원|교육감|도지사|시장|원장|소장|본부장|부시장|부지사|공무원|팀장|의사관)")


def tidy(text):
    return " ".join((text or "").split())


def text_with_breaks(node):
    # Preserve paragraph/line boundaries in utterance before normalizing.
    clone = html.fromstring(etree.tostring(node, encoding="unicode"))
    for tag in clone.xpath(".//br | .//hr"):
        tag.tail = "\n" + (tag.tail or "")
        tag.drop_tag()
    for tag in clone.xpath(".//p | .//div"):
        if tag is clone:
            continue
        tag.tail = "\n" + (tag.tail or "")
    return "\n".join(line.strip() for line in clone.text_content().splitlines() if line.strip())


def parse_speaker(block):
    header = block.xpath("./strong[1]")
    if not header:
        return None, None, None, "발언자 <strong> 없음"
    tag = header[0]
    title = tidy(tag.text_content()).lstrip("○◯● ")
    profile = tag.xpath('.//a[contains(concat(" ",normalize-space(@class)," ")," member_profile ")]')
    if profile:
        name = tidy(profile[0].text_content())
        role = tidy(title.replace(name, "", 1)).strip("↘ :")
    else:
        tokens = title.split()
        name, role = None, None
        if len(tokens) >= 2:
            if NAME_PATTERN.fullmatch(tokens[-1]):
                name, role = tokens[-1], " ".join(tokens[:-1])
            elif NAME_PATTERN.fullmatch(tokens[0]):
                name, role = tokens[0], " ".join(tokens[1:])
            elif SPACED_NAME.fullmatch(" ".join(tokens[-3:])):
                name, role = "".join(tokens[-3:]), " ".join(tokens[:-3])
    role = tidy(role) or None
    name = tidy(name) or None
    classes = block.get("class", "").split()
    member_code = tidy(block.get("data-member_code", ""))
    if "member-speech" in classes and member_code:
        speaker_type = "의원"
    elif role and CIVIL_ROLE.search(role):
        speaker_type = "공무원/공공기관(추정)"
    else:
        speaker_type = None
    warning = None if (role and name) else f"발언자 분리 확인: {title[:70]}"
    return role, name, speaker_type, warning


def parse_utterance(block):
    # Do not discard nested body text or br tails when removing the header.
    copy = html.fromstring(etree.tostring(block, encoding="unicode"))
    header = copy.xpath("./strong[1]")
    if header:
        header[0].drop_tree()  # lxml drop_tree preserves the element's tail
    for extra in copy.xpath('.//*[contains(concat(" ",normalize-space(@class)," ")," go-to-top ")]'):
        extra.drop_tree()
    return text_with_breaks(copy)


def parse_record(record, region, record_index):
    markup = record.get("MINTS_HTML") or ""
    if not markup.strip():
        return [], {"record_index": record_index, "warning": "HTML 비어 있음"}
    root = html.fromstring(markup)
    bodies = root.xpath('//*[@id="minutes-body"]')
    if not bodies:
        return [], {"record_index": record_index, "warning": "#minutes-body 없음"}
    body = bodies[0]
    # Both kinds appear inside #minutes-body; document order matters.
    items = body.xpath(
        './/*[contains(concat(" ",normalize-space(@class)," ")," item-in-contents ") '
        'or contains(concat(" ",normalize-space(@class)," ")," speaker-block ")]'
    )
    output = []
    current_agendas = []
    active_primary = None
    just_saw_speaker = False
    warnings = []
    for node in items:
        classes = node.get("class", "").split()
        if "item-in-contents" in classes:
            heading = tidy(node.text_content())
            if not heading:
                continue
            if PRIMARY_AGENDA.match(heading):
                # Several consecutive agenda headings can apply to the next turn.
                if just_saw_speaker:
                    current_agendas = []
                current_agendas.append(heading)
                active_primary = heading
            elif SUB_AGENDA.match(heading):
                if active_primary and active_primary not in current_agendas:
                    current_agendas = [active_primary]
                current_agendas.append(heading)
            else:
                # '위원장 인사', '5분 자유발언' etc: retain as a distinct
                # nonstandard item, but require human review of its meaning.
                if just_saw_speaker:
                    current_agendas = []
                current_agendas.append(heading)
                warnings.append(f"안건 형식 검토: {heading[:70]}")
            just_saw_speaker = False
        elif "speaker-block" in classes:
            role, name, speaker_type, warning = parse_speaker(node)
            utterance = parse_utterance(node)
            if warning:
                warnings.append(warning)
            if not utterance:
                warnings.append(f"빈 발언 본문: data-con_idx={node.get('data-con_idx')}")
            output.append({
                "date": record.get("MTG_DE"),
                "assembly": record.get("RASMBLY_NM"),
                "agenda_title": current_agendas.copy(),
                "speaker_role": role,
                "speaker_name": name,
                "speaker_type": speaker_type,
                "utterance": utterance,
                # Separate provenance helps trace extraction without changing 7 fields.
                "_source": {"region": region, "record_index": record_index,
                            "data_con_idx": node.get("data-con_idx")},
            })
            just_saw_speaker = True
    return output, {"record_index": record_index, "speaker_count": len(output),
                    "agenda_count": len(body.xpath('.//*[contains(concat(" ",normalize-space(@class)," ")," item-in-contents ")]')),
                    "warnings": list(dict.fromkeys(warnings))[:30]}


def selected(records, k, process_all):
    indices = [i for i, row in enumerate(records) if isinstance(row, dict) and (row.get("MINTS_HTML") or "").strip()]
    if process_all or len(indices) <= k:
        return indices
    positions = sorted({round(x * (len(indices) - 1) / (k - 1)) for x in range(k)}) if k > 1 else [0]
    return [indices[x] for x in positions]


def main():
    parser = argparse.ArgumentParser(description="공통 구조 4개 지역 회의록의 7개 항목 추출")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-per-region", type=int, default=5)
    parser.add_argument("--all", action="store_true", help="검증 후에만 사용: 원본 모든 레코드 처리")
    args = parser.parse_args()
    if args.sample_per_region < 1:
        parser.error("샘플 수는 1 이상이어야 합니다")
    if not args.input_dir.is_dir():
        parser.error(f"원본 폴더가 없습니다: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for region in REGIONS:
        source = args.input_dir / f"{region}.json"
        if not source.exists():
            print(f"[파일 없음] {source}")
            continue
        try:
            with source.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if not isinstance(data, list):
                print(f"[구조 오류] {region}: 최상위 list 아님")
                continue
            result, diagnostics = [], []
            for index in selected(data, args.sample_per_region, args.all):
                try:
                    extracted, diagnostic = parse_record(data[index], region, index)
                    result.extend(extracted)
                    diagnostics.append(diagnostic)
                except Exception as exc:
                    diagnostics.append({"record_index": index, "error": str(exc)})
            output = args.output_dir / f"{region}_extracted.json"
            with output.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            csv_path = args.output_dir / f"{region}_extracted.csv"
            with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
                for item in result:
                    csv_row = {column: item.get(column) for column in COLUMNS}
                    # A JSON array in one CSV cell preserves multiple linked agendas.
                    csv_row["agenda_title"] = json.dumps(item["agenda_title"], ensure_ascii=False)
                    writer.writerow(csv_row)
            report = args.output_dir / f"{region}_diagnostics.json"
            with report.open("w", encoding="utf-8") as f:
                json.dump(diagnostics, f, ensure_ascii=False, indent=2)
            print(f"[완료] {region}: 회의록 {len(diagnostics)}건, 발언 {len(result)}개 → {output.name}, {csv_path.name}")
            print("       미분리/검토:", sum(bool(x.get("warnings") or x.get("error")) for x in diagnostics), "개 회의록")
        except Exception as exc:
            print(f"[오류] {region}: {exc}")
    print(f"\n추출 결과 저장 폴더: {args.output_dir}")


if __name__ == "__main__":
    main()
