"""Section Definition 的安全加载、结构校验与渲染。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

SECTIONS_DIR = Path(__file__).resolve().parent / "sections"
_SECTION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _require_string(value: Any, label: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Section 定义中的 {label} 必须是字符串")
    result = value.strip()
    if not allow_empty and not result:
        raise ValueError(f"Section 定义中的 {label} 不能为空")
    return result


def validate_section(data: Any, expected_id: str | None = None) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Section 定义格式错误：YAML 顶层必须是映射对象")

    section_id = _require_string(data.get("id", expected_id or ""), "id", allow_empty=False)
    if expected_id and section_id != expected_id:
        raise ValueError(
            f"Section 文件 id 与请求不一致：请求 {expected_id}，文件声明 {section_id}"
        )

    name = _require_string(data.get("name", section_id), "name", allow_empty=False)
    objective = data.get("objective", "")
    if not isinstance(objective, str):
        raise ValueError("Section objective 必须是字符串")

    deliverables = data.get("deliverables", [])
    if not isinstance(deliverables, list):
        raise ValueError("Section deliverables 必须是数组")
    seen_ids: set[str] = set()
    normalized_deliverables = []
    for index, item in enumerate(deliverables, 1):
        if not isinstance(item, dict):
            raise ValueError(f"deliverables 第 {index} 项必须是映射对象")
        item_id = _require_string(item.get("id", f"item_{index}"), f"deliverables[{index}].id", allow_empty=False)
        if item_id in seen_ids:
            raise ValueError(f"Section deliverables 存在重复 id：{item_id}")
        seen_ids.add(item_id)
        normalized_deliverables.append({
            "id": item_id,
            "title": _require_string(item.get("title", item_id), f"{item_id}.title"),
            "description": _require_string(item.get("description", ""), f"{item_id}.description"),
        })

    quality = data.get("quality_requirements", [])
    if not isinstance(quality, list) or not all(isinstance(item, str) for item in quality):
        raise ValueError("Section quality_requirements 必须是字符串数组")

    return {
        **data,
        "id": section_id,
        "name": name,
        "objective": objective.strip(),
        "deliverables": normalized_deliverables,
        "quality_requirements": [item.strip() for item in quality if item.strip()],
    }


def load_section(section_id: str) -> dict:
    if not isinstance(section_id, str) or not _SECTION_ID_RE.fullmatch(section_id):
        raise ValueError("section_id 仅允许字母、数字、下划线和连字符")
    path = (SECTIONS_DIR / f"{section_id}.yaml").resolve()
    if path.parent != SECTIONS_DIR.resolve():
        raise ValueError("Section 路径越界")
    if not path.is_file():
        raise FileNotFoundError(f"未找到 Section 定义：{path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return validate_section(data, expected_id=section_id)


def render_section_brief(section: dict, company: str, report_period: str) -> str:
    section = validate_section(section, expected_id=section.get("id"))
    lines = [
        "## 本次任务的交付要求",
        "",
        f"公司：{company}",
        f"报告期间：{report_period}",
        f"Section：{section['name']}",
    ]

    if section["objective"]:
        lines.extend(["", "### 研究目标", section["objective"]])

    if section["deliverables"]:
        lines.extend(["", "### 必须交付的内容"])
        for item in section["deliverables"]:
            lines.append(
                f"- **{item['title']}**：{item['description']}"
            )

    if section["quality_requirements"]:
        lines.extend(["", "### 质量要求"])
        lines.extend(f"- {item}" for item in section["quality_requirements"])

    return "\n".join(lines)


def _deliverable_keywords(item: dict) -> list[str]:
    """从交付项标题与描述中提取用于覆盖判定的关键词。

    标题按顿号、连词切分为更细的短语提升命中率；描述按标点切分后仅保留
    2~12 字的中短短语，避免过长的完整句子永远无法命中。关键词按长度降序，
    优先精确短语匹配。
    """
    keywords: list[str] = []
    title = str(item.get("title", "")).strip()
    if title:
        keywords.append(title)
        for part in re.split(r"[、，,；;与及和/|]", title):
            part = part.strip()
            if len(part) >= 2:
                keywords.append(part)

    description = str(item.get("description", "")).strip()
    for part in re.split(r"[，。；：、\n]", description):
        part = part.strip()
        if 2 <= len(part) <= 12:
            keywords.append(part)

    seen: set[str] = set()
    unique: list[str] = []
    for keyword in keywords:
        if keyword and keyword not in seen:
            seen.add(keyword)
            unique.append(keyword)
    return sorted(unique, key=len, reverse=True)


def check_delivery(report: str, section: dict) -> dict:
    """校验报告正文是否覆盖 Section 定义的必交付内容。

    采用保守的关键词覆盖判定：任一关键词出现在报告标题或正文中即视为已覆盖。
    结果仅用于给出交付完整性提示与状态降级，不阻断成稿——关键词匹配无法替代
    语义判断，可能产生少量误报。
    """
    section = validate_section(section, expected_id=section.get("id"))
    report_text = report or ""
    headings = "\n".join(
        line for line in report_text.splitlines() if line.lstrip().startswith("#")
    )
    haystack = headings + "\n" + report_text

    covered: list[dict] = []
    missing: list[dict] = []
    for item in section.get("deliverables", []):
        keywords = _deliverable_keywords(item)
        matched = next((kw for kw in keywords if kw in haystack), None)
        entry = {"id": item["id"], "title": item["title"]}
        if matched:
            covered.append({**entry, "matched": matched})
        else:
            missing.append(entry)

    return {
        "deliverable_count": len(section.get("deliverables", [])),
        "covered_count": len(covered),
        "missing_count": len(missing),
        "covered": covered,
        "missing": missing,
        "complete": len(missing) == 0,
    }