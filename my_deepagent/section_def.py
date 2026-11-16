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
