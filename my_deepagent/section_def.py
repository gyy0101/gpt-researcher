"""Section Definition 的加载与渲染。

Section Definition 只定义「最终交付什么」以及基本质量要求，
不规定搜索词或具体研究步骤。研究路径由 Chief Researcher 自主决定。
"""
import os

import yaml

SECTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sections")


def load_section(section_id: str) -> dict:
    """读取指定 Section 的定义文件，返回 dict。

    定义文件为 YAML，位于 sections/<section_id>.yaml。
    """
    path = os.path.join(SECTIONS_DIR, f"{section_id}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 Section 定义：{path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Section 定义格式错误（应为 YAML 映射）：{path}")
    if not data.get("id"):
        data["id"] = section_id
    return data


def render_section_brief(section: dict, company: str, report_period: str) -> str:
    """把 Section 定义渲染为给 Chief Researcher 的交付要求文本。"""
    name = section.get("name") or section.get("id", "")
    objective = (section.get("objective") or "").strip()

    lines = [f"## 本次任务的交付要求", "", f"公司：{company}", f"报告期间：{report_period}", f"Section：{name}"]
    if objective:
        lines.append("")
        lines.append("### 研究目标")
        lines.append(objective)

    deliverables = section.get("deliverables") or []
    if deliverables:
        lines.append("")
        lines.append("### 必须交付的内容")
        for d in deliverables:
            title = d.get("title") or d.get("id", "")
            desc = (d.get("description") or "").strip()
            lines.append(f"- **{title}**：{desc}")

    quality = section.get("quality_requirements") or []
    if quality:
        lines.append("")
        lines.append("### 质量要求")
        for q in quality:
            lines.append(f"- {q}")

    return "\n".join(lines)