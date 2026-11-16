"""Run 级稳定来源标识与最终报告引用编号归一化。"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any


SOURCE_TOKEN_RE = re.compile(r"\[SRC:([A-Za-z0-9_-]+)\]")
NUMERIC_CITATION_RE = re.compile(r"(?<![A-Za-z0-9_])\[(\d+)\]")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class SourceRegistry:
    """同一 Run 中跨 query、跨 researcher 共享的来源注册表。"""

    def __init__(self, run_dir: str | Path):
        self.path = Path(run_dir) / "_internal" / "source_registry.json"
        self._lock = threading.RLock()
        stored = _read_json(self.path)
        self._sources: dict[str, dict[str, str]] = stored.get("sources", {})

    @staticmethod
    def _identity(chunk: dict) -> tuple[str, str]:
        doc_id = str(chunk.get("doc_id") or "").strip()
        title = str(chunk.get("title") or "").strip()
        content = str(
            chunk.get("content_with_weight")
            or chunk.get("content")
            or chunk.get("content_ltks")
            or ""
        ).strip()

        if doc_id:
            identity = "doc:" + doc_id
        elif title:
            identity = "title:" + title
        else:
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            identity = "content:" + digest
        display_title = title or "未命名文档（来源元数据缺失）"
        return identity, display_title

    def register(self, chunk: dict) -> str:
        identity, title = self._identity(chunk)
        source_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        with self._lock:
            current = self._sources.get(source_id)
            if current is None:
                self._sources[source_id] = {"title": title}
                _atomic_write_json(self.path, {"sources": self._sources})
            elif current.get("title") in {"", "未命名文档（来源元数据缺失）"} and title:
                current["title"] = title
                _atomic_write_json(self.path, {"sources": self._sources})
        return f"SRC:{source_id}"

    def snapshot(self) -> dict[str, dict[str, str]]:
        with self._lock:
            return {key: dict(value) for key, value in self._sources.items()}


def _strip_trailing_sources_section(report: str) -> str:
    """移除模型可能自行生成的尾部来源清单，统一由程序重建。"""
    pattern = re.compile(
        r"(?ims)^\s{0,3}#{1,3}\s*(?:来源|来源映射|Sources|参考来源|参考文献)\s*$.*\Z"
    )
    matches = list(pattern.finditer(report))
    if not matches:
        return report.rstrip()
    return report[:matches[-1].start()].rstrip()


def finalize_report_citations(
    report: str,
    registry_path: str | Path,
) -> tuple[str, dict[str, Any]]:
    """把稳定 SRC 标识转为最终 [n]，并校验无法映射的引用。"""
    path = Path(registry_path)
    registry = _read_json(path)
    sources = registry.get("sources", {})
    if not isinstance(sources, dict):
        sources = {}

    body = _strip_trailing_sources_section(report.strip())
    unresolved_numeric = sorted(set(NUMERIC_CITATION_RE.findall(body)))
    used_ids: list[str] = []
    for match in SOURCE_TOKEN_RE.finditer(body):
        source_id = match.group(1)
        if source_id not in used_ids:
            used_ids.append(source_id)

    known_ids = [item for item in used_ids if item in sources]
    unknown_ids = [item for item in used_ids if item not in sources]
    number_map = {source_id: index for index, source_id in enumerate(known_ids, 1)}

    def replace_source(match: re.Match) -> str:
        source_id = match.group(1)
        if source_id in number_map:
            return f"[{number_map[source_id]}]"
        return f"[待核验来源:{source_id}]"

    body = SOURCE_TOKEN_RE.sub(replace_source, body)

    # 旧式 [1] 编号没有跨 Findings 的可靠映射，不能擅自猜测其来源。
    warnings: list[str] = []
    if unknown_ids:
        warnings.append("存在无法在本次 Run 来源注册表中找到的 SRC 标识")
    if not used_ids:
        warnings.append("终稿未包含任何稳定来源标识，重要结论的引用支撑需要人工核对")
    if unresolved_numeric:
        warnings.append(
            "存在未映射的数字引用编号：" + "、".join(unresolved_numeric)
        )

    if warnings:
        body += (
            "\n\n> **引用校验提示**：" + "；".join(warnings)
            + "。这些引用未被自动归因，请人工核对原始 Findings。"
        )

    if known_ids:
        body += "\n\n## 来源\n\n"
        body += "\n".join(
            f"[{number_map[source_id]}] {sources[source_id].get('title') or '未命名文档'}"
            for source_id in known_ids
        )

    validation = {
        "source_count": len(known_ids),
        "registered_source_count": len(sources),
        "unknown_source_ids": unknown_ids,
        "unmapped_numeric_citations": unresolved_numeric,
        "has_warnings": bool(warnings),
    }
    return body.rstrip() + "\n", validation
