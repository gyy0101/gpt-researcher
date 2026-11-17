"""my_deepagent 第一阶段可靠性重构的单元测试。

覆盖范围（对应本次修改的五个核心问题）：

1. Findings 文件路径安全：``_locate_finding`` 拒绝目录越界与前缀串撞。
2. BudgetMiddleware 预算预留/结算/异常释放：``BudgetManager`` 的
   ``reserve_usage`` / ``settle_usage`` / ``consume_reservation``，以及
   模型调用异常与缺少 usage_metadata 时不泄漏预留。
3. Subagent 研究结果持久化：``BudgetMiddleware._persist_task_result`` 的
   落盘路径选择、越界兜底与短内容跳过。
4. Writer Findings 完整性：``_assemble_findings_context`` 超预算时不静默丢弃，
   省略条目保留明确清单，提炼失败时使用有损兜底保留 SRC 标识。
5. Section 交付完整性：``check_delivery`` 关键词覆盖判定与主流程状态降级。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

# 保证 my_deepagent 命名空间包可被导入（项目根目录加入 sys.path）。
_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from my_deepagent.budget import (  # noqa: E402
    BudgetManager,
    BudgetMiddleware,
    _locate_finding,
)
from my_deepagent.section_def import check_delivery, validate_section  # noqa: E402
from my_deepagent.writer import _assemble_findings_context  # noqa: E402
from my_deepagent.main import (  # noqa: E402
    STATUS_COMPLETED_BY_WRITER,
    STATUS_DEGRADED,
    _apply_degraded,
)


def make_budget(**overrides) -> BudgetManager:
    defaults = dict(
        max_subagent_tasks=6,
        max_quick_search_calls=30,
        max_deep_research_calls=12,
        per_task_quick_quota=4,
        per_task_deep_quota=3,
        chief_quick_reserve=8,
        max_context_chars=35000,
        max_tokens=None,
        max_cost=None,
        pricing_tier="auto",
    )
    defaults.update(overrides)
    return BudgetManager(**defaults)


# ---------------------------------------------------------------------------
# 1. Findings 文件路径安全
# ---------------------------------------------------------------------------


class TestLocateFinding:
    def test_normal_path_within_findings(self, tmp_path):
        run_dir = tmp_path / "run"
        fdir = run_dir / "findings"
        fdir.mkdir(parents=True)
        (fdir / "01-supply.md").write_text("数据", encoding="utf-8")

        assert _locate_finding(run_dir, "findings/01-supply.md") == (fdir / "01-supply.md").resolve()

    def test_non_findings_path_returns_none(self, tmp_path):
        run_dir = tmp_path / "run"
        (run_dir / "findings").mkdir(parents=True)

        assert _locate_finding(run_dir, "other/01-supply.md") is None
        assert _locate_finding(run_dir, "findings/plan.md") is None

    def test_rejects_directory_escape(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (tmp_path / "secret.md").write_text("越界", encoding="utf-8")

        # ../../../.. 上穿 run_dir 后应被拒绝，而不是定位到 run_dir 之外的 secret.md。
        assert _locate_finding(run_dir, "findings/../../secret.md") is None

    def test_rejects_prefix_collision(self, tmp_path):
        # 回归测试：旧实现用 str.startswith(run_dir) 判断，会把 run2 误判为 run 的子目录。
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        sibling = tmp_path / "run2" / "findings"
        sibling.mkdir(parents=True)
        (sibling / "secret.md").write_text("越界", encoding="utf-8")

        assert _locate_finding(run_dir, "findings/../../run2/findings/secret.md") is None


# ---------------------------------------------------------------------------
# 2. 预算预留 / 结算 / 异常释放
# ---------------------------------------------------------------------------


class TestBudgetReservation:
    def test_reserve_and_settle_usage(self):
        budget = make_budget(max_tokens=1000)
        ok, rid = budget.reserve_usage(600, "deepseek-flash")
        assert ok is True and rid is not None
        assert budget.summary()["reserved_tokens"] == 600

        budget.settle_usage(rid, 100, 50, "deepseek-flash")
        summary = budget.summary()
        assert summary["reserved_tokens"] == 0
        assert summary["settled_tokens"] == 150

    def test_reserve_rejects_over_budget(self):
        budget = make_budget(max_tokens=1000)
        ok1, _ = budget.reserve_usage(600, "deepseek-flash")
        ok2, rid2 = budget.reserve_usage(600, "deepseek-flash")

        assert ok1 is True
        assert ok2 is False and rid2 is None

    def test_consume_reservation_releases_without_settling(self):
        budget = make_budget(max_tokens=1000)
        ok, rid = budget.reserve_usage(600, "deepseek-flash")
        assert ok is True

        budget.consume_reservation(rid)
        summary = budget.summary()
        assert summary["reserved_tokens"] == 0
        assert summary["settled_tokens"] == 0

    def test_none_reservation_id_is_noop(self):
        budget = make_budget(max_tokens=1000)
        budget.consume_reservation(None)
        budget.settle_usage(None, 10, 10, "deepseek-flash")

        summary = budget.summary()
        assert summary["reserved_tokens"] == 0
        assert summary["settled_tokens"] == 0


class TestModelCallReservation:
    def _request(self):
        return SimpleNamespace(
            model=SimpleNamespace(model_name="deepseek-flash", max_tokens=None),
            messages=[],
            system_message=None,
            model_settings={},
        )

    @pytest.mark.asyncio
    async def test_exception_releases_reservation(self):
        budget = make_budget(max_tokens=100000)
        middleware = BudgetMiddleware(budget)

        async def bad_handler(_request):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await middleware.awrap_model_call(self._request(), bad_handler)

        summary = budget.summary()
        assert summary["reserved_tokens"] == 0
        assert summary["settled_tokens"] == 0

    @pytest.mark.asyncio
    async def test_missing_usage_metadata_releases_reservation(self):
        budget = make_budget(max_tokens=100000)
        middleware = BudgetMiddleware(budget)

        async def ok_handler(_request):
            return AIMessage(content="无 usage_metadata 的返回")

        response = await middleware.awrap_model_call(self._request(), ok_handler)

        assert response.content == "无 usage_metadata 的返回"
        summary = budget.summary()
        assert summary["reserved_tokens"] == 0
        assert summary["settled_tokens"] == 0


# ---------------------------------------------------------------------------
# 3. Subagent 研究结果持久化
# ---------------------------------------------------------------------------


class TestPersistTaskResult:
    def _request(self, description: str, call_id: str = "call123"):
        return SimpleNamespace(
            tool_call={"id": call_id, "args": {"description": description}}
        )

    def test_uses_description_path_and_replaces_content(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        middleware = BudgetMiddleware(make_budget(), run_dir=run_dir)

        result = ToolMessage(
            content="供给分析内容" * 20, tool_call_id="call123"
        )
        out = middleware._persist_task_result(
            self._request("请研究供给情况，写入 findings/01-supply.md"), result
        )

        fpath = run_dir / "findings" / "01-supply.md"
        assert fpath.exists()
        text = fpath.read_text(encoding="utf-8")
        assert "程序自动持久化" in text
        assert "供给分析内容" in text
        # 返回给 Chief 的是精简摘要，而非完整 Findings。
        assert isinstance(out, ToolMessage)
        assert "01-supply.md" in out.content
        assert "供给分析内容" * 20 not in out.content

    def test_rejects_traversal_and_falls_back_to_safe_name(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        middleware = BudgetMiddleware(make_budget(), run_dir=run_dir)

        result = ToolMessage(content="研究内容" * 30, tool_call_id="call123")
        out = middleware._persist_task_result(
            self._request("请研究并写入 findings/../../evil.md"), result
        )

        # 不得写入 run_dir 之外。
        assert not (tmp_path / "evil.md").exists()
        fallback = run_dir / "findings" / "task_call123.md"
        assert fallback.exists()
        assert isinstance(out, ToolMessage)
        assert "task_call123.md" in out.content

    def test_skips_short_content_without_writing(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        middleware = BudgetMiddleware(make_budget(), run_dir=run_dir)

        result = ToolMessage(content="太短", tool_call_id="call123")
        out = middleware._persist_task_result(
            self._request("写入 findings/a.md"), result
        )

        assert out is result
        assert not (run_dir / "findings").exists()


# ---------------------------------------------------------------------------
# 4. Writer Findings 完整性
# ---------------------------------------------------------------------------


class TestAssembleFindingsContext:
    def test_within_budget_returns_all(self):
        findings = [("a.md", "内容A", True), ("b.md", "内容B", True)]
        out = _assemble_findings_context(findings, 10000)

        assert "a.md" in out
        assert "b.md" in out

    def test_over_budget_lists_omitted_findings(self, monkeypatch):
        monkeypatch.setattr(
            "my_deepagent.writer._digest_finding",
            lambda name, content: f"[摘要]{name}",
        )
        findings = [
            ("a.md", "x" * 1000, True),
            ("b.md", "y" * 1000, True),
            ("c.md", "z" * 1000, True),
        ]

        out = _assemble_findings_context(findings, 40)

        # 被保留的第一份仍可见，其余 Findings 以明确清单列出，而非静默丢弃。
        assert "a.md" in out
        assert "以下 Findings 因上下文预算已省略" in out
        assert "b.md" in out
        assert "c.md" in out

    def test_digest_failure_uses_lossy_fallback_preserving_src(self, monkeypatch):
        def boom(_name, _content):
            raise RuntimeError("no llm")

        monkeypatch.setattr("my_deepagent.writer._digest_finding", boom)

        content = "## 小节\n\n正文内容 [SRC:abcdef123456] 更多内容" + "补" * 40
        out = _assemble_findings_context([("a.md", content, True)], 50)

        assert "压缩兜底" in out
        assert "[SRC:abcdef123456]" in out


# ---------------------------------------------------------------------------
# 5. Section 交付完整性校验
# ---------------------------------------------------------------------------


def _section() -> dict:
    return validate_section({
        "id": "sec",
        "name": "行业竞争",
        "deliverables": [
            {"id": "d1", "title": "供需分析", "description": "分析供给与需求的矛盾"},
            {"id": "d2", "title": "竞争优势", "description": "公司的竞争优势与劣势"},
        ],
    })


class TestCheckDelivery:
    def test_all_covered(self):
        report = (
            "# 行业竞争\n\n"
            "## 供需分析\n\n本文分析供给与需求的矛盾。\n\n"
            "## 竞争优势\n\n公司具备竞争优势。"
        )
        result = check_delivery(report, _section())

        assert result["complete"] is True
        assert result["covered_count"] == 2
        assert result["missing_count"] == 0

    def test_missing_reported(self):
        report = "# 别的\n\n完全无关的内容"
        result = check_delivery(report, _section())

        assert result["complete"] is False
        assert result["missing_count"] == 2
        assert {item["id"] for item in result["missing"]} == {"d1", "d2"}

    def test_structure_contains_expected_keys(self):
        result = check_delivery("", _section())

        assert set(result) == {
            "deliverable_count",
            "covered_count",
            "missing_count",
            "covered",
            "missing",
            "complete",
        }


class TestApplyDegraded:
    def test_delivery_gaps_trigger_degraded(self):
        assert (
            _apply_degraded(
                STATUS_COMPLETED_BY_WRITER, False, False, None, has_delivery_gaps=True
            )
            == STATUS_DEGRADED
        )

    def test_no_gaps_keeps_status(self):
        assert (
            _apply_degraded(STATUS_COMPLETED_BY_WRITER, False, False, None)
            == STATUS_COMPLETED_BY_WRITER
        )