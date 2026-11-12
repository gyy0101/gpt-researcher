"""tools.py 纯函数的单元测试（不调用任何外部 API）。

重点验证 Chunk 去重修复：同一文档下内容不同的 chunk 不应被误删。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from my_deepagent.tools import _chunk_key, _dedup_chunks  # noqa: E402


class TestChunkDedup(unittest.TestCase):
    def test_same_doc_different_content_not_dropped(self):
        """同一 doc_id、无 chunk_id、无位置信息、正文不同 → 两个都保留。"""
        chunks = [
            {"doc_id": "doc-1", "content": "动力电池出货量 300GWh"},
            {"doc_id": "doc-1", "content": "储能电池出货量 90GWh"},
        ]
        result = _dedup_chunks(chunks)
        self.assertEqual(len(result), 2)

    def test_same_chunk_id_is_deduped(self):
        """有稳定 chunk_id 且相同 → 去重为 1 个。"""
        chunks = [
            {"chunk_id": "c-1", "doc_id": "doc-1", "content": "内容A"},
            {"chunk_id": "c-1", "doc_id": "doc-1", "content": "内容A"},
        ]
        result = _dedup_chunks(chunks)
        self.assertEqual(len(result), 1)

    def test_same_doc_same_position_same_content_is_deduped(self):
        """同文档、同位置、同正文（真正的重复检索）→ 去重为 1 个。"""
        chunks = [
            {"doc_id": "doc-1", "page_num_int": 3, "content": "内容A"},
            {"doc_id": "doc-1", "page_num_int": 3, "content": "内容A"},
        ]
        result = _dedup_chunks(chunks)
        self.assertEqual(len(result), 1)

    def test_same_doc_different_position_not_dropped(self):
        """同一文档不同页码 → 视为不同 chunk，两个都保留。"""
        chunks = [
            {"doc_id": "doc-1", "page_num_int": 3, "content": "内容A"},
            {"doc_id": "doc-1", "page_num_int": 4, "content": "内容A"},
        ]
        result = _dedup_chunks(chunks)
        self.assertEqual(len(result), 2)

    def test_order_is_preserved(self):
        chunks = [
            {"doc_id": "doc-2", "content": "乙"},
            {"doc_id": "doc-1", "content": "甲"},
            {"doc_id": "doc-2", "content": "乙"},  # 重复
        ]
        result = _dedup_chunks(chunks)
        self.assertEqual([c["content"] for c in result], ["乙", "甲"])

    def test_chunk_key_uses_content_hash_without_position(self):
        """无 chunk_id、无位置信息时，键应包含正文摘要哈希，区分不同正文。"""
        a = {"doc_id": "doc-1", "content": "甲"}
        b = {"doc_id": "doc-1", "content": "乙"}
        self.assertNotEqual(_chunk_key(a), _chunk_key(b))


if __name__ == "__main__":
    unittest.main(verbosity=2)