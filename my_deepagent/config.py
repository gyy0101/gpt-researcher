"""集中管理配置：从环境变量读取 DeepSeek 与 RAG 的连接信息。"""
import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path)
except Exception:
    pass


def _parse_list(value: str):
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass
class Config:
    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_timeout: int = 120
    # RAG
    rag_base_url: str = ""
    rag_kb_ids: list = field(default_factory=list)
    rag_as_of_date: str = ""
    rag_timeout: int = 60

    @classmethod
    def from_env(cls):
        return cls(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            deepseek_timeout=int(os.getenv("DEEPSEEK_TIMEOUT", "120")),
            rag_base_url=os.getenv("RAG_BASE_URL", ""),
            rag_kb_ids=_parse_list(os.getenv("RAG_KB_IDS", "")),
            rag_as_of_date=os.getenv("RAG_AS_OF_DATE", ""),
            rag_timeout=int(os.getenv("RAG_TIMEOUT", "60")),
        )


_config = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config