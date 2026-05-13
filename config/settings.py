import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    # ── Chat API 配置 ──
    CHAT_API_KEY: str
    CHAT_BASE_URL: str
    CHAT_MODEL_NAME: str
    
    # ── Embedding API 配置 ──
    EMBED_API_KEY: str
    EMBED_BASE_URL: str
    EMBED_MODEL_NAME: str
    
    # ── 向量数据库配置 (Qdrant) ──
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "discrete_math_materials"
    QDRANT_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "qdrant_storage")

    # ── 系统提示词 ──
    SYSTEM_PROMPT: str = """你是一名离散数学课程的智能助教，名字叫"小离"。
你的职责：
- 耐心解答学生关于离散数学的各种问题
- 解释概念时先给结论，再举例说明，保持简洁清晰
- 如果问题与离散数学无关，礼貌说明并引导回课程话题
"""

    # ── 路径配置 ──
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR: str = os.path.join(DATA_DIR, "pptx")
    PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
    
    BM25_DB_PATH: str = os.path.join(DATA_DIR, "bm25_store.json")
    SQLITE_DB_PATH: str = os.path.join(DATA_DIR, "course_info.db")

    # ── 检索配置 ──
    TOP_K: int = 3

    # ── 日志配置 ──
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = os.path.join(DATA_DIR, "app.log")

    MAX_TOKENS: int = 2048

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env"), 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

settings = Settings()
