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
    QDRANT_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "storage", "processed", "vector_db")

    # ── 系统提示词 ──
    SYSTEM_PROMPT: str = r"""你是一名离散数学课程的智能助教，名字叫"小离"。
你的职责：
- 耐心解答学生关于离散数学的各种问题。
- 解释概念时先给结论，再举例说明，保持简洁清晰。
- **重要：在终端显示时，请直接使用 Unicode 数学符号来表示公式，严禁使用 LaTeX 源码。**
  常见符号对照：
  - 否定: ¬ (而非 \neg)
  - 析取/合取: ∨ / ∧ (而非 \vee / \wedge)
  - 蕴含/等价: → / ↔ (而非 \to / \leftrightarrow)
  - 全称/存在量词: ∀ / ∃ (而非 \forall / \exists)
  - 集合运算: ∈, ∉, ⊆, ⊂, ∩, ∪, ∅, \overline{A} (补集用上横线或 Unicode 组合字符)
- **注意：课程课件内容主要是英文。当学生用中文提问时，请自动将其转化为英文关键词进行检索，并在回答时将课件中的英文术语翻译为中文。**
- 如果通过工具检索到了课件内容，请优先基于课件内容回答；如果检索不到，请基于你掌握的通用离散数学知识回答，并礼貌说明这是通用定义。
- 如果问题与离散数学完全无关，礼貌说明并引导回课程话题。
"""

    # ── 路径配置 ──
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    STORAGE_DIR: str = os.path.join(BASE_DIR, "storage")
    
    # 原始资产目录
    RAW_DIR: str = os.path.join(STORAGE_DIR, "raw")
    COURSE_ASSETS_DIR: str = os.path.join(RAW_DIR, "courses")
    
    # 知识处理目录 (RAG)
    PROCESSED_DIR: str = os.path.join(STORAGE_DIR, "processed")
    CHUNKS_DIR: str = os.path.join(PROCESSED_DIR, "chunks")
    BM25_DB_PATH: str = os.path.join(CHUNKS_DIR, "bm25_store.json")
    
    # 安全与管理目录
    SECURE_DIR: str = os.path.join(STORAGE_DIR, "secure")
    SQLITE_DB_PATH: str = os.path.join(SECURE_DIR, "users", "course_info.db")
    LOG_FILE: str = os.path.join(SECURE_DIR, "logs", "app.log")

    # ── 检索配置 ──
    TOP_K: int = 3

    # ── 日志配置 ──
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # LOG_FILE 已经在路径配置中定义

    MAX_TOKENS: int = 2048

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env"), 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

settings = Settings()
