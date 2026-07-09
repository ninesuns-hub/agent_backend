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
    
    # ── 视觉识图 API（默认复用 EMBED 网关，需支持 vision 的模型）──
    VISION_API_KEY: str = ""
    VISION_BASE_URL: str = ""
    VISION_MODEL_NAME: str = "gpt-4o"
    
    # ── 向量数据库配置 (Qdrant) ──
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "discrete_math_materials"
    QDRANT_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "storage", "processed", "vector_db")

    # ── MySQL 配置 ──
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "L15834556205"
    MYSQL_DB: str = "Discrete"

    # ── Redis 配置 ──
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""

    # ── 鉴权配置 ──
    JWT_SECRET: str = "discrete_math_secret_key_2026"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # ── 邮箱配置 ──
    SMTP_HOST: str = "smtp.qq.com"
    SMTP_PORT: int = 465
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""

    # ── 系统提示词 ──
    SYSTEM_PROMPT: str = r"""你是一名离散数学课程的智能助教，名字叫"小离"。
你的职责：
- 耐心解答学生关于离散数学的各种问题。
- 解释概念时先给结论，再举例说明，保持简洁清晰。
- 回答可以使用标准 Markdown 排版；数学表达请使用标准 LaTeX，行内公式使用 `$...$`，块级公式使用 `$$...$$`。禁止用方括号 `[...]` 包裹公式。矩阵换行请写 `\\\\`。
- 如果需要画集合关系、Hasse 图、树、图论结构或流程图，优先输出 fenced Mermaid 代码块，例如 ```mermaid。
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
    CLASS_MATERIALS_DIR: str = os.path.join(RAW_DIR, "classes")
    HOMEWORK_DIR: str = os.path.join(RAW_DIR, "homework")
    CHAT_IMAGES_DIR: str = os.path.join(RAW_DIR, "chat_images")
    
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
