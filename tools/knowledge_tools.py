import logging
from .base import Tool

logger = logging.getLogger(__name__)

def create_knowledge_tool(vector_query_func, context_getter=None):
    """
    创建教学内容知识库查询工具
    """
    def wrapper(question: str, run_context: dict | None = None) -> str:
        logger.info("知识库查询工具收到查询，长度=%s", len(question))
        context = run_context or (context_getter() if context_getter else {}) or {}
        query_kwargs = {"class_id": context.get("class_id")}
        if context.get("request_id"):
            query_kwargs["request_id"] = context["request_id"]
        results = vector_query_func(question, **query_kwargs)
        if not results:
            logger.warning("知识库查询未找到结果")
            return "在课件资料中未找到相关知识点。"
        
        # 格式化 Observation，让模型知道可引用来源和正文片段
        formatted_results = []
        for idx, r in enumerate(results, start=1):
            score_info = f"相似度：{r['similarity']}" if 'similarity' in r and r['similarity'] > 0 else f"综合得分：{r.get('rrf_score', 'N/A')}"
            source_file = r.get("source_file", "未知资料")
            page = r.get("page", "未知页码")
            content = r.get("text", "")
            formatted_results.append(
                "\n".join([
                    f"### 检索片段 {idx}",
                    f"- 可引用来源：{source_file}，第 {page} 页",
                    f"- 检索得分：{score_info}",
                    "- 片段内容：",
                    content,
                ])
            )
        
        output = "\n\n".join(formatted_results)
        logger.info(f"知识库查询返回 {len(formatted_results)} 条结果")
        return output

    return Tool(
        name="query_lecture_knowledge",
        func=wrapper,
        accepts_context=True,
        description="非常有用！当你需要回答离散数学的具体知识点、定义、定理、公式或课件中的例题时，请调用此工具。输入应该是具体的数学概念或问题。"
    )
