import logging
from .base import Tool

logger = logging.getLogger(__name__)

def create_knowledge_tool(vector_query_func):
    """
    创建教学内容知识库查询工具
    """
    def wrapper(question: str) -> str:
        logger.info(f"知识库查询工具收到问题: {question}")
        results = vector_query_func(question)
        if not results:
            logger.warning("知识库查询未找到结果")
            return "在课件资料中未找到相关知识点。"
        
        # 格式化 Observation，让模型知道来源
        formatted_results = []
        for r in results:
            score_info = f"相似度：{r['similarity']}" if 'similarity' in r and r['similarity'] > 0 else f"综合得分：{r.get('rrf_score', 'N/A')}"
            source = f"【来源：{r['source_file']} 第{r['page']}页，{score_info}】"
            content = f"内容：{r['text']}"
            formatted_results.append(f"{source}\n{content}")
        
        output = "\n\n".join(formatted_results)
        logger.info(f"知识库查询返回 {len(formatted_results)} 条结果")
        return output

    return Tool(
        name="query_lecture_knowledge",
        func=wrapper,
        description="非常有用！当你需要回答离散数学的具体知识点、定义、定理、公式或课件中的例题时，请调用此工具。输入应该是具体的数学概念或问题。"
    )
