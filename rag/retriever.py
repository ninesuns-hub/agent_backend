import os
import json
import logging
import jieba
import uuid
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from ..config.settings import settings
from database import vector_repo

logger = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.corpus = []  # List of dicts: {"id": str, "text": str, "metadata": dict}
        self.bm25 = None
        self._load()

    def _tokenize(self, text: str) -> List[str]:
        # 针对离散数学专业词汇，jieba 分词可能需要自定义词典，目前先使用默认
        return list(jieba.cut(text))

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.corpus = json.load(f)
                if self.corpus:
                    tokenized_corpus = [self._tokenize(doc["text"]) for doc in self.corpus]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                logger.info(f"Loaded BM25 corpus from {self.storage_path}, size: {len(self.corpus)}")
            except Exception as e:
                logger.error(f"Failed to load BM25 corpus: {e}")
                self.corpus = []

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.corpus, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved BM25 corpus to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save BM25 corpus: {e}")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        documents: List of {"id": str, "text": str, "metadata": dict}
        """
        self.corpus.extend(documents)
        tokenized_corpus = [self._tokenize(doc["text"]) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self._save()

    def query(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.corpus:
            return []
        
        tokenized_query = self._tokenize(question)
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取得分最高的前 k 个索引
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for i in top_n:
            if scores[i] > 0:
                doc = self.corpus[i].copy()
                doc["bm25_score"] = float(scores[i])
                results.append(doc)
        return results

class HybridSearcher:
    def __init__(self):
        self.bm25_retriever = BM25Retriever(settings.BM25_DB_PATH)
        self.k = 60  # RRF 常数

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        同步向向量库和 BM25 索引添加文档
        """
        # 1. 写入向量库
        vector_repo.add_documents(chunks)
        
        # 2. 写入 BM25
        bm25_docs = []
        for c in chunks:
            bm25_docs.append({
                "id": c.get("id", str(uuid.uuid4())),
                "text": c["text"],
                "metadata": {
                    "source_type": c["source_type"],
                    "source_file": c["source_file"],
                    "chapter": c.get("chapter", ""),
                    "page": str(c.get("page", "")),
                }
            })
        self.bm25_retriever.add_documents(bm25_docs)

    def query(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = settings.TOP_K
            
        logger.info(f"执行混合检索: {question}")
        # 1. 获取向量检索结果 (取 2 倍 top_k 用于融合)
        vector_results = vector_repo.query(question, top_k=top_k * 2)
        logger.info(f"向量检索返回 {len(vector_results)} 条结果")
        
        # 2. 获取 BM25 检索结果
        bm25_results = self.bm25_retriever.query(question, top_k=top_k * 2)
        logger.info(f"BM25 检索返回 {len(bm25_results)} 条结果")
        
        # 3. RRF 融合 (Reciprocal Rank Fusion)
        rrf_scores = {} # doc_id -> score
        doc_map = {}    # doc_id -> doc_content
        
        # 处理向量结果
        for rank, res in enumerate(vector_results):
            # 使用内容摘要和元数据生成唯一 ID
            doc_id = f"{res['source_file']}_{res['page']}_{res['text'][:30]}"
            doc_map[doc_id] = res
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (self.k + rank + 1)
            
        # 处理 BM25 结果
        for rank, res in enumerate(bm25_results):
            doc_id = f"{res['metadata']['source_file']}_{res['metadata']['page']}_{res['text'][:30]}"
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "text": res["text"],
                    "source_file": res["metadata"]["source_file"],
                    "source_type": res["metadata"]["source_type"],
                    "chapter": res["metadata"]["chapter"],
                    "page": res["metadata"]["page"],
                    "similarity": 0.0 
                }
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (self.k + rank + 1)
            
        # 按 RRF 得分排序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        final_results = []
        for doc_id in sorted_ids:
            res = doc_map[doc_id]
            res["rrf_score"] = round(rrf_scores[doc_id], 4)
            final_results.append(res)
            
        logger.info(f"混合检索最终返回 {len(final_results)} 条结果")
        return final_results
