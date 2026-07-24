import os
import json
import logging
import jieba
import uuid
import time
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from ..config.settings import settings
from database import vector_repo

logger = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.corpus = []  # List of dicts: {"id": str, "text": str, "metadata": dict}
        self._tokenized_corpus: List[List[str]] = []
        self._scope_cache: Dict[tuple[str, ...], tuple[List[Dict[str, Any]], Any]] = {}
        self.bm25 = None
        self._load()

    def _tokenize(self, text: str) -> List[str]:
        # 针对离散数学专业词汇，jieba 分词可能需要自定义词典，目前先使用默认
        return list(jieba.cut(text))

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                needs_migration = not isinstance(payload, dict)
                if isinstance(payload, dict):
                    self.corpus = payload.get("documents", [])
                    self._tokenized_corpus = payload.get("tokenized_documents", [])
                else:
                    # Backward compatibility with the original list-only store.
                    self.corpus = payload
                    self._tokenized_corpus = []
                if self.corpus:
                    if len(self._tokenized_corpus) != len(self.corpus):
                        self._tokenized_corpus = [
                            self._tokenize(doc["text"]) for doc in self.corpus
                        ]
                        needs_migration = True
                    self.bm25 = BM25Okapi(self._tokenized_corpus)
                if needs_migration:
                    self._save()
                logger.info(f"Loaded BM25 corpus from {self.storage_path}, size: {len(self.corpus)}")
            except Exception as e:
                logger.error(f"Failed to load BM25 corpus: {e}")
                self.corpus = []

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        "version": 2,
                        "documents": self.corpus,
                        "tokenized_documents": self._tokenized_corpus,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(f"Saved BM25 corpus to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save BM25 corpus: {e}")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        documents: List of {"id": str, "text": str, "metadata": dict}
        """
        hashes = {
            doc.get("metadata", {}).get("document_hash")
            for doc in documents
            if doc.get("metadata", {}).get("document_hash")
        }
        if hashes:
            retained = [
                (doc, tokens)
                for doc, tokens in zip(self.corpus, self._tokenized_corpus)
                if doc.get("metadata", {}).get("document_hash") not in hashes
            ]
            self.corpus = [doc for doc, _ in retained]
            self._tokenized_corpus = [tokens for _, tokens in retained]
        self.corpus.extend(documents)
        self._tokenized_corpus.extend([
            self._tokenize(doc["text"]) for doc in documents
        ])
        self.bm25 = BM25Okapi(self._tokenized_corpus)
        self._scope_cache.clear()
        self._save()

    def delete_material_documents(self, class_id: int, material_id: int):
        retained = [
            (doc, tokens)
            for doc, tokens in zip(self.corpus, self._tokenized_corpus)
            if not (
                doc.get("metadata", {}).get("class_id") == class_id
                and doc.get("metadata", {}).get("material_id") == material_id
            )
        ]
        self.corpus = [doc for doc, _ in retained]
        self._tokenized_corpus = [tokens for _, tokens in retained]
        if self.corpus:
            self.bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self.bm25 = None
        self._invalidate_scope_cache([f"class:{class_id}"])
        self._save()

    def update_document_access(
        self,
        content_hash: str,
        scope_keys: list[str],
        sources: list[dict],
    ):
        changed = False
        changed_scopes = set(scope_keys)
        for doc in self.corpus:
            metadata = doc.get("metadata", {})
            if metadata.get("document_hash") == content_hash:
                changed_scopes.update(metadata.get("scope_keys", []))
                metadata["scope_keys"] = scope_keys
                metadata["sources"] = sources
                changed = True
        if changed:
            self._invalidate_scope_cache(changed_scopes)
            self._save()

    def delete_document(self, content_hash: str):
        removed_scopes = set()
        retained = []
        for doc, tokens in zip(self.corpus, self._tokenized_corpus):
            if doc.get("metadata", {}).get("document_hash") == content_hash:
                removed_scopes.update(doc.get("metadata", {}).get("scope_keys", []))
            else:
                retained.append((doc, tokens))
        self.corpus = [doc for doc, _ in retained]
        self._tokenized_corpus = [tokens for _, tokens in retained]
        if self.corpus:
            self.bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self.bm25 = None
        self._invalidate_scope_cache(removed_scopes)
        self._save()

    def _invalidate_scope_cache(self, scope_keys) -> None:
        affected = set(scope_keys or [])
        if not affected or "global" in affected:
            self._scope_cache.clear()
            return
        for key in list(self._scope_cache):
            if affected.intersection(key):
                self._scope_cache.pop(key, None)

    def query(
        self,
        question: str,
        top_k: int = 5,
        scope_keys: list[str] | None = None,
    ) -> List[Dict[str, Any]]:
        if not self.corpus:
            return []

        cache_key = tuple(sorted(scope_keys or []))
        cached = self._scope_cache.get(cache_key)
        if cached:
            eligible, bm25 = cached
        else:
            eligible = self.corpus
            eligible_tokens = self._tokenized_corpus
            if scope_keys:
                allowed = set(scope_keys)
                pairs = [
                    (doc, tokens)
                    for doc, tokens in zip(self.corpus, self._tokenized_corpus)
                    if allowed.intersection(
                        doc.get("metadata", {}).get("scope_keys", [])
                    )
                ]
                eligible = [doc for doc, _ in pairs]
                eligible_tokens = [tokens for _, tokens in pairs]
            bm25 = BM25Okapi(eligible_tokens) if eligible_tokens else None
            self._scope_cache[cache_key] = (eligible, bm25)
        if not eligible:
            return []

        tokenized_query = self._tokenize(question)
        scores = bm25.get_scores(tokenized_query)
        
        # 获取得分最高的前 k 个索引
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for i in top_n:
            if scores[i] > 0:
                doc = eligible[i].copy()
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
            extra_metadata = c.get("metadata", {})
            bm25_docs.append({
                "id": c.get("id", str(uuid.uuid4())),
                "text": c["text"],
                "metadata": {
                    "source_type": c["source_type"],
                    "source_file": c["source_file"],
                    "chapter": c.get("chapter", ""),
                    "page": str(c.get("page", "")),
                    **extra_metadata,
                }
            })
        self.bm25_retriever.add_documents(bm25_docs)

    def delete_material_documents(self, class_id: int, material_id: int):
        vector_repo.delete_material_documents(class_id, material_id)
        self.bm25_retriever.delete_material_documents(class_id, material_id)

    def update_document_access(
        self,
        content_hash: str,
        scope_keys: list[str],
        sources: list[dict],
    ):
        vector_repo.update_document_access(content_hash, scope_keys, sources)
        self.bm25_retriever.update_document_access(
            content_hash, scope_keys, sources
        )

    def delete_document(self, content_hash: str):
        vector_repo.delete_document(content_hash)
        self.bm25_retriever.delete_document(content_hash)

    def clear_all(self):
        """
        清空所有索引（向量库和 BM25）
        """
        logger.info("正在清空所有 RAG 索引...")
        # 1. 清空向量库
        vector_repo.clear_collection()
        
        # 2. 清空 BM25
        self.bm25_retriever.corpus = []
        self.bm25_retriever._tokenized_corpus = []
        self.bm25_retriever._scope_cache.clear()
        self.bm25_retriever.bm25 = None
        if os.path.exists(self.bm25_retriever.storage_path):
            os.remove(self.bm25_retriever.storage_path)
        logger.info("所有 RAG 索引已清空")

    def query(
        self,
        question: str,
        top_k: int = None,
        class_id: int | None = None,
        request_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = settings.TOP_K
            
        logger.info("执行混合检索，查询长度=%s", len(question))
        scope_keys = ["global"]
        if class_id is not None:
            scope_keys.append(f"class:{class_id}")
        # 1. 获取向量检索结果 (取 2 倍 top_k 用于融合)
        vector_started_at = time.perf_counter()
        vector_results = vector_repo.query(
            question,
            top_k=top_k * 2,
            scope_keys=scope_keys,
            request_id=request_id,
        )
        logger.info(json.dumps({
            "event": "chat_timing",
            "request_id": request_id or "-",
            "stage": "vector_retrieval",
            "elapsed_ms": round((time.perf_counter() - vector_started_at) * 1000, 2),
            "result_count": len(vector_results),
        }, ensure_ascii=False))
        logger.info(f"向量检索返回 {len(vector_results)} 条结果")
        
        # 2. 获取 BM25 检索结果
        bm25_started_at = time.perf_counter()
        bm25_results = self.bm25_retriever.query(
            question,
            top_k=top_k * 2,
            scope_keys=scope_keys,
        )
        logger.info(json.dumps({
            "event": "chat_timing",
            "request_id": request_id or "-",
            "stage": "bm25",
            "elapsed_ms": round((time.perf_counter() - bm25_started_at) * 1000, 2),
            "result_count": len(bm25_results),
        }, ensure_ascii=False))
        logger.info(f"BM25 检索返回 {len(bm25_results)} 条结果")
        
        # 3. RRF 融合 (Reciprocal Rank Fusion)
        rrf_scores = {} # doc_id -> score
        doc_map = {}    # doc_id -> doc_content
        
        # 处理向量结果
        for rank, res in enumerate(vector_results):
            # 使用内容摘要和元数据生成唯一 ID
            doc_id = (
                f"{res.get('document_hash', '')}_"
                f"{res['page']}_{res['text'][:30]}"
            )
            doc_map[doc_id] = res
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (self.k + rank + 1)
            
        # 处理 BM25 结果
        for rank, res in enumerate(bm25_results):
            doc_id = (
                f"{res['metadata'].get('document_hash', '')}_"
                f"{res['metadata']['page']}_{res['text'][:30]}"
            )
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "text": res["text"],
                    "source_file": res["metadata"]["source_file"],
                    "source_type": res["metadata"]["source_type"],
                    "chapter": res["metadata"]["chapter"],
                    "page": res["metadata"]["page"],
                    "document_hash": res["metadata"].get("document_hash", ""),
                    "sources": res["metadata"].get("sources", []),
                    "similarity": 0.0 
                }
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (self.k + rank + 1)
            
        # 按 RRF 得分排序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        final_results = []
        for doc_id in sorted_ids:
            res = doc_map[doc_id]
            sources = res.get("sources") or res.get("metadata", {}).get("sources", [])
            class_source = next(
                (
                    source for source in sources
                    if class_id is not None and source.get("class_id") == class_id
                ),
                None,
            )
            global_source = next(
                (
                    source for source in sources
                    if source.get("scope_type") == "global"
                ),
                None,
            )
            selected_source = class_source or global_source
            if selected_source:
                res["source_file"] = selected_source.get(
                    "filename", res.get("source_file", "未知资料")
                )
            res["rrf_score"] = round(rrf_scores[doc_id], 4)
            final_results.append(res)
            
        logger.info(f"混合检索最终返回 {len(final_results)} 条结果")
        return final_results
