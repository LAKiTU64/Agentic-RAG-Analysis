import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- Config ---
EMBEDDING_MODEL = "/workspaces/models/bge-small-zh-v1.5"
CHROMA_PATH = "/workspaces/ai-agent/AI_Agent_Complete/.chroma_db"
DOCS_DIR = "/workspaces/ai-agent/AI_Agent_Complete/documents"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 100
DEFAULT_SEARCH_K = 3
SIMILARITY_THRESHOLD = 0.5

# 北京时间
BEIJING_TZ = timezone(timedelta(hours=8))


class VectorKBManager:
    """
    向量知识库管理类（ChromaDB）：使用本地 Embedding 模型。
    变更点（POC最小改动）：
    - chunk 元数据从 doc_id(文件名) 改为 document_id(UUID)
    - 仍保留 filename 字段用于展示/追溯
    - 提供 add_document 返回结构化结果，便于 Web 接口使用
    """

    def __init__(self, persist_directory: str = CHROMA_PATH) -> None:
        self.persist_directory = persist_directory

        # 严格检查本地模型路径
        if not os.path.exists(EMBEDDING_MODEL):
            raise FileNotFoundError(
                f"❌ 找不到本地模型目录: {EMBEDDING_MODEL}。请确保模型已下载到该位置。"
            )

        # 初始化 Embedding：强制开启 local_files_only，禁止联网下载
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu",
                "local_files_only": True,
            },
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore: Optional[Chroma] = None
        self._load_or_create()

    def _load_or_create(self, is_reset: bool = False) -> None:
        """初始化加载。如果本地有数据则读取，否则创建一个空库。"""
        if is_reset and os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)

        os.makedirs(self.persist_directory, exist_ok=True)

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection",
            collection_metadata={"hnsw:space": "cosine"},
        )

    def _get_loader(self, file_path: str) -> TextLoader | Docx2txtLoader | PyPDFLoader:
        ext = file_path.split(".")[-1].lower()
        if ext in ("txt", "md"):
            return TextLoader(file_path, encoding="utf-8")
        if ext == "docx":
            return Docx2txtLoader(file_path)
        if ext == "pdf":
            return PyPDFLoader(file_path)
        raise ValueError(f"❌ 不支持的文件格式: {ext}")

    def add_document(
        self,
        file_path: str,
        *,
        document_id: Optional[str] = None,
        overwrite_document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        导入文档到向量库（切片后入库）。

        参数：
        - document_id: 可选，传入则使用指定 UUID（例如来自上层文档表）
        - overwrite_document_id: 可选，如果你想“更新”某个已有文档，传它会先按该 id 删除旧 chunks，再写入新 chunks

        返回：
        - 结构化结果，便于 Web/接口层使用
        """
        assert self.vectorstore is not None

        if not os.path.exists(file_path):
            return {"ok": False, "error": f"file not found: {file_path}"}

        filename = os.path.basename(file_path)
        add_time = datetime.now(BEIJING_TZ).isoformat()

        # 生成/校验 UUID
        if overwrite_document_id:
            try:
                uuid.UUID(overwrite_document_id)
            except Exception:
                return {
                    "ok": False,
                    "error": "overwrite_document_id is not a valid UUID",
                }
            doc_uuid = overwrite_document_id
        else:
            if document_id is None:
                doc_uuid = str(uuid.uuid4())
            else:
                try:
                    uuid.UUID(document_id)
                except Exception:
                    return {"ok": False, "error": "document_id is not a valid UUID"}
                doc_uuid = document_id

        # 若是更新：先删旧 chunks（按 document_id 删除）
        if overwrite_document_id:
            try:
                self.vectorstore.delete(where={"document_id": doc_uuid})
            except Exception as delete_error:
                return {
                    "ok": False,
                    "document_id": doc_uuid,
                    "filename": filename,
                    "error": f"failed to delete old chunks: {delete_error}",
                }

        try:
            loader = self._get_loader(file_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                add_start_index=True,
                separators=[
                    "\n### ",
                    "\n## ",
                    "\n\n",
                    "\n",
                    " ",
                    "",
                ],
            )
            splits = text_splitter.split_documents(docs)

            # 写入统一元数据（document_id 为主键；filename 为展示/追溯）
            for split in splits:
                split.metadata["document_id"] = doc_uuid
                split.metadata["filename"] = filename
                split.metadata["add_time"] = add_time

            self.vectorstore.add_documents(documents=splits)

            return {
                "ok": True,
                "document_id": doc_uuid,
                "filename": filename,
                "chunk_count": len(splits),
                "add_time": add_time,
            }

        except Exception as e:
            return {
                "ok": False,
                "document_id": doc_uuid,
                "filename": filename,
                "error": str(e),
            }

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """按 document_id(UUID) 删除文档的所有 chunks。"""
        assert self.vectorstore is not None
        try:
            uuid.UUID(document_id)
        except Exception:
            return {"ok": False, "error": "document_id is not a valid UUID"}

        try:
            self.vectorstore.delete(where={"document_id": document_id})
            return {"ok": True, "document_id": document_id}
        except Exception as e:
            return {"ok": False, "document_id": document_id, "error": str(e)}

    def search(
        self,
        query: str,
        k: int = DEFAULT_SEARCH_K,
        t: float = SIMILARITY_THRESHOLD,
        *,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        相似度检索。
        可选：按 document_id 过滤（实现方式：先检索再过滤；POC足够，但不是最优）。
        """
        assert self.vectorstore is not None

        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)

        results: List[Dict[str, Any]] = []
        for doc, score in docs_and_scores:
            # 注意：这里仍沿用你原逻辑：score 越小越相似（距离）。若你实际返回是相似度，需改为 >=
            if score <= t:
                meta = doc.metadata or {}
                if document_id and meta.get("document_id") != document_id:
                    continue

                results.append(
                    {
                        "content": doc.page_content,
                        "document_id": meta.get("document_id"),
                        "filename": meta.get("filename"),
                        "add_time": meta.get("add_time"),
                        "score": round(float(score), 4),
                        # 这些字段可能存在（取决于 loader/splitter），有则返回，便于前端定位
                        "page": meta.get("page"),
                        "start_index": meta.get("start_index"),
                        "source": meta.get("source"),
                    }
                )
        return results

    def get_overview(self) -> Dict[str, Any]:
        """
        轻量概览：返回切片数、文档数、最近导入时间等。
        仍通过 metadatas 推断（POC可用，数据大时建议改成独立文档表）。
        """
        assert self.vectorstore is not None

        all_data = self.vectorstore.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", []) or []

        if os.path.exists(self.persist_directory):
            ctime = os.path.getctime(self.persist_directory)
            create_time_str = datetime.fromtimestamp(ctime, BEIJING_TZ).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            create_time_str = "Unknown"

        # 统计每个 document_id 的最新 add_time 与 filename
        doc_stats: Dict[
            str, Tuple[str, str]
        ] = {}  # document_id -> (add_time, filename)
        for meta in metadatas:
            did = meta.get("document_id")
            atime = meta.get("add_time")
            fname = meta.get("filename")
            if did and atime:
                if did not in doc_stats or atime > doc_stats[did][0]:
                    doc_stats[did] = (atime, fname)

        sorted_docs = sorted(doc_stats.items(), key=lambda x: x[1][0], reverse=True)
        latest_update = sorted_docs[0][1][0] if sorted_docs else "N/A"

        return {
            "persist_directory": self.persist_directory,
            "create_time": create_time_str,
            "latest_update": latest_update,
            "total_chunks": len(metadatas),
            "total_documents": len(doc_stats),
            "documents": [
                {"document_id": did, "add_time": atime, "filename": fname}
                for did, (atime, fname) in sorted_docs
            ],
        }

    def reset_index(self) -> None:
        """重置向量库（物理删除持久化目录并重建）"""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        self._load_or_create(is_reset=False)

    def as_retriever(self, **kwargs):
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
        from pydantic import PrivateAttr

        class KBRetriever(BaseRetriever):
            _kb_manager: "VectorKBManager" = PrivateAttr()
            k: int = DEFAULT_SEARCH_K
            t: float = SIMILARITY_THRESHOLD

            def __init__(self, kb_manager, k, t, **data):
                super().__init__(**data)
                self._kb_manager = kb_manager
                self.k = k
                self.t = t

            def _get_relevant_documents(self, query: str) -> List[Document]:
                search_results = self._kb_manager.search(query, k=self.k, t=self.t)
                return [
                    Document(
                        page_content=res["content"],
                        metadata={
                            "document_id": res.get("document_id"),
                            "filename": res.get("filename"),
                            "add_time": res.get("add_time"),
                            "score": res.get("score"),
                            "page": res.get("page"),
                            "start_index": res.get("start_index"),
                            "source": res.get("source"),
                        },
                    )
                    for res in search_results
                ]

        k = kwargs.get("k", DEFAULT_SEARCH_K)
        t = kwargs.get("t", SIMILARITY_THRESHOLD)
        return KBRetriever(kb_manager=self, k=k, t=t)


if __name__ == "__main__":
    # 1. 初始化（确保模型路径正确）
    kb = VectorKBManager()

    # 2. 添加文档（准备 sample）
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR, exist_ok=True)
        with open(os.path.join(DOCS_DIR, "sample.txt"), "w", encoding="utf-8") as f:
            f.write("这是一个本地测试文档。")

    # 3. 批量导入
    for filename in os.listdir(DOCS_DIR):
        full_path = os.path.join(DOCS_DIR, filename)
        if os.path.isfile(full_path):
            result = kb.add_document(full_path)
            print(result)

    # 4. 概览与查询
    overview = kb.get_overview()
    print("overview:", overview)

    test_query = "L2缓存命中率低"
    results = kb.search(test_query)
    for res in results:
        print(
            f"source={res.get('filename')} document_id={res.get('document_id')} score={res.get('score')} content={res.get('content')}"
        )
