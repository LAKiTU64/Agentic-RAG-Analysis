import os

import shutil
import sys
import uuid
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
import yaml

# 北京时间
BEIJING_TZ = timezone(timedelta(hours=8))

# 分块策略
CHUNKING_STRATEGY = {
    "default": {
        "chunk_size": 300,
        "chunk_overlap": 60,
        "add_start_index": True,
        "separators": [
            "\n# ",
            "# ",
            "\n## ",
            "## ",
            "\n### ",
            "### ",
            "\n#### ",
            "#### ",
            "\n\n",
            ".\n",
            "?\n",
            "!\n",
            "\n",
            " ",
            "",
        ],
    }
}


class VectorKBManager:
    """
    向量知识库管理类（ChromaDB）：只使用本地 Embedding 模型。
    支持 document_id (UUID4) 和 doc_hash (SHA256) 双标识。
    """

    def __init__(self, config: Dict) -> None:
        kb_config = config.get("vector_store", {})
        self.embedding_model_path = kb_config.get("embedding_model_path")
        self.persist_directory = kb_config.get("persist_directory")
        self.chunk_size = kb_config.get("chunk_size")
        self.chunk_overlap = kb_config.get("chunk_overlap")
        self.default_search_k = kb_config.get("default_search_k", 8)
        self.max_distance = kb_config.get("max_distance", 0.5)

        # 检查本地模型路径
        if not os.path.exists(self.embedding_model_path):
            raise FileNotFoundError(
                f"❌ 找不到本地模型目录: {self.embedding_model_path}。请确保模型已下载到该位置。"
            )

        # 初始化 Embedding：强制开启 local_files_only，禁止联网下载 embedding 模型；增加 CUDA 环境的支持
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_path,
            model_kwargs={
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "local_files_only": True,
            },
            encode_kwargs={"normalize_embeddings": True},  # 开启向量归一化
        )

        self.vectorstore: Optional[Chroma] = None
        self._load_or_create()

    def _load_or_create(self, is_reset: bool = False) -> None:
        """初始化加载。如果本地有数据则读取，否则创建一个空库；is_reset=True 则强制创建新库"""
        if is_reset and os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)

        os.makedirs(self.persist_directory, exist_ok=True)

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection",
            collection_metadata={"hnsw:space": "cosine"},
            client_settings=Settings(anonymized_telemetry=False),  # 禁用遥测
        )

    def _get_loader(self, file_path: str) -> TextLoader:
        """根据文件类型，返回对应的 Loader 对象，支持 txt/md"""
        ext = file_path.split(".")[-1].lower()
        if ext in ("txt", "md"):
            return TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"不支持该文件类型：.{ext}")

    def _compute_file_hash(self, file_path: str) -> str:
        """计算文件的 SHA256 哈希（基于原始字节），用于内容指纹"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def add_document(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        overwrite_document_id: Optional[str] = None,
        chunking_strategy: Optional[str] = None,
        runtime_info: Optional[Dict[str, Any]] = None,
        override_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        导入文档到向量库（切片后入库）。

        参数：
        - document_id: 可选，传入则使用指定 UUID（例如来自上层文档表）
        - overwrite_document_id: 可选，如果想“更新”某个已有文档，传它会先按该 id 删除旧 chunks，再写入新 chunks
        - chunking_strategy: 可选，指定分块策略，默认策略default，后续将会维护其他策略
        - runtime_info: 可选，传入的sglang运行时元数据字典，写入每个 chunk 的 metadata 中
        - override_filename: 可选，覆盖文件名（用于前端展示）

        返回：
        - 结构化结果，便于 Web/接口层使用
        """
        assert self.vectorstore is not None

        if not os.path.exists(file_path):
            return {"ok": False, "error": f"file not found: {file_path}"}

        # 提取基础metadata；如果传了 override_filename 就用它作为实际文件名（以防文件名是临时文件tmpxxx）
        filename = (
            override_filename if override_filename else os.path.basename(file_path)
        )
        add_time = datetime.now(BEIJING_TZ).isoformat()
        doc_hash = self._compute_file_hash(file_path)

        # 生成/校验 UUID
        if overwrite_document_id:
            # 如果是更新文档，校验是否是合法UUID
            try:
                uuid.UUID(overwrite_document_id)
            except Exception:
                return {
                    "ok": False,
                    "error": "overwrite_document_id is not a valid UUID",
                }
            doc_uuid = overwrite_document_id
        else:
            # 如果是直接添加文档
            if document_id is None:
                # 没传参document_id的情况下生成新UUID
                doc_uuid = str(uuid.uuid4())
            else:
                # 传参document_id时，校验是否是合法UUID
                try:
                    uuid.UUID(document_id)
                except Exception:
                    return {"ok": False, "error": "document_id is not a valid UUID"}
                doc_uuid = document_id

        # 如果是更新文档，先删旧 chunks（按 document_id 删除）
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

            # 配置分块策略
            strategy_name = "default"
            strategy = CHUNKING_STRATEGY[strategy_name]

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=strategy["chunk_size"],
                chunk_overlap=strategy["chunk_overlap"],
                add_start_index=strategy["add_start_index"],
                separators=strategy["separators"],
            )
            splits = text_splitter.split_documents(docs)

            # 写入统一元数据
            for split in splits:
                split.metadata.update(
                    {
                        "document_id": doc_uuid,  # 基于UUID的document_id，标记唯一性
                        "doc_hash": doc_hash,  # 内容哈希
                        "filename": filename,  # 文件名
                        "add_time": add_time,  # 添加时间
                    }
                )
                # 写入runtime_info元数据（如果有）
                if isinstance(runtime_info, dict):
                    split.metadata.update(runtime_info)

            self.vectorstore.add_documents(documents=splits)

            return {
                "ok": True,
                "document_id": doc_uuid,
                "doc_hash": doc_hash,
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
        """按 document_id(UUID) 删除文档的所有 chunks"""
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
        k: int = None,
        max_distance: float = None,
        *,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        相似度检索

        params:
        - where_filter: 符合 Chroma 语法的过滤字典。
        例如: 找出使用 A100 或 H800 GPU、batch_size 大于 8、且 accuracy 小于等于 0.85 的文档
        where_filter = {
            "$and": [
                {
                    "$or": [
                        {"gpu": "A100"},
                        {"gpu": "H800"}
                    ]
                },
                {"batch_size": {"$gt": 8}},
                {"accuracy": {"$lte": 0.85}}
            ]
        }
        """
        assert self.vectorstore is not None

        if k is None:
            k = self.default_search_k
        if max_distance is None:
            max_distance = self.max_distance

        # 执行检索
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query, k=k, filter=where_filter
        )

        results: List[Dict[str, Any]] = []
        for doc, score in docs_and_scores:
            if score <= max_distance:
                meta = doc.metadata or {}
                res_item = {
                    "content": doc.page_content,
                    "score": round(float(score), 4),
                    **meta,  # 展开所有元数据
                }
                results.append(res_item)

        return results

    def get_overview(self) -> Dict[str, Any]:
        """
        知识库概览：返回切片数、文档数、文档元数据
        """
        assert self.vectorstore is not None
        all_data = self.vectorstore.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", []) or []

        # doc_stats 结构调整，存储完整的元数据字典
        doc_stats: Dict[str, Dict[str, Any]] = {}

        for meta in metadatas:
            did = meta.get("document_id")
            atime = meta.get("add_time")
            if did:
                # 记录该 document_id 下最新的元数据（或者第一条）
                if did not in doc_stats or (
                    atime and atime > doc_stats[did].get("add_time", "")
                ):
                    doc_stats[did] = meta

        sorted_docs = sorted(
            doc_stats.values(), key=lambda x: x.get("add_time", ""), reverse=True
        )

        return {
            "persist_directory": self.persist_directory,
            "total_chunks": len(metadatas),
            "total_documents": len(doc_stats),
            "documents": sorted_docs,  # 直接返回包含所有元数据的列表
        }

    def get_document_metadata(self, document_id: str) -> Dict[str, Any]:
        """返回该 document_id 的一份代表性元数据（用于前端编辑展示）"""
        assert self.vectorstore is not None
        try:
            uuid.UUID(document_id)
        except Exception:
            return {"ok": False, "error": "document_id is not a valid UUID"}

        try:
            data = self.vectorstore.get(
                where={"document_id": document_id}, include=["metadatas"]
            )
            metas = data.get("metadatas", []) or []
            if not metas:
                return {"ok": False, "error": "document not found"}
            # 取第一条chunk的元数据作为代表（正常情况下同一文档的chunk元数据应一致）
            meta = metas[0]
            return {"ok": True, "document_id": document_id, "metadata": meta}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def update_document_metadata(
        self,
        document_id: str,
        set_metadata: Dict[str, Any],
        *,
        delete_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """更新指定文档所有chunks元数据：支持 set(新增/修改) + delete(删除key)"""
        assert self.vectorstore is not None

        protected_keys = {"document_id", "doc_hash", "add_time"}
        delete_keys = delete_keys or []

        # 过滤 set 中的 protected
        filtered_set = {
            k: v for k, v in (set_metadata or {}).items() if k not in protected_keys
        }
        # 过滤 delete 中的 protected
        filtered_delete = [k for k in delete_keys if k not in protected_keys]

        try:
            existing_data = self.vectorstore.get(
                where={"document_id": document_id}, include=["metadatas"]
            )
            ids = existing_data.get("ids", [])
            if not ids:
                return {"ok": False, "error": "未找到对应的文档chunk"}

            metadatas = existing_data.get("metadatas", []) or []

            updated_metadatas = []
            for old_meta in metadatas:
                new_meta = dict(old_meta or {})

                # delete
                for k in filtered_delete:
                    new_meta.pop(k, None)

                # set/update
                new_meta.update(filtered_set)

                updated_metadatas.append(new_meta)

            self.vectorstore._collection.update(ids=ids, metadatas=updated_metadatas)
            return {"ok": True, "document_id": document_id}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def reset_index(self) -> None:
        """重置向量库（物理删除持久化目录并重建）"""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        self._load_or_create(is_reset=False)

    def as_retrieriter(self, **kwargs):
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
        from pydantic import PrivateAttr
        from typing import Optional, Dict, Any

        class KBRetriever(BaseRetriever):
            _kb_manager: "VectorKBManager" = PrivateAttr()
            k: int = self.default_search_k
            max_distance: float = self.max_distance

            def __init__(self, kb_manager, k, max_distance, **data):
                super().__init__(**data)
                self._kb_manager = kb_manager
                self.k = k
                self.max_distance = max_distance

            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager=None,  # LangChain 内部参数，可忽略
                **kwargs,
            ) -> List[Document]:
                # 从 kwargs 中提取 filter（符合 LangChain 规范）
                where_filter: Optional[Dict[str, Any]] = kwargs.get("filter")

                search_results = self._kb_manager.search(
                    query,
                    k=self.k,
                    max_distance=self.max_distance,
                    where_filter=where_filter,  # 👈 透传！
                )
                return [
                    Document(
                        page_content=res["content"],
                        metadata={
                            "document_id": res.get("document_id"),
                            "doc_hash": res.get("doc_hash"),
                            "filename": res.get("filename"),
                            "add_time": res.get("add_time"),
                            "score": res.get("score"),
                            "start_index": res.get("start_index"),
                            "source": res.get("source"),
                        },
                    )
                    for res in search_results
                ]

        k = kwargs.get("k", self.default_search_k)
        max_distance = kwargs.get("max_distance", self.max_distance)
        return KBRetriever(kb_manager=self, k=k, max_distance=max_distance)


if __name__ == "__main__":
    # 1. 初始化
    print("\nStep 1: 初始化向量库\n")
    config_path = "/workspaces/ai-agent/AI_Agent_Complete/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 错误: 找不到 {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)

    kb = VectorKBManager(config=config_yaml)

    # 2. 批量导入
    print("\nStep 2: 批量导入\n")
    # 模拟一个统一的 runtime_info（所有文档共享）
    mock_runtime_info = {
        "model": "qwen3-4b",
        "batch_size": 16,
        "gpu": "H800",
        "input_len": 512,
    }
    docs_dir = "/workspaces/ai-agent/AI_Agent_Complete/documents"
    for filename in os.listdir(docs_dir):
        full_path = os.path.join(docs_dir, filename)
        if os.path.isfile(full_path):
            result = kb.add_document(full_path, runtime_info=mock_runtime_info)
            print(result)

    # 3. 概览
    print("\nStep 3: 概览\n")
    overview = kb.get_overview()
    print("overview:", overview)

    # 4. 基础检索测试
    print("\nStep 4: 基础检索测试\n")
    test_query = "L2缓存命中率低"
    results = kb.search(test_query)
    for res in results:
        print(
            f"doc_hash={res.get('doc_hash')} | "
            f"filename={res.get('filename')} | "
            f"score={res.get('score')} | "
            f"content=\n{res.get('content')[:100]}..."
        )

    # 5. 带元数据过滤的检索测试
    print("\nStep 5: 元数据过滤检索测试 (filter: gpu=H800 且 batch_size > 8)\n")
    filtered_results = kb.search(
        test_query,
        where_filter={
            "$and": [
                {"gpu": "H800"},
                {"batch_size": {"$gt": 8}},
            ]
        },
    )
    for res in filtered_results:
        print(
            f"✅ 过滤命中 | gpu={res.get('gpu')} | batch_size={res.get('batch_size')} | "
            f"score={res.get('score')}"
        )

    # 6. 测试 as_retriever + filter（LangChain 标准用法）
    print("\nStep 6: 测试 as_retriever + filter\n")
    retriever = kb.as_retrieriter(k=3, max_distance=0.6)
    docs = retriever.invoke(
        "性能瓶颈",
        filter={"input_len": {"$gt": 511}},  # 利用 runtime_info 中的 accuracy
    )
    print(f"Retriever 返回 {len(docs)} 个文档")
    for doc in docs:
        print(
            f" - score={doc.metadata.get('score')}, accuracy={doc.metadata.get('accuracy')}"
        )
