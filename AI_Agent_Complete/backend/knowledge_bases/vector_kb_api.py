import uuid
from fastapi import Request, UploadFile, File, HTTPException, APIRouter
import tempfile
import os

from fastapi.responses import FileResponse

router = APIRouter()


def _get_kb_from_request(request: Request):
    kb = getattr(request.app.state, "kb", None)
    if kb is None:
        raise HTTPException(500, "知识库未初始化，请在应用启动时由 AIAgent 注入")
    return kb


@router.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    kb = _get_kb_from_request(request)

    # 1. 检查文件大小（安全限制，可保留）
    content = await file.read()
    if len(content) > 5_000_000:  # 5MB
        raise HTTPException(400, "文件超过5MB限制")

    # 2. 保存临时文件（保留原始扩展名）
    suffix = os.path.splitext(file.filename)[1]  # 保留原始大小写（如 .MD）
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # 3. 交给知识库管理器处理（它会校验类型、加载、切片、入库）
    try:
        # 指定override_filename以保留原始文件名
        result = kb.add_document(tmp_path, override_filename=file.filename)
        if not result["ok"]:
            raise HTTPException(400, result.get("error", "未知错误"))
        return {"status": "success", "document_id": result["document_id"]}
    finally:
        os.unlink(tmp_path)


# 更新元数据的接口
@router.patch("/document/{doc_id}/metadata")
async def update_metadata(request: Request, doc_id: str, payload: dict):
    """
    推荐格式：
    {
      "set": {"model_name":"xxx", "foo":123},
      "delete": ["bar", "baz"]
    }

    兼容旧格式：{"model_name":"xxx"}
    """
    kb = _get_kb_from_request(request)

    if "set" in payload or "delete" in payload:
        set_data = payload.get("set", {}) or {}
        delete_keys = payload.get("delete", []) or []
    else:
        set_data = payload
        delete_keys = []

    result = kb.update_document_metadata(doc_id, set_data, delete_keys=delete_keys)
    if not result["ok"]:
        raise HTTPException(400, result.get("error", "更新失败"))
    return {"status": "success"}


@router.post("/search")
async def search(request: Request, query: str):
    kb = _get_kb_from_request(request)
    results = kb.search(query)
    return {"results": results}


# 添加这些路由到你的 FastAPI app
@router.get("/documents")
async def list_documents(request: Request):
    """获取文档列表和概览"""
    kb = _get_kb_from_request(request)
    return kb.get_overview()


@router.delete("/document/{doc_id}")
async def delete_document_api(request: Request, doc_id: str):
    """删除文档（前端需要）"""
    kb = _get_kb_from_request(request)
    result = kb.delete_document(doc_id)
    if not result["ok"]:
        raise HTTPException(404, "文档不存在")
    return {"status": "deleted"}


@router.get("/document/{doc_id}/metadata")
async def get_metadata(request: Request, doc_id: str):
    """获取单个文档的元数据"""
    kb = _get_kb_from_request(request)
    result = kb.get_document_metadata(doc_id)
    if not result["ok"]:
        raise HTTPException(404, result.get("error", "未找到文档"))
    return result


@router.get("/document/{doc_id}/chunks")
async def get_document_chunks(request: Request, doc_id: str):
    """获取指定 document_id 的所有 chunks（用于前端预览切分效果）"""
    kb = _get_kb_from_request(request)
    try:
        uuid.UUID(doc_id)  # 校验 UUID
    except Exception:
        raise HTTPException(400, "无效的 document_id")

    # 调用 vectorstore.get 获取所有 chunks
    try:
        data = kb.vectorstore.get(
            where={"document_id": doc_id}, include=["metadatas", "documents"]
        )
        ids = data.get("ids", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        chunks = []
        for i in range(len(ids)):
            chunks.append(
                {
                    "id": ids[i],
                    "content": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }
            )

        if not chunks:
            raise HTTPException(404, "未找到该文档的 chunks")

        return {"chunks": chunks}
    except Exception as e:
        raise HTTPException(500, f"获取 chunks 失败: {str(e)}")


def _get_media_type_for_preview(filename: str) -> str:
    """
    根据文件名后缀返回适合浏览器预览的 MIME 类型。
    仅支持 .txt, .md, .json，其余返回 application/octet-stream（触发下载）。
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "txt":
        return "text/plain"
    elif ext in ("md", "markdown"):
        return "text/markdown"
    elif ext == "json":
        return "application/json"
    else:
        return "application/octet-stream"


@router.get("/document/{doc_id}/file")
async def download_document_file(request: Request, doc_id: str):
    kb = _get_kb_from_request(request)
    try:
        uuid.UUID(doc_id)
    except Exception:
        raise HTTPException(400, "无效的 document_id")

    # 获取文档元数据
    meta_result = kb.get_document_metadata(doc_id)
    if not meta_result["ok"]:
        raise HTTPException(404, "文档不存在或未保存源文件")

    rel_path = meta_result["metadata"].get("saved_file_relpath")
    if not rel_path:
        raise HTTPException(404, "该文档未保存源文件")

    full_path = os.path.join(kb.file_store_directory, rel_path)
    if not os.path.exists(full_path):
        raise HTTPException(404, "源文件已丢失")

    # 关键修改：使用 metadata 中的 'filename' 作为下载文件名
    filename_to_serve = meta_result["metadata"].get("filename") or os.path.basename(
        rel_path
    )

    media_type = _get_media_type_for_preview(
        filename_to_serve
    )  # 注意：这里用逻辑名判断 MIME（可接受）
    return FileResponse(
        path=full_path,
        filename=filename_to_serve,  # 这是核心！
        media_type=media_type,
    )


@router.get("/document/{doc_id}/content")
async def get_document_content(request: Request, doc_id: str):
    kb = _get_kb_from_request(request)
    try:
        uuid.UUID(doc_id)
    except Exception:
        raise HTTPException(400, "无效的 document_id")

    meta_result = kb.get_document_metadata(doc_id)
    if not meta_result["ok"]:
        raise HTTPException(404, "文档不存在或未保存源文件")

    rel_path = meta_result["metadata"].get("saved_file_relpath")
    if not rel_path:
        raise HTTPException(404, "该文档未保存源文件")

    full_path = os.path.abspath(os.path.join(kb.file_store_directory, rel_path))
    if not full_path.startswith(os.path.abspath(kb.file_store_directory)):
        raise HTTPException(403, "非法路径")
    if not os.path.exists(full_path):
        raise HTTPException(404, "源文件已丢失")

    # 只允许查看 .txt / .md / .json
    filename = os.path.basename(rel_path).lower()
    if not any(filename.endswith(ext) for ext in (".txt", ".md", ".markdown", ".json")):
        raise HTTPException(400, "仅支持查看 .txt / .md / .json 文件内容")

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        raise HTTPException(400, "文件不是有效的文本格式")

    display_filename = meta_result["metadata"].get("filename") or os.path.basename(
        rel_path
    )
    return {"filename": display_filename, "content": content}


__all__ = ["router"]
