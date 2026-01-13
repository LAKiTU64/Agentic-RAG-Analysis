from fastapi import UploadFile, File, HTTPException, APIRouter
import tempfile
import os
from .vector_kb_manager import VectorKBManager  # 替换为你的模块名

router = APIRouter()
kb = VectorKBManager()  # 初始化一次


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
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
        result = kb.add_document(tmp_path)
        if not result["ok"]:
            raise HTTPException(400, result.get("error", "未知错误"))
        return {"status": "success", "document_id": result["document_id"]}
    finally:
        os.unlink(tmp_path)


@router.post("/search")
async def search(query: str):
    results = kb.search(query)
    return {"results": results}


# 添加这些路由到你的 FastAPI app
@router.get("/documents")
async def list_documents():
    """获取文档列表（前端需要）"""
    return kb.get_overview()


@router.delete("/document/{doc_id}")
async def delete_document_api(doc_id: str):
    """删除文档（前端需要）"""
    result = kb.delete_document(doc_id)
    if not result["ok"]:
        raise HTTPException(404, "文档不存在")
    return {"status": "deleted"}


__all__ = ["router"]
