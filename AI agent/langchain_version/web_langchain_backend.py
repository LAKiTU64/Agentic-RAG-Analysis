#!/usr/bin/env python3
"""
基于LangChain的AI Agent Web后端服务器

集成LangChain框架，提供更智能的对话、工具调用和工作流程管理
"""

import os
import json
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 导入LangChain Agent
from langchain_agent import LangChainAgent

app = FastAPI(
    title="AI Agent LLM性能分析器 (LangChain版)",
    description="基于LangChain的智能大语言模型性能分析Web服务",
    version="2.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# 全局变量
langchain_agent: Optional[LangChainAgent] = None
active_sessions: Dict[str, Dict] = {}

# 数据模型
class ChatMessage(BaseModel):
    type: str  # "user", "assistant", "system", "file"
    content: str
    timestamp: datetime
    session_id: Optional[str] = None
    file_info: Optional[Dict] = None
    context: Optional[Dict] = None

class SessionManager:
    """会话管理器，支持多用户对话记忆"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
    
    def get_or_create_session(self, session_id: str) -> Dict:
        """获取或创建会话"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "created_at": datetime.now(),
                "messages": [],
                "uploaded_files": {},
                "context": {},
                "agent": LangChainAgent(use_openai=False)  # 为每个会话创建独立Agent
            }
        return self.sessions[session_id]
    
    def add_message(self, session_id: str, message: ChatMessage):
        """添加消息到会话历史"""
        session = self.get_or_create_session(session_id)
        session["messages"].append(message)
        
        # 限制消息历史长度
        if len(session["messages"]) > 50:
            session["messages"] = session["messages"][-50:]
    
    def add_file(self, session_id: str, file_id: str, file_data: Dict):
        """添加文件到会话上下文"""
        session = self.get_or_create_session(session_id)
        session["uploaded_files"][file_id] = file_data
        
        # 添加到上下文
        if "config_files" not in session["context"]:
            session["context"]["config_files"] = []
        session["context"]["config_files"].append(file_data)
    
    def get_agent(self, session_id: str) -> LangChainAgent:
        """获取会话的Agent"""
        session = self.get_or_create_session(session_id)
        return session["agent"]
    
    def get_context(self, session_id: str) -> Dict:
        """获取会话上下文"""
        session = self.get_or_create_session(session_id)
        return session.get("context", {})

# 会话管理器实例
session_manager = SessionManager()

class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """建立WebSocket连接"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        print(f"🔗 WebSocket连接已建立: {session_id}")
    
    def disconnect(self, session_id: str):
        """断开WebSocket连接"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            print(f"❌ WebSocket连接已断开: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        """发送消息给特定会话"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except Exception as e:
                print(f"发送消息失败 {session_id}: {e}")
                self.disconnect(session_id)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global langchain_agent
    langchain_agent = LangChainAgent(use_openai=False)
    print("🤖 LangChain AI Agent Web服务器启动完成")

@app.get("/", response_class=HTMLResponse)
async def root():
    """主页"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Agent LLM性能分析器 (LangChain版)</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>
    <body>
        <h1>🤖 AI Agent LLM性能分析器 (LangChain版)</h1>
        <p>基于LangChain的智能对话系统</p>
        <p>请访问 <a href="/chat">/chat</a> 开始使用</p>
        <p>API文档: <a href="/docs">/docs</a></p>
    </body>
    </html>
    """

@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    """聊天页面"""
    chat_file = Path("static/chat.html")
    if chat_file.exists():
        return chat_file.read_text(encoding='utf-8')
    else:
        return HTMLResponse(
            content="<h1>聊天页面未找到</h1><p>请确保 static/chat.html 文件存在</p>",
            status_code=404
        )

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket连接端点"""
    await manager.connect(websocket, session_id)
    
    # 发送欢迎消息
    await manager.send_message(session_id, {
        "type": "assistant_message",
        "content": """🤖 **欢迎使用LangChain版AI Agent！**

我现在具备更强的智能对话能力：
• 🧠 **智能推理**: 能够理解复杂的分析需求
• 🔧 **工具调用**: 自动选择合适的分析工具
• 💭 **记忆管理**: 记住我们的对话历史
• 🔄 **工作流优化**: 智能规划分析步骤

请告诉我您的性能分析需求，我会为您提供专业的帮助！""",
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # 处理不同类型的消息
            await handle_websocket_message(session_id, message_data)
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        print(f"WebSocket错误 {session_id}: {e}")
        manager.disconnect(session_id)

async def handle_websocket_message(session_id: str, message_data: dict):
    """处理WebSocket消息"""
    
    message_type = message_data.get("type", "")
    content = message_data.get("content", "")
    
    if message_type == "user_message":
        # 用户发送分析请求
        await process_user_message_with_langchain(session_id, content)
    
    elif message_type == "ping":
        # 心跳检测
        await manager.send_message(session_id, {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })

async def process_user_message_with_langchain(session_id: str, message: str):
    """使用LangChain处理用户消息"""
    
    try:
        # 获取会话Agent和上下文
        agent = session_manager.get_agent(session_id)
        context = session_manager.get_context(session_id)
        
        # 添加用户消息到会话历史
        user_message = ChatMessage(
            type="user",
            content=message,
            timestamp=datetime.now(),
            session_id=session_id
        )
        session_manager.add_message(session_id, user_message)
        
        # 发送思考状态
        await manager.send_message(session_id, {
            "type": "assistant_thinking",
            "content": "🤔 正在分析您的需求...",
            "timestamp": datetime.now().isoformat()
        })
        
        # 使用LangChain Agent处理消息
        result = await agent.process_message(message, context)
        
        if result["status"] == "success":
            response_content = result["response"]
            
            # 添加Assistant回复到会话历史
            assistant_message = ChatMessage(
                type="assistant",
                content=response_content,
                timestamp=datetime.now(),
                session_id=session_id
            )
            session_manager.add_message(session_id, assistant_message)
            
            # 发送回复
            await manager.send_message(session_id, {
                "type": "assistant_message",
                "content": response_content,
                "timestamp": datetime.now().isoformat(),
                "langchain_powered": True
            })
            
        else:
            # 错误处理
            await manager.send_message(session_id, {
                "type": "error",
                "content": f"❌ 处理消息时出错: {result['response']}",
                "timestamp": datetime.now().isoformat()
            })
            
    except Exception as e:
        await manager.send_message(session_id, {
            "type": "error",
            "content": f"❌ LangChain处理失败: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
        print(f"LangChain处理异常: {traceback.format_exc()}")

@app.post("/upload_config")
async def upload_config(file: UploadFile = File(...), session_id: str = Form(...)):
    """上传配置文件（LangChain版）"""
    
    try:
        # 检查文件类型
        if not file.filename.endswith(('.json', '.yaml', '.yml')):
            raise HTTPException(status_code=400, detail="只支持JSON和YAML格式文件")
        
        # 读取文件内容
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # 生成文件ID
        file_id = str(uuid.uuid4())
        
        # 获取会话Agent
        agent = session_manager.get_agent(session_id)
        
        # 使用LangChain Agent处理文件
        file_result = agent.add_uploaded_file(content_str, file.filename)
        
        if file_result["status"] == "success":
            # 保存到会话上下文
            file_data = {
                "file_id": file_id,
                "filename": file.filename,
                "content": content_str,
                "suggestions": file_result.get("suggestions", [])
            }
            
            session_manager.add_file(session_id, file_id, file_data)
            
            return {
                "filename": file.filename,
                "file_id": file_id,
                "message": file_result["message"],
                "suggestions": file_result.get("suggestions", []),
                "langchain_processed": True
            }
        else:
            raise HTTPException(status_code=400, detail=file_result["message"])
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件处理失败: {str(e)}")

@app.post("/intelligent_analysis")
async def intelligent_analysis(
    request_data: Dict[str, Any],
    session_id: str = Form(...)
):
    """智能分析接口（LangChain增强）"""
    
    try:
        # 获取会话Agent
        agent = session_manager.get_agent(session_id)
        context = session_manager.get_context(session_id)
        
        # 构建智能分析请求
        analysis_request = f"""
        基于以下信息进行智能分析：
        
        用户需求: {request_data.get('user_request', '')}
        配置参数: {json.dumps(request_data.get('config', {}), ensure_ascii=False)}
        上下文信息: {json.dumps(context, ensure_ascii=False)}
        
        请提供详细的分析计划和执行步骤。
        """
        
        # 使用LangChain Agent处理
        result = await agent.process_message(analysis_request, context)
        
        return {
            "status": result["status"],
            "analysis_plan": result["response"],
            "timestamp": result["timestamp"],
            "session_id": session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"智能分析失败: {str(e)}")

@app.get("/session_memory/{session_id}")
async def get_session_memory(session_id: str):
    """获取会话记忆"""
    try:
        agent = session_manager.get_agent(session_id)
        memory_summary = agent.get_memory_summary()
        
        return {
            "session_id": session_id,
            "memory_summary": memory_summary,
            "context": session_manager.get_context(session_id)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取记忆失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(session_manager.sessions),
        "langchain_agent_ready": langchain_agent is not None,
        "version": "2.0.0 (LangChain Enhanced)"
    }

@app.get("/agent_capabilities")
async def get_agent_capabilities():
    """获取Agent能力列表"""
    if langchain_agent:
        return {
            "tools": [
                {
                    "name": "prompt_analyzer", 
                    "description": "分析自然语言提示，提取分析需求"
                },
                {
                    "name": "config_analyzer",
                    "description": "解析配置文件，提供智能建议"
                },
                {
                    "name": "performance_analyzer",
                    "description": "执行LLM性能分析"
                },
                {
                    "name": "optimization_advisor",
                    "description": "提供性能优化建议"
                }
            ],
            "features": [
                "智能对话理解",
                "工具自动选择",
                "对话记忆管理",
                "上下文感知",
                "多会话支持"
            ]
        }
    else:
        return {"error": "LangChain Agent未初始化"}

if __name__ == "__main__":
    # 开发环境运行
    uvicorn.run(
        "web_langchain_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
