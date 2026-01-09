#!/usr/bin/env python3
"""
AI Agent Web后端服务器

基于FastAPI构建的Web服务器，支持：
1. 类ChatGPT的对话界面
2. 文件上传和解析
3. 实时分析进度推送
4. RESTful API接口

作者: AI助手
版本: 1.0
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

# 导入AI Agent组件
from ai_agent_analyzer import AIAgentAnalyzer, AnalysisRequest
import yaml

app = FastAPI(
    title="AI Agent LLM性能分析器",
    description="智能的大语言模型性能分析Web服务",
    version="1.0.0"
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
agent = None
active_connections: Dict[str, WebSocket] = {}
analysis_sessions: Dict[str, Dict] = {}

# 数据模型
class ChatMessage(BaseModel):
    type: str  # "user", "assistant", "system", "file"
    content: str
    timestamp: datetime
    session_id: Optional[str] = None
    file_info: Optional[Dict] = None

class AnalysisStatus(BaseModel):
    session_id: str
    status: str  # "running", "completed", "error"
    progress: int  # 0-100
    message: str
    results: Optional[Dict] = None

class FileUploadResponse(BaseModel):
    filename: str
    file_id: str
    content: Dict
    suggestions: List[str]

class ConfigFileParser:
    """配置文件解析器"""
    
    @staticmethod
    def parse_json_config(content: str) -> Dict:
        """解析JSON配置文件"""
        try:
            config = json.loads(content)
            return ConfigFileParser._extract_model_info(config)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON格式错误: {e}")
    
    @staticmethod
    def parse_yaml_config(content: str) -> Dict:
        """解析YAML配置文件"""
        try:
            config = yaml.safe_load(content)
            return ConfigFileParser._extract_model_info(config)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML格式错误: {e}")
    
    @staticmethod
    def _extract_model_info(config: Dict) -> Dict:
        """从配置中提取模型信息"""
        extracted = {
            "model_info": {},
            "analysis_params": {},
            "hardware_info": {},
            "suggestions": []
        }
        
        # 提取模型信息
        model_fields = ["model_name", "model_path", "model_type", "model_size", 
                       "architecture", "parameters", "precision"]
        
        for field in model_fields:
            if field in config:
                extracted["model_info"][field] = config[field]
        
        # 提取分析参数
        analysis_fields = ["batch_size", "input_len", "output_len", "temperature", 
                          "tp_size", "analysis_type", "profile_steps"]
        
        for field in analysis_fields:
            if field in config:
                extracted["analysis_params"][field] = config[field]
        
        # 提取硬件信息
        hardware_fields = ["gpu_type", "gpu_count", "memory_gb", "compute_capability",
                          "driver_version", "cuda_version"]
        
        for field in hardware_fields:
            if field in config:
                extracted["hardware_info"][field] = config[field]
        
        # 生成建议
        extracted["suggestions"] = ConfigFileParser._generate_suggestions(extracted)
        
        return extracted
    
    @staticmethod
    def _generate_suggestions(extracted_info: Dict) -> List[str]:
        """基于配置信息生成建议"""
        suggestions = []
        
        model_info = extracted_info.get("model_info", {})
        analysis_params = extracted_info.get("analysis_params", {})
        hardware_info = extracted_info.get("hardware_info", {})
        
        # 基于模型大小的建议
        model_size = model_info.get("model_size", "")
        if "7b" in model_size.lower():
            suggestions.append("🎯 7B模型推荐: batch_size=8-16, 适合单卡推理")
        elif "13b" in model_size.lower():
            suggestions.append("🎯 13B模型推荐: batch_size=4-8, 考虑使用tensor并行")
        elif "70b" in model_size.lower():
            suggestions.append("🎯 70B模型推荐: batch_size=1-2, 必须使用多卡并行")
        
        # 基于GPU类型的建议
        gpu_type = hardware_info.get("gpu_type", "").lower()
        if "a100" in gpu_type:
            suggestions.append("🚀 A100 GPU优化: 使用FP16/BF16精度, 启用Tensor Core")
        elif "h100" in gpu_type:
            suggestions.append("🚀 H100 GPU优化: 使用FP8精度, 充分利用Transformer Engine")
        elif "v100" in gpu_type:
            suggestions.append("⚠️ V100 GPU提醒: 内存较小，建议降低batch_size")
        
        # 基于精度的建议
        precision = model_info.get("precision", "").lower()
        if "fp32" in precision:
            suggestions.append("💾 FP32精度提醒: 内存占用较大，建议使用FP16")
        elif "int8" in precision:
            suggestions.append("⚡ INT8量化检测: 推理速度快，但可能影响精度")
        
        # 基于分析类型的建议
        analysis_type = analysis_params.get("analysis_type", "")
        if analysis_type == "ncu":
            suggestions.append("🔬 NCU深度分析: 关注kernel效率和内存带宽利用率")
        elif analysis_type == "nsys":
            suggestions.append("📊 NSys全局分析: 关注timeline和热点kernel识别")
        
        # 通用建议
        if not suggestions:
            suggestions.append("💡 建议先运行nsys进行全局分析，再针对热点进行ncu分析")
        
        return suggestions

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
    
    async def broadcast(self, message: dict):
        """广播消息给所有连接"""
        for session_id in list(self.active_connections.keys()):
            await self.send_message(session_id, message)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global agent
    agent = AIAgentAnalyzer()
    print("🤖 AI Agent Web服务器启动完成")

@app.get("/", response_class=HTMLResponse)
async def root():
    """主页"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Agent LLM性能分析器</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>
    <body>
        <h1>🤖 AI Agent LLM性能分析器</h1>
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
        await process_user_analysis_request(session_id, content)
    
    elif message_type == "ping":
        # 心跳检测
        await manager.send_message(session_id, {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })

async def process_user_analysis_request(session_id: str, prompt: str):
    """处理用户分析请求"""
    
    try:
        # 发送开始分析消息
        await manager.send_message(session_id, {
            "type": "assistant_message",
            "content": f"🔄 开始分析您的请求: {prompt}",
            "timestamp": datetime.now().isoformat()
        })
        
        # 解析提示词
        await manager.send_message(session_id, {
            "type": "progress",
            "progress": 10,
            "message": "正在解析提示词..."
        })
        
        # 异步执行分析
        asyncio.create_task(run_analysis_async(session_id, prompt))
        
    except Exception as e:
        await manager.send_message(session_id, {
            "type": "error",
            "content": f"处理请求失败: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })

async def run_analysis_async(session_id: str, prompt: str):
    """异步运行分析"""
    
    try:
        # 更新进度
        await manager.send_message(session_id, {
            "type": "progress", 
            "progress": 30,
            "message": "正在配置分析参数..."
        })
        
        # 运行分析 (这里需要在线程池中执行，避免阻塞)
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(agent.analyze_from_prompt, prompt)
            
            # 模拟进度更新
            for i in range(40, 90, 10):
                await asyncio.sleep(2)
                await manager.send_message(session_id, {
                    "type": "progress",
                    "progress": i,
                    "message": f"分析进行中... {i}%"
                })
            
            # 获取结果
            results = future.result()
        
        # 发送完成消息
        await manager.send_message(session_id, {
            "type": "progress",
            "progress": 100,
            "message": "分析完成!"
        })
        
        if 'error' not in results:
            # 成功完成
            output_dir = results.get('request', {}).get('output_dir', 'N/A')
            
            response_content = f"""✅ **分析完成!**
            
📁 **结果目录**: {output_dir}
🔬 **分析类型**: {results.get('request', {}).get('analysis_type', 'N/A')}
📊 **模型**: {results.get('request', {}).get('model_name', 'N/A')}

🎯 **主要发现**:
- 分析已成功完成
- 详细结果已保存到指定目录
- 可查看生成的可视化图表和报告

💡 **下一步建议**:
1. 查看生成的timeline图表
2. 分析性能瓶颈报告
3. 根据建议进行优化
"""
            
            await manager.send_message(session_id, {
                "type": "assistant_message",
                "content": response_content,
                "timestamp": datetime.now().isoformat(),
                "results": results
            })
        else:
            # 分析失败
            await manager.send_message(session_id, {
                "type": "error",
                "content": f"❌ 分析失败: {results['error']}",
                "timestamp": datetime.now().isoformat()
            })
            
    except Exception as e:
        # 异常处理
        await manager.send_message(session_id, {
            "type": "error", 
            "content": f"❌ 分析过程中出现错误: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
        print(f"分析异常: {traceback.format_exc()}")

@app.post("/upload_config", response_model=FileUploadResponse)
async def upload_config(file: UploadFile = File(...)):
    """上传配置文件"""
    
    try:
        # 检查文件类型
        if not file.filename.endswith(('.json', '.yaml', '.yml')):
            raise HTTPException(status_code=400, detail="只支持JSON和YAML格式文件")
        
        # 读取文件内容
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # 解析文件
        if file.filename.endswith('.json'):
            parsed_info = ConfigFileParser.parse_json_config(content_str)
        else:
            parsed_info = ConfigFileParser.parse_yaml_config(content_str)
        
        # 生成文件ID
        file_id = str(uuid.uuid4())
        
        # 保存到临时存储 (生产环境建议使用数据库)
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        with open(temp_dir / f"{file_id}.json", 'w', encoding='utf-8') as f:
            json.dump(parsed_info, f, indent=2, ensure_ascii=False)
        
        return FileUploadResponse(
            filename=file.filename,
            file_id=file_id,
            content=parsed_info,
            suggestions=parsed_info.get("suggestions", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件解析失败: {str(e)}")

@app.post("/generate_command")
async def generate_command(config_data: Dict[str, Any]):
    """基于配置生成分析命令"""
    
    try:
        # 提取配置信息
        model_info = config_data.get("model_info", {})
        analysis_params = config_data.get("analysis_params", {})
        
        # 构建提示词
        model_name = model_info.get("model_name", "unknown_model")
        batch_size = analysis_params.get("batch_size", [8])
        input_len = analysis_params.get("input_len", [512])
        output_len = analysis_params.get("output_len", [64])
        analysis_type = analysis_params.get("analysis_type", "auto")
        
        # 格式化batch_size等参数
        if isinstance(batch_size, list):
            batch_str = ",".join(map(str, batch_size))
        else:
            batch_str = str(batch_size)
        
        if isinstance(input_len, list):
            input_str = ",".join(map(str, input_len))
        else:
            input_str = str(input_len)
        
        if isinstance(output_len, list):
            output_str = ",".join(map(str, output_len))
        else:
            output_str = str(output_len)
        
        # 生成提示词
        prompt = f"""分析模型 {model_name}，
使用 {analysis_type} 分析，
batch_size: {batch_str}，
input_len: {input_str}，
output_len: {output_str}"""
        
        # 添加其他参数
        if "temperature" in analysis_params:
            prompt += f"，temperature: {analysis_params['temperature']}"
        
        if "tp_size" in analysis_params and analysis_params["tp_size"] > 1:
            prompt += f"，tp_size: {analysis_params['tp_size']}"
        
        return {
            "prompt": prompt,
            "command": f'python ai_agent_analyzer.py prompt "{prompt}"',
            "config_summary": {
                "model": model_name,
                "analysis_type": analysis_type,
                "batch_size": batch_size,
                "input_len": input_len,
                "output_len": output_len
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"生成命令失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(manager.active_connections),
        "agent_ready": agent is not None
    }

@app.get("/sessions")
async def list_sessions():
    """列出活动会话"""
    return {
        "active_sessions": list(manager.active_connections.keys()),
        "count": len(manager.active_connections)
    }

if __name__ == "__main__":
    # 开发环境运行
    uvicorn.run(
        "web_agent_backend:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )
