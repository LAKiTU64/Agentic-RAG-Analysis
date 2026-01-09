# AI Agent Web部署说明 🚀

## 📋 快速部署 (容器环境)

在您的服务器容器中，按以下步骤部署AI Agent Web服务：

### 1. 环境准备

```bash
# 确保Python 3.8+
python3 --version

# 安装依赖
pip3 install -r requirements_web.txt
```

### 2. 文件结构

确保以下文件存在：
```
AI/
├── web_agent_backend.py        # Web后端服务器
├── ai_agent_analyzer.py        # AI Agent核心
├── static/
│   └── chat.html              # 聊天界面
├── requirements_web.txt       # Web依赖
├── agent_config.yaml          # 配置文件
├── example_model_config.json  # 示例配置
└── TOOLS/Auto_Anlyze_tool/    # 分析工具(可选)
```

### 3. 一键部署

```bash
# 运行部署脚本
python3 deploy_web_agent.py

# 或手动启动
python3 web_agent_backend.py
```

### 4. 访问服务

- **聊天界面**: `http://localhost:8000/chat`
- **API文档**: `http://localhost:8000/docs`
- **健康检查**: `http://localhost:8000/health`

## 🔧 功能检查

```bash
# 检查功能完整性
python3 check_agent_features.py
```

## 💡 主要功能

### ✅ 智能对话
- 类ChatGPT界面
- 自然语言分析需求
- 实时进度显示

### ✅ 文件上传
- 支持JSON/YAML配置
- 自动解析模型信息
- 智能建议生成

### ✅ 性能分析
- NSys全局分析
- NCU kernel分析
- 集成分析模式

### ✅ 自动配置
- SGlang脚本参数
- 模型路径解析
- 命令生成

## 📝 使用示例

### 1. 文本对话
```
用户: "分析 llama-7b 模型，batch_size=8,16"
AI Agent: 自动配置参数并开始分析...
```

### 2. 配置文件上传
上传 `example_model_config.json`，AI Agent会：
- 解析模型信息
- 提供优化建议  
- 生成分析命令

### 3. 支持的提示词格式
```
# 中文
"分析 qwen-14b，进行ncu深度分析"
"对 chatglm-6b 进行综合性能分析，batch_size=1,4,8"

# 英文  
"analyze llama-7b with nsys profiling"
"ncu analysis for model with batch_size=16"
```

## 🛠️ 环境变量

```bash
# 可选配置
export PORT=8000                    # 服务端口
export CUDA_VISIBLE_DEVICES=0      # GPU设备
export WORKSPACE_ROOT=/app         # 工作目录
export MAX_UPLOAD_SIZE=10M         # 上传限制
```

## 📊 API接口

| 端点 | 方法 | 功能 |
|------|------|------|
| `/chat` | GET | 聊天界面 |
| `/ws/{session_id}` | WebSocket | 实时通信 |
| `/upload_config` | POST | 上传配置文件 |
| `/generate_command` | POST | 生成分析命令 |
| `/health` | GET | 健康检查 |

## 🔍 故障排除

### 端口占用
```bash
# 检查端口
lsof -i :8000

# 更换端口
PORT=8080 python3 web_agent_backend.py
```

### 依赖问题
```bash
# 重新安装
pip3 install -r requirements_web.txt --force-reinstall
```

### GPU支持
```bash
# 检查CUDA
nvidia-smi
nvcc --version
```

## 🎯 性能建议

1. **生产环境**: 使用 `gunicorn` 或 `uvicorn` workers
2. **负载均衡**: 配置多个实例
3. **文件存储**: 使用持久化存储卷
4. **监控**: 集成日志和监控系统

## 📞 支持

如有问题，请检查：
1. 依赖是否正确安装
2. 端口是否被占用
3. 分析工具是否存在
4. GPU环境是否正常

---

**🎉 AI Agent已准备就绪，开始您的LLM性能分析之旅！**
