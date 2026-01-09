# AI Agent - 快速配置参考卡片

## 📍 关键配置位置速查

### 1. SGlang服务地址

#### 主配置文件（推荐修改这里）
```yaml
文件: AI agent/configs_and_docs/agent_config.yaml
行数: 21-22

defaults:
  host: "127.0.0.1"      # ← 修改SGlang服务IP
  port: 30000            # ← 修改SGlang服务端口
```

**常见配置：**
- 本地运行: `127.0.0.1:30000`
- 远程服务器: `192.168.1.100:30000`

---

### 2. 模型文件路径

#### 模型目录配置
```yaml
文件: AI agent/configs_and_docs/agent_config.yaml
行数: 6

workspace:
  models_dir: "workspace/models"  # ← 修改模型存放目录
```

#### 模型路径映射
```yaml
文件: AI agent/configs_and_docs/agent_config.yaml
行数: 51-61

model_mappings:
  "llama-7b": "meta-llama/Llama-2-7b-hf"    # ← 修改为本地路径
  "qwen-14b": "Qwen/Qwen-14B-Chat"          # ← 或HuggingFace ID
```

**示例配置：**
```yaml
# 使用本地路径
model_mappings:
  "llama-7b": "D:/Models/Llama-2-7b-hf"
  "llama-7b": "workspace/models/Llama-2-7b-hf"  # 相对路径
```

---

### 3. SGlang代码路径

```yaml
文件: AI agent/configs_and_docs/agent_config.yaml
行数: 7

workspace:
  sglang_dir: "SGlang"   # ← 修改SGlang代码目录
```

**放置SGlang代码：**
```bash
# 方法1: 克隆到SGlang目录
git clone https://github.com/sgl-project/sglang.git SGlang

# 方法2: 使用已有的SGlang（Windows）
mklink /D SGlang "D:\path\to\your\sglang"

# 方法3: 修改配置指向其他位置
sglang_dir: "D:/Code/sglang"
```

---

### 4. 前端服务地址

#### 前端WebSocket连接
```javascript
文件: AI agent/web_interface/static/chat.html
     AI agent/langchain_version/static/chat.html
     AI agent/original_version/static/chat.html
行数: 约546-548

function initWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws/${sessionId}`;
    // ↑ 默认使用当前页面地址，通常不需要修改
}
```

**如需修改（前后端分离）：**
```javascript
const wsUrl = `ws://192.168.1.100:8000/ws/${sessionId}`;
```

#### 文件上传API
```javascript
文件: chat.html
行数: 约834

const response = await fetch('/upload_config', {
    method: 'POST',
    body: formData
});
// ↑ 使用相对路径，自动连接到当前服务器
```

---

## 🚀 快速启动命令

### 方法1: 使用启动脚本（最简单）
```bash
python start_ai_agent.py
# 然后选择: 1(LangChain版本) 或 2(原始版本)
```

### 方法2: 直接启动LangChain版本
```bash
cd "AI agent/langchain_version"
python web_langchain_backend.py
```

### 方法3: 直接启动原始版本
```bash
cd "AI agent/original_version"
python web_agent_backend.py
```

### 访问前端
```
浏览器打开: http://localhost:8000/chat
```

---

## ✅ 启动前必做检查

```bash
# 1. 检查Python依赖
pip install -r requirements_complete.txt

# 2. 检查NVIDIA工具
nvidia-smi    # 检查GPU
nsys --version  # 检查NSight Systems
ncu --version   # 检查NSight Compute

# 3. 检查目录结构
ls SGlang/     # 应该有SGlang代码
ls workspace/models/  # 应该有模型文件

# 4. 测试后端
python start_ai_agent.py
# 或
curl http://localhost:8000/health
```

---

## 🔍 测试配置

### 1. 健康检查
```bash
curl http://localhost:8000/health
```

预期响应:
```json
{
  "status": "healthy",
  "timestamp": "...",
  "active_sessions": 0,
  "langchain_agent_ready": true
}
```

### 2. 前端测试
浏览器访问: `http://localhost:8000/chat`
- 状态指示器应显示绿色"已连接"
- 应能看到欢迎消息

### 3. 对话测试
输入测试提示词:
```
分析 llama-7b 模型，batch_size=8
```

应该收到参数解析的回复。

---

## 📝 配置文件完整路径

| 配置项 | 文件路径 | 行数 |
|--------|---------|------|
| SGlang服务地址 | `AI agent/configs_and_docs/agent_config.yaml` | 21-22 |
| 模型目录 | `AI agent/configs_and_docs/agent_config.yaml` | 6 |
| 模型映射 | `AI agent/configs_and_docs/agent_config.yaml` | 51-61 |
| SGlang目录 | `AI agent/configs_and_docs/agent_config.yaml` | 7 |
| 前端WS地址 | `AI agent/*/static/chat.html` | 546-548 |

---

## 🐛 常见问题速查

### 问题: WebSocket连接失败
```
检查: 后端是否运行
解决: python start_ai_agent.py
```

### 问题: 找不到模型
```
检查: workspace/models/ 目录
解决: 
1. 下载模型到该目录
2. 或修改 agent_config.yaml 中的 model_mappings
```

### 问题: SGlang命令失败
```
检查: SGlang/ 目录是否有代码
解决: git clone https://github.com/sgl-project/sglang.git SGlang
```

### 问题: 导入错误
```
检查: 是否运行了路径修复
解决: python fix_paths.py
```

---

## 📊 目录结构参考

```
Agent/
├── AI agent/
│   ├── configs_and_docs/
│   │   └── agent_config.yaml        ← 主配置文件
│   ├── langchain_version/           ← LangChain版本（推荐）
│   │   ├── web_langchain_backend.py ← 启动这个
│   │   └── static/
│   │       └── chat.html            ← 前端页面
│   └── original_version/            ← 原始版本
│       └── web_agent_backend.py     ← 或启动这个
├── TOOLS/
│   └── Auto_Anlyze_tool/            ← 性能分析工具
│       ├── nsys_parser.py
│       └── ncu_parser.py
├── SGlang/                          ← 需要放置SGlang代码
├── workspace/
│   └── models/                      ← 需要放置模型文件
├── start_ai_agent.py                ← 快速启动脚本
├── fix_paths.py                     ← 路径修复脚本
├── requirements_complete.txt        ← 依赖列表
└── 配置指南.md                      ← 详细配置文档
```

---

## 💡 快速配置模板

### 本地开发环境
```yaml
# agent_config.yaml
workspace:
  models_dir: "workspace/models"
  sglang_dir: "SGlang"

defaults:
  host: "127.0.0.1"
  port: 30000

model_mappings:
  "llama-7b": "workspace/models/Llama-2-7b-hf"
```

### 使用远程SGlang服务器
```yaml
# agent_config.yaml
defaults:
  host: "192.168.1.100"    # 远程服务器IP
  port: 30000

model_mappings:
  "llama-7b": "/remote/path/to/model"  # 服务器上的路径
```

### 使用绝对路径
```yaml
# agent_config.yaml
workspace:
  models_dir: "D:/Models"
  sglang_dir: "D:/Code/sglang"

model_mappings:
  "llama-7b": "D:/Models/Llama-2-7b-hf"
  "qwen-14b": "D:/Models/Qwen-14B-Chat"
```

---

## 📞 获取帮助

1. 查看详细文档: `配置指南.md`
2. 检查日志输出: 运行时的终端信息
3. 验证配置: `cat "AI agent/configs_and_docs/agent_config.yaml"`

**项目状态检查：**
```bash
# 检查所有关键文件
python -c "
from pathlib import Path
files = [
    'AI agent/configs_and_docs/agent_config.yaml',
    'AI agent/langchain_version/web_langchain_backend.py',
    'TOOLS/Auto_Anlyze_tool/nsys_parser.py',
    'SGlang/',
    'workspace/models/'
]
for f in files:
    status = '✓' if Path(f).exists() else '✗'
    print(f'{status} {f}')
"
```

