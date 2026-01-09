# AI Agent LLM性能分析器 🤖

一个智能的大语言模型(LLM)性能分析助手，基于LangChain框架提供更智能的对话体验，能够自动配置SGlang脚本参数，运行nsys/ncu性能分析，并生成详细的分析报告。

## 🆕 版本选择

### 🔥 LangChain版 (推荐)
- **智能对话理解**: 基于LangChain的自然语言处理
- **工具链自动调用**: 智能选择合适的分析工具
- **记忆管理**: 记住对话历史和上下文
- **复杂工作流**: 支持多步骤链式分析流程
- **决策推理**: 智能分析需求和生成执行计划

### 📱 原版
- **快速启动**: 基于原生实现，启动迅速
- **功能完整**: 包含所有核心性能分析功能
- **轻量级**: 依赖较少，适合简单使用场景

## ✨ 主要功能

### 🎯 智能提示词解析
- 自动识别模型名称、脚本类型、分析需求
- 支持中英文提示词
- 智能参数配置和默认值填充

### ⚙️ 自动化配置
- 自动设置模型路径 (`workspace/models/`)
- 智能配置 `batch_size`, `input_len`, `output_len` 等参数
- 支持多种SGlang脚本 (`bench_one_batch_server`, `launch_server`)

### 🔬 多种分析模式
- **NSys分析**: 全局性能分析，timeline视图
- **NCU分析**: CUDA kernel深度分析
- **集成分析**: 先nsys识别热点，再ncu深度分析

### 📊 专业报告生成
- 自动生成可视化图表
- 详细的性能瓶颈分析
- 优化建议和专业报告

## 🚀 快速开始

### 安装依赖

```bash
# 安装Python依赖
pip install pandas matplotlib seaborn numpy requests

# 确保已安装NVIDIA工具
# - NVIDIA Nsight Systems
# - NVIDIA Nsight Compute
```

### 基本使用

#### 1. 启动服务

```bash
# 快速启动 (自动选择版本)
python start_agent.py --interactive

# 启动LangChain版 (推荐)
python start_agent.py --version langchain --port 8000

# 启动原版
python start_agent.py --version original --port 8000

# 直接启动LangChain版
python web_langchain_backend.py
```

#### 2. 命令行分析 (原版)

```bash
# 从提示词开始分析
python ai_agent_analyzer.py prompt "分析 llama-7b 模型，batch_size=8,16"

# 分析已有profile文件
python ai_agent_analyzer.py file profile.nsys-rep --analysis-type nsys

# 交互式模式
python ai_agent_analyzer.py interactive
```

#### 2. Python API

```python
from ai_agent_analyzer import AIAgentAnalyzer

# 创建分析器
agent = AIAgentAnalyzer(workspace_root=".")

# 从提示词分析
results = agent.analyze_from_prompt("分析 qwen-14b，进行ncu深度分析")

# 分析已有文件
results = agent.analyze_existing_files("profile.nsys-rep", "nsys")
```

## 📝 使用示例

### 基础分析

```python
# 原版AI Agent
from ai_agent_analyzer import AIAgentAnalyzer
agent = AIAgentAnalyzer()
results = agent.analyze_from_prompt("分析 llama-7b 模型，batch_size=8")

# LangChain版AI Agent (推荐)
from langchain_agent import LangChainAgent
agent = LangChainAgent()
result = await agent.process_message("分析 llama-7b 模型，batch_size=8")
```

### 不同分析类型

```python
# NSys全局分析
nsys_prompt = "对 qwen-14b 进行 nsys 全局性能分析，batch_size=16"

# NCU kernel分析  
ncu_prompt = "对 chatglm-6b 进行 ncu kernel深度分析，batch_size=4"

# 集成分析
auto_prompt = "对 baichuan-13b 进行综合分析，batch_size=8,16"
```

### 自定义参数

```python
prompt = """
分析模型 meta-llama/Llama-2-7b-hf，
batch_size: 1,4,8,16，
input_len: 256,512,1024，
output_len: 32,64,128，
temperature: 0.1，
tp_size: 2，
进行集成分析
"""
```

### 中文提示词支持

```python
chinese_prompts = [
    "分析 llama-7b 模型性能，批次大小8，输入长度512",
    "对 qwen-14b 进行深度kernel分析，使用ncu工具", 
    "综合分析 chatglm-6b 的性能瓶颈，包括nsys和ncu"
]
```

## 🏗️ 工作空间结构

推荐的项目结构：

```
workspace/
├── models/                          # 模型目录
│   ├── llama-7b/                   # 本地模型
│   ├── qwen-14b/
│   └── chatglm-6b/
├── TOOLS/
│   └── Auto_Anlyze_tool/           # 分析工具
│       ├── nsys_parser.py
│       ├── ncu_parser.py  
│       └── nsys_to_ncu_analyzer.py
├── SGlang/                         # SGlang源码
│   └── python/sglang/
├── ai_agent_analyzer.py            # AI Agent主程序
├── agent_config.yaml              # 配置文件
├── agent_examples.py               # 使用示例
└── analysis_*/                     # 分析结果目录
    ├── nsys_analysis_output/
    ├── ncu_analysis_output/
    └── integrated_analysis/
```

## ⚙️ 配置说明

### 默认参数 (agent_config.yaml)

```yaml
defaults:
  batch_size: [1, 8, 16]
  input_len: [512, 1024] 
  output_len: [64, 128]
  temperature: 0.0
  tp_size: 1
  analysis_type: "auto"
```

### 模型路径映射

```yaml
model_mappings:
  "llama-7b": "meta-llama/Llama-2-7b-hf"
  "qwen-14b": "Qwen/Qwen-14B-Chat"
  "chatglm-6b": "THUDM/chatglm-6b"
```

## 🔍 分析类型详解

### 1. NSys分析 (`nsys`)
- **用途**: 全局性能概览
- **输出**: Timeline、kernel统计、内存传输分析
- **适用**: 识别性能热点，了解整体执行流程

### 2. NCU分析 (`ncu`) 
- **用途**: CUDA kernel深度分析
- **输出**: SM效率、占用率、内存带宽、瓶颈分析
- **适用**: 优化特定kernel性能

### 3. 集成分析 (`auto`)
- **流程**: NSys识别热点 → NCU深度分析 → 综合报告
- **输出**: 完整的性能分析和优化建议
- **适用**: 全面的性能分析需求

## 🧠 LangChain版特有功能

### 🔗 智能工作流链
```python
from langchain_workflows import PerformanceAnalysisWorkflow

# 创建工作流
workflow = PerformanceAnalysisWorkflow()

# 运行智能分析链
result = await workflow.run_workflow(
    "分析llama-7b模型的性能瓶颈",
    config_data={"batch_size": [4, 8]}
)

# 自动生成: 模型分析 → 配置优化 → 执行计划 → 建议生成
```

### 💭 对话记忆管理
```python
# LangChain版本会记住对话历史
用户: "分析llama-7b模型"
AI: "好的，我来为您分析llama-7b模型..."

用户: "增加batch_size到16"  
AI: "明白，我会在之前的llama-7b分析基础上，将batch_size调整为16..."
```

### 🛠️ 智能工具选择
```python
# AI Agent会根据需求自动选择合适的工具
"我想了解模型的内存使用" → 自动调用 NSys分析工具
"kernel效率太低了" → 自动调用 NCU深度分析工具
"给我一些优化建议" → 自动调用 优化建议工具
```

## 📊 输出结果

每次分析会生成包含以下内容的结果目录：

```
analysis_llama-7b_20241011_143052/
├── benchmark_results.jsonl         # 基准测试结果
├── nsys_analysis_output/           # NSys分析结果
│   ├── kernel_timeline.png
│   ├── top_kernels.png
│   └── analysis_report.txt
├── ncu_analysis_output/            # NCU分析结果  
│   ├── gpu_utilization.png
│   ├── bottleneck_analysis.png
│   └── ncu_analysis_report.txt
└── integrated/                     # 集成分析结果
    ├── comprehensive_analysis.json
    └── integrated_performance_report.md
```

## 🎛️ 高级功能

### 交互式模式

```bash
python ai_agent_analyzer.py interactive
```

进入交互模式后，可以连续输入分析需求：

```
💬 请输入分析需求: 分析 llama-7b，batch_size=8
✅ 分析完成

💬 请输入分析需求: 对 qwen-14b 进行ncu分析
✅ 分析完成

💬 请输入分析需求: quit
👋 再见!
```

### 批量分析

```python
models = ["llama-7b", "qwen-14b", "chatglm-6b"]
for model in models:
    prompt = f"分析 {model}，进行集成分析，batch_size=8,16"
    results = agent.analyze_from_prompt(prompt)
```

## 🔧 故障排除

### 常见问题

1. **模型路径找不到**
   ```
   ⚠️  本地未找到模型，使用HuggingFace ID: llama-7b
   ```
   - 确保模型在 `workspace/models/` 目录下
   - 或者使用完整的HuggingFace模型ID

2. **NSys/NCU命令未找到**
   ```
   ❌ 未找到 nsys 命令
   ```
   - 安装NVIDIA Nsight Systems和Compute
   - 确保命令在PATH中

3. **分析超时**
   ```
   ⏰ ncu分析超时
   ```
   - 减少分析的kernel数量
   - 增加timeout设置

### 环境检查

```python
# 检查环境依赖
python -c "
import subprocess
import sys

required_commands = ['nsys', 'ncu', 'python']
for cmd in required_commands:
    try:
        subprocess.run([cmd, '--version'], capture_output=True, check=True)
        print(f'✅ {cmd} 可用')
    except:
        print(f'❌ {cmd} 不可用')
"
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发设置

```bash
git clone <repo>
cd ai-agent-analyzer
pip install -r requirements.txt

# 运行示例
python agent_examples.py

# 运行测试
python ai_agent_analyzer.py prompt "测试 llama-7b"
```

## 📄 许可证

MIT License

## 🔗 相关资源

- [NVIDIA Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/)
- [NVIDIA Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/)  
- [SGlang 项目](https://github.com/sgl-project/sglang)

---

**🎯 立即开始使用AI Agent自动化您的LLM性能分析！**
