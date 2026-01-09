#!/usr/bin/env python3
"""
AI Agent 文件路径修复脚本

功能：
1. 修复静态文件路径
2. 修复Python导入路径
3. 创建必要的目录结构
4. 复制文件到正确位置
"""

import os
import shutil
from pathlib import Path

def print_section(title):
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def create_directories():
    """创建必要的目录结构"""
    print_section("创建目录结构")
    
    directories = [
        "workspace/models",
        "analysis_results",
        "SGlang",  # 占位目录
        "AI agent/langchain_version/static",
        "AI agent/original_version/static",
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path}")
    
    # 创建SGlang目录说明
    sglang_readme = Path("SGlang/README.md")
    if not sglang_readme.exists():
        sglang_readme.write_text("""# SGlang 目录

这个目录需要放置SGlang的代码。

## 安装方法1: 克隆仓库
```bash
cd ..
git clone https://github.com/sgl-project/sglang.git
mv sglang/* SGlang/
```

## 安装方法2: 已有SGlang
如果你已经有SGlang代码在其他位置，可以：
1. 复制到这里
2. 或者在 `agent_config.yaml` 中修改 `sglang_dir` 指向你的SGlang目录

## 验证安装
```bash
cd SGlang
python -m sglang.launch_server --help
```
""", encoding='utf-8')

def copy_static_files():
    """复制静态文件到正确位置"""
    print_section("复制静态文件")
    
    source = Path("AI agent/web_interface/static/chat.html")
    
    targets = [
        Path("AI agent/langchain_version/static/chat.html"),
        Path("AI agent/original_version/static/chat.html"),
    ]
    
    if source.exists():
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"✅ 复制: {source} -> {target}")
    else:
        print(f"⚠️ 源文件不存在: {source}")

def fix_import_paths():
    """修复导入路径问题"""
    print_section("修复导入路径")
    
    # 修复 langchain_version/langchain_agent.py 的导入
    langchain_agent_file = Path("AI agent/langchain_version/langchain_agent.py")
    
    if langchain_agent_file.exists():
        content = langchain_agent_file.read_text(encoding='utf-8')
        
        # 替换导入路径
        old_imports = [
            "from ai_agent_analyzer import AIAgentAnalyzer, PromptParser, ConfigGenerator, AnalysisRequest",
            "from web_agent_backend import ConfigFileParser"
        ]
        
        new_imports = [
            "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).parent.parent / 'original_version'))\nsys.path.insert(0, str(Path(__file__).parent.parent.parent / 'TOOLS' / 'Auto_Anlyze_tool'))\nfrom ai_agent_analyzer import AIAgentAnalyzer, PromptParser, ConfigGenerator, AnalysisRequest",
            "from web_agent_backend import ConfigFileParser"
        ]
        
        # 检查是否需要修改
        if "sys.path.insert" not in content:
            # 在文件开头添加路径设置
            lines = content.split('\n')
            
            # 找到第一个import的位置
            import_line_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_line_idx = i
                    break
            
            # 插入路径设置
            path_setup = """
# 添加路径以正确导入模块
import sys
from pathlib import Path
_current_dir = Path(__file__).parent
sys.path.insert(0, str(_current_dir.parent / 'original_version'))
sys.path.insert(0, str(_current_dir.parent.parent / 'TOOLS' / 'Auto_Anlyze_tool'))
"""
            lines.insert(import_line_idx, path_setup)
            
            new_content = '\n'.join(lines)
            langchain_agent_file.write_text(new_content, encoding='utf-8')
            print(f"✅ 修复: {langchain_agent_file}")
        else:
            print(f"ℹ️ 已修复: {langchain_agent_file}")
    
    # 修复 original_version/ai_agent_analyzer.py 的导入
    ai_agent_file = Path("AI agent/original_version/ai_agent_analyzer.py")
    
    if ai_agent_file.exists():
        content = ai_agent_file.read_text(encoding='utf-8')
        
        # 检查TOOLS导入路径
        if 'tools_dir = Path("TOOLS/Auto_Anlyze_tool")' in content:
            # 修改为相对于项目根目录的路径
            content = content.replace(
                'tools_dir = Path("TOOLS/Auto_Anlyze_tool")',
                'tools_dir = Path(__file__).parent.parent.parent / "TOOLS" / "Auto_Anlyze_tool"'
            )
            
            ai_agent_file.write_text(content, encoding='utf-8')
            print(f"✅ 修复: {ai_agent_file}")
        else:
            print(f"ℹ️ 已修复: {ai_agent_file}")

def create_startup_scripts():
    """创建启动脚本"""
    print_section("创建启动脚本")
    
    # 创建根目录启动脚本
    start_script = Path("启动AI Agent.py")
    
    start_script.write_text("""#!/usr/bin/env python3
\"\"\"
AI Agent 快速启动脚本
\"\"\"

import sys
import subprocess
from pathlib import Path

def main():
    print("🤖 AI Agent LLM性能分析器 - 启动向导")
    print("="*50)
    print()
    print("请选择要启动的版本：")
    print("1. LangChain版本 (推荐) - 支持智能对话和工具链")
    print("2. 原始版本 - 基础功能版本")
    print()
    
    choice = input("请输入选项 (1/2) [默认: 1]: ").strip() or "1"
    
    if choice == "1":
        print("\\n🚀 启动 LangChain 版本...")
        backend_path = Path("AI agent/langchain_version/web_langchain_backend.py")
    elif choice == "2":
        print("\\n🚀 启动原始版本...")
        backend_path = Path("AI agent/original_version/web_agent_backend.py")
    else:
        print("❌ 无效选项")
        return
    
    if not backend_path.exists():
        print(f"❌ 文件不存在: {backend_path}")
        return
    
    print(f"📁 工作目录: {Path.cwd()}")
    print(f"🌐 服务地址: http://localhost:8000")
    print(f"💬 聊天界面: http://localhost:8000/chat")
    print()
    print("按 Ctrl+C 停止服务")
    print("="*50)
    print()
    
    try:
        subprocess.run([sys.executable, str(backend_path)], check=True)
    except KeyboardInterrupt:
        print("\\n\\n👋 服务已停止")
    except Exception as e:
        print(f"\\n❌ 启动失败: {e}")

if __name__ == "__main__":
    main()
""", encoding='utf-8')
    
    print(f"✅ 创建: {start_script}")
    
    # 创建工作目录说明
    workspace_readme = Path("workspace/models/README.md")
    if not workspace_readme.exists():
        workspace_readme.parent.mkdir(parents=True, exist_ok=True)
        workspace_readme.write_text("""# 模型文件目录

请将LLM模型文件放置在此目录下。

## 目录结构示例

```
models/
├── Llama-2-7b-hf/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer_config.json
├── Qwen-14B-Chat/
└── chatglm-6b/
```

## 模型下载方法

### 方法1: 使用 HuggingFace CLI
```bash
pip install huggingface_hub
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./Llama-2-7b-hf
```

### 方法2: 使用 Git LFS
```bash
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
```

### 方法3: 手动下载
访问 HuggingFace 模型页面，手动下载所有文件到相应目录。

## 配置模型路径

在 `AI agent/configs_and_docs/agent_config.yaml` 中配置：

```yaml
model_mappings:
  "llama-7b": "workspace/models/Llama-2-7b-hf"
  "qwen-14b": "workspace/models/Qwen-14B-Chat"
```

或使用绝对路径：

```yaml
model_mappings:
  "llama-7b": "D:/Models/Llama-2-7b-hf"
```

## 注意事项

1. 确保模型文件完整，包含所有必需文件
2. 大型模型需要足够的磁盘空间（70B模型可能需要100GB+）
3. 首次加载模型可能需要较长时间
""", encoding='utf-8')

def create_requirements_file():
    """创建requirements文件"""
    print_section("创建依赖文件")
    
    requirements = Path("requirements_complete.txt")
    requirements.write_text("""# AI Agent LLM性能分析器 - 完整依赖

# Web框架
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
websockets>=12.0

# 数据处理和可视化
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0

# 配置文件处理
pyyaml>=6.0

# HTTP请求
requests>=2.31.0

# LangChain (可选，用于LangChain版本)
langchain>=0.0.350
langchain-openai>=0.0.2
langchain-community>=0.0.10

# 其他工具
python-multipart>=0.0.6  # 文件上传支持
""", encoding='utf-8')
    
    print(f"✅ 创建: {requirements}")

def verify_environment():
    """验证环境配置"""
    print_section("验证环境")
    
    # 检查关键文件
    critical_files = [
        "AI agent/configs_and_docs/agent_config.yaml",
        "AI agent/langchain_version/langchain_agent.py",
        "AI agent/langchain_version/web_langchain_backend.py",
        "AI agent/original_version/ai_agent_analyzer.py",
        "TOOLS/Auto_Anlyze_tool/nsys_parser.py",
        "TOOLS/Auto_Anlyze_tool/ncu_parser.py",
    ]
    
    all_exist = True
    for file_path in critical_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 文件不存在")
            all_exist = False
    
    return all_exist

def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════╗
║   AI Agent LLM性能分析器 - 文件路径修复工具             ║
╚══════════════════════════════════════════════════════════╝
""")
    
    try:
        # 1. 创建目录
        create_directories()
        
        # 2. 复制静态文件
        copy_static_files()
        
        # 3. 修复导入路径
        fix_import_paths()
        
        # 4. 创建启动脚本
        create_startup_scripts()
        
        # 5. 创建依赖文件
        create_requirements_file()
        
        # 6. 验证环境
        all_ok = verify_environment()
        
        print_section("修复完成")
        
        if all_ok:
            print("✅ 所有关键文件检查通过")
            print()
            print("📋 下一步操作：")
            print()
            print("1. 安装依赖:")
            print("   pip install -r requirements_complete.txt")
            print()
            print("2. 配置SGlang:")
            print("   - 将SGlang代码放入 SGlang/ 目录")
            print("   - 或在 agent_config.yaml 中修改 sglang_dir 路径")
            print()
            print("3. 配置模型:")
            print("   - 将模型放入 workspace/models/ 目录")
            print("   - 或在 agent_config.yaml 中配置模型路径")
            print()
            print("4. 启动服务:")
            print("   python 启动AI Agent.py")
            print()
            print("5. 访问前端:")
            print("   浏览器打开 http://localhost:8000/chat")
            print()
            print("📚 详细配置请参考: 配置指南.md")
        else:
            print("⚠️ 部分文件缺失，请检查项目完整性")
        
    except Exception as e:
        print(f"\\n❌ 修复过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
""", encoding='utf-8')
    
    print(f"✅ 创建修复脚本: 修复文件路径.py")

