#!/usr/bin/env python3
"""
AI Agent环境设置脚本

自动检查和设置AI Agent运行环境
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_command_exists(command):
    """检查命令是否存在"""
    return shutil.which(command) is not None

def check_python_package(package):
    """检查Python包是否已安装"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def create_directory_structure():
    """创建推荐的目录结构"""
    
    directories = [
        "workspace",
        "workspace/models", 
        "analysis_results",
        "TOOLS/Auto_Anlyze_tool"
    ]
    
    print("📁 创建目录结构...")
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_path}")
    
    # 创建模型目录的README
    models_readme = Path("workspace/models/README.md")
    if not models_readme.exists():
        models_readme.write_text("""# 模型目录

请将LLM模型文件放置在此目录下。

## 支持的模型格式
- HuggingFace格式模型
- 本地模型文件

## 目录结构示例
```
models/
├── llama-7b/
├── qwen-14b/
└── chatglm-6b/
```

## 模型下载示例
```bash
# 使用HuggingFace Hub
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir llama-7b

# 使用Git LFS
git lfs clone https://huggingface.co/meta-llama/Llama-2-7b-hf llama-7b
```
""")

def check_environment():
    """检查环境依赖"""
    
    print("🔍 检查环境依赖...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"  Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("  ❌ Python版本过低，建议3.8+")
        return False
    else:
        print("  ✅ Python版本满足要求")
    
    # 检查必需的命令
    required_commands = {
        "nsys": "NVIDIA Nsight Systems",
        "ncu": "NVIDIA Nsight Compute", 
        "python": "Python解释器",
        "pip": "Python包管理器"
    }
    
    missing_commands = []
    for command, description in required_commands.items():
        if check_command_exists(command):
            print(f"  ✅ {command} ({description})")
        else:
            print(f"  ❌ {command} ({description}) - 未找到")
            missing_commands.append(command)
    
    # 检查Python包
    required_packages = [
        "pandas", "matplotlib", "seaborn", "numpy", "requests"
    ]
    
    missing_packages = []
    for package in required_packages:
        if check_python_package(package):
            print(f"  ✅ {package}")
        else:
            print(f"  ❌ {package} - 未安装")
            missing_packages.append(package)
    
    return len(missing_commands) == 0 and len(missing_packages) == 0

def install_python_requirements():
    """安装Python依赖"""
    
    requirements = [
        "pandas>=1.3.0",
        "matplotlib>=3.5.0", 
        "seaborn>=0.11.0",
        "numpy>=1.21.0",
        "requests>=2.25.0",
        "pyyaml>=5.4.0"
    ]
    
    print("📦 安装Python依赖...")
    
    # 创建requirements.txt
    requirements_file = Path("requirements_agent.txt")
    requirements_file.write_text("\n".join(requirements))
    print(f"  📝 已生成 {requirements_file}")
    
    # 安装依赖
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
        print("  ✅ Python依赖安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ 安装失败: {e}")
        return False

def setup_analysis_tools():
    """设置分析工具"""
    
    tools_dir = Path("TOOLS/Auto_Anlyze_tool")
    
    # 检查分析脚本是否存在
    required_scripts = [
        "nsys_parser.py",
        "ncu_parser.py", 
        "nsys_to_ncu_analyzer.py"
    ]
    
    print("🔧 检查分析工具...")
    
    missing_scripts = []
    for script in required_scripts:
        script_path = tools_dir / script
        if script_path.exists():
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script} - 未找到")
            missing_scripts.append(script)
    
    if missing_scripts:
        print("  ⚠️  请确保分析工具脚本已正确放置在 TOOLS/Auto_Anlyze_tool/ 目录下")
        return False
    
    return True

def check_gpu_environment():
    """检查GPU环境"""
    
    print("🖥️  检查GPU环境...")
    
    try:
        # 检查nvidia-smi
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        print("  ✅ NVIDIA GPU驱动正常")
        
        # 尝试提取GPU信息
        lines = result.stdout.split('\n')
        gpu_lines = [line for line in lines if 'GeForce' in line or 'Tesla' in line or 'Quadro' in line or 'A100' in line or 'H100' in line]
        
        for gpu_line in gpu_lines[:3]:  # 最多显示3个GPU
            gpu_info = gpu_line.split('|')[1].strip() if '|' in gpu_line else gpu_line.strip()
            print(f"    📱 {gpu_info}")
        
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ❌ 未检测到NVIDIA GPU或驱动未正确安装")
        return False

def create_example_config():
    """创建示例配置文件"""
    
    print("📄 创建示例文件...")
    
    # 创建示例启动脚本
    start_script = Path("start_agent.py")
    if not start_script.exists():
        start_script.write_text("""#!/usr/bin/env python3
\"\"\"
AI Agent快速启动脚本
\"\"\"

from ai_agent_analyzer import AIAgentAnalyzer

def main():
    print("🤖 AI Agent LLM性能分析器")
    print("=" * 40)
    
    agent = AIAgentAnalyzer(workspace_root=".")
    
    while True:
        try:
            prompt = input("\\n💬 请输入分析需求 (输入'quit'退出): ").strip()
            
            if prompt.lower() in ['quit', 'exit', '退出']:
                print("👋 再见!")
                break
            
            if not prompt:
                continue
            
            print("\\n🔄 开始分析...")
            results = agent.analyze_from_prompt(prompt)
            
            if 'error' not in results:
                print("✅ 分析完成!")
                if 'request' in results:
                    output_dir = results['request'].get('output_dir', 'N/A')
                    print(f"📁 结果目录: {output_dir}")
            else:
                print(f"❌ 分析失败: {results['error']}")
                
        except KeyboardInterrupt:
            print("\\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 意外错误: {e}")

if __name__ == "__main__":
    main()
""")
        print(f"  ✅ {start_script}")
    
    # 创建模型配置示例
    model_config = Path("model_config_example.py")
    if not model_config.exists():
        model_config.write_text("""# 模型配置示例

# 常见的提示词示例
EXAMPLE_PROMPTS = [
    # 基础分析
    "分析 llama-7b 模型，batch_size=8",
    "对 qwen-14b 进行 nsys 全局性能分析",
    "综合分析 chatglm-6b 的性能瓶颈",
    
    # 自定义参数
    "分析 baichuan-13b，batch_size=1,4,8，input_len=512,1024",
    "对 vicuna-7b 进行 ncu kernel深度分析，temperature=0.1",
    
    # 英文提示词
    "analyze llama-7b with batch_size=16, input_len=1024",
    "ncu analysis for qwen-14b model with tp_size=2"
]

# 模型路径映射 (如果使用本地模型)
MODEL_PATHS = {
    "llama-7b": "workspace/models/llama-7b",
    "qwen-14b": "workspace/models/qwen-14b", 
    "chatglm-6b": "workspace/models/chatglm-6b"
}

# 常用配置
DEFAULT_CONFIGS = {
    "small_model": {
        "batch_size": [1, 4, 8],
        "input_len": [256, 512],
        "output_len": [32, 64]
    },
    "large_model": {
        "batch_size": [1, 2, 4], 
        "input_len": [512, 1024],
        "output_len": [64, 128]
    }
}
""")
        print(f"  ✅ {model_config}")

def main():
    """主函数"""
    print("🚀 AI Agent环境设置向导")
    print("=" * 50)
    
    success = True
    
    # 1. 创建目录结构
    create_directory_structure()
    
    # 2. 检查环境
    if not check_environment():
        print("\n⚠️  发现环境问题，尝试修复...")
        success = install_python_requirements() and success
    
    # 3. 检查GPU环境
    gpu_ok = check_gpu_environment()
    if not gpu_ok:
        print("  ⚠️  GPU环境可能有问题，但不影响基础功能")
    
    # 4. 设置分析工具
    tools_ok = setup_analysis_tools()
    success = tools_ok and success
    
    # 5. 创建示例文件
    create_example_config()
    
    print("\n" + "=" * 50)
    
    if success:
        print("✅ 环境设置完成!")
        print("\n📋 后续步骤:")
        print("1. 将模型文件放入 workspace/models/ 目录")
        print("2. 确保 TOOLS/Auto_Anlyze_tool/ 包含分析脚本")
        print("3. 运行: python start_agent.py")
        print("4. 或使用: python ai_agent_analyzer.py interactive")
        
        print("\n💡 使用示例:")
        print('python ai_agent_analyzer.py prompt "分析 llama-7b，batch_size=8"')
        
    else:
        print("❌ 环境设置遇到问题!")
        print("\n🔧 手动检查:")
        print("1. 安装NVIDIA Nsight Systems和Compute")
        print("2. 运行: pip install -r requirements_agent.txt") 
        print("3. 确保分析工具脚本存在")
    
    print(f"\n📁 工作目录: {Path.cwd()}")

if __name__ == "__main__":
    main()

