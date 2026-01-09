#!/usr/bin/env python3
"""
AI Agent Web服务器简化部署脚本

用于在现有容器中快速部署AI Agent Web服务
"""

import os
import sys
import subprocess
from pathlib import Path
import json

def check_environment():
    """检查部署环境"""
    print("🔍 检查部署环境...")
    
    issues = []
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version < (3, 8):
        issues.append(f"Python版本过低: {python_version.major}.{python_version.minor}, 需要3.8+")
    else:
        print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要文件
    required_files = [
        "web_agent_backend.py",
        "ai_agent_analyzer.py", 
        "static/chat.html",
        "requirements_web.txt"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            issues.append(f"缺少文件: {file_path}")
    
    # 检查可选工具
    tools_dir = Path("TOOLS/Auto_Anlyze_tool")
    if tools_dir.exists():
        print(f"✅ 分析工具目录存在")
    else:
        print(f"⚠️  分析工具目录不存在，将使用模拟模式")
    
    return issues

def install_dependencies():
    """安装Python依赖"""
    print("\n📦 安装Python依赖...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements_web.txt"
        ], check=True)
        print("✅ 依赖安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False

def setup_directories():
    """创建必要的目录结构"""
    print("\n📁 设置目录结构...")
    
    directories = [
        "workspace/models",
        "temp_uploads", 
        "analysis_results",
        "static",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path}")

def create_config():
    """创建运行时配置"""
    print("\n⚙️ 创建配置文件...")
    
    config = {
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "log_level": "info"
        },
        "features": {
            "file_upload": True,
            "websocket": True,
            "analysis_tools": True
        },
        "limits": {
            "max_file_size_mb": 10,
            "max_concurrent_sessions": 50,
            "analysis_timeout_seconds": 600
        }
    }
    
    config_path = Path("web_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 配置文件: {config_path}")

def create_startup_script():
    """创建启动脚本"""
    print("\n🚀 创建启动脚本...")
    
    startup_content = '''#!/bin/bash

# AI Agent Web服务启动脚本

# 设置环境变量
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 检查端口
PORT=${PORT:-8000}
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "端口 $PORT 已被占用"
    exit 1
fi

echo "🚀 启动 AI Agent Web服务..."
echo "📱 访问地址: http://localhost:$PORT/chat"

# 启动服务
python3 web_agent_backend.py
'''
    
    startup_path = Path("start_web_agent.sh")
    startup_path.write_text(startup_content)
    startup_path.chmod(0o755)
    
    print(f"✅ 启动脚本: {startup_path}")

def run_health_check():
    """运行功能检查"""
    print("\n🔧 功能检查...")
    
    try:
        # 导入主模块检查
        sys.path.insert(0, '.')
        
        print("  检查后端模块...")
        import web_agent_backend
        print("  ✅ 后端模块导入成功")
        
        print("  检查AI Agent...")
        import ai_agent_analyzer
        print("  ✅ AI Agent模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ 模块导入失败: {e}")
        return False

def main():
    """主部署流程"""
    print("🤖 AI Agent Web服务部署向导")
    print("=" * 50)
    
    # 1. 环境检查
    issues = check_environment()
    if issues:
        print("\n❌ 发现环境问题:")
        for issue in issues:
            print(f"  • {issue}")
        
        choice = input("\n是否继续部署? (y/n): ").lower()
        if choice != 'y':
            print("部署已取消")
            return False
    
    # 2. 安装依赖
    if not install_dependencies():
        return False
    
    # 3. 设置目录
    setup_directories()
    
    # 4. 创建配置
    create_config()
    
    # 5. 创建启动脚本
    create_startup_script()
    
    # 6. 功能检查
    health_ok = run_health_check()
    
    print("\n" + "=" * 50)
    
    if health_ok:
        print("✅ 部署完成!")
        print("\n📋 启动服务:")
        print("  ./start_web_agent.sh")
        print("  或: python3 web_agent_backend.py")
        
        print("\n🌐 访问地址:")
        print("  聊天界面: http://localhost:8000/chat")
        print("  API文档: http://localhost:8000/docs")
        print("  健康检查: http://localhost:8000/health")
        
        print("\n💡 使用提示:")
        print("  1. 打开聊天界面开始对话")
        print("  2. 上传JSON/YAML配置文件获取建议") 
        print("  3. 使用自然语言描述分析需求")
        
    else:
        print("⚠️ 部署完成，但可能存在功能问题")
        print("请检查依赖安装和模块导入")
    
    return True

if __name__ == "__main__":
    main()


