#!/usr/bin/env python3
"""
AI Agent启动脚本

支持选择不同版本的AI Agent：
1. 原版AI Agent (基于原生实现)
2. LangChain版AI Agent (集成LangChain框架)
"""

import argparse
import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """检查依赖"""
    print("🔍 检查依赖...")
    
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI和Uvicorn已安装")
    except ImportError:
        print("❌ 缺少FastAPI或Uvicorn")
        return False
    
    return True

def check_langchain_dependencies():
    """检查LangChain依赖"""
    try:
        import langchain
        print("✅ LangChain已安装")
        return True
    except ImportError:
        print("⚠️ LangChain未安装，将使用原版AI Agent")
        return False

def start_original_agent(port=8000, host="0.0.0.0"):
    """启动原版AI Agent"""
    print("🚀 启动原版AI Agent...")
    
    cmd = [
        sys.executable, "web_agent_backend.py",
        "--host", host,
        "--port", str(port)
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 AI Agent已停止")

def start_langchain_agent(port=8000, host="0.0.0.0"):
    """启动LangChain版AI Agent"""
    print("🚀 启动LangChain版AI Agent...")
    
    cmd = [
        sys.executable, "web_langchain_backend.py",
        "--host", host, 
        "--port", str(port)
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 LangChain AI Agent已停止")

def interactive_mode():
    """交互式选择模式"""
    print("""
🤖 AI Agent LLM性能分析器启动向导
""" + "=" * 50)
    
    print("""
请选择AI Agent版本：

1️⃣ 原版AI Agent
   • 基于原生实现
   • 快速启动
   • 功能完整
   
2️⃣ LangChain版AI Agent (推荐)
   • 集成LangChain框架
   • 更智能的对话
   • 工具链自动调用
   • 记忆管理
   • 复杂工作流支持
""")
    
    while True:
        choice = input("请选择 (1/2): ").strip()
        
        if choice == "1":
            if check_dependencies():
                start_original_agent()
            break
        elif choice == "2":
            if check_dependencies() and check_langchain_dependencies():
                start_langchain_agent()
            elif check_dependencies():
                print("\n⚠️ LangChain未安装，是否安装? (y/n)")
                install_choice = input().strip().lower()
                if install_choice == 'y':
                    install_langchain()
                    start_langchain_agent()
                else:
                    start_original_agent()
            break
        else:
            print("请输入1或2")

def install_langchain():
    """安装LangChain依赖"""
    print("📦 正在安装LangChain...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "langchain>=0.0.350",
            "langchain-openai>=0.0.2", 
            "langchain-community>=0.0.10"
        ], check=True)
        print("✅ LangChain安装完成")
    except subprocess.CalledProcessError:
        print("❌ LangChain安装失败")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI Agent LLM性能分析器启动脚本')
    
    parser.add_argument('--version', choices=['original', 'langchain'], 
                       help='选择AI Agent版本')
    parser.add_argument('--port', type=int, default=8000, help='服务端口')
    parser.add_argument('--host', default='0.0.0.0', help='服务地址')
    parser.add_argument('--interactive', action='store_true', 
                       help='交互式选择模式')
    
    args = parser.parse_args()
    
    if args.interactive or not args.version:
        interactive_mode()
    else:
        if not check_dependencies():
            print("❌ 依赖检查失败，请安装必要的依赖")
            return
        
        if args.version == 'original':
            start_original_agent(args.port, args.host)
        elif args.version == 'langchain':
            if check_langchain_dependencies():
                start_langchain_agent(args.port, args.host)
            else:
                print("❌ LangChain未安装，请先安装或使用原版")

if __name__ == "__main__":
    main()


