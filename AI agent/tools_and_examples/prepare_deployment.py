#!/usr/bin/env python3
"""
AI Agent部署文件准备脚本

自动整理和打包部署所需的文件
"""

import os
import shutil
import tarfile
from pathlib import Path
import json

def create_deployment_package():
    """创建部署包"""
    
    print("📦 准备AI Agent部署文件...")
    
    # 创建部署目录
    deploy_dir = Path("ai-agent-deploy")
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir()
    
    # 核心文件列表
    core_files = [
        "web_langchain_backend.py",
        "langchain_agent.py", 
        "ai_agent_analyzer.py",
        "requirements_web.txt",
        "Dockerfile.simple"
    ]
    
    # Web界面文件
    web_files = [
        "static/chat.html"
    ]
    
    # 配置文件
    config_files = [
        "agent_config.yaml",
        "example_model_config.json"
    ]
    
    # 可选工具文件
    tool_files = [
        "TOOLS/Auto_Anlyze_tool/nsys_parser.py",
        "TOOLS/Auto_Anlyze_tool/ncu_parser.py", 
        "TOOLS/Auto_Anlyze_tool/nsys_to_ncu_analyzer.py"
    ]
    
    # 文档文件
    doc_files = [
        "README_AI_Agent.md",
        "LANGCHAIN_INTEGRATION.md",
        "DOCKER_DEPLOYMENT_GUIDE.md"
    ]
    
    # 复制核心文件
    print("📋 复制核心文件...")
    for file_path in core_files:
        if Path(file_path).exists():
            shutil.copy2(file_path, deploy_dir / Path(file_path).name)
            print(f"  ✅ {file_path}")
        else:
            print(f"  ⚠️  {file_path} (不存在)")
    
    # 复制Web界面文件
    print("\n🌐 复制Web界面文件...")
    static_dir = deploy_dir / "static"
    static_dir.mkdir(exist_ok=True)
    for file_path in web_files:
        if Path(file_path).exists():
            shutil.copy2(file_path, static_dir / Path(file_path).name)
            print(f"  ✅ {file_path}")
        else:
            print(f"  ⚠️  {file_path} (不存在)")
    
    # 复制配置文件
    print("\n⚙️ 复制配置文件...")
    for file_path in config_files:
        if Path(file_path).exists():
            shutil.copy2(file_path, deploy_dir / Path(file_path).name)
            print(f"  ✅ {file_path}")
        else:
            print(f"  ⚠️  {file_path} (可选，不存在)")
    
    # 复制工具文件
    print("\n🛠️ 复制分析工具...")
    tools_dir = deploy_dir / "TOOLS" / "Auto_Anlyze_tool"
    tools_dir.mkdir(parents=True, exist_ok=True)
    for file_path in tool_files:
        if Path(file_path).exists():
            shutil.copy2(file_path, tools_dir / Path(file_path).name)
            print(f"  ✅ {file_path}")
        else:
            print(f"  ⚠️  {file_path} (可选，不存在)")
    
    # 复制文档文件
    print("\n📚 复制文档文件...")
    for file_path in doc_files:
        if Path(file_path).exists():
            shutil.copy2(file_path, deploy_dir / Path(file_path).name)
            print(f"  ✅ {file_path}")
        else:
            print(f"  ⚠️  {file_path} (可选，不存在)")
    
    # 创建启动脚本
    create_startup_scripts(deploy_dir)
    
    # 创建部署说明
    create_deploy_readme(deploy_dir)
    
    # 创建tar包
    create_tar_package(deploy_dir)
    
    print(f"\n🎉 部署文件准备完成!")
    print(f"📁 部署目录: {deploy_dir}")
    print(f"📦 压缩包: ai-agent-deploy.tar.gz")

def create_startup_scripts(deploy_dir):
    """创建启动脚本"""
    
    print("\n🚀 创建启动脚本...")
    
    # Docker构建脚本
    build_script = deploy_dir / "build.sh"
    build_script.write_text("""#!/bin/bash

# AI Agent Docker构建脚本

echo "🐳 构建AI Agent Docker镜像..."

# 检查Dockerfile
if [ ! -f "Dockerfile.simple" ]; then
    echo "❌ Dockerfile.simple不存在"
    exit 1
fi

# 构建镜像
docker build -f Dockerfile.simple -t ai-agent:latest .

if [ $? -eq 0 ]; then
    echo "✅ 镜像构建成功"
    echo "📋 下一步: 运行 ./run.sh 启动容器"
else
    echo "❌ 镜像构建失败"
    exit 1
fi
""")
    build_script.chmod(0o755)
    
    # Docker运行脚本
    run_script = deploy_dir / "run.sh"
    run_script.write_text("""#!/bin/bash

# AI Agent Docker运行脚本

echo "🚀 启动AI Agent容器..."

# 停止并删除已存在的容器
if docker ps -a --format 'table {{.Names}}' | grep -q "ai-agent-container"; then
    echo "🔄 停止旧容器..."
    docker stop ai-agent-container
    docker rm ai-agent-container
fi

# 创建必要的目录
mkdir -p workspace/models analysis_results logs temp_uploads

# 启动新容器
docker run -d \\
  --name ai-agent-container \\
  -p 8000:8000 \\
  -v $(pwd)/workspace:/app/workspace \\
  -v $(pwd)/analysis_results:/app/analysis_results \\
  -v $(pwd)/logs:/app/logs \\
  ai-agent:latest

if [ $? -eq 0 ]; then
    echo "✅ 容器启动成功"
    echo "🌐 访问地址: http://localhost:8000/chat"
    echo "📊 API文档: http://localhost:8000/docs"
    echo "📋 查看日志: docker logs -f ai-agent-container"
else
    echo "❌ 容器启动失败"
    exit 1
fi
""")
    run_script.chmod(0o755)
    
    # 管理脚本
    manage_script = deploy_dir / "manage.sh"
    manage_script.write_text("""#!/bin/bash

# AI Agent管理脚本

case "$1" in
    "logs")
        echo "📋 查看容器日志..."
        docker logs -f ai-agent-container
        ;;
    "status")
        echo "📊 查看容器状态..."
        docker ps | grep ai-agent-container
        docker stats --no-stream ai-agent-container
        ;;
    "stop")
        echo "🛑 停止容器..."
        docker stop ai-agent-container
        ;;
    "restart")
        echo "🔄 重启容器..."
        docker restart ai-agent-container
        ;;
    "shell")
        echo "💻 进入容器shell..."
        docker exec -it ai-agent-container /bin/bash
        ;;
    "update")
        echo "🔄 更新部署..."
        ./build.sh && ./run.sh
        ;;
    *)
        echo "AI Agent 管理脚本"
        echo "使用方法: $0 {logs|status|stop|restart|shell|update}"
        echo ""
        echo "命令说明:"
        echo "  logs    - 查看容器日志"  
        echo "  status  - 查看容器状态"
        echo "  stop    - 停止容器"
        echo "  restart - 重启容器" 
        echo "  shell   - 进入容器shell"
        echo "  update  - 更新部署"
        ;;
esac
""")
    manage_script.chmod(0o755)
    
    print("  ✅ build.sh - Docker构建脚本")
    print("  ✅ run.sh - Docker运行脚本")
    print("  ✅ manage.sh - 容器管理脚本")

def create_deploy_readme(deploy_dir):
    """创建部署说明文件"""
    
    readme_content = """# AI Agent Docker部署包

## 🚀 快速部署

### 1. 上传到服务器
```bash
# 解压部署包
tar -xzf ai-agent-deploy.tar.gz
cd ai-agent-deploy
```

### 2. 构建Docker镜像
```bash
chmod +x *.sh
./build.sh
```

### 3. 启动服务
```bash
./run.sh
```

### 4. 访问服务
- 聊天界面: http://your-server:8000/chat
- API文档: http://your-server:8000/docs

## 🔧 管理命令

```bash
./manage.sh status   # 查看状态
./manage.sh logs     # 查看日志
./manage.sh restart  # 重启服务
./manage.sh stop     # 停止服务
./manage.sh shell    # 进入容器
./manage.sh update   # 更新部署
```

## 📁 文件说明

- `web_langchain_backend.py` - 主程序(LangChain版)
- `static/chat.html` - Web聊天界面
- `Dockerfile.simple` - Docker配置
- `requirements_web.txt` - Python依赖
- `build.sh` - 构建脚本
- `run.sh` - 启动脚本
- `manage.sh` - 管理脚本

## 🐛 故障排除

1. 端口被占用: 修改run.sh中的端口映射
2. 内存不足: 检查服务器资源
3. 依赖安装失败: 检查网络连接

## 📞 技术支持

如遇问题请检查:
- Docker是否正常运行
- 端口8000是否可用
- 服务器内存是否充足
"""
    
    (deploy_dir / "README.md").write_text(readme_content)
    print("  ✅ README.md - 部署说明")

def create_tar_package(deploy_dir):
    """创建tar压缩包"""
    
    print(f"\n📦 创建压缩包...")
    
    tar_path = "ai-agent-deploy.tar.gz"
    
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(deploy_dir, arcname="ai-agent-deploy")
    
    print(f"  ✅ {tar_path}")
    
    # 显示包大小
    size_mb = Path(tar_path).stat().st_size / (1024 * 1024)
    print(f"  📊 大小: {size_mb:.1f} MB")

if __name__ == "__main__":
    create_deployment_package()


