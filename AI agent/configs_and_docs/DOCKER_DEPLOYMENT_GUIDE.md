# Docker容器部署指南 🐳

## 📁 文件组织结构

### 1. 在服务器上创建项目目录

```bash
# 在服务器上创建项目目录
mkdir -p /opt/ai-agent
cd /opt/ai-agent
```

### 2. 需要上传的核心文件

```
/opt/ai-agent/
├── 🔥 核心文件 (必需)
│   ├── web_langchain_backend.py    # LangChain Web后端 (主程序)
│   ├── langchain_agent.py          # LangChain Agent核心
│   ├── ai_agent_analyzer.py        # AI Agent分析器
│   ├── requirements_web.txt        # Python依赖
│   └── Dockerfile.simple           # Docker配置
│
├── 🌐 Web界面 (必需)
│   └── static/
│       └── chat.html              # 聊天界面
│
├── ⚙️ 配置文件 (推荐)
│   ├── agent_config.yaml          # 配置文件
│   └── example_model_config.json  # 配置示例
│
├── 🛠️ 可选工具 (如果有分析需求)
│   └── TOOLS/
│       └── Auto_Anlyze_tool/
│           ├── nsys_parser.py
│           ├── ncu_parser.py
│           └── nsys_to_ncu_analyzer.py
│
└── 📋 可选文档
    ├── README_AI_Agent.md
    └── LANGCHAIN_INTEGRATION.md
```

## 🚀 部署步骤

### 步骤1: 上传文件到服务器

```bash
# 方式1: 使用scp上传
scp -r ./ai-agent/ user@your-server:/opt/

# 方式2: 使用rsync同步
rsync -avz --progress ./ai-agent/ user@your-server:/opt/ai-agent/

# 方式3: 先打包再上传
tar -czf ai-agent.tar.gz ai-agent/
scp ai-agent.tar.gz user@your-server:/opt/
# 在服务器上解压
ssh user@your-server "cd /opt && tar -xzf ai-agent.tar.gz"
```

### 步骤2: 在服务器上构建Docker镜像

```bash
# SSH到服务器
ssh user@your-server

# 进入项目目录
cd /opt/ai-agent

# 构建Docker镜像
docker build -f Dockerfile.simple -t ai-agent:latest .
```

### 步骤3: 运行Docker容器

```bash
# 运行容器
docker run -d \
  --name ai-agent-container \
  -p 8000:8000 \
  -v /opt/ai-agent/workspace:/app/workspace \
  -v /opt/ai-agent/analysis_results:/app/analysis_results \
  ai-agent:latest

# 检查容器状态
docker ps
docker logs ai-agent-container
```

## 📦 最小化文件清单

如果只想要核心功能，最少需要这些文件：

```
ai-agent/
├── web_langchain_backend.py        # 主程序
├── langchain_agent.py              # LangChain核心
├── ai_agent_analyzer.py            # 分析器
├── requirements_web.txt            # 依赖
├── Dockerfile.simple               # Docker配置
└── static/chat.html                # Web界面
```

## 🔧 Docker命令参考

### 常用管理命令

```bash
# 查看日志
docker logs -f ai-agent-container

# 进入容器
docker exec -it ai-agent-container /bin/bash

# 重启容器
docker restart ai-agent-container

# 停止容器
docker stop ai-agent-container

# 删除容器
docker rm ai-agent-container

# 查看容器资源使用
docker stats ai-agent-container
```

### 更新部署

```bash
# 停止并删除旧容器
docker stop ai-agent-container
docker rm ai-agent-container

# 重新构建镜像
docker build -f Dockerfile.simple -t ai-agent:latest .

# 启动新容器
docker run -d --name ai-agent-container -p 8000:8000 ai-agent:latest
```

## 🌐 访问服务

部署完成后，通过以下方式访问：

```
# 聊天界面
http://your-server-ip:8000/chat

# API文档
http://your-server-ip:8000/docs

# 健康检查
http://your-server-ip:8000/health
```

## ⚙️ 环境变量配置

可以在运行容器时设置环境变量：

```bash
docker run -d \
  --name ai-agent-container \
  -p 8000:8000 \
  -e PORT=8000 \
  -e PYTHONPATH=/app \
  -e LOG_LEVEL=INFO \
  ai-agent:latest
```

## 🔒 安全建议

```bash
# 1. 限制容器资源
docker run -d \
  --name ai-agent-container \
  -p 8000:8000 \
  --memory=2g \
  --cpus=2 \
  ai-agent:latest

# 2. 使用非root用户运行 (在Dockerfile中已配置)

# 3. 只暴露必要端口
# 不要使用 -p 0.0.0.0:8000:8000，而是使用 -p 127.0.0.1:8000:8000

# 4. 定期更新镜像
docker pull python:3.9-slim
docker build --no-cache -f Dockerfile.simple -t ai-agent:latest .
```

## 🐛 故障排除

### 常见问题

1. **容器启动失败**
```bash
# 检查日志
docker logs ai-agent-container

# 检查镜像构建
docker build --no-cache -f Dockerfile.simple -t ai-agent:latest .
```

2. **端口被占用**
```bash
# 查看端口占用
netstat -tlnp | grep 8000

# 使用其他端口
docker run -d --name ai-agent-container -p 8080:8000 ai-agent:latest
```

3. **内存不足**
```bash
# 监控资源使用
docker stats ai-agent-container

# 增加swap或升级服务器配置
```

4. **依赖安装失败**
```bash
# 在Dockerfile中添加代理设置
ENV http_proxy=http://proxy.company.com:8080
ENV https_proxy=http://proxy.company.com:8080
```

## 📊 性能优化

```bash
# 1. 使用多阶段构建减少镜像大小
# 2. 配置合适的内存限制
# 3. 使用卷挂载持久化数据
docker run -d \
  --name ai-agent-container \
  -p 8000:8000 \
  -v /opt/ai-agent/data:/app/data \
  -v /opt/ai-agent/logs:/app/logs \
  ai-agent:latest
```

---

**🎯 按照以上步骤，您就可以在服务器的Docker容器中成功部署AI Agent！**
