# AI Agent - Reorganized Architecture

This directory contains the reorganized AI Agent LLM Performance Analyzer with a clean English-named structure.

## Directory Structure

```
AI agent/
├── langchain_version/          # LangChain-powered version
│   ├── langchain_agent.py
│   ├── langchain_workflows.py
│   ├── web_langchain_backend.py
│   └── start_agent.py
├── original_version/           # Original implementation
│   ├── ai_agent_analyzer.py
│   ├── web_agent_backend.py
│   └── check_agent_features.py
├── web_interface/              # Web UI files
│   └── static/
│       └── chat.html
├── configs_and_docs/           # Configuration and documentation
│   ├── requirements_web.txt
│   ├── README_AI_Agent.md
│   ├── LANGCHAIN_INTEGRATION.md
│   ├── FEATURES_SUMMARY.md
│   └── DEPLOYMENT.md
└── tools_and_examples/         # Utilities and examples
    ├── example_model_config.json
    ├── agent_examples.py
    └── deploy_web_agent.py
```

## Quick Start

### LangChain Version (Recommended)
```bash
cd langchain_version
python web_langchain_backend.py
```

### Original Version
```bash
cd original_version  
python web_agent_backend.py
```

## Documentation

- See `configs_and_docs/README_AI_Agent.md` for complete usage guide
- See `configs_and_docs/LANGCHAIN_INTEGRATION.md` for LangChain features
- See `configs_and_docs/DEPLOYMENT.md` for deployment instructions

## Version Comparison

| Feature | Original | LangChain |
|---------|----------|-----------|
| Startup Speed | Fast | Normal |
| Intelligence | Basic | Advanced |
| Memory | None | Full |
| Tool Selection | Manual | Automatic |

---

Generated from: D:\Desk\Xiong\AI
