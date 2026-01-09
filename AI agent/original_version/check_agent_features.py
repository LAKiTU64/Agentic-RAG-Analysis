#!/usr/bin/env python3
"""
AI Agent功能检查脚本

检查AI Agent的各项功能是否符合要求
"""

import json
from pathlib import Path
import re

def check_prompt_parsing():
    """检查提示词解析功能"""
    print("🔍 检查提示词解析功能...")
    
    test_prompts = [
        "分析 llama-7b 模型，batch_size=8",
        "对 qwen-14b 进行 nsys 全局性能分析",
        "综合分析 chatglm-6b 的性能瓶颈，batch_size=1,4,8",
        "使用 ncu 深度分析 kernel 性能，input_len=512,1024",
        "analyze meta-llama/Llama-2-7b-hf with batch_size=16"
    ]
    
    try:
        from ai_agent_analyzer import PromptParser
        parser = PromptParser()
        
        results = []
        for prompt in test_prompts:
            try:
                request = parser.parse_prompt(prompt)
                results.append({
                    "prompt": prompt,
                    "parsed": True,
                    "model": request.model_name,
                    "analysis_type": request.analysis_type,
                    "batch_size": request.batch_size
                })
            except Exception as e:
                results.append({
                    "prompt": prompt,
                    "parsed": False,
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r["parsed"])
        print(f"  ✅ 提示词解析: {success_count}/{len(test_prompts)} 成功")
        
        return results
        
    except ImportError:
        print("  ❌ 无法导入提示词解析器")
        return []

def check_config_parsing():
    """检查配置文件解析功能"""
    print("\n📄 检查配置文件解析功能...")
    
    # 示例JSON配置
    json_config = {
        "model_name": "llama-7b",
        "model_path": "/path/to/model",
        "batch_size": [1, 4, 8],
        "input_len": [512, 1024],
        "output_len": [64, 128],
        "analysis_type": "auto",
        "gpu_type": "A100",
        "precision": "fp16"
    }
    
    # 示例YAML配置
    yaml_config = """
model_name: qwen-14b
model_path: /path/to/qwen
batch_size: [1, 2, 4]
input_len: 1024
output_len: 64
analysis_type: ncu
gpu_type: H100
memory_gb: 80
"""
    
    try:
        from web_agent_backend import ConfigFileParser
        parser = ConfigFileParser()
        
        # 测试JSON解析
        json_result = parser.parse_json_config(json.dumps(json_config))
        print(f"  ✅ JSON解析成功，建议数: {len(json_result.get('suggestions', []))}")
        
        # 测试YAML解析
        yaml_result = parser.parse_yaml_config(yaml_config)
        print(f"  ✅ YAML解析成功，建议数: {len(yaml_result.get('suggestions', []))}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 配置文件解析失败: {e}")
        return False

def check_web_backend():
    """检查Web后端功能"""
    print("\n🌐 检查Web后端功能...")
    
    try:
        from web_agent_backend import app
        
        # 检查路由
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/upload_config", "/generate_command", "/health", "/ws/{session_id}"]
        
        found_routes = []
        for expected in expected_routes:
            if any(expected.replace("{session_id}", "") in route for route in routes):
                found_routes.append(expected)
        
        print(f"  ✅ 路由检查: {len(found_routes)}/{len(expected_routes)} 找到")
        
        # 检查WebSocket支持
        websocket_routes = [route for route in routes if "ws" in route.lower()]
        if websocket_routes:
            print("  ✅ WebSocket支持: 已启用")
        else:
            print("  ⚠️ WebSocket支持: 未检测到")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Web后端检查失败: {e}")
        return False

def check_file_structure():
    """检查文件结构完整性"""
    print("\n📁 检查文件结构...")
    
    required_files = {
        "web_agent_backend.py": "Web后端服务器",
        "ai_agent_analyzer.py": "AI Agent核心",
        "static/chat.html": "聊天界面",
        "requirements_web.txt": "Web依赖",
        "agent_config.yaml": "配置文件"
    }
    
    optional_files = {
        "TOOLS/Auto_Anlyze_tool/nsys_parser.py": "NSys解析器",
        "TOOLS/Auto_Anlyze_tool/ncu_parser.py": "NCU解析器",
        "SGlang/python/sglang/bench_one_batch_server.py": "SGlang基准脚本"
    }
    
    # 检查必需文件
    missing_required = []
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"  ✅ {description}: {file_path}")
        else:
            print(f"  ❌ {description}: {file_path} (缺失)")
            missing_required.append(file_path)
    
    # 检查可选文件
    for file_path, description in optional_files.items():
        if Path(file_path).exists():
            print(f"  ✅ {description}: {file_path}")
        else:
            print(f"  ⚠️ {description}: {file_path} (可选)")
    
    return len(missing_required) == 0

def check_ai_agent_features():
    """检查AI Agent核心功能"""
    print("\n🤖 检查AI Agent核心功能...")
    
    features_check = {
        "提示词解析": False,
        "参数配置": False, 
        "多种分析模式": False,
        "文件上传支持": False,
        "实时进度推送": False
    }
    
    try:
        # 检查AI Agent类
        from ai_agent_analyzer import AIAgentAnalyzer, PromptParser, ConfigGenerator
        
        # 基础功能检查
        features_check["提示词解析"] = hasattr(PromptParser, 'parse_prompt')
        features_check["参数配置"] = hasattr(ConfigGenerator, 'generate_sglang_config')
        
        # 检查分析模式
        analyzer = AIAgentAnalyzer()
        if hasattr(analyzer, 'analyze_from_prompt'):
            features_check["多种分析模式"] = True
        
        print("  AI Agent功能:")
        for feature, available in features_check.items():
            status = "✅" if available else "❌"
            print(f"    {status} {feature}")
        
    except Exception as e:
        print(f"  ❌ AI Agent检查失败: {e}")
    
    try:
        # 检查Web功能
        from web_agent_backend import ConfigFileParser, ConnectionManager
        
        features_check["文件上传支持"] = hasattr(ConfigFileParser, 'parse_json_config')
        features_check["实时进度推送"] = hasattr(ConnectionManager, 'send_message')
        
        print("  Web功能:")
        for feature in ["文件上传支持", "实时进度推送"]:
            status = "✅" if features_check[feature] else "❌"
            print(f"    {status} {feature}")
            
    except Exception as e:
        print(f"  ❌ Web功能检查失败: {e}")
    
    return sum(features_check.values()) / len(features_check)

def generate_feature_summary():
    """生成功能摘要"""
    print("\n📊 AI Agent功能摘要:")
    
    features = {
        "🔤 智能提示词解析": [
            "• 自动识别模型名称 (llama-7b, qwen-14b等)",
            "• 提取分析参数 (batch_size, input_len等)",  
            "• 支持中英文提示词",
            "• 智能默认参数填充"
        ],
        "📁 配置文件支持": [
            "• JSON/YAML格式解析",
            "• 模型信息自动提取",
            "• 基于配置的智能建议",
            "• 拖拽上传界面"
        ],
        "🔬 多种分析模式": [
            "• NSys全局性能分析",
            "• NCU深度kernel分析", 
            "• 集成分析(自动热点识别)",
            "• 自定义分析参数"
        ],
        "💻 Web界面": [
            "• 类ChatGPT对话界面",
            "• WebSocket实时通信",
            "• 进度条和状态显示",
            "• 响应式设计"
        ],
        "⚙️ 自动化配置": [
            "• SGlang脚本参数配置",
            "• 模型路径自动解析",
            "• 性能分析命令生成",
            "• 结果文件组织"
        ]
    }
    
    for category, feature_list in features.items():
        print(f"\n{category}:")
        for feature in feature_list:
            print(f"  {feature}")

def main():
    """主检查流程"""
    print("🔍 AI Agent功能检查")
    print("=" * 60)
    
    # 文件结构检查
    structure_ok = check_file_structure()
    
    # 提示词解析检查
    prompt_results = check_prompt_parsing()
    
    # 配置文件解析检查  
    config_ok = check_config_parsing()
    
    # Web后端检查
    web_ok = check_web_backend()
    
    # AI Agent功能检查
    agent_score = check_ai_agent_features()
    
    # 生成功能摘要
    generate_feature_summary()
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 检查结果总结:")
    
    print(f"  📁 文件结构: {'✅ 完整' if structure_ok else '❌ 缺失文件'}")
    print(f"  🔤 提示词解析: {'✅ 正常' if prompt_results else '❌ 异常'}")
    print(f"  📄 配置文件解析: {'✅ 正常' if config_ok else '❌ 异常'}")
    print(f"  🌐 Web后端: {'✅ 正常' if web_ok else '❌ 异常'}")
    print(f"  🤖 AI Agent功能: {agent_score*100:.0f}% 完整")
    
    overall_score = (
        structure_ok * 0.2 + 
        bool(prompt_results) * 0.2 + 
        config_ok * 0.2 + 
        web_ok * 0.2 + 
        agent_score * 0.2
    )
    
    print(f"\n🎯 总体功能完整度: {overall_score*100:.0f}%")
    
    if overall_score >= 0.8:
        print("✅ AI Agent功能完备，可以正常使用!")
    elif overall_score >= 0.6:
        print("⚠️ AI Agent基本功能可用，建议修复部分问题")
    else:
        print("❌ AI Agent存在较多问题，建议检查部署")
    
    return overall_score >= 0.6

if __name__ == "__main__":
    main()


