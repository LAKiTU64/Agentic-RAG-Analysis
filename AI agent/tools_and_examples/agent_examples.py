#!/usr/bin/env python3
"""
AI Agent性能分析器使用示例

展示如何使用AI Agent进行各种LLM性能分析
"""

from ai_agent_analyzer import AIAgentAnalyzer

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    agent = AIAgentAnalyzer(workspace_root=".")
    
    # 示例1: 简单的模型分析
    prompt1 = "分析 llama-7b 模型，使用 batch_size=8, input_len=512, output_len=64"
    results1 = agent.analyze_from_prompt(prompt1)
    
    print("示例1结果:")
    if 'error' not in results1:
        print(f"✅ 分析完成，输出目录: {results1.get('request', {}).get('output_dir', 'N/A')}")
    else:
        print(f"❌ 分析失败: {results1['error']}")

def example_different_analysis_types():
    """不同分析类型示例"""
    print("\n=== 不同分析类型示例 ===")
    
    agent = AIAgentAnalyzer()
    
    # nsys分析
    prompt_nsys = "对 qwen-14b 进行 nsys 全局性能分析，batch_size=16"
    print("执行nsys分析...")
    
    # ncu分析  
    prompt_ncu = "对 chatglm-6b 进行 ncu kernel深度分析，batch_size=4"
    print("执行ncu分析...")
    
    # 集成分析
    prompt_auto = "对 baichuan-13b 进行综合分析，batch_size=8,16"
    print("执行集成分析...")

def example_custom_parameters():
    """自定义参数示例"""
    print("\n=== 自定义参数示例 ===")
    
    agent = AIAgentAnalyzer()
    
    # 多种batch size和长度组合
    prompt = """
    分析模型 meta-llama/Llama-2-7b-hf，
    batch_size: 1,4,8,16，
    input_len: 256,512,1024，
    output_len: 32,64,128，
    temperature: 0.1，
    tp_size: 2，
    进行集成分析
    """
    
    results = agent.analyze_from_prompt(prompt)
    print(f"自定义参数分析结果: {'成功' if 'error' not in results else '失败'}")

def example_sglang_scripts():
    """不同SGLang脚本示例"""
    print("\n=== 不同SGLang脚本示例 ===")
    
    agent = AIAgentAnalyzer()
    
    # bench_one_batch_server
    prompt1 = "使用 bench_one_batch_server 脚本分析 vicuna-7b，batch_size=8"
    print("使用bench_one_batch_server脚本...")
    
    # launch_server
    prompt2 = "使用 launch_server 启动 llama-13b 服务器进行分析"
    print("使用launch_server脚本...")

def example_analyze_existing_files():
    """分析已有文件示例"""
    print("\n=== 分析已有文件示例 ===")
    
    agent = AIAgentAnalyzer()
    
    # 分析nsys文件
    nsys_file = "path/to/profile.nsys-rep"
    results1 = agent.analyze_existing_files(nsys_file, "nsys")
    print(f"nsys文件分析: {'成功' if 'error' not in results1 else '失败'}")
    
    # 分析ncu文件
    ncu_file = "path/to/kernel_profile.ncu-rep"
    results2 = agent.analyze_existing_files(ncu_file, "ncu")
    print(f"ncu文件分析: {'成功' if 'error' not in results2 else '失败'}")

def example_chinese_prompts():
    """中文提示词示例"""
    print("\n=== 中文提示词示例 ===")
    
    agent = AIAgentAnalyzer()
    
    prompts = [
        "分析 llama-7b 模型性能，批次大小8，输入长度512",
        "对 qwen-14b 进行深度kernel分析，使用ncu工具",
        "综合分析 chatglm-6b 的性能瓶颈，包括nsys和ncu",
        "启动 baichuan-13b 服务器并进行性能测试"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"中文提示词 {i}: {prompt}")
        # 这里只是展示，实际运行时取消注释
        # results = agent.analyze_from_prompt(prompt)

def example_interactive_mode():
    """交互模式使用示例"""
    print("\n=== 交互模式示例 ===")
    
    print("""
    启动交互模式:
    python ai_agent_analyzer.py interactive
    
    然后输入类似以下的提示词:
    - "分析 llama-7b，batch_size=8"
    - "对 qwen-14b 进行ncu分析" 
    - "综合分析 chatglm-6b 的性能"
    - "quit" (退出)
    """)

def example_command_line_usage():
    """命令行使用示例"""
    print("\n=== 命令行使用示例 ===")
    
    examples = [
        # 从提示词分析
        'python ai_agent_analyzer.py prompt "分析 llama-7b 模型，batch_size=8,16"',
        
        # 分析已有文件
        'python ai_agent_analyzer.py file profile.nsys-rep --analysis-type nsys',
        'python ai_agent_analyzer.py file kernel_profile.ncu-rep --analysis-type ncu',
        
        # 交互式模式
        'python ai_agent_analyzer.py interactive --workspace /path/to/workspace'
    ]
    
    print("命令行使用示例:")
    for i, cmd in enumerate(examples, 1):
        print(f"{i}. {cmd}")

def example_workspace_structure():
    """工作空间结构示例"""
    print("\n=== 推荐的工作空间结构 ===")
    
    print("""
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
    └── analysis_*/                     # 分析结果目录
        ├── nsys_analysis_output/
        ├── ncu_analysis_output/
        └── integrated_analysis/
    """)

def run_all_examples():
    """运行所有示例(演示用)"""
    print("🤖 AI Agent性能分析器使用示例")
    print("=" * 50)
    
    # 注意: 以下示例仅用于演示，实际运行需要真实的模型和环境
    
    example_basic_usage()
    example_different_analysis_types()  
    example_custom_parameters()
    example_sglang_scripts()
    example_analyze_existing_files()
    example_chinese_prompts()
    example_interactive_mode()
    example_command_line_usage()
    example_workspace_structure()
    
    print("\n🎉 所有示例展示完毕!")
    print("💡 提示: 在实际使用前，请确保:")
    print("   1. 已安装SGlang和相关依赖")
    print("   2. 已安装NVIDIA Nsight Systems和Compute")
    print("   3. 模型文件已下载到workspace/models/目录")
    print("   4. GPU环境配置正确")

if __name__ == "__main__":
    run_all_examples()

