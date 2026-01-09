#!/usr/bin/env python3
"""
集成分析工具使用示例
演示如何从nsys结果提取热点kernels，再用ncu深度分析

使用场景：
1. SGlang性能分析
2. PyTorch模型分析  
3. 自定义CUDA程序分析
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from nsys_to_ncu_analyzer import NSysToNCUAnalyzer, create_sglang_analysis_workflow

def example_sglang_analysis():
    """SGlang集成分析示例"""
    print("="*80)
    print("🚀 SGlang 集成性能分析示例")
    print("="*80)
    
    # 使用SGlang专用工作流
    sglang_workflow = create_sglang_analysis_workflow()
    
    try:
        # 运行分析
        result_dir = sglang_workflow(
            model_path="meta-llama/Meta-Llama-3-8B-Instruct",
            batch_size=8,
            input_len=512, 
            output_len=64
        )
        
        print(f"✅ SGlang分析完成，结果保存在: {result_dir}")
        
    except Exception as e:
        print(f"❌ SGlang分析失败: {e}")

def example_pytorch_analysis():
    """PyTorch模型分析示例"""
    print("\n" + "="*80)
    print("🔥 PyTorch 模型集成分析示例")
    print("="*80)
    
    # PyTorch训练脚本示例
    pytorch_cmd = [
        'python', '-c', '''
import torch
import torch.nn as nn

# 创建一个简单的模型
model = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 256), 
    nn.ReLU(),
    nn.Linear(256, 10)
).cuda()

# 模拟训练过程
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for i in range(10):
    x = torch.randn(32, 1024).cuda()  # batch_size=32
    y = torch.randint(0, 10, (32,)).cuda()
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    if i % 5 == 0:
        print(f"Step {i}, Loss: {loss.item():.4f}")

print("PyTorch training completed!")
        '''
    ]
    
    try:
        # 创建分析器
        analyzer = NSysToNCUAnalyzer("pytorch_analysis")
        
        # 步骤1: nsys全局分析
        print("🔍 步骤1: nsys全局分析...")
        nsys_file = analyzer.step1_nsys_analysis(pytorch_cmd, "pytorch_training")
        
        # 步骤2: 提取热点kernels
        print("🔥 步骤2: 提取热点kernels...")
        hot_kernels = analyzer.step2_extract_hot_kernels(nsys_file, top_k=6)
        
        if hot_kernels:
            # 步骤3: ncu深度分析（只分析前3个最重要的）
            print("⚡ 步骤3: ncu深度分析...")
            ncu_files = analyzer.step3_ncu_targeted_analysis(pytorch_cmd, hot_kernels, max_kernels=2)
            
            # 步骤4: 综合分析
            print("📊 步骤4: 综合分析...")
            results = analyzer.step4_comprehensive_analysis(ncu_files)
            
            # 生成报告
            report_file = analyzer.generate_final_report(results)
            print(f"✅ PyTorch分析完成，报告: {report_file}")
        else:
            print("⚠️  未发现热点kernels")
            
    except Exception as e:
        print(f"❌ PyTorch分析失败: {e}")

def example_custom_analysis():
    """自定义程序分析示例"""
    print("\n" + "="*80)
    print("🛠️ 自定义程序集成分析示例")
    print("="*80)
    
    # 这里可以替换为您的自定义CUDA程序
    custom_cmd = [
        'python', '-c', '''
import torch

print("Running custom CUDA operations...")

# 创建大矩阵进行计算
a = torch.randn(2048, 2048).cuda()
b = torch.randn(2048, 2048).cuda()

# 执行多种CUDA操作
for i in range(5):
    # 矩阵乘法
    c = torch.matmul(a, b)
    
    # 激活函数
    c = torch.relu(c)
    c = torch.sigmoid(c)
    
    # 归一化
    c = torch.layer_norm(c, c.shape[-1:])
    
    # 统计操作
    mean_val = torch.mean(c)
    max_val = torch.max(c)
    
    print(f"Iteration {i}: mean={mean_val:.4f}, max={max_val:.4f}")

print("Custom operations completed!")
        '''
    ]
    
    try:
        analyzer = NSysToNCUAnalyzer("custom_analysis")
        
        # 完整的四步分析流程
        nsys_file = analyzer.step1_nsys_analysis(custom_cmd, "custom_ops")
        hot_kernels = analyzer.step2_extract_hot_kernels(nsys_file, top_k=5, min_duration_ms=0.05)
        
        if hot_kernels:
            ncu_files = analyzer.step3_ncu_targeted_analysis(custom_cmd, hot_kernels, max_kernels=3)
            results = analyzer.step4_comprehensive_analysis(ncu_files)
            report_file = analyzer.generate_final_report(results)
            
            print(f"✅ 自定义程序分析完成")
            print(f"📁 结果目录: {analyzer.output_dir}")
            print(f"📄 分析报告: {report_file}")
        else:
            print("⚠️  未发现符合条件的热点kernels")
            
    except Exception as e:
        print(f"❌ 自定义程序分析失败: {e}")

def example_kernel_extraction_only():
    """仅提取热点kernel名称的示例"""
    print("\n" + "="*80)
    print("📋 仅提取热点Kernel名称示例")
    print("="*80)
    
    # 假设您已有一个nsys profile文件
    existing_nsys_file = "existing_profile.nsys-rep"
    
    # 检查文件是否存在
    if not Path(existing_nsys_file).exists():
        print(f"⚠️  示例文件不存在: {existing_nsys_file}")
        print("创建一个快速profile作为示例...")
        
        # 创建一个快速示例
        quick_cmd = ['python', '-c', 'import torch; a=torch.randn(100,100).cuda(); b=torch.matmul(a,a); print("Done")']
        
        analyzer = NSysToNCUAnalyzer("kernel_extraction_demo")
        nsys_file = analyzer.step1_nsys_analysis(quick_cmd, "quick_demo")
        existing_nsys_file = nsys_file
    
    try:
        # 只进行kernel提取，不运行ncu
        analyzer = NSysToNCUAnalyzer("kernel_extraction")
        hot_kernels = analyzer.step2_extract_hot_kernels(existing_nsys_file, top_k=10)
        
        print(f"\n🔥 提取的热点kernel名称列表:")
        print("-" * 100)
        
        for i, kernel in enumerate(hot_kernels, 1):
            print(f"{i:2d}. {kernel['name']}")
            print(f"    总时间: {kernel['total_time_ms']:8.3f} ms, "
                  f"调用次数: {kernel['count']:4d}, "
                  f"平均时间: {kernel['avg_time_ms']:6.3f} ms")
            print()
        
        # 生成ncu命令建议
        print("💡 建议的NCU分析命令:")
        print("-" * 60)
        
        for i, kernel in enumerate(hot_kernels[:3], 1):  # 只显示前3个
            safe_name = kernel['name'].replace(' ', '_').replace('(', '').replace(')', '')[:30]
            print(f"# 分析kernel {i}: {kernel['name'][:50]}...")
            print(f"ncu --kernel-name \"{kernel['name']}\" --set full -o hotspot_{i}_{safe_name} your_program")
            print()
            
    except Exception as e:
        print(f"❌ Kernel提取失败: {e}")

def main():
    """主函数 - 运行所有示例"""
    print("🎯 NVIDIA 集成性能分析工具 - 使用示例")
    print("这个工具展示了如何结合 nsys 和 ncu 进行高效的性能分析")
    
    examples = [
        ("SGlang分析", example_sglang_analysis),
        ("PyTorch分析", example_pytorch_analysis), 
        ("自定义程序分析", example_custom_analysis),
        ("仅提取Kernel名称", example_kernel_extraction_only)
    ]
    
    print(f"\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    try:
        choice = input(f"\n请选择要运行的示例 (1-{len(examples)}, 或按Enter运行所有): ").strip()
        
        if choice == "":
            # 运行所有示例
            for name, func in examples:
                print(f"\n{'='*20} 运行 {name} {'='*20}")
                try:
                    func()
                except KeyboardInterrupt:
                    print(f"\n⚠️  跳过 {name}")
                    continue
                except Exception as e:
                    print(f"❌ {name} 执行失败: {e}")
                    continue
        else:
            # 运行指定示例
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                name, func = examples[idx]
                print(f"\n运行示例: {name}")
                func()
            else:
                print("❌ 无效的选择")
                
    except KeyboardInterrupt:
        print("\n⚠️  用户中断执行")
    except ValueError:
        print("❌ 请输入有效数字")
    except Exception as e:
        print(f"❌ 执行失败: {e}")
    
    print(f"\n✅ 示例演示完成!")

if __name__ == "__main__":
    main()

