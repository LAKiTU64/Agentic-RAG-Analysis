#!/usr/bin/env python3
"""
Nsys 解析工具使用示例

演示如何使用 nsys_parser.py 来分析 NVIDIA Nsight Systems 输出文件
"""

from nsys_parser import NsysParser, NsysAnalyzer, NsysVisualizer, NsysReporter
import os
import sys

def example_basic_usage():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 假设您有一个 nsys 输出文件
    input_file = "profile_output.nsys-rep"  # 或 .sqlite, .csv, .json
    
    if not os.path.exists(input_file):
        print(f"示例文件 {input_file} 不存在")
        print("请使用以下命令生成 nsys 文件:")
        print("nsys profile -o profile_output your_cuda_program")
        return
    
    try:
        # 1. 创建解析器并解析文件
        parser = NsysParser(input_file)
        parser.parse()
        
        # 2. 创建分析器并分析数据
        analyzer = NsysAnalyzer(parser)
        stats = analyzer.analyze()
        
        # 3. 打印基本统计信息
        print(f"解析到 {len(parser.kernels)} 个CUDA kernels")
        print(f"解析到 {len(parser.memory_transfers)} 个内存传输")
        
        # 4. 生成可视化（可选）
        visualizer = NsysVisualizer(parser, analyzer)
        visualizer.create_visualizations()
        
        # 5. 生成报告（可选）
        reporter = NsysReporter(parser, analyzer)
        reporter.generate_report()
        
        print("✅ 分析完成! 结果保存在 nsys_analysis_output/ 目录")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")

def example_programmatic_analysis():
    """程序化分析示例"""
    print("\n=== 程序化分析示例 ===")
    
    # 假设您已经有了解析后的数据
    input_file = "profile_output.sqlite"
    
    if not os.path.exists(input_file):
        print(f"示例文件 {input_file} 不存在")
        return
    
    try:
        # 解析数据
        parser = NsysParser(input_file)
        parser.parse()
        
        # 手动分析特定 kernels
        print("\n🔥 Kernel 分析:")
        kernel_times = {}
        for kernel in parser.kernels:
            if kernel.name not in kernel_times:
                kernel_times[kernel.name] = []
            kernel_times[kernel.name].append(kernel.duration * 1000)  # ms
        
        # 找出最耗时的 kernels
        avg_times = {name: sum(times)/len(times) for name, times in kernel_times.items()}
        top_kernels = sorted(avg_times.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for i, (name, avg_time) in enumerate(top_kernels, 1):
            print(f"  {i}. {name[:50]}... : {avg_time:.3f} ms (平均)")
        
        # 分析内存传输
        if parser.memory_transfers:
            print("\n💾 内存传输分析:")
            total_h2d = sum(m.size for m in parser.memory_transfers if m.kind == "H2D")
            total_d2h = sum(m.size for m in parser.memory_transfers if m.kind == "D2H")
            
            print(f"  Host->Device: {total_h2d / (1024*1024):.2f} MB")
            print(f"  Device->Host: {total_d2h / (1024*1024):.2f} MB")
            
            avg_bandwidth = sum(m.bandwidth for m in parser.memory_transfers if m.bandwidth) / len(parser.memory_transfers)
            print(f"  平均带宽: {avg_bandwidth:.2f} GB/s")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")

def example_generate_nsys_profile():
    """示例：如何生成 nsys profile 文件"""
    print("\n=== 如何生成 nsys profile 文件 ===")
    
    print("""
1. 对于普通CUDA程序:
   nsys profile -o my_profile ./your_cuda_program

2. 对于Python程序 (如PyTorch):
   nsys profile -o torch_profile python train.py

3. 对于SGLang服务:
   nsys profile -o sglang_profile python -m sglang.launch_server ...

4. 高级选项 (收集更多信息):
   nsys profile -o detailed_profile -t cuda,nvtx,osrt,cudnn,cublas ./program

生成的 .nsys-rep 文件可以直接用本工具分析:
   python nsys_parser.py my_profile.nsys-rep
""")

def example_batch_analysis():
    """批量分析示例"""
    print("\n=== 批量分析示例 ===")
    
    # 分析目录中的所有 nsys 文件
    profile_dir = "profiles/"
    if not os.path.exists(profile_dir):
        print(f"创建示例目录: {profile_dir}")
        os.makedirs(profile_dir, exist_ok=True)
        print("请将 .nsys-rep 或 .sqlite 文件放入此目录")
        return
    
    nsys_files = [f for f in os.listdir(profile_dir) 
                  if f.endswith(('.nsys-rep', '.sqlite', '.db'))]
    
    if not nsys_files:
        print("未找到 nsys 文件")
        return
    
    print(f"发现 {len(nsys_files)} 个文件:")
    
    batch_results = {}
    
    for filename in nsys_files:
        filepath = os.path.join(profile_dir, filename)
        print(f"\n处理: {filename}")
        
        try:
            parser = NsysParser(filepath)
            parser.parse()
            
            analyzer = NsysAnalyzer(parser)
            stats = analyzer.analyze()
            
            # 保存关键指标
            batch_results[filename] = {
                'kernels': len(parser.kernels),
                'memory_transfers': len(parser.memory_transfers),
                'total_kernel_time': stats['kernel_analysis'].get('total_kernel_time', 0)
            }
            
            print(f"  ✅ {len(parser.kernels)} kernels, {len(parser.memory_transfers)} 内存传输")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            batch_results[filename] = {'error': str(e)}
    
    # 汇总结果
    print(f"\n📊 批量分析汇总:")
    print(f"{'文件名':<30} {'Kernels':<10} {'内存传输':<10} {'总时间(ms)':<15}")
    print("-" * 70)
    
    for filename, result in batch_results.items():
        if 'error' not in result:
            print(f"{filename:<30} {result['kernels']:<10} {result['memory_transfers']:<10} {result['total_kernel_time']:<15.2f}")
        else:
            print(f"{filename:<30} {'错误':<10} {'-':<10} {'-':<15}")

def main():
    """主函数 - 运行所有示例"""
    print("🚀 Nsys 解析工具使用示例")
    print("=" * 50)
    
    # 检查依赖
    try:
        import pandas
        import matplotlib
        import seaborn
        print("✅ 所有依赖已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return
    
    # 运行示例
    example_generate_nsys_profile()
    example_basic_usage()
    example_programmatic_analysis()
    example_batch_analysis()
    
    print(f"\n🎉 示例完成!")
    print(f"💡 提示: 使用 'python nsys_parser.py your_file.nsys-rep' 直接分析文件")

if __name__ == "__main__":
    main()


