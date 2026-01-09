#!/usr/bin/env python3
"""
NCU Parser 使用示例

演示如何使用 ncu_parser.py 分析 NVIDIA Nsight Compute 输出文件
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from ncu_parser import NCUParser, NCUAnalyzer, NCUVisualizer, NCUReporter

def example_basic_usage():
    """基础使用示例"""
    print("="*60)
    print("NCU Parser 基础使用示例")
    print("="*60)
    
    # 示例输入文件路径 (请根据实际情况修改)
    input_files = [
        "sample_profile.ncu-rep",  # NCU report 文件
        "kernel_metrics.csv",     # CSV 导出文件
        "profile_data.json"       # JSON 导出文件
    ]
    
    for input_file in input_files:
        if Path(input_file).exists():
            print(f"\n📁 处理文件: {input_file}")
            
            try:
                # 1. 解析NCU文件
                parser = NCUParser(input_file)
                parser.parse()
                
                # 2. 分析性能数据
                analyzer = NCUAnalyzer(parser)
                stats = analyzer.analyze()
                
                # 3. 生成可视化
                visualizer = NCUVisualizer(parser, analyzer)
                visualizer.create_visualizations()
                
                # 4. 生成分析报告
                reporter = NCUReporter(parser, analyzer)
                reporter.generate_report()
                
                print(f"✅ 分析完成! 解析了 {len(parser.kernels)} 个kernels")
                
            except Exception as e:
                print(f"❌ 处理失败: {e}")
        else:
            print(f"⚠️  文件不存在: {input_file}")

def example_advanced_analysis():
    """高级分析示例"""
    print("\n" + "="*60)
    print("NCU Parser 高级分析示例")
    print("="*60)
    
    # 假设有一个示例文件
    input_file = "advanced_profile.ncu-rep"
    
    if not Path(input_file).exists():
        print(f"⚠️  示例文件不存在: {input_file}")
        print("创建模拟数据进行演示...")
        
        # 创建模拟数据
        create_sample_data()
        input_file = "sample_data.json"
    
    try:
        # 解析和分析
        parser = NCUParser(input_file)
        parser.parse()
        
        analyzer = NCUAnalyzer(parser)
        stats = analyzer.analyze()
        
        # 详细分析结果
        print(f"\n📊 分析结果摘要:")
        print(f"• 总kernel数: {len(parser.kernels)}")
        
        # GPU利用率分析
        gpu_stats = stats.get('gpu_utilization', {})
        if 'average_sm_efficiency' in gpu_stats:
            print(f"• 平均SM效率: {gpu_stats['average_sm_efficiency']:.1f}%")
            print(f"• 低效率kernel数: {gpu_stats.get('kernels_below_50_percent', 0)}")
        
        # 内存性能分析
        memory_stats = stats.get('memory_analysis', {})
        if 'bandwidth_stats' in memory_stats:
            bandwidth = memory_stats['bandwidth_stats']
            print(f"• 平均DRAM带宽: {bandwidth.get('average_bandwidth', 0):.1f} GB/s")
        
        # 瓶颈分析
        bottleneck_stats = stats.get('bottleneck_analysis', {})
        print(f"• 识别瓶颈数: {bottleneck_stats.get('total_bottlenecks', 0)}")
        
        if 'top_issues' in bottleneck_stats:
            print(f"\n🚫 主要性能问题:")
            for i, issue in enumerate(bottleneck_stats['top_issues'][:3], 1):
                print(f"  {i}. {issue['description']} ({issue['severity']})")
        
        # 生成报告和可视化
        visualizer = NCUVisualizer(parser, analyzer)
        visualizer.create_visualizations()
        
        reporter = NCUReporter(parser, analyzer)
        reporter.generate_report()
        
        print(f"\n✅ 高级分析完成!")
        
    except Exception as e:
        print(f"❌ 高级分析失败: {e}")

def create_sample_data():
    """创建示例数据用于演示"""
    import json
    
    sample_data = [
        {
            "name": "sample_kernel_1",
            "smEfficiency": 85.2,
            "achievedOccupancy": 75.6,
            "theoreticalOccupancy": 87.3,
            "dramBandwidth": 650.4,
            "l2HitRate": 82.1,
            "l1HitRate": 78.5,
            "tensorActive": 12.3,
            "warpExecutionEfficiency": 91.7,
            "duration": 2.34,
            "registersPerThread": 32
        },
        {
            "name": "sample_kernel_2", 
            "smEfficiency": 45.1,
            "achievedOccupancy": 34.2,
            "theoreticalOccupancy": 62.8,
            "dramBandwidth": 285.7,
            "l2HitRate": 45.3,
            "l1HitRate": 67.2,
            "tensorActive": 0.0,
            "warpExecutionEfficiency": 67.4,
            "duration": 5.67,
            "registersPerThread": 48
        },
        {
            "name": "sample_kernel_3",
            "smEfficiency": 72.8,
            "achievedOccupancy": 68.4,
            "theoreticalOccupancy": 75.2,
            "dramBandwidth": 478.3,
            "l2HitRate": 88.7,
            "l1HitRate": 92.1,
            "tensorActive": 85.6,
            "warpExecutionEfficiency": 89.3,
            "duration": 1.23,
            "registersPerThread": 24
        }
    ]
    
    with open("sample_data.json", "w", encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)
    
    print("✅ 创建示例数据: sample_data.json")

def example_custom_analysis():
    """自定义分析示例"""
    print("\n" + "="*60)
    print("NCU Parser 自定义分析示例")
    print("="*60)
    
    # 创建示例数据
    create_sample_data()
    
    try:
        parser = NCUParser("sample_data.json")
        parser.parse()
        
        print(f"\n🔍 自定义分析:")
        
        # 自定义指标分析
        high_efficiency_kernels = []
        low_efficiency_kernels = []
        tensor_core_kernels = []
        
        for kernel in parser.kernels:
            # 分类高效和低效kernel
            if kernel.sm_efficiency and kernel.sm_efficiency > 70:
                high_efficiency_kernels.append(kernel)
            elif kernel.sm_efficiency and kernel.sm_efficiency < 50:
                low_efficiency_kernels.append(kernel)
            
            # 使用Tensor Core的kernel
            if kernel.tensor_active and kernel.tensor_active > 10:
                tensor_core_kernels.append(kernel)
        
        print(f"• 高效率kernels (>70%): {len(high_efficiency_kernels)}")
        for k in high_efficiency_kernels:
            print(f"  - {k.name}: SM效率 {k.sm_efficiency:.1f}%")
        
        print(f"• 低效率kernels (<50%): {len(low_efficiency_kernels)}")
        for k in low_efficiency_kernels:
            print(f"  - {k.name}: SM效率 {k.sm_efficiency:.1f}%")
        
        print(f"• 使用Tensor Core的kernels: {len(tensor_core_kernels)}")
        for k in tensor_core_kernels:
            print(f"  - {k.name}: Tensor活跃度 {k.tensor_active:.1f}%")
        
        # 自定义优化建议
        print(f"\n💡 自定义优化建议:")
        
        if low_efficiency_kernels:
            print("• 对于低效率kernels:")
            print("  - 检查算法复杂度和工作负载分布")
            print("  - 考虑增加每个线程的计算量")
            print("  - 检查是否存在分支分歧")
        
        if not tensor_core_kernels:
            print("• 未检测到Tensor Core使用:")
            print("  - 考虑将适合的操作迁移到Tensor Core")
            print("  - 使用半精度或混合精度计算")
        
        # 内存优化建议
        low_bandwidth_kernels = [k for k in parser.kernels 
                               if k.dram_bandwidth and k.dram_bandwidth < 400]
        if low_bandwidth_kernels:
            print("• 对于低带宽利用率kernels:")
            print("  - 优化内存访问模式")
            print("  - 考虑使用共享内存")
            print("  - 检查内存合并访问")
        
    except Exception as e:
        print(f"❌ 自定义分析失败: {e}")

def cleanup_sample_files():
    """清理示例文件"""
    sample_files = ["sample_data.json"]
    
    for file in sample_files:
        if Path(file).exists():
            Path(file).unlink()
            print(f"🗑️  清理文件: {file}")

def main():
    """主函数 - 运行所有示例"""
    print("🚀 NCU Parser 使用示例")
    print("这个脚本演示了如何使用NCU分析工具")
    
    try:
        # 基础使用示例
        example_basic_usage()
        
        # 高级分析示例  
        example_advanced_analysis()
        
        # 自定义分析示例
        example_custom_analysis()
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断执行")
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
    finally:
        # 清理示例文件
        cleanup_sample_files()
        print(f"\n✅ 示例执行完成!")

if __name__ == "__main__":
    main()

