#!/usr/bin/env python3
"""
NVIDIA 性能分析集成工具
先用 nsys 识别热点kernels，再用 ncu 深度分析

工作流程：
1. nsys profile -> 获取全局性能overview  
2. 提取热点kernel名称
3. ncu profile -> 针对热点kernels深度分析
4. 综合分析报告

作者: AI助手
版本: 1.0
"""

import os
import sys
import json
import subprocess
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# 导入我们的分析工具
sys.path.append(str(Path(__file__).parent))
from nsys_parser import NsysParser, NsysAnalyzer
from ncu_parser import NCUParser, NCUAnalyzer, NCUVisualizer, NCUReporter

class NSysToNCUAnalyzer:
    """集成 nsys 和 ncu 的分析工具"""
    
    def __init__(self, output_dir: str = "integrated_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.hot_kernels = []
        self.nsys_stats = {}
        self.ncu_results = {}
        
    def step1_nsys_analysis(self, target_command: List[str], profile_name: str = "overview") -> str:
        """第一步：使用nsys进行全局性能分析"""
        
        nsys_profile = self.output_dir / f"{profile_name}.nsys-rep"
        
        # 构建nsys命令
        nsys_cmd = [
            'nsys', 'profile',
            '-o', str(nsys_profile.with_suffix('')),  # nsys会自动添加.nsys-rep
            '-t', 'cuda,nvtx,osrt',
            '--cuda-memory-usage=true',
            '--force-overwrite=true'
        ] + target_command
        
        print("🚀 步骤1: 运行nsys全局性能分析...")
        print(f"命令: {' '.join(nsys_cmd)}")
        
        try:
            result = subprocess.run(nsys_cmd, capture_output=True, text=True, check=True)
            print(f"✅ nsys分析完成: {nsys_profile}")
            return str(nsys_profile)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ nsys分析失败: {e.stderr}")
            raise
    
    def step2_extract_hot_kernels(self, nsys_file: str, 
                                  top_k: int = 10, 
                                  min_duration_ms: float = 0.1) -> List[Dict]:
        """第二步：从nsys结果中提取热点kernels"""
        
        print("🔍 步骤2: 从nsys结果提取热点kernels...")
        
        # 使用我们的nsys解析器
        parser = NsysParser(nsys_file)
        parser.parse()
        
        if not parser.kernels:
            print("⚠️  未发现CUDA kernels")
            return []
        
        # 分析kernels
        analyzer = NsysAnalyzer(parser)
        self.nsys_stats = analyzer.analyze()
        
        # 提取热点kernels
        hot_kernels = []
        
        # 方法1: 按执行时间排序
        kernels_by_time = sorted(parser.kernels, 
                               key=lambda k: k.duration, reverse=True)
        
        # 方法2: 按调用次数统计
        kernel_stats = {}
        for kernel in parser.kernels:
            name = kernel.name
            if name not in kernel_stats:
                kernel_stats[name] = {
                    'count': 0,
                    'total_time': 0,
                    'max_time': 0,
                    'avg_time': 0
                }
            
            kernel_stats[name]['count'] += 1
            kernel_stats[name]['total_time'] += kernel.duration * 1000  # 转为ms
            kernel_stats[name]['max_time'] = max(kernel_stats[name]['max_time'], 
                                                kernel.duration * 1000)
        
        # 计算平均时间
        for name, stats in kernel_stats.items():
            stats['avg_time'] = stats['total_time'] / stats['count']
        
        # 按总执行时间排序，提取top-k
        sorted_kernels = sorted(kernel_stats.items(), 
                              key=lambda x: x[1]['total_time'], reverse=True)
        
        for kernel_name, stats in sorted_kernels[:top_k]:
            # 过滤掉执行时间太短的kernels
            if stats['avg_time'] >= min_duration_ms:
                hot_kernels.append({
                    'name': kernel_name,
                    'total_time_ms': stats['total_time'],
                    'avg_time_ms': stats['avg_time'],
                    'count': stats['count'],
                    'max_time_ms': stats['max_time']
                })
        
        self.hot_kernels = hot_kernels
        
        print(f"🔥 识别到 {len(hot_kernels)} 个热点kernels:")
        for i, kernel in enumerate(hot_kernels[:5], 1):
            print(f"  {i}. {kernel['name'][:60]}... "
                  f"(总计: {kernel['total_time_ms']:.2f}ms, "
                  f"平均: {kernel['avg_time_ms']:.3f}ms, "
                  f"调用: {kernel['count']}次)")
        
        # 保存热点kernels到文件
        hot_kernels_file = self.output_dir / "hot_kernels.json"
        with open(hot_kernels_file, 'w', encoding='utf-8') as f:
            json.dump(hot_kernels, f, indent=2, ensure_ascii=False)
        
        print(f"📋 热点kernels列表已保存: {hot_kernels_file}")
        return hot_kernels
    
    def step3_ncu_targeted_analysis(self, target_command: List[str], 
                                   kernels_to_analyze: List[Dict],
                                   max_kernels: int = 5) -> List[str]:
        """第三步：使用ncu对热点kernels进行深度分析"""
        
        print("⚡ 步骤3: 使用ncu深度分析热点kernels...")
        
        ncu_results = []
        
        # 限制分析的kernel数量（ncu分析很耗时）
        kernels_to_analyze = kernels_to_analyze[:max_kernels]
        
        for i, kernel_info in enumerate(kernels_to_analyze):
            kernel_name = kernel_info['name']
            
            # 清理kernel名称，用于文件名
            safe_name = re.sub(r'[^\w\-_]', '_', kernel_name)[:50]
            ncu_profile = self.output_dir / f"ncu_kernel_{i}_{safe_name}"
            
            # 构建ncu命令
            ncu_cmd = [
                'ncu',
                '--kernel-name', kernel_name,
                '--set', 'full',  # 收集完整指标集
                '-o', str(ncu_profile),
                '--force-overwrite'
            ] + target_command
            
            print(f"🎯 正在分析kernel {i+1}/{len(kernels_to_analyze)}: {kernel_name[:60]}...")
            
            try:
                # 运行ncu
                result = subprocess.run(ncu_cmd, capture_output=True, text=True, 
                                      timeout=300, check=True)  # 5分钟超时
                
                ncu_file = str(ncu_profile) + ".ncu-rep"
                if Path(ncu_file).exists():
                    print(f"✅ kernel分析完成: {ncu_file}")
                    ncu_results.append(ncu_file)
                    
                    # 立即导出为CSV以便分析
                    self._export_ncu_to_csv(ncu_file)
                else:
                    print(f"⚠️  NCU文件未生成: {ncu_file}")
                    
            except subprocess.TimeoutExpired:
                print(f"⏰ kernel分析超时: {kernel_name[:60]}")
            except subprocess.CalledProcessError as e:
                print(f"❌ kernel分析失败: {kernel_name[:60]} - {e.stderr}")
            except Exception as e:
                print(f"❌ 意外错误: {e}")
        
        return ncu_results
    
    def _export_ncu_to_csv(self, ncu_file: str) -> Optional[str]:
        """导出ncu结果为CSV格式"""
        csv_file = ncu_file.replace('.ncu-rep', '.csv')
        
        export_cmd = [
            'ncu', '--csv',
            '--log-file', csv_file,
            '--import', ncu_file
        ]
        
        try:
            subprocess.run(export_cmd, capture_output=True, text=True, check=True)
            if Path(csv_file).exists():
                return csv_file
        except:
            pass
        
        return None
    
    def step4_comprehensive_analysis(self, ncu_files: List[str]) -> Dict:
        """第四步：综合分析nsys和ncu结果"""
        
        print("📊 步骤4: 综合分析结果...")
        
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'nsys_overview': self.nsys_stats,
            'hot_kernels_count': len(self.hot_kernels),
            'ncu_detailed_analysis': {}
        }
        
        # 分析每个ncu结果
        for ncu_file in ncu_files:
            csv_file = ncu_file.replace('.ncu-rep', '.csv')
            
            if Path(csv_file).exists():
                try:
                    # 使用我们的ncu分析器
                    parser = NCUParser(csv_file)
                    parser.parse()
                    
                    analyzer = NCUAnalyzer(parser)
                    stats = analyzer.analyze()
                    
                    kernel_name = Path(ncu_file).stem
                    comprehensive_results['ncu_detailed_analysis'][kernel_name] = {
                        'kernels_analyzed': len(parser.kernels),
                        'bottlenecks_found': len(analyzer.bottlenecks),
                        'gpu_utilization': stats.get('gpu_utilization', {}),
                        'memory_analysis': stats.get('memory_analysis', {}),
                        'bottleneck_summary': [
                            {
                                'type': b.type,
                                'severity': b.severity,
                                'description': b.description
                            }
                            for b in analyzer.bottlenecks[:3]  # 只取前3个
                        ]
                    }
                    
                    # 生成详细的可视化报告
                    visualizer = NCUVisualizer(parser, analyzer)
                    vis_output_dir = self.output_dir / f"visualization_{kernel_name}"
                    visualizer.output_dir = vis_output_dir
                    vis_output_dir.mkdir(exist_ok=True)
                    visualizer.create_visualizations()
                    
                except Exception as e:
                    print(f"⚠️  分析 {ncu_file} 失败: {e}")
        
        # 保存综合分析结果
        results_file = self.output_dir / "comprehensive_analysis.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📋 综合分析结果已保存: {results_file}")
        return comprehensive_results
    
    def generate_final_report(self, comprehensive_results: Dict) -> str:
        """生成最终分析报告"""
        
        report_file = self.output_dir / "integrated_performance_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 集成性能分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # nsys概览
            f.write("## 🔍 Nsys 全局性能概览\n\n")
            nsys_overview = comprehensive_results.get('nsys_overview', {})
            
            if 'kernel_analysis' in nsys_overview:
                kernel_stats = nsys_overview['kernel_analysis']
                f.write(f"- **总kernels数量**: {kernel_stats.get('total_kernels', 0)}\n")
                f.write(f"- **总kernel执行时间**: {kernel_stats.get('total_kernel_time', 0):.2f} ms\n")
                f.write(f"- **平均kernel执行时间**: {kernel_stats.get('avg_kernel_time', 0):.3f} ms\n")
            
            # 热点kernels
            f.write(f"\n## 🔥 识别的热点Kernels ({comprehensive_results.get('hot_kernels_count', 0)}个)\n\n")
            for i, kernel in enumerate(self.hot_kernels[:10], 1):
                f.write(f"{i}. **{kernel['name'][:80]}**\n")
                f.write(f"   - 总执行时间: {kernel['total_time_ms']:.2f} ms\n")
                f.write(f"   - 平均执行时间: {kernel['avg_time_ms']:.3f} ms\n") 
                f.write(f"   - 调用次数: {kernel['count']}\n\n")
            
            # ncu深度分析
            f.write("## ⚡ NCU 深度分析结果\n\n")
            ncu_analysis = comprehensive_results.get('ncu_detailed_analysis', {})
            
            for kernel_name, analysis in ncu_analysis.items():
                f.write(f"### {kernel_name}\n\n")
                
                gpu_util = analysis.get('gpu_utilization', {})
                if gpu_util:
                    f.write(f"- **平均SM效率**: {gpu_util.get('average_sm_efficiency', 0):.1f}%\n")
                
                memory_analysis = analysis.get('memory_analysis', {})
                if 'bandwidth_stats' in memory_analysis:
                    bw = memory_analysis['bandwidth_stats']
                    f.write(f"- **平均内存带宽**: {bw.get('average_bandwidth', 0):.1f} GB/s\n")
                
                # 瓶颈分析
                bottlenecks = analysis.get('bottleneck_summary', [])
                if bottlenecks:
                    f.write(f"- **主要性能瓶颈**:\n")
                    for bottleneck in bottlenecks:
                        f.write(f"  - {bottleneck['description']} ({bottleneck['severity']})\n")
                
                f.write("\n")
            
            # 优化建议
            f.write("## 💡 优化建议\n\n")
            f.write("### 基于nsys分析:\n")
            f.write("- 关注上述热点kernels的优化\n")
            f.write("- 检查kernel调用的时间间隙，优化overlap\n\n")
            
            f.write("### 基于ncu分析:\n")
            f.write("- 对SM效率低的kernels进行算法优化\n")
            f.write("- 对内存带宽低的kernels优化访问模式\n")
            f.write("- 根据具体瓶颈类型采取针对性优化措施\n")
        
        print(f"📄 最终报告已生成: {report_file}")
        return str(report_file)

def create_sglang_analysis_workflow():
    """创建SGlang专用的分析工作流"""
    
    def run_sglang_integrated_analysis(model_path: str, 
                                      batch_size: int = 8,
                                      input_len: int = 512, 
                                      output_len: int = 64):
        """运行SGlang的集成分析"""
        
        # 构建SGlang命令
        sglang_cmd = [
            'python', '-m', 'sglang.bench_one_batch',
            '--model-path', model_path,
            '--batch-size', str(batch_size),
            '--input-len', str(input_len),
            '--output-len', str(output_len),
            '--load-format', 'dummy'
        ]
        
        # 创建分析器
        analyzer = NSysToNCUAnalyzer(f"sglang_analysis_b{batch_size}_i{input_len}_o{output_len}")
        
        # 步骤1: nsys全局分析
        nsys_file = analyzer.step1_nsys_analysis(sglang_cmd, "sglang_overview")
        
        # 步骤2: 提取热点kernels
        hot_kernels = analyzer.step2_extract_hot_kernels(nsys_file, top_k=8)
        
        if not hot_kernels:
            print("❌ 未发现热点kernels，分析终止")
            return
        
        # 步骤3: ncu深度分析（限制分析数量）
        ncu_files = analyzer.step3_ncu_targeted_analysis(sglang_cmd, hot_kernels, max_kernels=3)
        
        # 步骤4: 综合分析
        results = analyzer.step4_comprehensive_analysis(ncu_files)
        
        # 生成最终报告
        report_file = analyzer.generate_final_report(results)
        
        print(f"\n🎉 SGlang集成分析完成!")
        print(f"📁 分析结果目录: {analyzer.output_dir}")
        print(f"📄 分析报告: {report_file}")
        
        return analyzer.output_dir
    
    return run_sglang_integrated_analysis

def main():
    parser = argparse.ArgumentParser(description='集成 nsys 和 ncu 的性能分析工具')
    parser.add_argument('command', nargs='+', help='要分析的命令')
    parser.add_argument('--output-dir', default='integrated_analysis', help='输出目录')
    parser.add_argument('--top-k', type=int, default=10, help='提取的热点kernel数量')
    parser.add_argument('--max-ncu-kernels', type=int, default=5, help='ncu分析的最大kernel数量')
    parser.add_argument('--min-duration', type=float, default=0.1, help='最小kernel执行时间(ms)')
    
    # SGlang特殊参数
    parser.add_argument('--sglang-model', type=str, help='SGlang模型路径')
    parser.add_argument('--sglang-batch', type=int, default=8, help='SGlang批次大小')
    parser.add_argument('--sglang-input-len', type=int, default=512, help='SGlang输入长度')
    parser.add_argument('--sglang-output-len', type=int, default=64, help='SGlang输出长度')
    
    args = parser.parse_args()
    
    try:
        if args.sglang_model:
            # SGlang专用分析
            sglang_workflow = create_sglang_analysis_workflow()
            sglang_workflow(
                args.sglang_model,
                args.sglang_batch,
                args.sglang_input_len, 
                args.sglang_output_len
            )
        else:
            # 通用分析
            analyzer = NSysToNCUAnalyzer(args.output_dir)
            
            # 步骤1-4
            nsys_file = analyzer.step1_nsys_analysis(args.command)
            hot_kernels = analyzer.step2_extract_hot_kernels(nsys_file, args.top_k, args.min_duration)
            
            if hot_kernels:
                ncu_files = analyzer.step3_ncu_targeted_analysis(args.command, hot_kernels, args.max_ncu_kernels)
                results = analyzer.step4_comprehensive_analysis(ncu_files)
                analyzer.generate_final_report(results)
            else:
                print("❌ 未发现符合条件的热点kernels")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断分析")
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

