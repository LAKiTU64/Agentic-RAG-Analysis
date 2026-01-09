#!/usr/bin/env python3
"""
NVIDIA 性能分析集成工具
先用 nsys 识别热点kernels，再用 ncu 深度分析

工作流程：
1. nsys profile -> 获取全局性能overview  
2. 提取热点kernel名称
3. ncu profile -> 针对热点kernels深度分析
4. 综合分析报告

作者: xjw
版本: 1.0
"""

import os
import sys
import json
import subprocess
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import pandas as pd

# 导入我们的分析工具
sys.path.append(str(Path(__file__).parent))
from nsys_parser import NsysParser, NsysAnalyzer
from ncu_parser import NCUParser, NCUAnalyzer, NCUVisualizer, NCUReporter

# 引入高阶报告与知识库摄取模块（可选）
try:
    from backend.advanced_report import generate_advanced_report
except Exception:
    generate_advanced_report = None  # type: ignore
try:
    from backend.knowledge_bases.kb_ingest import ingest_json_to_faiss, flatten_json
except Exception:
    ingest_json_to_faiss = None  # type: ignore
    flatten_json = None  # type: ignore

INTEGRATED_REPORT_PATH = Path("/workspace/Agent/AI_Agent_Complete/sglang_analysis_b8_i512_o64/integrated_performance_report.md")

class NSysToNCUAnalyzer:
    """集成 nsys 和 ncu 的分析工具

    统一输出目录:
        默认使用 /workspace/Agent/AI_Agent_Complete 作为根路径下的 integrated_analysis 子目录，
        便于 Agent 读取所有生成的报告和中间产物。
    """
    DEFAULT_BASE_DIR = Path("/workspace/Agent/AI_Agent_Complete")

    def __init__(self, output_dir: str = "integrated_analysis"):
        # 如果用户传入的是绝对路径则使用原值，否则拼接到默认基路径下
        base = self.DEFAULT_BASE_DIR
        if output_dir.startswith('/'):
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = base / output_dir
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
            # 让 nsys 的输出直接打印到终端，便于实时查看（不再 capture_output）
            # result = subprocess.run(nsys_cmd, check=True)
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
            # kernel 可能为 dict，也可能为其它类型，做防护处理
            try:
                name_str = kernel['name'] if isinstance(kernel, dict) else kernel
            except Exception:
                name_str = kernel

            # 强制转为字符串，防止非字符串类型导致切片错误
            name_str = str(name_str)
            name_short = name_str[:60]
            total = kernel.get('total_time_ms', 0) if isinstance(kernel, dict) else 0
            avg = kernel.get('avg_time_ms', 0) if isinstance(kernel, dict) else 0
            count = kernel.get('count', 0) if isinstance(kernel, dict) else 0

            print(f"  {i}. {name_short}... (总计: {total:.2f}ms, 平均: {avg:.3f}ms, 调用: {count}次)")
        
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

        # 限制分析数量
        kernels_to_analyze = kernels_to_analyze[:max_kernels]
        
        for i, kernel_info in enumerate(kernels_to_analyze):
            kernel_name = str(kernel_info.get('name', 'kernel')).strip()

            # 清理kernel名称，用于文件名
            safe_name = re.sub(r'[^\w\-_]', '_', kernel_name)[:50]
            ncu_profile = self.output_dir / f"ncu_kernel_{i}_{safe_name}"
            
            # print(f"🎯 正在分析kernel {i+1}/{len(kernels_to_analyze)}: {kernel_name[:60]}...")

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
                print(ncu_cmd)
                # result = subprocess.run(ncu_cmd, capture_output=True, text=True, 
                #                        check=True)  # 5分钟超时
                result = subprocess.run(ncu_cmd)  # 5分钟超时
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

    def step3_ncu_global_focus(self, target_command: List[str], hot_kernels: List[Dict], top_focus: int = 5,
                               set_name: str = 'compute', launch_limit: Optional[int] = None) -> Tuple[Optional[str], Dict[str, Dict]]:
        """替代定向分析：一次全量 ncu 采集，然后仅针对 nsys 发现的前 top_focus 个热点 kernel 提取与归并指标。

        返回: (全量 ncu 报告路径或 None, focus_metrics dict)
        focus_metrics 结构 (键为热点 kernel 原名):
            {
              kernel_display_name: {
                  'kernels_analyzed': int,
                  'gpu_utilization': {...},
                  'memory_analysis': {...},
                  'bottleneck_summary': [...]
              }
            }
        """
        if not hot_kernels:
            print("⚠️ 无热点 kernel，跳过全量 NCU 采集")
            return None, {}
        # 运行一次全量采集
        full_rep = self.full_ncu_capture(target_command, profile_name='ncu_full_capture_global', set_name=set_name, launch_limit=launch_limit)
        if not full_rep:
            return None, {}
        csv_file = full_rep.replace('.ncu-rep', '.csv')
        if not Path(csv_file).exists() or Path(csv_file).stat().st_size == 0:
            print("⚠️ 全量采集 CSV 不存在或为空，无法提取焦点内核指标")
            return full_rep, {}
        # 解析 CSV
        try:
            parser = NCUParser(csv_file)
            parser.parse()
        except Exception as e:
            print(f"⚠️ 全量 CSV 解析失败: {e}")
            return full_rep, {}
        # 构建焦点分析
        focus = {}
        # 建立快速列表
        metrics_list = parser.kernels  # List[KernelMetrics]
        def _match_entries(target: str) -> List[Any]:
            t_low = target.lower()
            matched = [km for km in metrics_list if t_low in km.name.lower() or km.name.lower() in t_low]
            # 若无直接包含匹配，尝试按分词公共子串 >=5 char
            if not matched:
                # 简单切割非字母数字
                import re
                tokens = [tok for tok in re.split(r'[^A-Za-z0-9_]+', t_low) if len(tok) >= 5]
                if tokens:
                    for tok in tokens:
                        part = [km for km in metrics_list if tok in km.name.lower()]
                        matched.extend(part)
            # 去重
            uniq = []
            seen = set()
            for m in matched:
                if id(m) not in seen:
                    seen.add(id(m)); uniq.append(m)
            return uniq[:50]  # 防止过多
        def _avg(vals: List[Optional[float]]) -> Optional[float]:
            nums = [v for v in vals if isinstance(v, (int, float))]
            return sum(nums)/len(nums) if nums else None
        focus_targets = hot_kernels[:top_focus]
        for hk in focus_targets:
            kname = str(hk.get('name',''))
            entries = _match_entries(kname)
            if not entries:
                continue
            sm_eff = _avg([e.sm_efficiency for e in entries])
            occ = _avg([e.achieved_occupancy for e in entries])
            dram = _avg([e.dram_bandwidth for e in entries])
            l2 = _avg([e.l2_hit_rate for e in entries])
            warp_eff = _avg([e.warp_execution_efficiency for e in entries])
            tensor_active = _avg([e.tensor_active for e in entries])
            # 瓶颈判定 (启发式)
            bottlenecks = []
            def add_bottleneck(cond: bool, desc: str, severity: str):
                if cond:
                    bottlenecks.append({'type': 'heuristic', 'severity': severity, 'description': desc})
            add_bottleneck(sm_eff is not None and sm_eff < 40, 'SM效率偏低', 'high')
            add_bottleneck(dram is not None and dram < 150, '内存带宽可能受限', 'medium')
            add_bottleneck(occ is not None and occ < 25, 'Occupancy较低', 'medium')
            add_bottleneck(warp_eff is not None and warp_eff < 70, 'Warp执行效率一般', 'low')
            focus[kname] = {
                'kernels_analyzed': len(entries),
                'gpu_utilization': {
                    'average_sm_efficiency': sm_eff,
                    'achieved_occupancy': occ,
                    'tensor_core_active': tensor_active,
                },
                'memory_analysis': {
                    'bandwidth_stats': {
                        'average_bandwidth': dram,
                        'l2_hit_rate': l2,
                    }
                },
                'bottleneck_summary': bottlenecks
            }
        print(f"🔎 全量采集中已生成 {len(focus)} 个焦点内核的聚合指标")
        return full_rep, focus
    
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
    
    def step4_comprehensive_analysis(self, ncu_files: List[str], focus_metrics: Optional[Dict[str, Dict]] = None) -> Dict:
        """第四步：综合分析nsys和ncu结果"""
        
        print("📊 步骤4: 综合分析结果...")
        
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'nsys_overview': self.nsys_stats,
            'hot_kernels_count': len(self.hot_kernels),
            'ncu_detailed_analysis': {},
            'ncu_focus_analysis': focus_metrics or {}
        }
        
        # 分析每个ncu结果
        # 若提供焦点聚合指标，则不必对全量 ncu_full_capture_global 逐文件做标准分析（仍可保留 targeted 文件分析）
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
        """生成最终分析报告，保留已有的 NCU 报告块"""
        report_file = self.output_dir / "integrated_performance_report.md"
        start_tag = "<!-- NCU_REPORT_START -->"
        end_tag = "<!-- NCU_REPORT_END -->"

        existing_ncu_block = ""
        if report_file.exists():
            old = report_file.read_text(encoding='utf-8')
            import re
            m = re.search(f"{start_tag}.*?{end_tag}", old, flags=re.DOTALL)
            if m:
                existing_ncu_block = m.group(0)

        lines = []
        lines.append("# 集成性能分析报告\n")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # nsys 概览
        lines.append("## 🔍 Nsys 全局性能概览\n")
        nsys_overview = comprehensive_results.get('nsys_overview', {})
        if 'kernel_analysis' in nsys_overview:
            ks = nsys_overview['kernel_analysis']
            lines.append(f"- 总kernels数量: {ks.get('total_kernels', 0)}")
            lines.append(f"- 总kernel执行时间: {ks.get('total_kernel_time', 0):.2f} ms")
            lines.append(f"- 平均kernel执行时间: {ks.get('avg_kernel_time', 0):.3f} ms\n")

        # 热点 kernels
        lines.append(f"## 🔥 识别的热点Kernels ({comprehensive_results.get('hot_kernels_count', 0)}个)\n")
        for i, kernel in enumerate(self.hot_kernels[:10], 1):
            name = kernel.get('name', '')[:80]
            lines.append(f"{i}. {name}")
            lines.append(f"   - 总执行时间: {kernel.get('total_time_ms',0):.2f} ms")
            lines.append(f"   - 平均执行时间: {kernel.get('avg_time_ms',0):.3f} ms")
            lines.append(f"   - 调用次数: {kernel.get('count',0)}\n")

        # NCU 深度（占位，真正内容由 ncu_parser 插入块保留）
        lines.append("## ⚡ NCU 深度分析结果\n")
        if existing_ncu_block:
            lines.append("（保留已有 NCU 报告块）\n")
        else:
            lines.append("（尚未生成 NCU 报告，运行 ncu_parser.py 后会自动插入）\n")

        # 焦点聚合
        focus_analysis = comprehensive_results.get('ncu_focus_analysis', {})
        if focus_analysis:
            lines.append("## 🎯 焦点内核聚合指标\n")
            for kname, a in focus_analysis.items():
                lines.append(f"### {kname}")
                gu = a.get('gpu_utilization', {})
                mem = a.get('memory_analysis', {}).get('bandwidth_stats', {})
                if gu:
                    lines.append(f"- 平均SM效率: {gu.get('average_sm_efficiency','N/A')}")
                    lines.append(f"- Occupancy: {gu.get('achieved_occupancy','N/A')}")
                if mem:
                    lines.append(f"- 平均带宽: {mem.get('average_bandwidth','N/A')} GB/s")
                    lines.append(f"- L2命中率: {mem.get('l2_hit_rate','N/A')}%")
                bsum = a.get('bottleneck_summary', [])
                if bsum:
                    lines.append("- 主要瓶颈:")
                    for b in bsum:
                        lines.append(f"  - {b['description']} ({b['severity']})")
                lines.append("")

        # 优化建议
        lines.append("## 💡 优化建议\n")
        lines.append("- 关注热点kernel的调度与并行重叠")
        lines.append("- 针对低SM效率kernel优化算法/批次")
        lines.append("- 针对低带宽/低命中率kernel优化内存访问\n")

        # 合并文本
        report_text = "\n".join(lines).rstrip() + "\n"
        # 若有旧 NCU 块，附加在末尾（保持标记完整）
        if existing_ncu_block:
            report_text += "\n" + existing_ncu_block + "\n"

        report_file.write_text(report_text, encoding='utf-8')
        print(f"📄 最终报告已生成(保留 NCU 块): {report_file}")
        return str(report_file)

def create_sglang_analysis_workflow():
    """创建SGlang专用的分析工作流"""
    DEFAULT_MODEL_DIR = os.getenv('SGLANG_MODEL_PATH') or os.getenv('MODEL_PATH') or '/workspace/models/'

    def run_sglang_integrated_analysis(model_path: Optional[str] = None, 
                                      batch_size: int = 8,
                                      input_len: int = 512, 
                                      output_len: int = 64):
        """运行SGlang的集成分析

        参数:
            model_path: 模型路径，若未提供则使用环境变量 SGLANG_MODEL_PATH / MODEL_PATH，最后回退 /workspace/models/
        """
        if not model_path:
            model_path = DEFAULT_MODEL_DIR.rstrip('/')
            print(f"ℹ️ 未提供 model_path，使用默认路径: {model_path}")
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

# --- 辅助函数: 将高阶报告 Markdown 粗略结构化为 JSON ---
def _extract_advanced_json(md_text: str) -> Dict[str, Any]:  # type: ignore
    sections: Dict[str, Any] = {}
    current = None
    for line in md_text.splitlines():
        if line.startswith('#'):
            # 获取标题
            title = line.strip('# ').strip()
            current = title
            sections[current] = []
        else:
            if current is not None:
                sections[current].append(line)
    # 简单抽取任务列表与分类
    tasks = []
    for k, v in sections.items():
        if '任务列表' in k or '细粒度' in k:
            tasks.extend([ln for ln in v if ln.strip().startswith('- ')])
    summary = sections.get('6. 总结 (Summary)', [])
    return {
        'sections': list(sections.keys()),
        'tasks_lines': tasks,
        'summary': '\n'.join(summary[:10]),
        'raw_length': len(md_text)
    }

class NCUParser:
    """NCU 报告解析器（支持新旧格式）"""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.kernels = []  # type: List[KernelMetrics]
        self.metadata = {}
        self._is_legacy = False  # 标记是否为旧版宽表格式

    def parse(self) -> None:
        """主解析函数"""
        if not self.file_path.exists():
            print(f"❌ NCU 文件不存在: {self.file_path}")
            return
        
        # 尝试解析为 CSV 格式
        if str(self.file_path).endswith('.csv'):
            self._parse_csv()
        else:
            # 尝试新版 JSON 格式
            json_file = self.file_path.with_suffix('.json')
            if json_file.exists():
                self._parse_json(json_file)
            else:
                print("⚠️ 无法识别的文件格式或文件缺失 (需手动检查):", self.file_path)
    
    def _parse_csv(self) -> None:
        """解析 CSV 格式的 NCU 报告"""
        try:
            # 尝试读取为长表格式
            df = pd.read_csv(self.file_path)
            print(f"📊 检测到长表格式，行数: {len(df)}")
            self._parse_csv_kernels(df)
            self.metadata['total_kernels'] = len(self.kernels)
            return
        except Exception as e:
            print(f"⚠️ 长表格式解析失败: {e}")
        
        try:
            # 尝试读取为宽表格式
            df = pd.read_csv(self.file_path, header=None)
            print(f"📊 检测到宽表格式，行数: {len(df)}")
            self._parse_csv_wide(df)
            self.metadata['total_kernels'] = len(self.kernels)
            return
        except Exception as e:
            print(f"⚠️ 宽表格式解析失败: {e}")
        
        print("❌ CSV 文件解析失败，请检查文件格式")
    
    def _parse_csv_kernels(self, df: pd.DataFrame) -> None:
        """兼容 Nsight Compute 导出的长表/宽表 CSV"""
        kernel_name_col = None
        for cand in ['Kernel Name','KernelName','Name']:
            if cand in df.columns:
                kernel_name_col = cand
                break
        if kernel_name_col is None:
            print("⚠️ 未找到 Kernel Name 列，退回通用解析")
            return

        # 长表检测
        is_long = {'Section Name','Metric Name','Metric Value'}.issubset(df.columns)

        if is_long:
            # 清理数值
            df['Metric Value'] = df['Metric Value'].astype(str).str.replace(',', '', regex=False)
            df['Metric Value'] = pd.to_numeric(df['Metric Value'], errors='coerce')

            for kname, kdf in df.groupby(kernel_name_col):
                km = KernelMetrics(name=str(kname))

                def get(section, metric):
                    sel = kdf[(kdf['Section Name'] == section) & (kdf['Metric Name'] == metric)]['Metric Value']
                    return None if sel.empty else float(sel.mean())

                # 基础形状（只取第一次出现的 Block / Grid）
                try:
                    b = str(kdf['Block Size'].dropna().iloc[0])
                    g = str(kdf['Grid Size'].dropna().iloc[0])
                    def parse_shape(s):
                        s = s.strip().strip('()')
                        parts = [int(p.strip()) for p in s.split(',')]
                        return tuple(parts)
                    km.block_size = parse_shape(b)
                    km.grid_size = parse_shape(g)
                except Exception:
                    pass

                # 映射核心指标
                km.sm_efficiency = get('Compute Workload Analysis','SM Busy') \
                                   or get('GPU Speed Of Light Throughput','Compute (SM) Throughput')

                km.achieved_occupancy = get('Occupancy','Achieved Occupancy')
                km.theoretical_occupancy = get('Occupancy','Theoretical Occupancy')

                # 时长(us)
                dur_us = get('GPU Speed Of Light Throughput','Duration')
                if dur_us is not None:
                    km.duration = dur_us / 1000.0  # ms

                # 带宽与命中率
                bw_gbps = get('Memory Workload Analysis','Memory Throughput')  # Gbyte/s
                if bw_gbps is None:
                    bw_gbps = get('Memory Workload Analysis','DRAM Throughput')  # 可能是 %
                km.dram_bandwidth = bw_gbps
                km.l2_hit_rate = get('Memory Workload Analysis','L2 Hit Rate')
                l1 = get('Memory Workload Analysis','L1/TEX Hit Rate')
                km.l1_hit_rate = l1

                # 寄存器与共享内存（动态+静态）
                regs = get('Launch Statistics','Registers Per Thread')
                if regs is not None:
                    km.registers_per_thread = int(regs)
                dyn_shm = get('Launch Statistics','Dynamic Shared Memory Per Block')
                sta_shm = get('Launch Statistics','Static Shared Memory Per Block')
                if dyn_shm is not None or sta_shm is not None:
                    dyn_bytes = (dyn_shm or 0) * 1024.0  # Kbyte -> byte
                    sta_bytes = sta_shm or 0
                    km.shared_memory_per_block = int(dyn_bytes + sta_bytes)

                # Warp 执行效率（粗略：Avg. Not Predicated Off Threads /32）
                not_off = get('Warp State Statistics','Avg. Not Predicated Off Threads Per Warp')
                if not_off is not None:
                    km.warp_execution_efficiency = min(100.0, (not_off / 32.0) * 100.0)

                self.kernels.append(km)

            print(f"✅ 长表解析完成: {len(self.kernels)} kernels")
            return

        # 宽表旧逻辑（保留）
        for kernel_name in df[kernel_name_col].unique():
            row = df[df[kernel_name_col] == kernel_name].iloc[0]
            km = KernelMetrics(name=str(kernel_name))
            mapping = {
                'SM Efficiency':'sm_efficiency',
                'Achieved Occupancy':'achieved_occupancy',
                'Theoretical Occupancy':'theoretical_occupancy',
                'DRAM Bandwidth':'dram_bandwidth',
                'L2 Hit Rate':'l2_hit_rate',
                'L1 Hit Rate':'l1_hit_rate',
                'Duration':'duration',
                'Registers Per Thread':'registers_per_thread'
            }
            for col, attr in mapping.items():
                if col in df.columns and pd.notna(row[col]):
                    setattr(km, attr, row[col])
            self.kernels.append(km)
        print(f"✅ 宽表解析完成: {len(self.kernels)} kernels")

class NCUReporter:
    """NCU 报告生成器"""

    def __init__(self, parser: NCUParser, analyzer: NCUAnalyzer, output_dir: str):
        self.parser = parser
        self.analyzer = analyzer
        self.output_dir = output_dir

    def _safe_pct(self, v):
        try:
            return f"{float(v):.1f}%"
        except:
            return "N/A"

    def _safe_num(self, v, unit=""):
        try:
            return f"{float(v):.2f}{unit}"
        except:
            return "N/A"

    def generate_report(self) -> str:
        """生成文本报告并写入集成 Markdown"""
        stats = self.analyzer.stats.get('gpu_utilization', {})
        lines = []
        lines.append("### NCU 分析结果 (自动插入)\n")
        lines.append(f"- Kernel 数量: {len(self.parser.kernels)}")
        if stats:
            lines.append(f"- 平均 SM Busy: {self._safe_pct(stats.get('average_sm_efficiency'))}")
            lines.append(f"- SM Busy 最低: {self._safe_pct(stats.get('min_sm_efficiency'))}")
            lines.append(f"- 低于50% 的 Kernel 数: {stats.get('kernels_below_50_percent',0)}")
        # 简要列出前若干 kernel
        lines.append("\n#### 关键 Kernel 指标 (前10)")
        for k in self.parser.kernels[:10]:
            lines.append(
                f"- {k.name[:80]} | SM Busy: {self._safe_pct(k.sm_efficiency)} | Occ: {self._safe_pct(k.achieved_occupancy)} | L2 Hit: {self._safe_pct(k.l2_hit_rate)} | BW: {self._safe_num(k.dram_bandwidth,'')}"
            )

        report_text = "\n".join(lines)

        # 写入单独文件（保留原行为）
        out_path = Path(self.output_dir) / "ncu_report.txt"
        out_path.parent.mkdir(exist_ok=True, parents=True)
        out_path.write_text(report_text, encoding='utf-8')

        # 集成写入 integrated_performance_report.md
        self._update_integrated_markdown(report_text)

        print(f"📄 NCU报告已生成: {out_path}")
        print(f"📎 已更新集成报告: {INTEGRATED_REPORT_PATH}")
        return report_text

    def _update_integrated_markdown(self, ncu_section: str):
        start_tag = "<!-- NCU_REPORT_START -->"
        end_tag   = "<!-- NCU_REPORT_END -->"
        block = f"{start_tag}\n{ncu_section}\n{end_tag}\n"

        if INTEGRATED_REPORT_PATH.exists():
            content = INTEGRATED_REPORT_PATH.read_text(encoding='utf-8')
            if start_tag in content and end_tag in content:
                # 替换旧块
                content = re.sub(
                    f"{start_tag}.*?{end_tag}",
                    block,
                    content,
                    flags=re.DOTALL
                )
            else:
                # 追加
                content += "\n" + block
        else:
            content = "# 集成性能分析报告\n\n" + block
        INTEGRATED_REPORT_PATH.write_text(content, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description='集成 nsys 和 ncu 的性能分析工具',
                                     epilog='示例: python nsys_to_ncu_analyzer.py -- python -m sglang.bench_one_batch --model-path /path --batch-size 8 --input-len 512 --output-len 64 --load-format dummy')
    parser.add_argument('command', nargs='*', help='要分析的命令 (如未使用 --raw-cmd, 可用 "--" 分隔)')
    parser.add_argument('--output-dir', default='integrated_analysis', help='输出目录 (默认根: /workspace/Agent/AI_Agent_Complete)')
    parser.add_argument('--top-k', type=int, default=10, help='提取的热点kernel数量')
    parser.add_argument('--max-ncu-kernels', type=int, default=5, help='ncu分析的最大kernel数量')
    parser.add_argument('--min-duration', type=float, default=0.1, help='最小kernel执行时间(ms)')
    parser.add_argument('--full-ncu', action='store_true', help='执行一次不做过滤的全量 NCU 采集 (与热点分析并行或替代)')
    parser.add_argument('--full-ncu-set', default='compute', help='全量采集使用的 NCU 指标集合 (--set 值)')
    parser.add_argument('--full-ncu-launch-limit', type=int, default=None, help='限制全量采集的 kernel 次数 (--launch-count)')
    parser.add_argument('--save-hot-kernels', action='store_true', help='保存热点 kernel 名称到 hot_kernels.txt (纯文本)')

    # 高阶报告相关参数
    parser.add_argument('--advanced-report', action='store_true', help='生成高阶优化建议报告 (advanced_performance_report.md)')
    parser.add_argument('--advanced-detailed', action='store_true', help='高阶报告包含详细指标快照与细粒度 Kernel 任务')
    parser.add_argument('--advanced-json', action='store_true', help='同时导出高阶报告为 JSON (advanced_performance_report.json) 以供知识库摄取')
    parser.add_argument('--ingest-advanced', action='store_true', help='将高阶报告 JSON 写入知识库 (需要 embedding 环境可用)')
    parser.add_argument('--kb-path', type=str, default='knowledge_store', help='知识库存储目录 (FAISS)')
    
    # SGlang特殊参数
    parser.add_argument('--sglang-model', type=str, help='SGlang模型路径 (默认: 环境变量 SGLANG_MODEL_PATH / MODEL_PATH 或 /workspace/models/)')
    parser.add_argument('--force-model-path', type=str, help='强制覆盖目标命令中的 --model-path 参数为指定路径 (可用环境变量 FORCE_MODEL_PATH)')
    parser.add_argument('--sglang-batch', type=int, default=8, help='SGlang批次大小')
    parser.add_argument('--sglang-input-len', type=int, default=512, help='SGlang输入长度')
    parser.add_argument('--sglang-output-len', type=int, default=64, help='SGlang输出长度')
    
    # 允许未知参数保留给目标命令
    known_args, unknown_tail = parser.parse_known_args()
    if getattr(known_args, 'command', None):
        base_cmd = known_args.command
    else:
        base_cmd = []
    # 如果用户使用 -- 分隔形式: python script.py -- <target command parts>
    # unknown_tail 就是后续的真实命令参数
    target_command = []
    if unknown_tail:
        target_command = unknown_tail
    else:
        target_command = base_cmd
    # 可选强制模型路径替换
    force_model_path = known_args.force_model_path or os.getenv('FORCE_MODEL_PATH')
    if force_model_path:
        def _rewrite_model_path(cmd: List[str]) -> List[str]:
            out = []
            skip_next = False
            for i, token in enumerate(cmd):
                if skip_next:
                    skip_next = False
                    continue
                if token == '--model-path':
                    out.append('--model-path')
                    out.append(force_model_path)
                    skip_next = True  # 跳过原路径
                    # 原路径丢弃
                elif token.startswith('--model-path='):
                    out.append(f'--model-path={force_model_path}')
                else:
                    out.append(token)
            return out
        before = ' '.join(target_command)
        target_command = _rewrite_model_path(target_command)
        after = ' '.join(target_command)
        if before != after:
            print(f"🔧 已强制覆盖 --model-path: {force_model_path}")
        else:
            # 如果原命令未包含 --model-path，直接追加
            target_command += ['--model-path', force_model_path]
            print(f"🔧 原命令未包含 --model-path，已追加: {force_model_path}")

    if not target_command:
        print('❌ 未提供待分析的目标命令。示例: python nsys_to_ncu_analyzer.py -- python -m sglang.bench_one_batch --model-path ...')
        return
    args = known_args
    
    try:
        # SGlang 专用路径默认处理
        sglang_workflow = create_sglang_analysis_workflow()
        if args.sglang_model or os.getenv('SGLANG_MODEL_PATH') or os.getenv('MODEL_PATH'):
            sglang_workflow(
                args.sglang_model or os.getenv('SGLANG_MODEL_PATH') or os.getenv('MODEL_PATH'),
                args.sglang_batch,
                args.sglang_input_len,
                args.sglang_output_len
            )
        else:
            # 通用分析
            analyzer = NSysToNCUAnalyzer(args.output_dir)
            
            # 步骤1-4
            nsys_file = analyzer.step1_nsys_analysis(target_command)
            hot_kernels = analyzer.step2_extract_hot_kernels(nsys_file, args.top_k, args.min_duration)
            if args.save_hot_kernels and hot_kernels:
                # 已在 step2 内部生成 hot_kernels.txt，这里只做提示
                print("📎 --save-hot-kernels 已启用，hot_kernels.txt 已生成。")

            full_capture_file = None
            if args.full_ncu:
                full_capture_file = analyzer.full_ncu_capture(
                    target_command,
                    profile_name='ncu_full_capture',
                    set_name=args.full_ncu_set,
                    launch_limit=args.full_ncu_launch_limit
                )

            ncu_files = []
            if hot_kernels:
                targeted = analyzer.step3_ncu_targeted_analysis(target_command, hot_kernels, args.max_ncu_kernels)
                ncu_files.extend(targeted)
            else:
                print('❌ 未发现符合条件的热点kernels (跳过定向 NCU)')

            if full_capture_file:
                ncu_files.append(full_capture_file)

            advanced_json_obj = None
            if ncu_files:
                results = analyzer.step4_comprehensive_analysis(ncu_files)
                analyzer.generate_final_report(results)
                # 生成高阶报告
                if args.advanced_report and generate_advanced_report:
                    try:
                        adv_path = generate_advanced_report(analyzer.output_dir, detailed=args.advanced_detailed)
                        print(f"🧠 高阶优化报告已生成: {adv_path}")
                        if args.advanced_json:
                            # 将生成的 markdown 转为简单结构化 JSON (提取部分段落)
                            md_text = Path(adv_path).read_text(encoding='utf-8')
                            advanced_json_obj = _extract_advanced_json(md_text)
                            json_path = analyzer.output_dir / 'advanced_performance_report.json'
                            json_path.write_text(json.dumps(advanced_json_obj, ensure_ascii=False, indent=2), encoding='utf-8')
                            print(f"📦 高阶报告 JSON 已导出: {json_path} ➡️ 可用于知识库摄取")
                        # 可选摄取知识库
                        if args.ingest_advanced and advanced_json_obj and ingest_json_to_faiss and flatten_json:
                            try:
                                texts = [json.dumps(advanced_json_obj, ensure_ascii=False)]
                                ingest_json_to_faiss(json.dumps(advanced_json_obj, ensure_ascii=False), kb_path=args.kb_path)
                                print("📥 已尝试将高阶报告写入知识库向量库")
                            except Exception as e:
                                print(f"⚠️ 高阶报告知识库摄取失败: {e}")
                    except Exception as e:
                        print(f"⚠️ 高阶报告生成失败: {e}")
                elif args.advanced_report and not generate_advanced_report:
                    print("⚠️ advanced_report 模块不可用，跳过高阶报告生成")
            else:
                print('⚠️ 未生成任何 NCU 报告, 结束。')
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断分析")
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

