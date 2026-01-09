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
import sqlite3
import subprocess
import argparse
import re
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timezone

# 导入我们的分析工具
sys.path.append(str(Path(__file__).parent))
from nsys_parser import NsysParser, NsysAnalyzer
from ncu_parser import NCUParser, NCUAnalyzer, NCUVisualizer, NCUReporter
try:
    from .roofline_estimator import compute_roofline  # type: ignore
except Exception:
    try:
        from backend.utils.roofline_estimator import compute_roofline  # type: ignore
    except Exception:
        local_dir = Path(__file__).parent
        if str(local_dir) not in sys.path:
            sys.path.insert(0, str(local_dir))
        from roofline_estimator import compute_roofline  # type: ignore

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

class NSysToNCUAnalyzer:
    """集成 nsys 和 ncu 的分析工具

    统一输出目录:
        默认使用 /workspace/Agent/AI_Agent_Complete 作为根路径下的 integrated_analysis 子目录，
        便于 Agent 读取所有生成的报告和中间产物。
    """
    DEFAULT_BASE_DIR = Path("/workspace/Agent/AI_Agent_Complete")

    def __init__(self, output_dir: str = "integrated_analysis", env: Optional[Dict[str, str]] = None):
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
        # 为子进程调用（nsys/ncu）预先保存环境变量，便于控制 GPU 绑定
        self.env = env or os.environ.copy()
        # 默认补充 FP16/F16 Tensor Core 指标，便于后续分析
        self.ncu_metrics = [
            'smsp__inst_executed_pipe_tensor_op_hf.sum',
            'smsp__inst_executed_pipe_fp16.sum',
            'sm__inst_executed_pipe_tensor_op_hf.sum',
            'sm__sass_thread_inst_executed_op_hf.sum',
        ]
        self.roofline_estimate: Optional[Dict[str, Any]] = None
        
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
            # '--capture-range=nvtx',
            # '--capture-range-end=stop',
        ] + target_command 
        
        print("🚀 步骤1: 运行nsys全局性能分析...")
        print(f"命令: {' '.join(nsys_cmd)}")
        
        try:
            result = subprocess.run(nsys_cmd, capture_output=True, text=True, check=True, env=self.env)
            print(f"✅ nsys分析完成: {nsys_profile}")
            return str(nsys_profile)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ nsys分析失败: {e.stderr}")
            raise
    
    def step2_extract_hot_kernels(self, nsys_file: str, 
                                  top_k: int = 10, 
                                  min_duration_ms: float = 0.1) -> List[Dict]:
        """第二步：解析 nsys 并生成 layer_kernels.csv（跳过热点提取，直接使用 NVTX 关联的 kernels）
        现在默认仅使用 Run[2] 的子集进行后续分析与报告。
        """
        
        print("🔍 步骤2: 解析 nsys 并生成 layer_kernels.csv（跳过热点提取）...")
        
        parser = NsysParser(nsys_file)
        parser.parse()
        # 导出由三表关联得到的 layer_kernels.csv（位于 .sqlite 同目录）
        lk_csv = parser.export_kernel_summary_csv(nsys_file, self.output_dir / f"{Path(nsys_file).stem}_kernels")
        target_csv = self.output_dir / "layer_kernels.csv"
        if lk_csv:
            try:
                src = Path(lk_csv)
                if src.resolve() == target_csv.resolve():
                    # 源目标一致，视为成功
                    print(f"📄 layer_kernels.csv 已存在于输出目录: {target_csv}")
                else:
                    # 强制覆盖
                    import shutil, os
                    if target_csv.exists():
                        os.remove(target_csv)
                    shutil.copy2(src, target_csv)
                    print(f"📄 已生成 layer_kernels.csv: {target_csv}")
            except Exception as e:
                print(f"⚠️ 拷贝 layer_kernels.csv 失败: {e}. 原路径: {lk_csv}")
        else:
            print("⚠️ 未生成 layer_kernels.csv，请确认 NVTX_EVENTS/StringIds/CUPTI 表存在且有 Layer[...] 标签")

        # 仅保留 Run[2] 的子集
        run_tag = "#Run[2]"
        run_csv = self.output_dir / "layer_kernels_run2.csv"
        try:
            import csv
            kept = 0
            with open(target_csv, newline='', encoding='utf-8') as f, open(run_csv, 'w', newline='', encoding='utf-8') as g:
                rdr = csv.DictReader(f)
                w = csv.DictWriter(g, fieldnames=rdr.fieldnames)
                w.writeheader()
                for r in rdr:
                    if run_tag in (r.get("layer") or ""):
                        w.writerow(r)
                        kept += 1
            print(f"📄 已生成仅 Run[2] 的子集: {run_csv}（{kept} 行）")
        except Exception as e:
            print(f"⚠️ 生成 Run[2] 子集失败: {e}")

        # 用 Run[2] 子集生成精简 JSON（仅 name、dur_ms），并以此作为后续唯一数据源
        rows_run2 = []
        try:
            import csv, json
            with open(run_csv, newline='', encoding='utf-8') as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    rows_run2.append({
                        'name': r.get('kernel_name',''),
                        'dur_ms': float(r.get('dur_ms', 0.0) or 0.0)
                    })
            hot_json_path = self.output_dir / "layer_kernels_run2_hot.json"
            hot_json_path.write_text(json.dumps(rows_run2, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"📄 已生成精简热点文件(顺序保留，不去重): {hot_json_path}")
        except Exception as e:
            print(f"⚠️ 读取 Run[2] 子集失败: {e}")
            rows_run2 = []

        # 用 Run[2] JSON 更新 nsys 概览（total_time 为 dur_ms 求和，不统计 count/max）
        total_time_ms = sum(item.get('dur_ms', 0.0) for item in rows_run2)
        total_kernels = len(rows_run2)
        avg_time_ms = (total_time_ms / total_kernels) if total_kernels else 0.0

        analyzer = NsysAnalyzer(parser)
        self.nsys_stats = {
            'kernel_analysis': {
                'total_kernels': total_kernels,
                'total_kernel_time': total_time_ms,
                'avg_kernel_time': avg_time_ms,
            },
            'layer_kernels_rows': rows_run2,           # 用精简结构替代
            'layer_kernels_source': str(hot_json_path) # 指向 run2_hot.json
        }

        # 记录关键 NVTX 范围的起止与时长，便于报告引用
        nvtx_tag = "Layer[0]#Run[2]"
        nvtx_span = self._query_nvtx_range(parser.sqlite_file, nvtx_tag)
        if nvtx_span:
            self.nsys_stats.setdefault('nvtx_ranges', {})[nvtx_tag] = nvtx_span

        # 将“热点”列表设置为 run2_hot.json 的顺序列表（不去重、不排序）
        self.hot_kernels = rows_run2[:]
        print(f"ℹ️ 已按要求使用 Run[2] JSON（{len(self.hot_kernels)} 条），顺序保留且不去重。")
        return self.hot_kernels
    
    def step3_ncu_targeted_analysis(self, target_command: List[str], 
                                   kernels_to_analyze: List[Dict],
                                   max_kernels: Optional[int] = None) -> List[str]:
        """第三步：使用ncu对热点kernels进行深度分析
        现在默认对传入列表的每一个 kernel 进行分析（保持顺序），除非显式提供 max_kernels。
        """
        
        print("⚡ 步骤3: 使用ncu深度分析热点kernels...")
        
        ncu_results = []

        # 允许仅包含 {name, dur_ms} 的最简结构
        # 若检测到占位/数字 kernel 名，尝试 list-kernels 发现真实名
        if any(self._is_placeholder_name(str(k.get('name',''))) for k in kernels_to_analyze):
            print("🔍 检测到占位/数字kernel名，触发 ncu --list-kernels 进行真实名称发现...")
            discovered = self.list_kernels_with_ncu(target_command)
            selected = self._select_real_kernels(discovered, len(kernels_to_analyze) if max_kernels is None else max_kernels)
            print(f"🧭 选择用于深度分析的真实kernel名称: {selected}")
            # 替换前若长度不足则补齐
            for i, real_name in enumerate(selected):
                if i < len(kernels_to_analyze):
                    kernels_to_analyze[i]['name'] = real_name
                    kernels_to_analyze[i]['discovered'] = True
                else:
                    kernels_to_analyze.append({'name': real_name, 'discovered': True})

        # 在最终分析前去重，避免重复 kernel 造成冗余采集
        def _deduplicate(entries: List[Dict]) -> List[Dict]:
            seen = set()
            unique: List[Dict] = []
            duplicates = 0
            for entry in entries:
                name = str(entry.get('name', '')).strip()
                if not name:
                    unique.append(entry)
                    continue
                if name in seen:
                    duplicates += 1
                    continue
                seen.add(name)
                unique.append(entry)
            if duplicates:
                print(f"♻️  去重后减少 {duplicates} 个重复 kernel")
            return unique

        kernels_to_analyze = _deduplicate(kernels_to_analyze)

        # 分析数量：默认全部；若指定 max_kernels 则截断
        if max_kernels is not None:
            kernels_to_analyze = kernels_to_analyze[:max_kernels]
        
        for i, kernel_info in enumerate(kernels_to_analyze):
            kernel_name = str(kernel_info.get('name', 'kernel')).strip()

            # 清理kernel名称，用于文件名
            safe_name = re.sub(r'[^\w\-_]', '_', kernel_name)[:50]
            ncu_profile = self.output_dir / f"ncu_kernel_{i}_{safe_name}"
            
            print(f"🎯 正在分析kernel {i+1}/{len(kernels_to_analyze)}: {kernel_name[:60]}...")

            def attempt_profile(attempt_cmd: List[str], attempt_tag: str) -> Optional[str]:
                """封装一次 ncu 尝试，返回 .ncu-rep 路径或 None"""
                try:
                    res = subprocess.run(attempt_cmd, env=self.env, capture_output=True, text=True)
                    if res.returncode != 0:
                        snippet = (res.stderr or '')[:200].replace('\n', ' ')
                        print(f"⚠️ 尝试 {attempt_tag} 失败(returncode={res.returncode}): {snippet}")
                        return None
                    ncu_file = str(ncu_profile) + '.ncu-rep'
                    if Path(ncu_file).exists():
                        print(f"✅ 成功生成 NCU 报告 ({attempt_tag}): {ncu_file}")
                        return ncu_file
                    else:
                        print(f"⚠️ 尝试 {attempt_tag} 未生成 .ncu-rep 文件: {ncu_file}")
                except subprocess.TimeoutExpired:
                    print(f"⏰ 尝试 {attempt_tag} 超时")
                except Exception as e:
                    print(f"❌ 尝试 {attempt_tag} 异常: {e}")
                return None

            # 回退策略：精确匹配 -> 正则前缀 -> 无过滤
            attempts = []
            attempts.append({
                'tag': 'exact-demangled',
                'cmd': ['ncu', '--kernel-name-base', 'demangled', '--kernel-name', kernel_name,
                        '--rename-kernels=0', '--set', 'full']
            })
            prefix_raw = re.sub(r'"', '', kernel_name)[:60]
            prefix_regex = re.sub(r'([\\.^$|?*+\[\](){}])', r'\\\1', prefix_raw)
            attempts.append({
                'tag': 'regex-prefix',
                'cmd': ['ncu', '--kernel-name-base', 'demangled', '--kernel-name', f'regex:^{prefix_regex}',
                        '--rename-kernels=0', '--set', 'full']
            })
            attempts.append({
                'tag': 'unfiltered-basic',
                'cmd': ['ncu', '--launch-count', '50', '--set', 'compute']
            })

            for att in attempts:
                metrics_segment = self._metrics_args()
                att['cmd'] = att['cmd'] + metrics_segment + ['-o', str(ncu_profile), '--force-overwrite'] + target_command

            produced = None
            for att in attempts:
                print(f"🔁 NCU 尝试: {att['tag']}")
                print(att['cmd'])
                produced = attempt_profile(att['cmd'], att['tag'])
                if produced:
                    break

            if produced:
                ncu_results.append(produced)
                self._export_ncu_to_csv(produced)
            else:
                print(f"❌ 所有尝试均未生成 NCU 报告: {kernel_name[:80]}")
        
        return ncu_results


    def list_kernels_with_ncu(self, target_command: List[str]) -> List[str]:
        """运行 ncu --list-kernels 以获取实际可分析的 kernel 名称列表"""
        cmd = ['ncu', '--list-kernels'] + target_command
        print(f"🧪 运行: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=self.env)
            output = result.stdout + '\n' + result.stderr
        except Exception as e:
            print(f"❌ list-kernels 失败: {e}")
            return []

        # 解析输出：每行可能包含 kernel 名称。我们过滤出含 'Kernel', 'cuda', 'cutlass', 'flash', 'aten', 'cublas', 'gemm', 'matmul', 'triton'
        lines = [l.strip() for l in output.splitlines() if l.strip()]
        kernels = []
        import re
        pattern = re.compile(r'(Kernel|cuda|cutlass|flash|aten|cublas|gemm|matmul|triton)', re.IGNORECASE)
        for line in lines:
            # 常见格式: index + name 或 直接 name
            # 排除太短行
            if len(line) < 4:
                continue
            if pattern.search(line):
                # 去掉前导编号或装饰符
                cleaned = re.sub(r'^\s*\d+\s*[:\-]?\s*', '', line)
                kernels.append(cleaned)
        # 去重保持顺序
        seen = set(); uniq = []
        for k in kernels:
            if k not in seen:
                seen.add(k); uniq.append(k)
        print(f"📋 list-kernels 获得候选 {len(uniq)} 个 (前10): {uniq[:10]}")
        return uniq

    def _select_real_kernels(self, discovered: List[str], max_kernels: int) -> List[str]:
        """根据优先级从发现的 kernel 名称列表中挑选用于分析的名称"""
        if not discovered:
            return []
        priority_patterns = [
            'FlashAttn', 'flash', 'cutlass', 'triton', 'gemm', 'matmul', 'cublas', 'aten', 'reduce', 'norm'
        ]
        scored = []
        for name in discovered:
            low = name.lower()
            score = 0
            for idx, pat in enumerate(priority_patterns):
                if pat.lower() in low:
                    score += (100 - idx)  # earlier pattern higher score
            # 长度和包含 Kernel 字样加一点分
            if 'kernel' in low:
                score += 5
            if len(name) > 30:
                score += 1
            scored.append((score, name))
        # 排序，分数高的靠前
        scored.sort(reverse=True)
        selected = [n for s, n in scored[:max_kernels]]
        return selected

    def _metrics_args(self) -> List[str]:
        metrics = getattr(self, 'ncu_metrics', [])
        if not metrics:
            return []
        return ['--metrics', ','.join(metrics)]

    def _is_placeholder_name(self, name: str) -> bool:
        # 将数字、__unnamed_ 以及若干低信号 / 框架性名字视为占位。增加 __cudart_ 前缀，以便后续用 CSV 真实 kernel 名替换。
        return (
            name.isdigit()
            or name.startswith('__unnamed_')
            or name in ('cudafe++', 'sleep', 'python', 'node')
            or name.startswith('__cudart_')
        )
    
    def _query_nvtx_range(self, sqlite_path: Optional[Path], tag: str) -> Optional[Dict[str, Any]]:
        if not sqlite_path or not Path(sqlite_path).exists():
            return None
        pattern = f"%{tag}%"
        sql = (
            "SELECT n.start, n.end, (n.end - n.start) AS span_ns "
            "FROM NVTX_EVENTS n "
            "LEFT JOIN StringIds s ON n.textId = s.id "
            "WHERE (n.text LIKE ? OR s.value LIKE ?) AND n.start IS NOT NULL AND n.end IS NOT NULL "
            "ORDER BY span_ns DESC LIMIT 1"
        )
        try:
            with sqlite3.connect(str(sqlite_path)) as conn:
                row = conn.execute(sql, (pattern, pattern)).fetchone()
        except Exception as exc:
            print(f"⚠️ NVTX 范围查询失败: {exc}")
            return None
        if not row:
            return None
        start_ns, end_ns, span_ns = row
        if start_ns is None or end_ns is None or span_ns is None:
            return None
        return {
            'start_ns': int(start_ns),
            'end_ns': int(end_ns),
            'duration_ns': int(span_ns),
            'duration_ms': float(span_ns) / 1_000_000.0,
        }

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
            'ncu_focus_analysis': focus_metrics or {},
            'roofline_estimate': self.roofline_estimate,
        }
        
        # 分析每个ncu结果
        # 若提供焦点聚合指标，则不必对全量 ncu_full_capture_global 逐文件做标准分析（仍可保留 targeted 文件分析）
        for ncu_file in ncu_files:
            csv_file = ncu_file.replace('.ncu-rep', '.csv')
            print(csv_file)
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
            f.write("集成性能分析报告\n\n")
            f.write(f"生成时间: {datetime.now().astimezone().strftime('%Y年%m月%d日')}\n\n")
            roofline = comprehensive_results.get('roofline_estimate') or {}
            if roofline:
                def fmt_time(seconds: Optional[float]) -> str:
                    if seconds is None:
                        return "N/A"
                    if isinstance(seconds, float) and math.isinf(seconds):
                        return "∞"
                    return f"{seconds * 1000:.3f} ms"

                def fmt_perf(value: float) -> str:
                    return f"{value / 1e12:.2f} TOPS"

                def fmt_mem(value: float) -> str:
                    return f"{value / 1e9:.2f} GB"

                def fmt_ops(value: float) -> str:
                    return f"{value / 1e12:.2f} TOPs"

                def fmt_ai(value: float) -> str:
                    return f"{value:.2f} OPs/Byte"

                bits = roofline.get('precision_bits', {})
                params = roofline.get('params', {})
                prefill_r = roofline.get('prefill', {})
                decode_r = roofline.get('decode', {})
                overall_r = roofline.get('overall', {})
                f.write("一、Roofline 预测指标\n")
                f.write(f" 硬件: {roofline.get('hardware', 'unknown')}\n")
                f.write(
                    f" 精度: W{bits.get('w', 'N/A')} / A{bits.get('a', 'N/A')} / KV{bits.get('kv', 'N/A')}\n"
                )
                f.write(
                    f" Prefill 阶段: 强度 {fmt_ai(prefill_r.get('arithmetic_intensity', 0.0))}, "
                    f"受限类型 {prefill_r.get('bound', 'N/A')}, "
                    f"性能 {fmt_perf(prefill_r.get('performance', 0.0))}, "
                    f"预计时长 {fmt_time(prefill_r.get('inference_time'))}, "
                    f"内存访问 {fmt_mem(prefill_r.get('memory_access', 0.0))}\n"
                )
                decode_total_time = decode_r.get('total_time')
                f.write(
                    f" Decode 单token: 强度 {fmt_ai(decode_r.get('arithmetic_intensity', 0.0))}, "
                    f"受限类型 {decode_r.get('bound', 'N/A')}, "
                    f"性能 {fmt_perf(decode_r.get('performance', 0.0))}, "
                    f"单token时长 {fmt_time(decode_r.get('inference_time'))}, "
                    f"总时长(@输出{params.get('output_len', 0)} token) {fmt_time(decode_total_time)}, "
                    f"总内存 {fmt_mem(decode_r.get('total_memory', 0.0))}\n"
                )
                f.write(
                    f" 总体估计: 强度 {fmt_ai(overall_r.get('arithmetic_intensity', 0.0))}, "
                    f"受限类型 {overall_r.get('bound', 'N/A')}, "
                    f"性能 {fmt_perf(overall_r.get('performance', 0.0))}, "
                    f"预计总时长 {fmt_time(overall_r.get('inference_time'))}, "
                    f"总体OPs {fmt_ops(overall_r.get('total_ops', 0.0))}\n\n"
                )
            # nsys概览（优先展示 layer_kernels）
            f.write("二、Nsys 全局性能概览\n")
            nsys_overview = comprehensive_results.get('nsys_overview', {})
            # layer_rows = nsys_overview.get('layer_kernels_rows', [])
            # src_hint = nsys_overview.get('layer_kernels_source')
            # if layer_rows:
            #     if src_hint:
            #         f.write(f"- 来源：{src_hint}\n")
            #     else:
            #         f.write("- 来源：NVTX_EVENTS + StringIds + CUPTI_ACTIVITY_KIND_KERNEL 三表关联\n")
            #     f.write("- 明细见: layer_kernels_run2_hot.json\n\n")
            #     preview = layer_rows[:20]
            #     for r in preview:
            #         f.write(f"- {str(r.get('name',''))[:80]} | {r.get('dur_ms',0)} ms\n")
            #     f.write("\n")
            # # 概览
            kernel_stats = nsys_overview.get('kernel_analysis')
            kernel_time = 0.0
            if kernel_stats:
                kernel_time = kernel_stats.get('total_kernel_time', 0.0)
                f.write(f" 总kernels数量: {kernel_stats.get('total_kernels', 0)}\n")
                f.write(f" 总kernel执行时间: {kernel_time:.2f} ms\n")
                # f.write(f"- 平均kernel执行时间: {kernel_stats.get('avg_kernel_time', 0):.3f} ms\n")
            nvtx_ranges = nsys_overview.get('nvtx_ranges', {})
            for tag, span in nvtx_ranges.items():
                start_ns = span.get('start_ns')
                end_ns = span.get('end_ns')
                duration_ms = span.get('duration_ms')
                if duration_ms is None and isinstance(start_ns, (int, float)) and isinstance(end_ns, (int, float)):
                    duration_ms = (end_ns - start_ns) / 1_000_000.0
                if duration_ms is None:
                    continue
                idle_ratio = 0.0 if not duration_ms else max(duration_ms - kernel_time, 0.0) / duration_ms * 100.0
                f.write(
                    f" {tag} 范围持续时间: {duration_ms:.2f} ms, 空泡率 {idle_ratio:.2f} %\n"
                )
            f.write("\n")
            # 热点kernels（保持顺序，不去重，只显示每条的 dur_ms）
            # f.write(f"## 🔥 Run[2] Kernels（按出现顺序）\n\n")
            total_kernel_time_ms = sum(k.get('dur_ms', 0.0) for k in self.hot_kernels)
            for i, kernel in enumerate(self.hot_kernels, 1):
                display_name = str(kernel.get('name',''))
                dur = float(kernel.get('dur_ms', 0.0))
                percent = (dur / total_kernel_time_ms * 100.0) if total_kernel_time_ms > 0 else 0.0
                f.write(f"{i}. {display_name}\n")
                f.write(f"   - 执行时间: {dur:.3f} ms\n")
                f.write(f"   - 时间占比: {percent:.2f}%\n\n")
            
            # NCU 深度分析
            f.write("三、 NCU 深度分析结果\n\n")
            ncu_analysis = comprehensive_results.get('ncu_detailed_analysis', {})
            focus_analysis = comprehensive_results.get('ncu_focus_analysis', {})
            
            # 写入逐 kernel 详细
            items = []
            for kernel_name, analysis in ncu_analysis.items():
                m = re.match(r'^ncu_kernel_(\d+)_', kernel_name)
                idx = int(m.group(1)) if m else 10**9  # 无前缀的放在最后
                items.append((idx, kernel_name, analysis))
            items.sort(key=lambda x: x[0])

            for i, (idx, kernel_name, analysis) in enumerate(items, 1):
                # 标题改为编号行
                f.write(f"{i}. {kernel_name}\n\n")
                # 基本统计
                f.write(f"   - 识别瓶颈数: {analysis.get('bottlenecks_found', 0)}\n")
                # GPU 利用率
                gu = analysis.get('gpu_utilization', {})
                if gu:
                    f.write(f"   - 平均SM效率: {gu.get('average_sm_efficiency', 'N/A')}\n")
                    f.write(f"   - 最高SM效率: {gu.get('max_sm_efficiency', 'N/A')}\n")
                    f.write(f"   - 最低SM效率: {gu.get('min_sm_efficiency', 'N/A')}\n")
                    f.write(f"   - 低于50%数量: {gu.get('kernels_below_50_percent', 0)} / {gu.get('total_kernels', 0)}\n")
                # 内存分析
                mem = analysis.get('memory_analysis', {})
                bw = mem.get('bandwidth_stats', {})
                if bw:
                    f.write(f"   - 平均带宽: {bw.get('average_bandwidth', 'N/A')} GB/s\n")
                    f.write(f"   - 最高带宽: {bw.get('max_bandwidth', 'N/A')} GB/s\n")
                    f.write(f"   - 最低带宽: {bw.get('min_bandwidth', 'N/A')} GB/s\n")
                l2 = mem.get('l2_cache_stats', {})
                if l2:
                    f.write(f"   - 平均L2命中率: {l2.get('average_l2_hit_rate', 'N/A')}\n")
                    f.write(f"   - 低L2命中率kernel数: {l2.get('kernels_low_l2_hit_rate', 0)}\n")
                l1 = mem.get('l1_cache_stats', {})
                if l1:
                    f.write(f"   - 平均L1命中率: {l1.get('average_l1_hit_rate', 'N/A')}\n")
                    f.write(f"   - 低L1命中率kernel数: {l1.get('kernels_low_l1_hit_rate', 0)}\n")
                # 瓶颈
                bsum = analysis.get('bottleneck_summary', [])
                if bsum:
                    f.write("   - 主要瓶颈:\n")
                    for b in bsum:
                        f.write(f"     - {b.get('description','')} ({b.get('severity','')})\n")
                f.write("\n")

            # 写入焦点聚合（若有）
            if focus_analysis:
                f.write("## 🎯 焦点内核聚合指标 (全量采集提取)\n\n")
                for kname, fan in focus_analysis.items():
                    f.write(f"### {kname}\n\n")
                    gu = fan.get('gpu_utilization', {})
                    if gu:
                        f.write(f"- 平均SM效率: {gu.get('average_sm_efficiency','N/A')}\n")
                        f.write(f"- Occupancy: {gu.get('achieved_occupancy','N/A')}\n")
                    mem = fan.get('memory_analysis', {}).get('bandwidth_stats', {})
                    if mem:
                        f.write(f"- 平均带宽: {mem.get('average_bandwidth','N/A')} GB/s\n")
                        f.write(f"- L2命中率: {mem.get('l2_hit_rate','N/A')}%\n")
                    bsum = fan.get('bottleneck_summary', [])
                    if bsum:
                        f.write("- 主要瓶颈:\n")
                        for b in bsum:
                            f.write(f"  - {b.get('description','')} ({b.get('severity','')})\n")
                    f.write("\n")
                

        print(f"📄 最终报告已生成: {report_file}")
        return str(report_file)

def create_sglang_analysis_workflow():
    """创建SGlang专用的分析工作流"""
    DEFAULT_MODEL_DIR = os.getenv('SGLANG_MODEL_PATH') or os.getenv('MODEL_PATH') or '/workspace/models/'

    def run_sglang_integrated_analysis(model_path: Optional[str] = None, 
                                      batch_size: int = 1,
                                      input_len: int = 128, 
                                      output_len: int = 1,
                                      disable_chunked_prefill: bool = True,
                                      gpu_ids: Optional[List[str]] = None,
                                      hardware: Optional[str] = None,
                                      w_bit: int = 16,
                                      a_bit: int = 16,
                                      kv_bit: Optional[int] = None,
                                      use_flashattention: bool = False):
        print(f"[DEBUG] run_sglang_integrated_analysis entered: bs={batch_size}, in={input_len}, out={output_len}")
        """运行SGlang的集成分析（固定设置：bs=1, in=128, out=1），可在多个GPU上顺序执行"""
        if not model_path:
            model_path = DEFAULT_MODEL_DIR.rstrip('/')
            print(f"ℹ️ 未提供 model_path，使用默认路径: {model_path}")

        gpu_list = gpu_ids or ["0", "1"]
        base_env = os.environ.copy()
        base_env['SGLANG_ENABLE_CHUNKED_PREFILL'] = '0'

        resolved_hardware = hardware or os.getenv('ROOFLINE_HARDWARE') or 'nvidia_H800_SXM5_80G'
        outputs: List[Dict[str, str]] = []
        for gpu_id in gpu_list:
            env = base_env.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            print(f"🔁 在 GPU {gpu_id} 上运行 nsys/ncu 流程 ...")
            sglang_cmd = [
                'python', '-m', 'sglang.bench_one_batch',
                '--model-path', model_path,
                '--batch-size', str(batch_size),
                '--input-len', str(input_len),
                '--output-len', str(output_len),
                '--load-format', 'dummy',
                '--chunked-prefill-size', '0',
                '--disable-cuda-graph'
            ]

            analyzer = NSysToNCUAnalyzer(
                f"sglang_analysis_b{batch_size}_i{input_len}_o{output_len}_gpu{gpu_id}",
                env=env,
            )
            try:
                roofline = compute_roofline(
                    Path(model_path),
                    resolved_hardware,
                    batch_size,
                    input_len,
                    output_len,
                    w_bit=w_bit,
                    a_bit=a_bit,
                    kv_bit=kv_bit,
                    use_flashattention=use_flashattention,
                )
                analyzer.roofline_estimate = roofline
                try:
                    estimate_path = analyzer.output_dir / "roofline_estimate.json"
                    with open(estimate_path, "w", encoding="utf-8") as rf:
                        json.dump(roofline, rf, ensure_ascii=False, indent=2)
                    print(f"📐 Roofline 预测已写入: {estimate_path}")
                except Exception as write_exc:
                    print(f"⚠️ Roofline 结果写入失败: {write_exc}")
                print("📐 Roofline 预测已完成，结果将在最终报告中展示。")
            except Exception as exc:
                print(f"⚠️ Roofline 预测失败: {exc}")
            nsys_file = analyzer.step1_nsys_analysis(sglang_cmd, "sglang_overview")
            hot = analyzer.step2_extract_hot_kernels(nsys_file, top_k=8)
            if not hot:
                print("❌ 未发现热点kernels，分析终止");
                continue
            ncu_files = analyzer.step3_ncu_targeted_analysis(sglang_cmd, hot, max_kernels=len(hot))
            results = analyzer.step4_comprehensive_analysis(ncu_files)
            report_file = analyzer.generate_final_report(results)
            outputs.append({
                "gpu": str(gpu_id),
                "dir": str(analyzer.output_dir)
            })
            print(f"📁 输出目录: {analyzer.output_dir}\n📄 报告: {report_file}")

        return outputs
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

def main():
    import argparse, os
    parser = argparse.ArgumentParser(description='集成 nsys 和 ncu 的性能分析工具')
    # SGlang 参数
    parser.add_argument('--sglang-model', type=str, default=os.getenv('SGLANG_MODEL_PATH') or os.getenv('MODEL_PATH'),
                        help='SGlang模型路径')
    parser.add_argument('--sglang-batch', type=int, default=1, help='SGlang批次大小')
    parser.add_argument('--sglang-input-len', type=int, default=128, help='SGlang输入长度')
    parser.add_argument('--sglang-output-len', type=int, default=1, help='SGlang输出长度')
    # 兼容：保留未知参数但不使用
    known_args, unknown_tail = parser.parse_known_args()
    # 忽略 unknown_tail，统一走工作流
    if unknown_tail:
        print(f"[WARN] 忽略原始目标命令（unknown_tail）：{' '.join(unknown_tail)}")

    try:
        run_workflow = create_sglang_analysis_workflow()
        print("[DEBUG] 调用 create_sglang_analysis_workflow()")
        run_workflow(
            model_path=known_args.sglang_model,
            batch_size=known_args.sglang_batch,
            input_len=known_args.sglang_input_len,
            output_len=known_args.sglang_output_len,
            disable_chunked_prefill=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断分析")
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()

