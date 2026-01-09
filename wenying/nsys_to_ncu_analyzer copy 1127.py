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
        # 纯文本列表（仅名称），便于前端直接读取
        try:
            txt_list_path = self.output_dir / "hot_kernels.txt"
            with open(txt_list_path, 'w', encoding='utf-8') as f_txt:
                for hk in hot_kernels:
                    name_val = hk.get('name') if isinstance(hk, dict) else str(hk)
                    f_txt.write(str(name_val).strip() + '\n')
            print(f"📝 已生成纯文本热点 kernel 名称列表: {txt_list_path}")
        except Exception as e:
            print(f"⚠️ 生成纯文本热点列表失败: {e}")
        # JSON 导出 + 名称增强
        try:
            json_path = parser.export_to_json(self.output_dir / f"{Path(nsys_file).stem}.json")
            json_names = parser.extract_kernel_names_from_json(json_path, limit=800)
            gpu_candidates = parser.filter_gpu_kernel_candidates(json_names)
            self._augment_hot_kernels_with_json_names(gpu_candidates)
        except Exception as e:
            print(f"⚠️ JSON kernel 名增强失败: {e}")
        # CSV kernel summary 导出（更可靠名称来源）
        try:
            csv_summary = parser.export_kernel_summary_csv(nsys_file, self.output_dir / f"{Path(nsys_file).stem}_kernels")
            if csv_summary:
                summary_rows = parser.parse_kernel_summary_csv(csv_summary)
                self._replace_with_csv_kernel_names(summary_rows)
        except Exception as e:
            print(f"⚠️ CSV kernel 汇总获取失败: {e}")
        return hot_kernels

    def _augment_hot_kernels_with_json_names(self, gpu_candidates: List[str]):
        """如果热点 kernel 名都是数字或 __unnamed_，尝试用真实候选名替换前 N 个以便 ncu 匹配。
        保留原字段 'original_name'.
        """
        if not self.hot_kernels or not gpu_candidates:
            return
        # 判断需要增强的比例
        def is_placeholder(name: str):
            return name.isdigit() or name.startswith('__unnamed_')
        placeholders = [hk for hk in self.hot_kernels if is_placeholder(str(hk.get('name','')))]
        if not placeholders:
            return
        replace_count = min(len(placeholders), len(gpu_candidates))
        for i in range(replace_count):
            hk = placeholders[i]
            candidate = gpu_candidates[i]
            hk['original_name'] = hk['name']
            hk['name'] = candidate
        print(f"🔁 已用 {replace_count} 个 JSON 候选名称替换占位/数字 hotspot kernel 名，示例: {[(hk.get('original_name'), hk['name']) for hk in placeholders[:3]]}")

    def _replace_with_csv_kernel_names(self, summary_rows: List[Dict]):
        """根据 kernel summary CSV 中的真实名称，对热点列表进行精准替换。
        优先匹配占位/数字名，或时间显著的非真实名。
        """
        if not summary_rows:
            return
        real_names = [r['name'] for r in summary_rows if r['name']]
        if not real_names:
            return
        # 构建一个迭代器
        idx = 0
        for hk in self.hot_kernels:
            name = str(hk.get('name',''))
            if self._is_placeholder_name(name):
                if idx < len(real_names):
                    hk['original_name'] = name
                    hk['name'] = real_names[idx]
                    hk['csv_substituted'] = True
                    idx += 1
        if idx > 0:
            print(f"🧬 已用 CSV 汇总中的真实名称替换 {idx} 个 hotspot kernel: {[hk['name'] for hk in self.hot_kernels[:idx]]}")
    
    def step3_ncu_targeted_analysis(self, target_command: List[str], 
                                   kernels_to_analyze: List[Dict],
                                   max_kernels: int = 5) -> List[str]:
        """第三步：使用ncu对热点kernels进行深度分析"""
        
        print("⚡ 步骤3: 使用ncu深度分析热点kernels...")
        
        ncu_results = []

        # 若热点kernel名称可疑（数字/unnamed），先用 ncu --list-kernels 自动发现真实名称
        if any(self._is_placeholder_name(str(k.get('name',''))) for k in kernels_to_analyze):
            print("🔍 检测到占位/数字kernel名，触发 ncu --list-kernels 进行真实名称发现...")
            discovered = self.list_kernels_with_ncu(target_command)
            selected = self._select_real_kernels(discovered, max_kernels)
            print(f"🧭 选择用于深度分析的真实kernel名称: {selected}")
            # 替换 kernels_to_analyze 中的 name 字段
            for i, real_name in enumerate(selected):
                if i < len(kernels_to_analyze):
                    kernels_to_analyze[i]['discovered'] = True
                    kernels_to_analyze[i]['original_name'] = kernels_to_analyze[i]['name']
                    kernels_to_analyze[i]['name'] = real_name
                else:
                    # 如果热点列表不够，补充
                    kernels_to_analyze.append({'name': real_name, 'discovered': True, 'total_time_ms': 0, 'avg_time_ms': 0, 'count': 0})

        # 限制分析数量
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
                    res = subprocess.run(attempt_cmd)
                    if res.returncode != 0:
                        # 打印stderr的一部分便于调试
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

            # 构建多层回退策略：
            # 1) 原始名称 + demangled 基准
            # 2) 正则前缀匹配 (减少过长名称精确匹配失败概率)
            # 3) 去掉过滤 (采集所有内核，限制 launch-count 以降开销)
            attempts = []
            # 1 原始精确匹配（demangled）
            attempts.append({
                'tag': 'exact-demangled',
                'cmd': ['ncu', '--kernel-name-base', 'demangled', '--kernel-name', kernel_name,
                        '--rename-kernels=0', '--set', 'full', '-o', str(ncu_profile), '--force-overwrite'] + target_command
            })
            # 2 正则前缀（取前 60 可见字符，去除引号，只保留安全字符）
            prefix_raw = re.sub(r'"', '', kernel_name)[:60]
            # 适度裁剪到第一个右括号或模板结束符，避免过长
            m_end = re.search(r'[)>]$', prefix_raw)
            prefix_clean = prefix_raw
            # 转义正则特殊字符
            prefix_regex = re.sub(r'([\\.^$|?*+\[\](){}])', r'\\\1', prefix_clean)
            attempts.append({
                'tag': 'regex-prefix',
                'cmd': ['ncu', '--kernel-name-base', 'demangled', '--kernel-name', f'regex:^{prefix_regex}',
                        '--rename-kernels=0', '--set', 'full', '-o', str(ncu_profile), '--force-overwrite'] + target_command
            })
            # 3 无过滤（可能较多数据，使用基础指标集 + launch-count 限制）
            attempts.append({
                'tag': 'unfiltered-basic',
                'cmd': ['ncu', '--launch-count', '50', '--set', 'compute', '-o', str(ncu_profile), '--force-overwrite'] + target_command
            })

            produced = None
            for att in attempts:
                print(f"🔁 NCU 尝试: {att['tag']}")
                produced = attempt_profile(att['cmd'], att['tag'])
                if produced:
                    break

            if produced:
                ncu_results.append(produced)
                self._export_ncu_to_csv(produced)
            else:
                print(f"❌ 所有尝试均未生成 NCU 报告: {kernel_name[:80]}")
        
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

    def full_ncu_capture(self, target_command: List[str], profile_name: str = "ncu_full_capture",
                          set_name: str = "compute", launch_limit: Optional[int] = None,
                          timeout: int = 1200) -> Optional[str]:
        """执行一次不做 kernel 过滤的完整 NCU 采集。

        参数:
            target_command: 原始待分析命令 (['python', 'script.py', ...])
            profile_name: 输出报告基名
            set_name: 使用的 NCU 指标集合 (--set)。可选: 'compute', 'full' 等
            launch_limit: 使用 --launch-count 限制采集的 kernel 次数 (降低长任务开销)
            timeout: 超时时间 (秒)

        行为:
            生成 <profile_name>.ncu-rep 及对应的 CSV/JSON (若可能)
            输出路径位于统一的 self.output_dir 下。
        """
        ncu_profile_base = self.output_dir / profile_name
        ncu_rep = str(ncu_profile_base) + '.ncu-rep'
        cmd = ['ncu', '--set', set_name, '-o', str(ncu_profile_base), '--force-overwrite']
        if launch_limit:
            cmd += ['--launch-count', str(launch_limit)]
        # 不加 --kernel-name 过滤, 捕获全部可见内核
        cmd += target_command
        print(f"🌀 全量 NCU 采集: {' '.join(cmd)}")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if res.returncode != 0:
                print(f"⚠️ 全量采集失败(returncode={res.returncode}): {(res.stderr or '')[:300].replace('\n',' ')}")
                return None
            if not Path(ncu_rep).exists():
                print(f"⚠️ 未生成 ncu 报告文件: {ncu_rep}")
                return None
            print(f"✅ 全量 NCU 采集完成: {ncu_rep}")
            # 尝试导出 CSV
            self._export_ncu_to_csv(ncu_rep)
            return ncu_rep
        except subprocess.TimeoutExpired:
            print("⏳ 全量 NCU 采集超时")
        except Exception as e:
            print(f"❌ 全量 NCU 采集异常: {e}")
        return None

    def list_kernels_with_ncu(self, target_command: List[str]) -> List[str]:
        """运行 ncu --list-kernels 以获取实际可分析的 kernel 名称列表"""
        cmd = ['ncu', '--list-kernels'] + target_command
        print(f"🧪 运行: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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

    def _is_placeholder_name(self, name: str) -> bool:
        # 将数字、__unnamed_ 以及若干低信号 / 框架性名字视为占位。增加 __cudart_ 前缀，以便后续用 CSV 真实 kernel 名替换。
        return (
            name.isdigit()
            or name.startswith('__unnamed_')
            or name in ('cudafe++', 'sleep', 'python', 'node')
            or name.startswith('__cudart_')
        )
    
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
                display_name = kernel.get('name','')[:80]
                original_name = kernel.get('original_name')
                csv_flag = ' (CSV替换)' if kernel.get('csv_substituted') else ''
                if original_name and original_name != kernel.get('name'):
                    display_name = f"{original_name} → {kernel.get('name')}" + csv_flag
                f.write(f"{i}. **{display_name}**\n")
                f.write(f"   - 总执行时间: {kernel.get('total_time_ms',0):.2f} ms\n")
                f.write(f"   - 平均执行时间: {kernel.get('avg_time_ms',0):.3f} ms\n") 
                f.write(f"   - 调用次数: {kernel.get('count',0)}\n")
                if kernel.get('discovered'):
                    f.write(f"   - 名称来源: ncu --list-kernels 发现\n")
                f.write("\n")
            
            # ncu深度分析
            f.write("## ⚡ NCU 深度分析结果\n\n")
            ncu_analysis = comprehensive_results.get('ncu_detailed_analysis', {})
            focus_analysis = comprehensive_results.get('ncu_focus_analysis', {})
            
            for kernel_name, analysis in ncu_analysis.items():
                f.write(f"### {kernel_name}\n\n")
            if focus_analysis:
                f.write("## 🎯 焦点内核聚合指标 (全量采集提取)\n\n")
                for kname, analysis in focus_analysis.items():
                    f.write(f"### {kname}\n\n")
                    gu = analysis.get('gpu_utilization', {})
                    if gu:
                        f.write(f"- 平均SM效率: {gu.get('average_sm_efficiency','N/A')}\n")
                        f.write(f"- Occupancy: {gu.get('achieved_occupancy','N/A')}\n")
                    mem = analysis.get('memory_analysis', {}).get('bandwidth_stats', {})
                    if mem:
                        f.write(f"- 平均带宽: {mem.get('average_bandwidth','N/A')} GB/s\n")
                        f.write(f"- L2命中率: {mem.get('l2_hit_rate','N/A')}%\n")
                    bsum = analysis.get('bottleneck_summary', [])
                    if bsum:
                        f.write("- 主要瓶颈:\n")
                        for b in bsum:
                            f.write(f"  - {b['description']} ({b['severity']})\n")
                    f.write('\n')
                
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

