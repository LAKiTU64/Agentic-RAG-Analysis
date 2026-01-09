#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent核心模块 - 集成NSys和NCU性能分析
"""

import re
import os
import sys
import json
import asyncio
import subprocess
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from math import gcd, lcm
from fractions import Fraction

# 导入分析工具
from utils.nsys_to_ncu_analyzer import NSysToNCUAnalyzer, create_sglang_analysis_workflow
from .offline_llm import get_offline_qwen_client

try:
    from .utils.roofline_estimator import compute_roofline  # type: ignore
except Exception:
    try:
        from utils.roofline_estimator import compute_roofline  # type: ignore
    except Exception:
        compute_roofline = None  # type: ignore

OFFLINE_QWEN_PATH = Path(os.getenv("QWEN_LOCAL_MODEL_PATH", "/workspace/Qwen3-32B"))

class AIAgent:
    """AI Agent核心类 - 自动化性能分析"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sglang_path = Path(config.get('sglang_path', 'SGlang'))
        self.models_path = Path(config.get('models_path', 'models'))
        self.model_mappings = config.get('model_mappings', {})
        self.results_dir = Path(config.get('output', {}).get('results_dir', 'analysis_results'))
        self.results_dir.mkdir(exist_ok=True)
        
        # 分析工具配置
        self.profiling_config = config.get('profiling_tools', {})
        self.analysis_defaults = config.get('analysis_defaults', {})

        # 缓存最近一次分析的关键信息，便于对外接口复用
        self.last_analysis_dir: Optional[str] = None
        self.last_analysis_dirs: List[str] = []
        self.last_analysis_reports: List[str] = []
        self.last_analysis_table: Optional[str] = None
        self.last_analysis_suggestions: Optional[str] = None
        self.last_roofline_estimate: Optional[Dict[str, Any]] = None
        
    async def process_message(self, message: str) -> str:
        """处理用户消息并执行分析"""
        
        # 提取模型名称
        model_name = self._extract_model_name(message)
        
        # 提取分析类型
        analysis_type = self._extract_analysis_type(message)
        
        # 提取参数
        params = self._extract_parameters(message)
        
        # 如果没有提供参数，使用默认值
        if not params.get('batch_size'):
            params['batch_size'] = self.analysis_defaults.get('batch_size', [1])
        if not params.get('input_len'):
            params['input_len'] = self.analysis_defaults.get('input_len', [128])
        if not params.get('output_len'):
            params['output_len'] = self.analysis_defaults.get('output_len', [1])
        
        # 生成初始响应
        response = f"""✅ **已解析您的请求**

🤖 **模型**: {model_name or '未指定'}
🔬 **分析类型**: {analysis_type}
📊 **参数**:
  • batch_size: {params.get('batch_size', [])}
  • input_len: {params.get('input_len', [])}
  • output_len: {params.get('output_len', [])}

"""
        
        # 如果模型名称明确，执行实际分析
        if model_name:
            # 获取模型路径
            model_path = self._resolve_model_path(model_name)
            
            if not model_path:
                response += f"""
❌ **错误**: 未找到模型 '{model_name}'
📋 可用模型: {', '.join(self.model_mappings.keys())}

💡 **提示**: 请在 config.yaml 中配置模型路径
"""
                return response
            
            response += f"""🚀 **开始分析...**

📁 模型路径: {model_path}
⏳ 预计时间: 3-10分钟（取决于参数组合数量）

"""
            
            # 执行分析（异步）
            try:
                analysis_results = await self._run_analysis(
                    model_path=model_path,
                    analysis_type=analysis_type,
                    params=params
                )
                
                response += analysis_results
                
            except Exception as e:
                response += f"""
❌ **分析失败**: {str(e)}

💡 **可能原因**:
1. NSys/NCU工具未安装或未在PATH中
2. 模型路径不正确
3. GPU不可用或驱动问题
4. 参数配置错误

🔧 **调试步骤**:
1. 运行 `nsys --version` 和 `ncu --version` 检查工具
2. 运行 `nvidia-smi` 检查GPU
3. 检查模型路径是否存在
"""
        else:
            response += """
💡 **下一步**:
请指定要分析的模型名称，例如：
• "分析 llama-7b"
• "对 qwen-14b 进行性能分析"
• "使用 ncu 深度分析 chatglm-6b"

📋 **可用模型**: """ + ', '.join(self.model_mappings.keys())
        
        return response
    
    async def _run_analysis(self, model_path: str, analysis_type: str, params: Dict) -> str:
        """执行实际的性能分析"""
        
        results = []

        # 重置缓存，防止使用陈旧结果
        self.last_analysis_table = None
        self.last_analysis_reports = []
        self.last_analysis_dirs = []
        self.last_analysis_dir = None
        self.last_analysis_suggestions = None
        self.last_roofline_estimate = None
        
        # 获取参数组合
        batch_sizes = params.get('batch_size', [1])
        input_lens = params.get('input_len', [128])
        output_lens = params.get('output_len', [1])
        
        # 只分析第一组参数（避免时间过长）
        batch_size = batch_sizes[0] if isinstance(batch_sizes, list) else batch_sizes
        input_len = input_lens[0] if isinstance(input_lens, list) else input_lens
        output_len = output_lens[0] if isinstance(output_lens, list) else output_lens

        precision_cfg = self.analysis_defaults.get('precision', {}) if isinstance(self.analysis_defaults.get('precision', {}), dict) else {}

        def _parse_int(value, default):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        w_bit = _parse_int(precision_cfg.get('w_bit'), 16)
        a_bit = _parse_int(precision_cfg.get('a_bit'), 16)
        kv_bit_candidate = precision_cfg.get('kv_bit')
        parsed_kv_bit = _parse_int(kv_bit_candidate, None) if kv_bit_candidate is not None else None
        kv_bit = parsed_kv_bit if isinstance(parsed_kv_bit, int) else None
        use_flashattention = bool(precision_cfg.get('use_flashattention', False))
        hardware_key = self.analysis_defaults.get('hardware') or os.getenv('ROOFLINE_HARDWARE') or 'nvidia_H800_SXM5_80G'
        
        try:
            # 创建分析工作流
            analysis_workflow = create_sglang_analysis_workflow()
            
            # 执行分析
            loop = asyncio.get_event_loop()
            workflow_callable = partial(
                analysis_workflow,
                str(model_path),
                batch_size,
                input_len,
                output_len,
                True,
                None,
                hardware_key,
                w_bit,
                a_bit,
                kv_bit,
                use_flashattention,
            )
            workflow_output = await loop.run_in_executor(None, workflow_callable)

            run_records: List[Tuple[str, Path]] = []
            if isinstance(workflow_output, list):
                for idx, item in enumerate(workflow_output):
                    gpu_label: str
                    output_path: Optional[str] = None
                    if isinstance(item, dict):
                        gpu_label = str(item.get('gpu', idx))
                        output_path = item.get('dir') or item.get('path')
                    else:
                        gpu_label = str(idx)
                        output_path = str(item)
                    if output_path:
                        run_records.append((gpu_label, Path(output_path)))
            elif workflow_output:
                run_records.append(("0", Path(str(workflow_output))))

            if not run_records:
                results.append("⚠️ **分析已完成，但未找到输出目录**")
                return '\n'.join(results)

            self.last_analysis_dirs = [str(path) for _, path in run_records]

            report_infos = []
            roofline_infos: List[Tuple[str, Path, Dict[str, Any]]] = []
            missing_reports = []
            for idx, (gpu_label, output_dir) in enumerate(run_records):
                report_path = output_dir / "integrated_performance_report.md"
                if report_path.exists():
                    report_text = report_path.read_text(encoding='utf-8')
                    report_infos.append({
                        'gpu': gpu_label,
                        'dir': output_dir,
                        'report': report_path,
                        'text': report_text,
                        'index': idx
                    })
                    roofline_path = output_dir / "roofline_estimate.json"
                    if roofline_path.exists():
                        try:
                            with open(roofline_path, 'r', encoding='utf-8') as rf:
                                roofline_data = json.load(rf)
                            roofline_infos.append((gpu_label, roofline_path, roofline_data))
                        except Exception as roof_exc:
                            print(f"⚠️ 读取 Roofline 预测失败 ({roofline_path}): {roof_exc}")
                else:
                    missing_reports.append(output_dir)

            if not report_infos:
                dir_lines = '\n'.join(f"  • {path}" for _, path in run_records)
                results.append(f"""
⚠️ **分析已完成，但未生成报告文件**

📁 结果目录:
{dir_lines}
💡 请检查目录中的其他输出文件
""")
                return '\n'.join(results)

            primary_info = report_infos[0]
            self.last_analysis_dir = str(primary_info['dir'])
            self.last_analysis_reports = [str(info['report']) for info in report_infos]
            summary = self._extract_report_summary(primary_info['text'])

            try:
                loop = asyncio.get_event_loop()
                if len(report_infos) > 1:
                    table_markdown = self._generate_multi_gpu_table(
                        [info['text'] for info in report_infos],
                        [info['gpu'] for info in report_infos]
                    )
                else:
                    table_markdown = await loop.run_in_executor(
                        None,
                        self._generate_report_table,
                        primary_info['text']
                    )
            except Exception as table_exc:
                table_markdown = f"⚠️ 表格生成失败: {table_exc}"

            self.last_analysis_table = table_markdown

            suggestions = self._generate_optimization_suggestions(report_infos)
            self.last_analysis_suggestions = suggestions if suggestions else None

            dir_lines = '\n'.join(
                f"  • {self._format_gpu_label(info['gpu'], info['index'])}: {info['dir']}" for info in report_infos
            )

            missing_lines = ''
            if missing_reports:
                missing_lines = '\n'.join(f"  • {path}" for path in missing_reports)
                missing_lines = f"\n⚠️ 未找到以下目录的报告文件:\n{missing_lines}\n"

            roofline_section = "📐 **Roofline 预测**:\n暂未生成 Roofline 预测\n"
            if roofline_infos:
                self.last_roofline_estimate = roofline_infos[0][2]
                roofline_preview = self._render_roofline_preview(self.last_roofline_estimate)
                roofline_source = str(roofline_infos[0][1])
                roofline_section = (
                    f"📐 **Roofline 预测** (来源: {roofline_source}):\n{roofline_preview}\n"
                )

            results.append(f"""
✅ **分析完成!**

📁 **结果目录**:
{dir_lines}
📄 **报告文件**: {primary_info['report']}
{missing_lines}
{summary}

{roofline_section}

📌 **热点Kernel表格预览**:
{table_markdown}

💡 **优化建议**:
{suggestions or '暂未生成优化建议'}

🔍 **详细报告**: 请查看 {primary_info['report']}
📊 **可视化图表**: 请查看对应结果目录中的图片文件
""")
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            results.append(f"""
❌ **分析执行失败**

错误信息: {str(e)}

详细错误:
```
{error_detail}
```

💡 **常见问题解决**:
1. 确保已安装 nsys 和 ncu 工具
2. 确保 SGlang 已正确安装
3. 确保模型文件路径正确
4. 确保有足够的 GPU 内存
""")
        
        return '\n'.join(results)

    def _render_roofline_preview(self, roofline: Dict[str, Any]) -> str:
        if not roofline:
            return "暂未生成 Roofline 数据"

        def _fmt_time(seconds: Optional[float]) -> str:
            if seconds is None:
                return "N/A"
            if isinstance(seconds, (int, float)):
                if seconds == float('inf'):
                    return "∞"
                return f"{seconds * 1000:.3f} ms"
            return str(seconds)

        def _fmt_perf(value: Optional[float]) -> str:
            if not isinstance(value, (int, float)):
                return "N/A"
            return f"{value / 1e12:.2f} TOPS"

        def _fmt_ai(value: Optional[float]) -> str:
            if not isinstance(value, (int, float)):
                return "N/A"
            return f"{value:.2f} OPs/Byte"

        def _fmt_mem(value: Optional[float]) -> str:
            if not isinstance(value, (int, float)):
                return "N/A"
            return f"{value / 1e9:.2f} GB"

        bits = roofline.get('precision_bits', {})
        params = roofline.get('params', {})
        prefill = roofline.get('prefill', {})
        decode = roofline.get('decode', {})
        overall = roofline.get('overall', {})

        lines = []
        lines.append(f"- 模型硬件: {roofline.get('hardware', 'unknown')}")
        lines.append(
            f"- 精度: W{bits.get('w', 'N/A')} / A{bits.get('a', 'N/A')} / KV{bits.get('kv', 'N/A')}"
        )
        lines.append(
            f"- Prefill: 强度 {_fmt_ai(prefill.get('arithmetic_intensity'))}, "
            f"性能 {_fmt_perf(prefill.get('performance'))}, "
            f"耗时 {_fmt_time(prefill.get('inference_time'))}, "
            f"内存访问 {_fmt_mem(prefill.get('memory_access'))}"
        )
        lines.append(
            f"- Decode(单 token): 强度 {_fmt_ai(decode.get('arithmetic_intensity'))}, "
            f"性能 {_fmt_perf(decode.get('performance'))}, "
            f"耗时 {_fmt_time(decode.get('inference_time'))}"
        )
        lines.append(
            f"- 总体估计: 强度 {_fmt_ai(overall.get('arithmetic_intensity'))}, "
            f"性能 {_fmt_perf(overall.get('performance'))}, "
            f"总耗时 {_fmt_time(overall.get('inference_time'))}"
        )
        if params:
            lines.append(
                f"- 分析参数: batch={params.get('batch_size')}, prompt={params.get('prompt_len')}, output={params.get('output_len')}"
            )
        return '\n'.join(lines)

    @staticmethod
    def _generate_report_table(report_text: str) -> str:
        client = get_offline_qwen_client(OFFLINE_QWEN_PATH)
        return client.report_to_table(report_text)

    @staticmethod
    def _collect_ncu_csv_snippets(output_dir: Path, limit: int = 1200) -> List[Tuple[str, str]]:
        snippets: List[Tuple[str, str]] = []
        if not output_dir.exists():
            return snippets
        for csv_path in sorted(output_dir.glob("ncu_kernel*.csv")):
            try:
                raw = csv_path.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            snippet = raw[:limit]
            if snippet.strip():
                snippets.append((csv_path.name, snippet))
        return snippets

    def _generate_optimization_suggestions(self, report_infos: List[Dict[str, str]]) -> str:
        if not report_infos:
            return ""

        try:
            client = get_offline_qwen_client(OFFLINE_QWEN_PATH)
        except Exception as exc:
            return f"⚠️ 优化建议生成失败: {exc}"

        labeled_reports: List[Tuple[str, str]] = []
        raw_snippets: List[Tuple[str, str]] = []
        for info in report_infos:
            label = self._format_gpu_label(info['gpu'], info['index'])
            labeled_reports.append((label, info.get('text', '')))
            output_dir = Path(info['dir'])
            for name, data in self._collect_ncu_csv_snippets(output_dir):
                raw_snippets.append((f"{label} / {name}", data))

        output_sections: List[str] = []

        try:
            if labeled_reports:
                suggestions = client.suggest_optimizations(labeled_reports)
                if suggestions:
                    output_sections.append(suggestions)
        except Exception as exc:
            output_sections.append(f"⚠️ 优化建议生成失败: {exc}")

        try:
            if raw_snippets:
                raw_suggestions = client.suggest_raw_data_optimizations(raw_snippets, max_new_tokens=1024)
                if raw_suggestions:
                    output_sections.append(f"📊 原始数据建议:\n{raw_suggestions}")
        except Exception as exc:
            output_sections.append(f"⚠️ 原始数据建议生成失败: {exc}")

        return "\n\n".join(output_sections).strip()

    def _generate_multi_gpu_table(self, report_texts: List[str], gpu_labels: List[str]) -> str:
        if not report_texts:
            return "⚠️ 未找到可用的报告内容"

        formatted_labels: List[str] = []
        per_gpu_tables: List[Tuple[str, str]] = []
        for idx, report_text in enumerate(report_texts):
            raw_label = gpu_labels[idx] if idx < len(gpu_labels) else str(idx)
            label = self._format_gpu_label(raw_label, idx)
            formatted_labels.append(label)
            try:
                table_md = self._generate_report_table(report_text)
            except Exception:
                table_md = ""
            per_gpu_tables.append((label, table_md))

        try:
            client = get_offline_qwen_client(OFFLINE_QWEN_PATH)
            if any(table for _, table in per_gpu_tables):
                merged = client.merge_gpu_tables(per_gpu_tables)
                if merged and merged.count('|') >= len(formatted_labels) * 2:
                    return merged
        except Exception:
            pass

        return self._generate_multi_gpu_table_python(report_texts, gpu_labels)

    def _generate_multi_gpu_table_python(self, report_texts: List[str], gpu_labels: List[str]) -> str:
        if not report_texts:
            return "⚠️ 未找到可用的报告内容"

        parsed_entries = [self._parse_kernel_entries_from_report(text) for text in report_texts]
        if not parsed_entries or not parsed_entries[0]:
            return "⚠️ 未能解析多GPU表格数据"

        label_cells = [self._format_gpu_label(lbl, idx) for idx, lbl in enumerate(gpu_labels)]
        header_cells = ["Kernel"]
        last_index = len(label_cells) - 1
        for idx_lbl, lbl in enumerate(label_cells):
            header_cells.append(f"{lbl} Duration(ms)")
            add_ratio = len(label_cells) == 1 or idx_lbl != last_index
            if add_ratio:
                header_cells.append(f"{lbl} Ratio(%)")
        if len(label_cells) >= 2:
            header_cells.append(f"{label_cells[0]}：{label_cells[1]} 时间占比")

        header = "| " + " | ".join(header_cells) + " |"
        divider = "| " + " | ".join(["---"] * len(header_cells)) + " |"

        max_len = max(len(entries) for entries in parsed_entries)
        rows = []
        def _parse_duration(val: str) -> float:
            try:
                return float(val)
            except (TypeError, ValueError):
                return 0.0
        def _parse_ratio_component(val: str) -> Optional[Fraction]:
            if val is None:
                return None
            text = str(val).strip()
            if not text:
                return None
            try:
                frac = Fraction(text)
            except (ValueError, ZeroDivisionError):
                return None
            if frac < 0:
                return None
            return frac
        def _fractions_to_ints(fracs: List[Optional[Fraction]]) -> List[Optional[int]]:
            positives = [f for f in fracs if isinstance(f, Fraction) and f > 0]
            scale = None
            if positives:
                scale = positives[0].denominator
                for frac in positives[1:]:
                    scale = lcm(scale, frac.denominator)
            ints: List[Optional[int]] = []
            for frac in fracs:
                if frac is None:
                    ints.append(None)
                elif frac == 0:
                    ints.append(0)
                else:
                    if scale is None:
                        scale = frac.denominator
                    ints.append(frac.numerator * (scale // frac.denominator))
            if positives:
                common = None
                for val in ints:
                    if isinstance(val, int) and val > 0:
                        common = val if common is None else gcd(common, val)
                if common and common > 1:
                    ints = [val // common if isinstance(val, int) and val > 0 else val for val in ints]
            return ints
        for idx in range(max_len):
            name_candidates = []
            for entries in parsed_entries:
                if idx < len(entries) and entries[idx]['name']:
                    name_candidates.append(entries[idx]['name'])
            base_name = name_candidates[0] if name_candidates else f"Kernel {idx + 1}"
            alt_names = {nm for nm in name_candidates if nm != base_name}
            if alt_names:
                merged_name = base_name + " / " + " / ".join(sorted(alt_names))
            else:
                merged_name = base_name

            row_cells = [merged_name]
            duration_values: List[float] = []
            pair_ratios: Optional[List[Optional[Fraction]]] = [None, None] if len(parsed_entries) >= 2 else None
            for gpu_idx, entries in enumerate(parsed_entries):
                if idx < len(entries):
                    duration = entries[idx]['duration']
                    ratio = entries[idx]['ratio']
                    row_cells.append(duration)
                    duration_values.append(_parse_duration(duration))
                    add_ratio = len(parsed_entries) == 1 or gpu_idx != last_index
                    if add_ratio:
                        row_cells.append(ratio)
                    if pair_ratios is not None and gpu_idx < 2:
                        pair_ratios[gpu_idx] = _parse_ratio_component(ratio)
                else:
                    row_cells.append('')
                    add_ratio = len(parsed_entries) == 1 or gpu_idx != last_index
                    if add_ratio:
                        row_cells.append('')
                    if pair_ratios is not None and gpu_idx < 2:
                        pair_ratios[gpu_idx] = Fraction(0, 1)
            if pair_ratios is not None:
                simplified_ints = _fractions_to_ints(pair_ratios)
                pair_strings = []
                for val in simplified_ints:
                    if val is None:
                        pair_strings.append('')
                    else:
                        pair_strings.append(str(val))
                combined = f"{pair_strings[0]}：{pair_strings[1]}" if any(pair_strings) else ''
                row_cells.append(combined)
            sort_key = max(duration_values) if duration_values else 0.0
            rows.append((sort_key, row_cells))
        sorted_rows = ["| " + " | ".join(cells) + " |" for _, cells in sorted(rows, key=lambda item: item[0], reverse=True)]

        return "\n".join([header, divider, *sorted_rows])

    def _parse_kernel_entries_from_report(self, report_text: str) -> List[Dict[str, str]]:
        entries: List[Dict[str, str]] = []
        lines = report_text.splitlines()
        idx = 0
        total_lines = len(lines)
        while idx < total_lines:
            raw_line = lines[idx]
            if raw_line.strip().startswith('二、'):
                break
            match = re.match(r'^\s*\d+\.\s+(.*)$', raw_line)
            if match:
                name = match.group(1).strip()
                duration = ''
                ratio = ''
                idx += 1
                while idx < total_lines:
                    line = lines[idx].strip()
                    if line.startswith('- 执行时间'):
                        dur_match = re.search(r'([0-9.]+)\s*ms', line)
                        if dur_match:
                            duration = dur_match.group(1)
                    elif line.startswith('- 时间占比'):
                        ratio_match = re.search(r'([0-9.]+)\s*%', line)
                        if ratio_match:
                            ratio = ratio_match.group(1)
                    elif re.match(r'^\s*\d+\.', lines[idx]) or line.startswith('二、'):
                        break
                    idx += 1
                entries.append({
                    'name': name,
                    'duration': duration,
                    'ratio': ratio
                })
            else:
                idx += 1
        return entries

    @staticmethod
    def _format_gpu_label(label: str, index: int) -> str:
        if not label:
            return f"GPU{index}"
        normalized = label.strip()
        if not normalized:
            return f"GPU{index}"
        if normalized.lower().startswith('gpu'):
            return normalized.upper()
        return f"GPU{normalized}"
    
    def _extract_report_summary(self, report_content: str) -> str:
        """从报告中提取关键摘要信息"""
        
        lines = report_content.split('\n')
        summary_lines = []
        
        # 提取关键统计信息
        for i, line in enumerate(lines):
            if '总kernels数量' in line or '总kernel执行时间' in line:
                summary_lines.append(line)
            elif '🔥 识别的热点Kernels' in line:
                # 提取前3个热点kernel
                summary_lines.append("\n**🔥 热点Kernels (Top 3):**")
                for j in range(i+1, min(i+10, len(lines))):
                    if lines[j].strip() and lines[j].startswith(('1.', '2.', '3.')):
                        summary_lines.append(lines[j][:100])
                break
        
        if summary_lines:
            return '\n'.join(summary_lines)
        else:
            return "**📊 分析报告已生成，请查看详细文件**"
    
    def _resolve_model_path(self, model_name: str) -> Optional[str]:
        """解析模型路径"""
        
        # 检查是否在映射表中
        if model_name in self.model_mappings:
            mapped_path = self.model_mappings[model_name]
            
            # 如果是绝对路径，直接返回
            if Path(mapped_path).is_absolute():
                return mapped_path
            
            # 否则，相对于 models_path
            full_path = self.models_path / mapped_path
            return str(full_path)
        
        # 如果不在映射表中，尝试直接作为路径
        if Path(model_name).exists():
            return model_name
        
        # 尝试相对于 models_path
        potential_path = self.models_path / model_name
        if potential_path.exists():
            return str(potential_path)
        
        return None
    
    def _extract_model_name(self, prompt: str) -> Optional[str]:
        """提取模型名称"""
        
        # 首先检查已知的模型别名
        for model_name in self.model_mappings.keys():
            if model_name.lower() in prompt.lower():
                return model_name
        
        # 然后使用正则表达式匹配通用模型名称模式
        patterns = [
            r'llama[^/\s]*-?\d*[^/\s]*-?\d+[bB]?',
            r'qwen[^/\s]*-?\d*[^/\s]*-?\d+[bB]?',
            r'chatglm[^/\s]*-?\d+[bB]?',
            r'baichuan[^/\s]*-?\d+[bB]?',
            r'vicuna[^/\s]*-?\d+[bB]?',
            r'mistral[^/\s]*-?\d+[bB]?',
            r'mixtral[^/\s]*-?\d+[bB]?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _extract_analysis_type(self, prompt: str) -> str:
        """提取分析类型"""
        prompt_lower = prompt.lower()
        
        if 'ncu' in prompt_lower or 'kernel' in prompt_lower or '深度' in prompt_lower or 'nsight compute' in prompt_lower:
            return 'ncu (深度kernel分析)'
        elif 'nsys' in prompt_lower or '全局' in prompt_lower or 'nsight systems' in prompt_lower:
            return 'nsys (全局性能分析)'
        elif '集成' in prompt_lower or '综合' in prompt_lower or '完整' in prompt_lower:
            return 'auto (集成分析: nsys + ncu)'
        else:
            return 'auto (集成分析: nsys + ncu)'
    
    def _extract_parameters(self, prompt: str) -> Dict:
        """提取参数"""
        params = {}
        
        # 提取batch_size
        batch_match = re.search(r'batch[-_\s]*size?[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)', prompt, re.IGNORECASE)
        if batch_match:
            batch_sizes = [int(x.strip()) for x in re.split(r'[,，\s]+', batch_match.group(1)) if x.strip()]
            params['batch_size'] = batch_sizes
        
        # 提取input_len
        input_match = re.search(r'input[-_\s]*len[gth]*[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)', prompt, re.IGNORECASE)
        if input_match:
            input_lens = [int(x.strip()) for x in re.split(r'[,，\s]+', input_match.group(1)) if x.strip()]
            params['input_len'] = input_lens
        
        # 提取output_len
        output_match = re.search(r'output[-_\s]*len[gth]*[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)', prompt, re.IGNORECASE)
        if output_match:
            output_lens = [int(x.strip()) for x in re.split(r'[,，\s]+', output_match.group(1)) if x.strip()]
            params['output_len'] = output_lens
        
        return params

    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        return list(self.model_mappings.keys())
    
    def get_analysis_status(self) -> Dict:
        """获取当前分析状态"""
        return {
            'available_models': self.get_available_models(),
            'results_directory': str(self.results_dir),
            'nsys_enabled': self.profiling_config.get('nsys', {}).get('enabled', True),
            'ncu_enabled': self.profiling_config.get('ncu', {}).get('enabled', True),
        }
