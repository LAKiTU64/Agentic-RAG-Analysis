import asyncio
import os
import re
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from offline_llm_v3 import get_offline_qwen_client
from utils.nsys_to_ncu_analyzer import create_sglang_analysis_workflow


class PerformanceAnalyzer:
    """
    从原 agent_core.py 迁移出来的性能分析模块，封装成class。
    方便性能分析的管理。
    尽量不改变原有逻辑与输出内容，只做最小封装与依赖注入。

    依赖：
    - llm_client: 用于生成表格/建议（保持原实现）
    - workflow_factory: create_sglang_analysis_workflow (函数)
    """

    def __init__(
        self,
        llm_client: Any,
        workflow_factory: Any,
        results_dir: Union[str, Path] = "results",
    ):
        self.llm_client = llm_client

        # SGLang workflow 函数
        self.workflow_factory = workflow_factory

        # 输出目录
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # 分析结果缓存
        self.last_analysis_dir: Optional[str] = None
        self.last_analysis_dirs: List[str] = []
        self.last_analysis_reports: List[str] = []
        self.last_analysis_table: Optional[str] = None
        self.last_analysis_suggestions: Optional[str] = None

    async def run_analysis(
        self, model_path: str, analysis_type: str, params: Dict
    ) -> str:
        """
        执行性能分析
        注意：analysis_type 参数不参与逻辑分支（保持原代码行为）。
        """

        results = []

        # 参数提取（只取第一组参数；analysis_type 不生效）
        batch_sizes = params.get("batch_size", [1])
        input_lens = params.get("input_len", [128])
        output_lens = params.get("output_len", [1])

        # 只分析第1组参数（避免时间过长）
        batch_size = batch_sizes[0] if isinstance(batch_sizes, list) else batch_sizes
        input_len = input_lens[0] if isinstance(input_lens, list) else input_lens
        output_len = output_lens[0] if isinstance(output_lens, list) else output_lens

        try:
            # 创建分析工作流
            analysis_workflow = self.workflow_factory()
            workflow_output = await asyncio.get_event_loop().run_in_executor(
                None,
                analysis_workflow,
                str(model_path),
                batch_size,
                input_len,
                output_len,
            )

            run_records: List[Tuple[str, Path]] = []

            # 检查输出是否为列表（处理多卡或多次运行的情况）
            if isinstance(workflow_output, list):
                for idx, item in enumerate(workflow_output):
                    output_path = (
                        item.get("dir") or item.get("path")
                        if isinstance(item, dict)
                        else str(item)
                    )
                    gpu_label = (
                        str(item.get("gpu", idx))
                        if isinstance(item, dict)
                        else str(idx)
                    )
                    if output_path:
                        run_records.append((gpu_label, Path(output_path)))
            elif workflow_output:
                run_records.append(("0", Path(str(workflow_output))))

            if not run_records:
                results.append("⚠️ **分析已完成，但未找到输出目录**")
                return "\n".join(results)

            report_infos = []
            missing_reports = []
            for idx, (gpu_label, output_dir) in enumerate(run_records):
                report_path = output_dir / "integrated_performance_report.md"
                if report_path.exists():
                    report_text = report_path.read_text(encoding="utf-8")
                    report_infos.append(
                        {
                            "gpu": gpu_label,
                            "dir": output_dir,
                            "report": report_path,
                            "text": report_text,
                            "index": idx,
                        }
                    )
                else:
                    missing_reports.append(output_dir)

            if not report_infos:
                dir_lines = "\n".join(f"  • {path}" for _, path in run_records)
                results.append(
                    f"""
                    ⚠️ **分析已完成，但未生成报告文件**

                    📁 结果目录:
                    {dir_lines}
                    💡 请检查目录中的其他输出文件
                """
                )
                return "\n".join(results)

            primary_info = report_infos[0]
            self.last_analysis_dir = str(primary_info["dir"])
            self.last_analysis_reports = [str(info["report"]) for info in report_infos]
            summary = self._extract_report_summary(primary_info["text"])

            try:
                loop = asyncio.get_event_loop()
                if len(report_infos) > 1:
                    table_markdown = self._generate_multi_gpu_table(
                        [info["text"] for info in report_infos],
                        [info["gpu"] for info in report_infos],
                    )
                else:
                    table_markdown = await loop.run_in_executor(
                        None, self._generate_report_table, primary_info["text"]
                    )
            except Exception as table_exc:
                table_markdown = f"⚠️ 表格生成失败: {table_exc}"

            self.last_analysis_table = table_markdown

            suggestions = self._generate_optimization_suggestions(report_infos)
            self.last_analysis_suggestions = suggestions if suggestions else None

            dir_lines = "\n".join(
                f"  • {self._format_gpu_label(info['gpu'], info['index'])}: {info['dir']}"
                for info in report_infos
            )

            missing_lines = ""
            if missing_reports:
                missing_lines = "\n".join(f"  • {path}" for path in missing_reports)
                missing_lines = f"\n⚠️ 未找到以下目录的报告文件:\n{missing_lines}\n"

            results.append(
                f"""
                ✅ **分析完成!**

                📁 **结果目录**:
                {dir_lines}
                📄 **报告文件**: {primary_info["report"]}
                {missing_lines}
                {summary}

                📌 **热点Kernel表格预览**:
                {table_markdown}

                💡 **优化建议**:
                {suggestions or "暂未生成优化建议"}

                🔍 **详细报告**: 请查看 {primary_info["report"]}
                📊 **可视化图表**: 请查看对应结果目录中的图片文件
            """
            )

        except Exception as e:
            import traceback

            error_detail = traceback.format_exc()
            results.append(
                f"""
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
            """
            )

        return "\n".join(results)

    def _extract_report_summary(self, report_content: str) -> str:
        """从报告中提取关键摘要信息"""

        lines = report_content.split("\n")
        summary_lines = []

        # 提取关键统计信息
        for i, line in enumerate(lines):
            if "总kernels数量" in line or "总kernel执行时间" in line:
                summary_lines.append(line)
            elif "🔥 识别的热点Kernels" in line:
                # 提取前3个热点kernel
                summary_lines.append("\n**🔥 热点Kernels (Top 3):**")
                for j in range(i + 1, min(i + 10, len(lines))):
                    if lines[j].strip() and lines[j].startswith(("1.", "2.", "3.")):
                        summary_lines.append(lines[j][:100])
                break

        return "\n".join(summary_lines) if summary_lines else ""

    def _generate_multi_gpu_table(
        self, report_texts: List[str], gpu_labels: List[str]
    ) -> str:
        """生成多GPU报告表格（注意：原实现并未真正对齐多 GPU 数据）"""

        if not report_texts:
            return "⚠️ 未找到可用的报告内容"

        entries = self._parse_kernel_entries_from_report(report_texts[0])
        header = (
            "| Kernel | " + " | ".join([f"{lbl} Duration" for lbl in gpu_labels]) + " |"
        )
        sep = "|---" * (len(gpu_labels) + 1) + "|"
        rows = []
        for entry in entries[:5]:  # Top 5
            rows.append(
                f"| {entry['name']} | {entry['duration']} |"
                + " ... |" * (len(gpu_labels) - 1)
            )
        return f"{header}\n{sep}\n" + "\n".join(rows)

    def _parse_kernel_entries_from_report(
        self, report_text: str
    ) -> List[Dict[str, str]]:
        entries: List[Dict[str, str]] = []
        lines = report_text.splitlines()
        idx = 0
        total_lines = len(lines)
        while idx < total_lines:
            raw_line = lines[idx]
            if raw_line.strip().startswith("二、"):
                break
            match = re.match(r"^\s*\d+\.\s+(.*)$", raw_line)
            if match:
                name = match.group(1).strip()
                duration = ""
                ratio = ""
                idx += 1
                while idx < total_lines:
                    line = lines[idx].strip()
                    if line.startswith("- 执行时间"):
                        dur_match = re.search(r"([0-9.]+)\s*ms", line)
                        if dur_match:
                            duration = dur_match.group(1)
                    elif line.startswith("- 时间占比"):
                        ratio_match = re.search(r"([0-9.]+)\s*%", line)
                        if ratio_match:
                            ratio = ratio_match.group(1)
                    elif re.match(r"^\s*\d+\.", lines[idx]) or line.startswith("二、"):
                        break
                    idx += 1
                entries.append({"name": name, "duration": duration, "ratio": ratio})
            else:
                idx += 1
        return entries

    @staticmethod
    def _collect_ncu_csv_snippets(
        output_dir: Path, limit: int = 1500
    ) -> List[Tuple[str, str]]:
        """
        从输出目录收集 NCU 生成的 CSV 文件片段，用于辅助 LLM 分析。
        Args:
            output_dir: 结果目录
            limit: 每个文件读取的字符数限制（防止 Prompt 爆炸）
        """
        
        snippets: List[Tuple[str, str]] = []
        if not output_dir.exists():
            return snippets

        # 查找 ncu_kernel*.csv 文件
        for csv_path in sorted(output_dir.glob("*.csv")):
            # 简单过滤，只看可能有用的数据文件
            if (
                "ncu" not in csv_path.name.lower()
                and "profile" not in csv_path.name.lower()
            ):
                continue

            try:
                # 只读取前 N 个字符
                raw = csv_path.read_text(encoding="utf-8", errors="ignore")
                snippet = raw[:limit]
                if snippet.strip():
                    snippets.append((csv_path.name, snippet))
            except Exception:
                continue

        return snippets

    def _generate_optimization_suggestions(
        self, report_infos: List[Dict[str, Any]]
    ) -> str:
        """
        基于分析报告和原始 CSV 数据生成优化建议。（原样迁移：仍调用 llm_client）
        """
        if not report_infos:
            return ""

        # 1. 准备上下文数据
        context_parts = []

        for info in report_infos:
            gpu_lbl = self._format_gpu_label(info["gpu"], info["index"])
            context_parts.append(
                f"=== 报告 ({gpu_lbl}) ===\n{info.get('text', '')[:2000]}"
            )

            output_dir = Path(info["dir"])
            csv_data = self._collect_ncu_csv_snippets(output_dir)
            if csv_data:
                for name, content in csv_data:
                    context_parts.append(
                        f"=== 原始数据 ({gpu_lbl}/{name}) ===\n{content}\n..."
                    )

        full_context = "\n\n".join(context_parts)

        # 2. 构建 Prompt
        prompt = f"""
        你是一位 CUDA 性能优化专家。请根据以下提供的【性能分析报告】和【原始采样数据】，给出具体的优化建议。

        ### 分析数据
        {full_context}

        ### 你的任务
        请分析上述数据，并输出一段 Markdown 格式的优化建议。

        要求：
        1. **识别瓶颈**：指出最耗时的 Kernel 及其可能的原因（如 Memory Bound, Compute Bound, Latency Bound）。
        2. **具体建议**：
           - 如果是 Memory Bound，建议检查显存合并访问、Shared Memory 使用等。
           - 如果是 Compute Bound，建议检查 Tensor Core 利用率。
           - 如果发现大量小 Kernel，建议考虑 Kernel Fusion。
        3. **简洁专业**：不要废话，直接列出 Top 3 建议。

        ### 优化建议：
        """.strip()

        # 3. 调用 LLM
        try:
            suggestion = self.llm_client.generate(
                prompt, max_tokens=1024, mode="conversation"
            )
            return suggestion
        except Exception as e:
            return f"⚠️ 无法生成优化建议: {e}"

    @staticmethod
    def _generate_report_table(report_text: str) -> str:
        """
        [Legacy] 静态方法，供外部旧代码调用。（原样迁移）
        注意：原实现会在内部重新构造 client（保持不改）
        用report_to_table方法生成表格。
        """
        from offline_llm_v3 import get_offline_qwen_client

        model_path = Path(
            os.getenv(
                "QWEN_LOCAL_MODEL_PATH",
                "/workspaces/ai-agent/AI_Agent_Complete/.models/Qwen3-4B-Instruct-2507",
            )
        )

        try:
            client = get_offline_qwen_client(model_path)
            return client.report_to_table(report_text)
        except Exception as e:
            return f"⚠️ 表格生成失败 (Legacy Mode): {e}"

    @staticmethod
    def _format_gpu_label(label: str, index: int) -> str:
        if not label:
            return f"GPU{index}"
        normalized = label.strip()
        if not normalized:
            return f"GPU{index}"
        if normalized.lower().startswith("gpu"):
            return normalized.upper()
        return f"GPU{normalized}"


if __name__ == "__main__":
    # 1. 创建测试模型
    test_model_path = "/workspace/models/Llama-2-7b-hf"

    # 2. 初始化 LLM Client (使用真实客户端，但不依赖实际生成结果)
    # 注意: 这里使用 offline_llm_v3 的客户端，但实际测试不依赖其输出
    offline_qwen_path = Path("/workspace/models/Qwen3-30B-A3B-Instruct-2507")
    llm_client = get_offline_qwen_client(offline_qwen_path)

    # 3. 创建 PerformanceAnalyzer 实例
    analyzer = PerformanceAnalyzer(
        llm_client=llm_client,
        workflow_factory=create_sglang_analysis_workflow,
        results_dir="test_results",
    )

    # 4. 执行性能分析（使用测试模型）
    print("\n" + "=" * 60)
    print("🚀 开始执行性能分析测试 (使用真实 sglang 分析流程)")
    print(f"测试模型路径: {test_model_path}")
    print("=" * 60)

    try:
        # 执行分析（使用默认参数）
        result = asyncio.run(
            analyzer.run_analysis(
                model_path=test_model_path,
                analysis_type="all",
                params={"batch_size": 1, "input_len": 128, "output_len": 1},
            )
        )

        print("\n" + "=" * 60)
        print("✅ 分析完成! 结果摘要:")
        print("=" * 60)
        print(result)

        # 5. 验证结果目录
        if analyzer.last_analysis_dir:
            print(f"\n🔍 结果目录: {analyzer.last_analysis_dir}")
            print(f"📄 报告文件: {analyzer.last_analysis_reports[0]}")
        else:
            print("⚠️ 未找到分析结果目录 (可能分析失败)")

    except Exception as e:
        print(f"\n❌ 测试执行失败: {str(e)}")
        print("💡 请检查以下事项:")
        print("1. 确保已安装 sglang 和 nsys/ncu 工具")
        print("2. 确保测试模型路径存在 (当前: test_model)")
        print("3. 确保有足够的 GPU 内存")
        sys.exit(1)
