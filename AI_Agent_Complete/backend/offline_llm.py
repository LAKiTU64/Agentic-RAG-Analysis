from __future__ import annotations
import threading
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

_PROMPT_TEMPLATE = (
    "你是GPU性能分析专家。阅读以下性能报告，为其中列出的每个 kernel 生成 Markdown 表格，"
    "列出 Kernel 名称、执行时长(ms)、时间占比(%)。\n"
    "表头固定为: Kernel | Duration(ms) | Ratio(%)。\n"
    "报告内容如下:\n\n{report}\n\n 请仅输出 Markdown 表格。"
)


def _truncate_kernel_column(markdown: str, max_len: int = 30) -> str:
    """截断过长的 Kernel 名称以防止表格变形"""

    lines = markdown.splitlines()
    truncated = []
    for line in lines:
        stripped = line.strip()
        if "|" not in stripped:
            truncated.append(line)
            continue

        core = stripped.strip("|")
        cells = [cell.strip() for cell in core.split("|")]
        if not cells:
            truncated.append(line)
            continue

        header_or_separator = cells[0].lower() == "kernel" or all(
            set(cell) <= {"-", ":", " "} for cell in cells
        )
        if header_or_separator:
            truncated.append(line)
            continue

        # cells[0] = cells[0][:max_len]
        rebuilt = "| " + " | ".join(cells) + " |"
        truncated.append(rebuilt)
    return "\n".join(truncated)


def _extract_table_only(markdown: str) -> str:
    """从混合文本中提取 Markdown 表格部分"""

    lines = markdown.splitlines()
    collected = []
    table_started = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if table_started:
                break
            continue
        if "|" not in stripped:
            if table_started:
                break
            continue
        if not table_started:
            header_cells = [
                cell.strip().lower() for cell in stripped.strip("|").split("|")
            ]
            if not header_cells or "kernel" not in header_cells[0]:
                continue
            table_started = True
        collected.append(stripped if stripped.startswith("|") else f"| {stripped} |")
    return "\n".join(collected).strip()


def _parse_kernel_entries(report_text: str) -> List[Dict[str, str]]:
    """正则兜底：从报告文本中直接提取数据"""

    entries: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for line in report_text.splitlines():
        number_match = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if number_match:
            if current.get("name") and current.get("duration") and current.get("ratio"):
                entries.append(current)
            current = {"name": number_match.group(1).strip()}
            continue
        if not current:
            continue
        if "duration" not in current:
            time_match = re.search(r"执行时间[:：]\s*([0-9.]+)\s*ms", line)
            if time_match:
                current["duration"] = time_match.group(1)
                continue
        if "ratio" not in current:
            ratio_match = re.search(r"时间占比[:：]\s*([0-9.]+)%", line)
            if ratio_match:
                current["ratio"] = ratio_match.group(1)
                continue
    if current.get("name") and current.get("duration") and current.get("ratio"):
        entries.append(current)
    return entries


def _build_table_from_entries(entries: List[Dict[str, str]]) -> str:
    """将字典列表转换为 Markdown 表格"""

    if not entries:
        return ""
    rows = ["| Kernel | Duration(ms) | Ratio(%) |", "| --- | --- | --- |"]
    for item in entries:
        rows.append(f"| {item['name']} | {item['duration']} | {item['ratio']} |")
    return "\n".join(rows)


class OfflineQwenClient:
    def __init__(self, model_dir: Path):
        if not model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")
        print(f"加载本地LLM模型路径: {model_dir} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.pipeline = TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # 将 pytorch 设置为eval模式，确保推理结果的一致
        self.model.eval()

    # 统一的LLM生成接口
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        mode: str = "conversation",
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        统一的LLM生成接口

        Args:
            prompt: 用户输入
            max_tokens: 最大生成长度
            mode:
                - "structured": 强制 JSON 输出，Temp=0
                - "conversation": 自然对话，Temp=0.3
            temperature: 覆盖默认温度
            system_prompt: 覆盖默认 System Prompt
        """

        # 1. 配置默认参数
        do_sample = True
        default_temp = 0.3

        # 2. 构建 Messages
        messages = []

        if mode == "structured":
            # 如果是structure模式则覆盖温度和system_prompt
            default_temp = 0.0
            do_sample = False
            sys_content = system_prompt or (
                "你是一个高性能计算专家。请直接输出结果，"
                "严禁输出 Markdown 代码块标记，只输出纯 JSON 字符串。"
            )
            messages.append({"role": "system", "content": sys_content})

        elif mode == "conversation":
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            # 如果没有指定 system_prompt，Instruct 模型通常不需要显式 System 也能工作良好
            # 或者可以加一个通用的: "You are a helpful assistant."

        messages.append({"role": "user", "content": prompt})

        # 3. 应用 Chat Template (核心：适配 Instruct 模型)
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        # 4. 确定生成参数
        actual_temp = temperature if temperature is not None else default_temp

        # 防止温度为0时 do_sample=True 导致报错
        if actual_temp == 0.0:
            do_sample = False

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            gen_kwargs["temperature"] = actual_temp
            gen_kwargs["top_p"] = 0.9

        # 5. 执行生成
        with torch.no_grad():
            outputs = self.model.generate(input_ids, **gen_kwargs)

        generated_ids = outputs[0][input_ids.shape[1] :]
        generated = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 6. 全局清洗 <think> 标签 (防止思维链污染结果)
        # 即使在 conversation 模式下，如果不需要展示思考过程，清洗掉也是安全的
        # 如果你想保留 conversation 的思考过程，可以将此逻辑移到 mode check 内
        generated = re.sub(
            r"<think>.*?</think>", "", generated, flags=re.DOTALL
        ).strip()

        return generated

    def _generate_text(self, prompt: str, max_new_tokens: int = 2000) -> str:
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=False,
        )
        generated = outputs[0]["generated_text"]
        if generated.startswith(prompt):
            generated = generated[len(prompt) :]
        return generated.strip()

    def report_to_table(self, report_text: str) -> str:
        prompt = _PROMPT_TEMPLATE.format(report=report_text)
        generated = self._generate_text(prompt, max_new_tokens=10000)
        cleaned = _extract_table_only(generated)
        if not cleaned or cleaned.count("|") < 6:
            parsed_entries = _parse_kernel_entries(report_text)
            cleaned = _build_table_from_entries(parsed_entries)
        if not cleaned:
            cleaned = generated
        return _truncate_kernel_column(cleaned)

    def merge_gpu_tables(self, labeled_tables: List[Tuple[str, str]]) -> str:
        print("调用离线llm进行多GPU表格合并，表格数量:", len(labeled_tables))
        if not labeled_tables:
            return ""

        labels = [label for label, _ in labeled_tables]
        header_parts = ["Kernel"]
        last_index = len(labels) - 1
        for idx, label in enumerate(labels):
            header_parts.append(f"{label} Duration(ms)")
            add_ratio = len(labels) == 1 or idx != last_index
            if add_ratio:
                header_parts.append(f"{label} Ratio(%)")
        if len(labels) >= 2:
            header_parts.append(f"{labels[0]}：{labels[1]} 时间")

        header_spec = " | ".join(header_parts)
        pair_instruction = ""
        if len(labels) >= 2:
            pair_instruction = (
                f"请计算 {labels[0]} 与 {labels[1]} 的持续时间最简整数比，例如 0.004 和 0.004 输出 1：1；"
                "若任一缺失则填入 0：0 或留空。"
            )

        table_sections = []
        for label, table in labeled_tables:
            table_clean = table.strip()
            if not table_clean:
                table_sections.append(f"### {label}\n(无可用表格数据)\n")
            else:
                table_sections.append(f"### {label}\n{table_clean}")

        prompt = (
            "你是一名 GPU 性能分析专家。下面提供了多张单 GPU 的 kernel 性能表格。\n"
            "请将它们合并为一张 Markdown 表格，满足以下要求：\n"
            f"1. 表头固定为: {header_spec}\n"
            "2. 按 kernel 名称对齐；若某 GPU 缺失该 kernel，留空即可。\n"
            "3. 每行按照所有 GPU 中最大的持续时间从大到小排序。\n"
            "4. Duration(ms) 和 Ratio(%) 保留原有数值格式，不要额外四舍五入。\n"
            f"5. {pair_instruction if pair_instruction else '若只有一个 GPU，则忽略最简整数比的需求。'}\n"
            "6. 仅输出最终的 Markdown 表格，勿添加说明文字。\n\n"
            "以下是输入表格：\n"
            + ("\n\n".join(table_sections) if table_sections else "(无有效表格)")
        )

        generated = self._generate_text(prompt, max_new_tokens=4000)
        merged = _extract_table_only(generated)
        if not merged:
            merged = generated
        return _truncate_kernel_column(merged)

    def suggest_optimizations(
        self,
        labeled_reports: List[Tuple[str, str]],
        max_new_tokens: int = 4096,
    ) -> str:
        if not labeled_reports:
            return ""

        report_sections = []
        for label, report in labeled_reports:
            label_text = label.strip() if label else "GPU"
            sanitized_report = report.strip()
            if not sanitized_report:
                continue
            report_sections.append(f"### {label_text}\n{sanitized_report}")

        if not report_sections:
            return ""

        prompt = (
            "你是一名 GPU 性能调优专家。以下是多份 GPU 性能分析报告片段，"
            "请结合它们给出 3-5 条具备可执行性的优化建议。"
            "每条建议需包含: 关注的瓶颈、优化思路、可能的验证方式。"
            "若多个 GPU 之间存在差异，请指出并给出针对性的措施。"
            "格式要求使用 Markdown 列表。\n\n"
            "报告内容如下：\n" + "\n\n".join(report_sections)
        )

        generated = self._generate_text(prompt, max_new_tokens=max_new_tokens)
        cleaned = generated.strip()
        if not cleaned:
            return "⚠️ 离线模型未返回有效的优化建议"
        return cleaned

    def suggest_raw_data_optimizations(
        self,
        labeled_snippets: List[Tuple[str, str]],
        max_new_tokens: int = 2048,
    ) -> str:
        if not labeled_snippets:
            return ""

        snippet_sections: List[str] = []
        for label, snippet in labeled_snippets:
            title = (label or "数据片段").strip()
            content = snippet.strip()
            if not content:
                continue
            snippet_sections.append(f"### {title}\n```\n{content}\n```")

        if not snippet_sections:
            return ""

        prompt = (
            "你是一名 GPU 性能调优专家。以下提供的是来自 Nsight Compute/Nsys 的原始 CSV 数据片段，"
            "每个片段包含多个 kernel 的指标。请基于这些原始数值，给出 3-5 条可执行的优化建议。"
            "每条建议需要说明关注的指标、优化方向，以及后续验证方式。若不同 GPU 或 kernel 之间指标存在差异，"
            "请指出差异并给出针对性的建议。仅输出 Markdown 列表，不要包含额外说明。\n\n"
            "原始数据片段如下：\n" + "\n\n".join(snippet_sections)
        )

        generated = self._generate_text(prompt, max_new_tokens=max_new_tokens)
        cleaned = generated.strip()
        if not cleaned:
            return "⚠️ 离线模型未返回有效的原始数据建议"
        return cleaned


_client_lock = threading.Lock()
_cached_client: Optional[OfflineQwenClient] = None


def get_offline_qwen_client(model_dir: Path) -> OfflineQwenClient:
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                _cached_client = OfflineQwenClient(model_dir)
    return _cached_client
