from __future__ import annotations
import threading
from pathlib import Path
from typing import Optional, List, Dict
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 辅助函数：Markdown 表格解析与清洗
# ==========================================

_PROMPT_TEMPLATE = (
    "你是GPU性能分析专家。阅读以下性能报告，为其中列出的每个 kernel 生成 Markdown 表格，"
    "列出 Kernel 名称、执行时长(ms)、时间占比(%)。\n"
    "表头固定为: Kernel | Duration(ms) | Ratio(%)。\n"
    "报告内容如下:\n\n{report}\n\n 请仅输出 Markdown 表格，不要包含任何解释或 <think> 思考过程。"
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

        # 简单的表头或分割线检测
        header_or_separator = cells[0].lower() == "kernel" or all(
            set(cell) <= {"-", ":", " "} for cell in cells
        )
        if header_or_separator:
            truncated.append(line)
            continue

        # 重建行
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
            # 检测表头
            clean_line = stripped.strip("|").lower()
            if "kernel" in clean_line and "duration" in clean_line:
                table_started = True
            else:
                continue
        collected.append(stripped if stripped.startswith("|") else f"| {stripped} |")
    return "\n".join(collected).strip()


def _parse_kernel_entries(report_text: str) -> List[Dict[str, str]]:
    """正则兜底：从报告文本中直接提取数据"""
    entries: List[Dict[str, str]] = []
    current: Dict[str, str] = {}

    for line in report_text.splitlines():
        # 匹配 "1. kernel_name"
        number_match = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if number_match:
            if current.get("name") and current.get("duration") and current.get("ratio"):
                entries.append(current)
            current = {"name": number_match.group(1).strip()}
            continue

        if not current:
            continue

        # 匹配时长
        if "duration" not in current:
            time_match = re.search(r"执行时间[:：]\s*([0-9.]+)\s*ms", line)
            if time_match:
                current["duration"] = time_match.group(1)
                continue

        # 匹配占比
        if "ratio" not in current:
            ratio_match = re.search(r"时间占比[:：]\s*([0-9.]+)%", line)
            if ratio_match:
                current["ratio"] = ratio_match.group(1)
                continue

    # 添加最后一项
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


# ==========================================
# 核心类：OfflineQwenClient (Instruct 适配版)
# ==========================================


class OfflineQwenClient:
    def __init__(self, model_dir: Path):
        if not model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")

        print(f"Loading local Qwen model from: {model_dir} ...")
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
        self.model.eval()  # 显式设为 eval 模式

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        mode: str = "conversation",
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        统一生成入口，支持 Chat Template 和 <think> 清洗。

        Args:
            prompt: 用户输入
            max_tokens: 最大生成长度
            mode:
                - "structured": 强制 JSON 输出，Temp=0，禁用思考链
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
            default_temp = 0.0
            do_sample = False
            sys_content = system_prompt or (
                "你是一个高性能计算专家。请直接输出结果，"
                "严禁输出 <think>...</think> 思考过程，"
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

    def report_to_table(self, report_text: str) -> str:
        """
        利用 LLM 将文本报告转为 Markdown 表格，带正则兜底。
        """
        prompt = _PROMPT_TEMPLATE.format(report=report_text)

        # 复用 generate 方法，设定 mode="conversation" 但温度为 0 以保证稳定
        # 注意：这里虽然用 conversation 模式，但目的是生成格式化文本
        try:
            generated = self.generate(
                prompt,
                max_tokens=2048,
                mode="conversation",
                temperature=0.0,  # 强制确定性
                system_prompt="你是一个数据处理助手，只负责格式化输出。",
            )
        except Exception as e:
            print(f"[Warning] LLM 表格生成失败: {e}")
            generated = ""

        # 提取表格
        cleaned = _extract_table_only(generated)

        # 验证表格有效性 (至少要有表头和分隔线，或者若干行)
        if not cleaned or cleaned.count("|") < 4:
            # LLM 失败，启用正则兜底
            parsed_entries = _parse_kernel_entries(report_text)
            cleaned = _build_table_from_entries(parsed_entries)

        return _truncate_kernel_column(cleaned)


# ==========================================
# 单例模式管理
# ==========================================

_client_lock = threading.Lock()
_cached_client: Optional[OfflineQwenClient] = None


def get_offline_qwen_client(model_dir: Path) -> OfflineQwenClient:
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                _cached_client = OfflineQwenClient(model_dir)
    return _cached_client
