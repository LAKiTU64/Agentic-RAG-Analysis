#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent核心模块 - 集成NSys和NCU性能分析
"""

import re
import os
import json
import asyncio
import sys
import yaml
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from math import gcd, lcm
from fractions import Fraction

# 导入分析工具
from utils.nsys_to_ncu_analyzer import create_sglang_analysis_workflow
from offline_llm import get_offline_qwen_client
from knowledge_bases.vector_kb_manager import VectorKBManager

try:
    from .utils.roofline_estimator import compute_roofline
except Exception:
    try:
        from utils.roofline_estimator import compute_roofline
    except Exception:
        compute_roofline = None

OFFLINE_QWEN_PATH = Path(os.getenv("QWEN_LOCAL_MODEL_PATH", "/workspace/Qwen3-32B"))


class AIAgent:
    """AI Agent核心类 - 自动化性能分析"""

    def __init__(self, config: Dict):
        self.config = config

        # sglang 和模型路径
        self.sglang_path = Path(config.get("sglang_path"))
        self.models_path = Path(config.get("models_path"))
        self.model_mappings = config.get("model_mappings", {})

        # 分析结果输出目录
        self.results_dir = Path(config.get("output", {}).get("results_dir", "results"))
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # 分析工具配置
        self.profiling_config = config.get("profiling_tools", {})
        self.analysis_defaults = config.get("analysis_defaults", {})

        # 缓存最近一次分析的关键信息，便于对外接口复用
        self.last_analysis_dir: Optional[str] = None
        self.last_analysis_dirs: List[str] = []
        self.last_analysis_reports: List[str] = []
        self.last_analysis_table: Optional[str] = None
        self.last_analysis_suggestions: Optional[str] = None
        self.last_roofline_estimate: Optional[Dict[str, Any]] = None

        # 本地 LLM 客户端
        self.offline_qwen_path = Path(config.get("offline_qwen_path"))
        self.llm_client = get_offline_qwen_client(self.offline_qwen_path)

        # 向量知识库相关
        # kb_config = config.get("vector_store", {})
        # self.embedding_model = kb_config.get("embedding_model")
        # self.persist_directory = kb_config.get("persist_directory")
        # self.chunk_size = kb_config.get("chunk_size")
        # self.chunk_overlap = kb_config.get("chunk_overlap")
        # self.default_search_k = kb_config.get("default_search_k", 8)
        # self.max_distance = kb_config.get("max_distance", 0.5)
        self.kb = VectorKBManager(config=config)

        # 对话历史缓冲区
        self.chat_history: List[Dict[str, str]] = []  # 对话历史（完整保留）
        self.default_history_turns = 3  # 默认拉取最近对话轮数（按需拉取）

        # intent-prompt 映射
        self.intent_mappings = {
            "analysis": "用户希望**立即执行**性能分析任务（如“跑一下qwen”、“nsys/ncu分析”）。必须包含**动作意图**（运行/测试/分析等）。",
            "rag-qa": "用户在**询问知识、数据或建议**（如“瓶颈是什么”、“推荐batch_size”、“某个Kernel的运行情况”）。无执行意图。",
            "chat": "打招呼、感谢、闲聊以及一切无法归类的内容（如“你好”、“你是谁”）。",
        }

    def _format_history_str(
        self, history: List[Dict[str, str]], limit: int = -1
    ) -> str:
        """
        取出并格式化历史对话。
        Args:
            history: 完整历史列表
            limit: 取最近多少轮对话。1轮=用户+Agent共2条对话。
                   -1   : (默认) 使用 self.default_history_turns
                   0 : 不取历史对话
                   int  : 指定具体轮数
        """
        if not history or limit == 0:
            return ""

        if limit == -1:
            limit = self.default_history_turns

        # 取最近 limit 轮对话
        target_history = history[-(limit * 2) :] if limit else history

        lines = []
        for msg in target_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            # 限制单条消息长度，防止 Prompt 过长
            content = msg["content"].replace("\n", " ").strip()[:512]
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    async def _parse_intent(self, user_query: str, intent: str = "auto") -> str:
        """
        意图识别函数。

        Args:
            user_query: 用户当前输入
            history: 对话历史（用于上下文，但非必需）
            intent: 指定意图模式。若为 "auto"，则由 LLM 判断；否则直接返回 intent。

        Returns:
            str: 意图类别，取值为 self.intent_mappings 的键之一（如 "rag-qa", "analysis", "chat"）
        """
        supported_intents = set(self.intent_mappings.keys())

        # 如果 intent 不是 auto，直接返回
        if intent != "auto":
            return intent

        # === LLM 自动意图识别 ===
        # 告诉 LLM 当前可用模型有哪些
        available_models = list(self.model_mappings.keys())
        models_str = ", ".join([f'"{m}"' for m in available_models])

        # 构建历史对话
        history_str = self._format_history_str(self.chat_history)

        # 生成意图定义描述
        intent_definitions = "\n".join(
            f"- **{intent}**: {desc}" for intent, desc in self.intent_mappings.items()
        )

        prompt = f"""
            你是一个意图分类器。请仅根据用户当前输入的**语义**判断其意图类别。
            
            ### 意图定义（重要）
            {intent_definitions}
            
            ### 输出要求（重要）
            仅输出以下单词之一：{" | ".join(supported_intents)}，不要解释，不要标点，不要JSON。
            
            ### 判断原则
            1. 优先看**用户当前输入**，历史仅作辅助。
            2. 若无法明确归类，请返回"chat"。

            ### 当前可用模型参考（仅作背景参考，不影响分类）
            [{models_str}]
            
            ### 用户当前输入（主要判断依据）
            {user_query}

            {"### 最近对话历史（可忽略的次要判断依据）\n" + history_str if history_str else "（无历史记录）"}
        """

        raw_output = self.llm_client.generate(prompt, max_tokens=32).strip()
        raw_output = raw_output.lower().strip(" .,!?\"'")
        if raw_output in supported_intents:
            return raw_output
        else:
            # 规则兜底
            user_query_lower = user_query.lower()
            if any(
                kw in user_query_lower
                for kw in ["分析", "跑", "测", "profile", "运行", "执行", "测试"]
            ):
                return "analysis"
            elif any(
                kw in user_query_lower
                for kw in [
                    "什么",
                    "多少",
                    "是否",
                    "为什么",
                    "如何",
                    "推荐",
                    "文档",
                    "查询",
                    "是多少",
                ]
            ):
                return "rag-qa"
            else:
                return "chat"

    async def _parse_raw_params(
        self, user_query: str, rewrite_query: bool = False
    ) -> Dict[str, Any]:
        """
        从用户查询中提取原始的模型名称、分析参数和改写后的query。后续将其转为性能分析的参数，或者用于RAG-QA的filter。
        把提取json和改写query放在一起，因为这样做语义更连贯。
        返回示例请参考下面的prompt。

        """

        available_models = list(self.model_mappings.keys())
        models_str = ", ".join([f'"{m}"' for m in available_models])

        # 构建历史对话
        history_str = self._format_history_str(self.chat_history, limit=1)

        prompt = f"""
            你是一个专业的参数提取助手。请从用户的自然语言输入中提取执行参数，并同时生成用于向量检索的“改写查询”。
            
            ### 0. 输出格式（非常重要）
            你必须**仅输出一个标准 JSON 对象**（不能有任何解释、不能有 Markdown 代码块、不能有多余文本）。
            JSON 必须仅包含以下 4 个字段：model, params, analysis_type, search_query。

            ### 1. 字段定义
            - **model**: 模型名称。必须严格匹配列表：[{models_str}]。如果找不到匹配项则为null。
            - **params**: 一个必须存在的 JSON 对象，允许为空对象，包含整数键：batch_size/input_len/output_len。如果用户没提到对应键，则直接忽略。
            - **analysis_type**: 分析类型，取值只能是 "nsys", "ncu"或null。判断逻辑如下：
                1. **nsys**: 用户提到 "nsys"、"全局"、"整体"、"profile"、"timeline"。
                2. **ncu**: 用户提到 "ncu"、"深度"、"kernel细节"、"指令级"。
                3. null: 用户未明确提及分析类型，或用户明确提及了nsys和ncu两种分析类型。
            - **search_query**: 改写后的“干净问题”，用于向量检索。
                1. 只在rewrite_query为true时改写。如果rewrite_query为false，则search_query输出一个空字符串。
                2. 改写时：移除本次输出 JSON 中的 model/params/analysis_type 等约束信息（以及与其语义等价的限定表达），只保留用户要查询的“知识点/指标/结论”。
                3. search_query 尽量短（5~30 个汉字或单词）。如果用户问题本身已经很干净，可与原问题等价或更短。
                4. 不要在 search_query 中添加原问题里不存在的信息。
                
            ### 2. 参考示例
            user_query: "用nsys和ncu跑一下qwen3-4b，batch_size设为1，output_len=1"
            rewrite_query: false
            Output: {{"model": "qwen3-4b", "params": {{"batch_size": 1, "output_len": 1}}, "analysis_type": null, "search_query": ""}}

            user_query: "帮我给模型做个ncu深度分析"
            rewrite_query: false
            Output: {{"model": null, "params": {{}}, "analysis_type": "ncu", "search_query": ""}}
            
            user_query: "llama-7b在batch_size=1、input_len=128的全局分析报告中，总kernel数有多少？"
            rewrite_query: true
            Output: {{"model": "llama-7b", "params": {{"batch_size": 1, "input_len": 128}}, "analysis_type": "nsys", "search_query": "总kernel数有多少？"}}

            user_query: "qwen-7b在batch_size=16的情况下瓶颈最多的kernel有哪些？"
            rewrite_query: true
            Output: {{"model": "qwen-7b", "params": {{"batch_size": 16}}, "analysis_type": null, "search_query": "瓶颈最多的kernel有哪些？"}}
            
            ### 3. 本次是否需要改写query（非常重要）
            {str(rewrite_query).lower()}

            ### 4. 用户输入（主要依据）
            {user_query}
            
            ### 5. 最近对话历史（可能为空）
            #### tips: 仅当用户提到“之前/上一次/上个问题/刚才/沿用/跟前面一样”等需要你查询对话历史的关键词时，才可以纳入参考，否则忽略对话历史
            {history_str if history_str else "对话历史为空。"}

            ### 6. 你的JSON输出：
        """.strip()

        def _strip_code_fence(text: str) -> str:
            """去除代码块标记"""
            text = text.strip()
            # 处理开头
            if text.startswith("```"):
                # 找到第一行换行符
                if "\n" in text:
                    text = text.split("\n", 1)[1]
                else:
                    # 极其罕见情况：只有 ```json 没有换行
                    text = text.lstrip("`").lstrip("json").strip()

            # 处理结尾
            if text.endswith("```"):
                text = text[:-3]

            return text.strip()

        def _extract_first_json_object(text: str) -> Optional[str]:
            """
            从任意文本中提取第一个完整的 JSON 对象（从第一个 { 开始做括号配对）。
            只做轻量提取：不处理字符串内花括号的复杂情形，但对 LLM 常见输出足够稳。
            """
            s = text
            start = s.find("{")
            if start < 0:
                return None

            depth = 0
            for i in range(start, len(s)):
                if s[i] == "{":
                    depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]
            return None

        parsed_params: Dict[str, Any] = {}
        model_name: Optional[str] = None
        analysis_type: Optional[str] = None
        search_query: Optional[str] = None

        try:
            raw = self.llm_client.generate(
                prompt, max_tokens=512, mode="structured"
            ).strip()
            json_text = _strip_code_fence(raw)

            # 如果 raw 不是纯 JSON，则尝试抽取其中第一个 JSON 对象
            if not (json_text.startswith("{") and json_text.endswith("}")):
                extracted = _extract_first_json_object(json_text)
                if extracted:
                    json_text = extracted

            result = json.loads(json_text)
            model_name = result.get("model")
            parsed_params = result.get("params", {}) or {}
            raw_type = result.get("analysis_type")
            analysis_type = raw_type if raw_type in ("nsys", "ncu") else None
            search_query = (
                str(result.get("search_query", "")).strip() if rewrite_query else ""
            )
        except Exception as e:
            print(
                f"[_parse_raw_params] LLM parse failed: {e}. raw={locals().get('raw', None)!r}"
            )

        # 规则兜底：尝试从 query 中提取模型名（如果 LLM 没拿到）
        if not model_name:
            q = user_query.lower()
            # 最长优先，避免短名字误命中
            for model in sorted(available_models, key=len, reverse=True):
                if model.lower() in q:
                    model_name = model
                    break

        # 合并参数，只保留用户提到的字段
        final_params: Dict[str, int] = {}
        if isinstance(parsed_params, dict):
            for key in ("batch_size", "input_len", "output_len"):
                if key in parsed_params and parsed_params[key] not in (None, ""):
                    try:
                        final_params[key] = int(parsed_params[key])
                    except (ValueError, TypeError):
                        pass

        return {
            "model": model_name,
            "params": final_params,
            "analysis_type": analysis_type,
            "search_query": search_query,
        }

    def _finalize_params_for_analysis(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """对message解析的参数做默认值补全，用于性能分析"""

        defaults = {"batch_size": 1, "input_len": 128, "output_len": 1}
        params = {**defaults, **(raw.get("params") or {})}
        analysis_type = raw.get("analysis_type")

        return {
            "model": raw.get("model"),
            "params": params,
            "analysis_type": analysis_type,
        }

    def _finalize_params_for_rag_filter(
        self, raw: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        将message解析的参数转化为chroma语法的where_fileter
         - 把解析出来的所有字段放在一个 dict 中，作为 where_filter
         - 如果value是 None/null/空字符串，则忽略该字段
         - _parse_raw_params返回的是一个嵌套字典，而chroma的metadata是扁平的，转换成where_filter时需要把params拍平
         - 目前只实现了=，暂不支持>、<等范围查询
        """

        if not isinstance(raw, dict):
            return None

        def is_empty(v: Any) -> bool:
            """判断值是否为空（None、空字符串、"none"、"null"）"""
            if v is None:
                return True
            if isinstance(v, str) and v.strip() in ("", "none", "null"):
                return True
            return False

        def coerce_scalar(v: Any) -> Any:
            """试图把字符串形式的数字转为实际格式，例如 "128" -> 128 or "1.28" -> 1.28"""
            if isinstance(v, str):
                s = v.strip()
                # int
                if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                    try:
                        return int(s)
                    except Exception:
                        return v
                # float
                try:
                    if "." in s:
                        return float(s)
                except Exception:
                    pass
            return v

        where: Dict[str, Any] = {}

        # 1) 顶层字段（除了 params）
        for k, v in raw.items():
            if k == "params":
                continue
            if is_empty(v):
                continue
            where[k] = coerce_scalar(v)

        # 2) params 拍平
        params = raw.get("params")
        if isinstance(params, dict):
            for k, v in params.items():
                if is_empty(v):
                    continue
                where[k] = coerce_scalar(v)

        print(where)
        return where or None

    def _resolve_model_path(self, model_name: str) -> Optional[str]:
        """
        解析并返回模型路径
        """

        if not model_name:
            return None
        # 1. 如果 model_mappings 配置的是绝对路径，则直接返回该路径
        if model_name in self.model_mappings:
            mapped_path = self.model_mappings[model_name]
            if Path(mapped_path).is_absolute():
                return mapped_path
            return str(self.models_path / mapped_path)

        # 2. 如果 model_mappings 配置的是相对路径，则与 models_path 拼接成绝对路径，若路径存在则返回，否则返回None
        if Path(model_name).exists():
            return model_name
        potential_path = self.models_path / model_name
        if potential_path.exists():
            return str(potential_path)
        return None

    async def _agent_analysis(self, message: str) -> str:
        """
        处理性能分析意图。
        从用户消息中提取模型和参数，启动分析流程，并返回结果摘要或错误信息。
        """
        try:
            response = """✅ **已解析您的请求**\n"""

            # Step 1: 解析分析参数（model + kwargs）
            parsed_raw = await self._parse_raw_params(message)
            parsed = self._finalize_params_for_analysis(parsed_raw)
            print(parsed)

            model_name = parsed.get("model")
            params = parsed.get("params", {})
            analysis_type = parsed.get("analysis_type", None)

            response += f"🤖 **模型**: {model_name or '未指定'}\n🔬 **分析类型**: {analysis_type or '未指定 (默认nsys+ncu)'}\n📊 **参数**: {params}\n"

            available = ", ".join(self.model_mappings.keys())

            if not model_name or model_name not in self.model_mappings:
                return (
                    response
                    + f"❌ **分析失败**: 未指定模型或模型不可用。可用模型：{available}"
                )

            # Step 2: 执行分析流程
            model_path = self._resolve_model_path(model_name)
            if not model_path:
                # 明确抛出错误，让用户知道是模型配置问题
                raise ValueError(
                    f"模型路径解析失败: '{model_name}'。\n"
                    f"请检查 config.yaml 中的 'model_mappings' 是否包含该模型，"
                    f"或者模型文件是否存在于: {self.models_path}"
                )

            analysis_result = await self._run_analysis(
                model_path=model_path,
                analysis_type=analysis_type,
                params=params,
            )
            return response + analysis_result

        except Exception as e:
            return response + f"❌ **分析执行异常**: {str(e)}"

    async def _agent_rag_qa(self, message: str) -> str:
        """
        处理专业问答（RAG-QA）意图。
        执行知识库检索，并基于检索结果生成严谨、有依据的回答。
        """

        # Step 1: 构建where_filter，改写query，检索相关知识片段
        parsed_raw = await self._parse_raw_params(message, rewrite_query=True)
        where_filter = self._finalize_params_for_rag_filter(
            {k: v for k, v in parsed_raw.items() if k != "search_query"}
        )
        search_query = parsed_raw.get("search_query") or message
        retrieved_contexts = self.kb.search(
            query=search_query, where_filter=where_filter
        )
        # debug用：打印重写后的query和用于元数据过滤的where_filter
        # print(f"search_query: {search_query}")
        # print(f"where_filter: {where_filter}")

        # Step 2: 构建 RAG 上下文和历史对话
        rag_context = ""
        if retrieved_contexts:
            rag_snippets = [
                f"【文档片段 {i + 1}】\n{res['content']}"
                for i, res in enumerate(retrieved_contexts)
            ]
            rag_context = "\n\n".join(rag_snippets)
        # debug用：打印RAG召回结果
        # print(rag_context)

        history_str = self._format_history_str(self.chat_history, limit=1)

        # Step 3: 严格约束的 RAG-QA 生成
        prompt = f"""
            你是一个严谨的数据分析员。你必须完全依据【参考资料】回答用户关于 GPU 性能数据的提问。

            ### 参考资料
            {rag_context if rag_context else "（警告：未检索到相关文档，可能需要告知用户资料缺失）"}

            ### 用户问题（主要的用户意图依据）
            {message}

            ### 对话历史（次要、可忽略的用户意图补充）
            {history_str if history_str else "（无历史记录）"}

            ### 严格约束 (Strict Rules)
            1. **数据精确性**：如果用户询问某个 Kernel 的具体指标（如瓶颈数、带宽），**必须**在参考资料中找到**完全匹配**的 Kernel 名称后才能回答。
            2. **拒绝猜测**：如果资料里有 "Kernel A" 和 "Kernel B"，但用户问 "Kernel C"，你必须回答："资料中未找到 Kernel C 的数据"。**严禁**把 A 的数据安在 C 上。
            3. **原文引用**：回答时尽量使用资料中的原话或数据。
            4. **空值处理**：如果资料为空或不相关，直接回答：“抱歉，知识库中没有相关信息。”

            ### 回答：
        """.strip()

        try:
            answer = self.llm_client.generate(prompt, max_tokens=1024).strip()
            ref_count = len(retrieved_contexts) if retrieved_contexts else 0
            return f"🤖 **RAG-QA**\n{answer}\n\n---\n💡 *基于 {ref_count} 条知识库片段回答*"
        except Exception as e:
            return f"❌ **RAG-QA生成失败**: {str(e)}"

    async def _agent_chat(self, message: str) -> str:
        """
        闲聊模式Agent
        """

        # 构建历史对话
        history_str = self._format_history_str(self.chat_history)

        prompt = f"""
            你是一个专业的 AI 性能分析专家。
            不要胡编乱造技术数据，可以进行简短的日常对话。
            
            对话历史（可忽略的用户意图补充）
            {history_str if history_str else "（无历史记录）"}
            
            用户: {message}
            
            你的回复:
        """
        raw = self.llm_client.generate(prompt, max_tokens=256).strip()
        return f"🤖 **闲聊模式**\n{raw}"

    async def process_message(self, message: str, intent: str = "auto") -> str:
        """
        Agentic-RAG 意图路由:
        1. [Router] 意图识别 → 返回一个intent_mappings中的key （例如 "analysis" | "rag-qa" | "chat"）
        2. [Branch] 根据意图进行分支处理
        3. [History] 保存对话历史
        """

        # Step1: 意图识别（返回一个intent_mappings中的key）
        try:
            intent = await self._parse_intent(message, intent)
        except Exception as e:
            return f"❌ **意图识别失败**: {str(e)}"

        response_text = ""

        # Step2: 根据意图路由到对应agent
        if intent == "analysis":
            # === 分支 A: 性能分析 (Analysis) ===
            print("[analysis] 识别为分析意图")
            response_text = await self._agent_analysis(message)

        elif intent == "rag-qa":
            # === 分支 B: RAG问答 (RAG-QA) ===
            print("[rag-qa] 识别为RAG问答意图")
            response_text = await self._agent_rag_qa(message)

        elif intent == "chat":
            # === 分支 C: 闲聊 (Chat) ===
            print("[chat] 识别为闲聊意图")
            response_text = await self._agent_chat(message)

        else:
            # 理论上不会走到这里
            response_text = "❓ 无法理解您的意图，请换种方式提问。"

        # Step3: 保存对话历史，如果intent=analysis，则只返回摘要
        self.chat_history.append({"role": "user", "content": message})

        history_response = (
            response_text if intent != "analysis" else "已完成性能分析任务。"
        )
        self.chat_history.append({"role": "assistant", "content": history_response})

        return response_text

    # 已整合进_agent_analysis()
    async def _execute_analysis_flow(
        self, model_name: str, analysis_type: str, params: Dict
    ) -> str:
        model_path = self._resolve_model_path(model_name)
        if not model_path:
            # 明确抛出错误，让用户知道是模型配置问题
            raise ValueError(
                f"模型路径解析失败: '{model_name}'。\n"
                f"请检查 config.yaml 中的 'model_mappings' 是否包含该模型，"
                f"或者模型文件是否存在于: {self.models_path}"
            )
        return await self._run_analysis(
            model_path=model_path, analysis_type=analysis_type, params=params
        )

    async def _run_analysis(
        self, model_path: str, analysis_type: str, params: Dict
    ) -> str:
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
        batch_sizes = params.get("batch_size", [1])
        input_lens = params.get("input_len", [128])
        output_lens = params.get("output_len", [1])

        # 只分析第一组参数（避免时间过长）
        batch_size = batch_sizes[0] if isinstance(batch_sizes, list) else batch_sizes
        input_len = input_lens[0] if isinstance(input_lens, list) else input_lens
        output_len = output_lens[0] if isinstance(output_lens, list) else output_lens

        precision_cfg = (
            self.analysis_defaults.get("precision", {})
            if isinstance(self.analysis_defaults.get("precision", {}), dict)
            else {}
        )

        def _parse_int(value, default):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        w_bit = _parse_int(precision_cfg.get("w_bit"), 16)
        a_bit = _parse_int(precision_cfg.get("a_bit"), 16)
        kv_bit_candidate = precision_cfg.get("kv_bit")
        parsed_kv_bit = (
            _parse_int(kv_bit_candidate, None) if kv_bit_candidate is not None else None
        )
        kv_bit = parsed_kv_bit if isinstance(parsed_kv_bit, int) else None
        use_flashattention = bool(precision_cfg.get("use_flashattention", False))
        hardware_key = (
            self.analysis_defaults.get("hardware")
            or os.getenv("ROOFLINE_HARDWARE")
            or "nvidia_H800_SXM5_80G"
        )

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
                        gpu_label = str(item.get("gpu", idx))
                        output_path = item.get("dir") or item.get("path")
                    else:
                        gpu_label = str(idx)
                        output_path = str(item)
                    if output_path:
                        run_records.append((gpu_label, Path(output_path)))
            elif workflow_output:
                run_records.append(("0", Path(str(workflow_output))))

            if not run_records:
                results.append("⚠️ **分析已完成，但未找到输出目录**")
                return "\n".join(results)

            self.last_analysis_dirs = [str(path) for _, path in run_records]

            report_infos = []
            roofline_infos: List[Tuple[str, Path, Dict[str, Any]]] = []
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
                    roofline_path = output_dir / "roofline_estimate.json"
                    if roofline_path.exists():
                        try:
                            with open(roofline_path, "r", encoding="utf-8") as rf:
                                roofline_data = json.load(rf)
                            roofline_infos.append(
                                (gpu_label, roofline_path, roofline_data)
                            )
                        except Exception as roof_exc:
                            print(
                                f"⚠️ 读取 Roofline 预测失败 ({roofline_path}): {roof_exc}"
                            )
                else:
                    missing_reports.append(output_dir)

            if not report_infos:
                dir_lines = "\n".join(f"  • {path}" for _, path in run_records)
                results.append(f"""
⚠️ **分析已完成，但未生成报告文件**

📁 结果目录:
{dir_lines}
💡 请检查目录中的其他输出文件
""")
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

            roofline_section = "📐 **Roofline 预测**:\n暂未生成 Roofline 预测\n"
            if roofline_infos:
                self.last_roofline_estimate = roofline_infos[0][2]
                roofline_preview = self._render_roofline_preview(
                    self.last_roofline_estimate
                )
                roofline_source = str(roofline_infos[0][1])
                roofline_section = f"📐 **Roofline 预测** (来源: {roofline_source}):\n{roofline_preview}\n"

            results.append(f"""
✅ **分析完成!**

📁 **结果目录**:
{dir_lines}
📄 **报告文件**: {primary_info["report"]}
{missing_lines}
{summary}

{roofline_section}

📌 **热点Kernel表格预览**:
{table_markdown}

💡 **优化建议**:
{suggestions or "暂未生成优化建议"}

🔍 **详细报告**: 请查看 {primary_info["report"]}
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

        return "\n".join(results)

    def _render_roofline_preview(self, roofline: Dict[str, Any]) -> str:
        if not roofline:
            return "暂未生成 Roofline 数据"

        def _fmt_time(seconds: Optional[float]) -> str:
            if seconds is None:
                return "N/A"
            if isinstance(seconds, (int, float)):
                if seconds == float("inf"):
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

        bits = roofline.get("precision_bits", {})
        params = roofline.get("params", {})
        prefill = roofline.get("prefill", {})
        decode = roofline.get("decode", {})
        overall = roofline.get("overall", {})
        observed = (
            roofline.get("observed", {})
            if isinstance(roofline.get("observed"), dict)
            else {}
        )
        comparison = (
            roofline.get("comparison", {})
            if isinstance(roofline.get("comparison"), dict)
            else {}
        )

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
        if observed:
            avg_sm = observed.get("avg_sm_efficiency")
            avg_sm_str = f"{avg_sm:.1f}%" if isinstance(avg_sm, (int, float)) else "N/A"
            mem_gbps = observed.get("observed_memory_throughput_gbps")
            mem_str = (
                f"{mem_gbps:.1f} GB/s" if isinstance(mem_gbps, (int, float)) else "N/A"
            )
            lines.append(
                f"- 实测: SM效率 {avg_sm_str}, 内存 {mem_str}, "
                f"算术强度 {_fmt_ai(observed.get('observed_arithmetic_intensity'))}, 计算 {_fmt_perf(observed.get('observed_compute_flops'))}"
            )
            util_c = observed.get("compute_utilization")
            util_m = observed.get("memory_utilization")
            util_parts = []
            if isinstance(util_c, (int, float)):
                util_parts.append(f"计算利用率 {util_c * 100:.1f}%")
            if isinstance(util_m, (int, float)):
                util_parts.append(f"内存利用率 {util_m * 100:.1f}%")
            if util_parts:
                lines.append(f"- 利用率: {'，'.join(util_parts)}")
        if comparison:
            perf_gap = comparison.get("performance_gap_ratio")
            ai_gap = comparison.get("arithmetic_intensity_gap_ratio")
            bound_alignment = comparison.get("bound_alignment")
            gap_parts = []
            if isinstance(perf_gap, (int, float)):
                gap_parts.append(f"性能差 {perf_gap * 100:.1f}%")
            if isinstance(ai_gap, (int, float)):
                gap_parts.append(f"强度差 {ai_gap * 100:.1f}%")
            if bound_alignment:
                gap_parts.append(f"边界 {bound_alignment}")
            if gap_parts:
                lines.append(f"- 与预测差异: {'，'.join(gap_parts)}")
        return "\n".join(lines)

    @staticmethod
    def _generate_report_table(report_text: str) -> str:
        client = get_offline_qwen_client(OFFLINE_QWEN_PATH)
        return client.report_to_table(report_text)

    @staticmethod
    def _collect_ncu_csv_snippets(
        output_dir: Path, limit: int = 1200
    ) -> List[Tuple[str, str]]:
        snippets: List[Tuple[str, str]] = []
        if not output_dir.exists():
            return snippets
        for csv_path in sorted(output_dir.glob("ncu_kernel*.csv")):
            try:
                raw = csv_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            snippet = raw[:limit]
            if snippet.strip():
                snippets.append((csv_path.name, snippet))
        return snippets

    def _generate_optimization_suggestions(
        self, report_infos: List[Dict[str, str]]
    ) -> str:
        if not report_infos:
            return ""

        try:
            client = get_offline_qwen_client(OFFLINE_QWEN_PATH)
        except Exception as exc:
            return f"⚠️ 优化建议生成失败: {exc}"

        labeled_reports: List[Tuple[str, str]] = []
        raw_snippets: List[Tuple[str, str]] = []
        for info in report_infos:
            label = self._format_gpu_label(info["gpu"], info["index"])
            labeled_reports.append((label, info.get("text", "")))
            output_dir = Path(info["dir"])
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
                raw_suggestions = client.suggest_raw_data_optimizations(
                    raw_snippets, max_new_tokens=1024
                )
                if raw_suggestions:
                    output_sections.append(f"📊 原始数据建议:\n{raw_suggestions}")
        except Exception as exc:
            output_sections.append(f"⚠️ 原始数据建议生成失败: {exc}")

        return "\n\n".join(output_sections).strip()

    def _generate_multi_gpu_table(
        self, report_texts: List[str], gpu_labels: List[str]
    ) -> str:
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
                if merged and merged.count("|") >= len(formatted_labels) * 2:
                    return merged
        except Exception:
            pass

        return self._generate_multi_gpu_table_python(report_texts, gpu_labels)

    def _generate_multi_gpu_table_python(
        self, report_texts: List[str], gpu_labels: List[str]
    ) -> str:
        if not report_texts:
            return "⚠️ 未找到可用的报告内容"

        parsed_entries = [
            self._parse_kernel_entries_from_report(text) for text in report_texts
        ]
        if not parsed_entries or not parsed_entries[0]:
            return "⚠️ 未能解析多GPU表格数据"

        label_cells = [
            self._format_gpu_label(lbl, idx) for idx, lbl in enumerate(gpu_labels)
        ]
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
                    ints = [
                        val // common if isinstance(val, int) and val > 0 else val
                        for val in ints
                    ]
            return ints

        for idx in range(max_len):
            name_candidates = []
            for entries in parsed_entries:
                if idx < len(entries) and entries[idx]["name"]:
                    name_candidates.append(entries[idx]["name"])
            base_name = name_candidates[0] if name_candidates else f"Kernel {idx + 1}"
            alt_names = {nm for nm in name_candidates if nm != base_name}
            if alt_names:
                merged_name = base_name + " / " + " / ".join(sorted(alt_names))
            else:
                merged_name = base_name

            row_cells = [merged_name]
            duration_values: List[float] = []
            pair_ratios: Optional[List[Optional[Fraction]]] = (
                [None, None] if len(parsed_entries) >= 2 else None
            )
            for gpu_idx, entries in enumerate(parsed_entries):
                if idx < len(entries):
                    duration = entries[idx]["duration"]
                    ratio = entries[idx]["ratio"]
                    row_cells.append(duration)
                    duration_values.append(_parse_duration(duration))
                    add_ratio = len(parsed_entries) == 1 or gpu_idx != last_index
                    if add_ratio:
                        row_cells.append(ratio)
                    if pair_ratios is not None and gpu_idx < 2:
                        pair_ratios[gpu_idx] = _parse_ratio_component(ratio)
                else:
                    row_cells.append("")
                    add_ratio = len(parsed_entries) == 1 or gpu_idx != last_index
                    if add_ratio:
                        row_cells.append("")
                    if pair_ratios is not None and gpu_idx < 2:
                        pair_ratios[gpu_idx] = Fraction(0, 1)
            if pair_ratios is not None:
                simplified_ints = _fractions_to_ints(pair_ratios)
                pair_strings = []
                for val in simplified_ints:
                    if val is None:
                        pair_strings.append("")
                    else:
                        pair_strings.append(str(val))
                combined = (
                    f"{pair_strings[0]}：{pair_strings[1]}" if any(pair_strings) else ""
                )
                row_cells.append(combined)
            sort_key = max(duration_values) if duration_values else 0.0
            rows.append((sort_key, row_cells))
        sorted_rows = [
            "| " + " | ".join(cells) + " |"
            for _, cells in sorted(rows, key=lambda item: item[0], reverse=True)
        ]

        return "\n".join([header, divider, *sorted_rows])

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
    def _format_gpu_label(label: str, index: int) -> str:
        if not label:
            return f"GPU{index}"
        normalized = label.strip()
        if not normalized:
            return f"GPU{index}"
        if normalized.lower().startswith("gpu"):
            return normalized.upper()
        return f"GPU{normalized}"

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

        if summary_lines:
            return "\n".join(summary_lines)
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

    # 改为LLM识别
    def _extract_model_name(self, prompt: str) -> Optional[str]:
        """提取模型名称"""

        # 首先检查已知的模型别名
        for model_name in self.model_mappings.keys():
            if model_name.lower() in prompt.lower():
                return model_name

        # 然后使用正则表达式匹配通用模型名称模式
        patterns = [
            r"llama[^/\s]*-?\d*[^/\s]*-?\d+[bB]?",
            r"qwen[^/\s]*-?\d*[^/\s]*-?\d+[bB]?",
            r"chatglm[^/\s]*-?\d+[bB]?",
            r"baichuan[^/\s]*-?\d+[bB]?",
            r"vicuna[^/\s]*-?\d+[bB]?",
            r"mistral[^/\s]*-?\d+[bB]?",
            r"mixtral[^/\s]*-?\d+[bB]?",
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(0)

        return None

    # 改为LLM识别
    def _extract_analysis_type(self, prompt: str) -> str:
        """提取分析类型"""
        prompt_lower = prompt.lower()

        if (
            "ncu" in prompt_lower
            or "kernel" in prompt_lower
            or "深度" in prompt_lower
            or "nsight compute" in prompt_lower
        ):
            return "ncu (深度kernel分析)"
        elif (
            "nsys" in prompt_lower
            or "全局" in prompt_lower
            or "nsight systems" in prompt_lower
        ):
            return "nsys (全局性能分析)"
        elif "集成" in prompt_lower or "综合" in prompt_lower or "完整" in prompt_lower:
            return "auto (集成分析: nsys + ncu)"
        else:
            return "auto (集成分析: nsys + ncu)"

    # 改为LLM识别
    def _extract_parameters(self, prompt: str) -> Dict:
        """提取参数"""
        params = {}

        # 提取batch_size
        batch_match = re.search(
            r"batch[-_\s]*size?[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)", prompt, re.IGNORECASE
        )
        if batch_match:
            batch_sizes = [
                int(x.strip())
                for x in re.split(r"[,，\s]+", batch_match.group(1))
                if x.strip()
            ]
            params["batch_size"] = batch_sizes

        # 提取input_len
        input_match = re.search(
            r"input[-_\s]*len[gth]*[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)",
            prompt,
            re.IGNORECASE,
        )
        if input_match:
            input_lens = [
                int(x.strip())
                for x in re.split(r"[,，\s]+", input_match.group(1))
                if x.strip()
            ]
            params["input_len"] = input_lens

        # 提取output_len
        output_match = re.search(
            r"output[-_\s]*len[gth]*[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)",
            prompt,
            re.IGNORECASE,
        )
        if output_match:
            output_lens = [
                int(x.strip())
                for x in re.split(r"[,，\s]+", output_match.group(1))
                if x.strip()
            ]
            params["output_len"] = output_lens

        return params

    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        return list(self.model_mappings.keys())

    # 没用到
    def get_analysis_status(self) -> Dict:
        """获取当前分析状态"""
        return {
            "available_models": self.get_available_models(),
            "results_directory": str(self.results_dir),
            "nsys_enabled": self.profiling_config.get("nsys", {}).get("enabled", True),
            "ncu_enabled": self.profiling_config.get("ncu", {}).get("enabled", True),
        }


if __name__ == "__main__":
    # 0. CLI style 依赖项
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.styles import Style

    # 1. 加载 Config
    config_path = "/workspaces/ai-agent/AI_Agent_Complete/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 错误: 找不到 {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)

    # 2. 初始化 Agent
    print("🔄 正在初始化 AI Agent...")
    try:
        agent = AIAgent(config_yaml)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)

    # # 3. 加载知识库
    # document_dir = Path("/workspaces/ai-agent/AI_Agent_Complete/documents")
    # if document_dir.exists():
    #     print("📚 正在加载知识库文档...")
    #     count = 0
    #     for file_path in document_dir.iterdir():
    #         if file_path.is_file() and file_path.suffix in [".md", ".txt"]:
    #             agent.kb.add_document(str(file_path))
    #             count += 1
    #     print(f"✅ 已加载 {count} 个文档。")
    # else:
    #     print("⚠️ 文档目录不存在，跳过加载。")

    # 4. 对话测试
    async def interactive_chat_loop():
        style = Style.from_dict({"user-prompt": "#00aa00 bold", "text": "#ffffff"})
        session = PromptSession(history=InMemoryHistory())

        print("\n" + "=" * 60)
        print("🤖 AI 性能分析器")
        print("💡 支持指令: '分析 llama-7b' | 提问: 'kernel 有多少' | 闲聊: '你是谁'")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = await session.prompt_async(
                    HTML("<user-prompt>User ></user-prompt> "), style=style
                )
                user_input = user_input.strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\n👋 再见！")
                    break

                print("\n⏳ Agent 正在思考...")
                response = await agent.process_message(user_input)
                print("-" * 20 + " Agent 回复 " + "-" * 20)
                print(response)
                print("-" * 52 + "\n")

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")

    asyncio.run(interactive_chat_loop())
