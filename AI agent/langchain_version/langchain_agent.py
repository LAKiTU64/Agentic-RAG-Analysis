#!/usr/bin/env python3
"""
基于LangChain的AI Agent LLM性能分析器

集成LangChain框架，提供更智能的对话、工具调用和工作流程管理
"""

# Path setup for imports
import sys
from pathlib import Path
_current_dir = Path(__file__).parent
sys.path.insert(0, str(_current_dir.parent / 'original_version'))
sys.path.insert(0, str(_current_dir.parent.parent / 'TOOLS' / 'Auto_Anlyze_tool'))

import json
import asyncio
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
from pathlib import Path
import traceback

# LangChain imports
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.agents.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import tool
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import OutputParserException
from pydantic import BaseModel, Field

# 导入现有的AI Agent组件
from ai_agent_analyzer import AIAgentAnalyzer, PromptParser, ConfigGenerator, AnalysisRequest
from web_agent_backend import ConfigFileParser

# 尝试导入OpenAI，如果没有则使用本地模拟
try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class MockLLM:
    """模拟LLM类，用于没有OpenAI API时的测试"""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, prompt: str) -> str:
        return self._generate_mock_response(prompt)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """基于提示生成模拟回答"""
        if "分析" in prompt or "analyze" in prompt.lower():
            return "我将为您进行LLM性能分析。请提供模型名称和分析参数。"
        elif "配置" in prompt or "config" in prompt.lower():
            return "我已解析您的配置文件，并生成了相应的建议。"
        elif "建议" in prompt or "recommend" in prompt.lower():
            return "基于分析结果，我建议优化内存使用和计算效率。"
        else:
            return "我是LLM性能分析助手，可以帮您分析模型性能、解析配置文件、提供优化建议。"

# LangChain工具定义
class PromptAnalysisTool(BaseTool):
    """提示词分析工具"""
    
    name = "prompt_analyzer"
    description = """
    分析用户输入的自然语言提示，提取LLM性能分析需求。
    输入：用户的自然语言描述
    输出：结构化的分析请求参数
    
    示例输入：'分析llama-7b模型，batch_size=8,16'
    """
    
    def __init__(self):
        super().__init__()
        self.parser = PromptParser()
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """运行提示词分析"""
        try:
            request = self.parser.parse_prompt(query)
            result = {
                "status": "success",
                "model_name": request.model_name,
                "analysis_type": request.analysis_type,
                "batch_size": request.batch_size,
                "input_len": request.input_len,
                "output_len": request.output_len,
                "script_type": request.script_type
            }
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"提示词解析失败: {str(e)}"
            }, ensure_ascii=False)

class ConfigAnalysisTool(BaseTool):
    """配置文件分析工具"""
    
    name = "config_analyzer"
    description = """
    分析上传的JSON/YAML配置文件，提取模型信息和分析参数。
    输入：配置文件内容(JSON格式字符串)
    输出：解析的配置信息和智能建议
    """
    
    def __init__(self):
        super().__init__()
        self.parser = ConfigFileParser()
    
    def _run(
        self, 
        config_content: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """运行配置文件分析"""
        try:
            # 解析配置文件
            parsed_info = self.parser.parse_json_config(config_content)
            
            result = {
                "status": "success",
                "model_info": parsed_info["model_info"],
                "analysis_params": parsed_info["analysis_params"],
                "hardware_info": parsed_info["hardware_info"],
                "suggestions": parsed_info["suggestions"]
            }
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "error", 
                "message": f"配置文件分析失败: {str(e)}"
            }, ensure_ascii=False)

class PerformanceAnalysisTool(BaseTool):
    """性能分析执行工具"""
    
    name = "performance_analyzer"
    description = """
    执行LLM性能分析，支持nsys全局分析、ncu深度分析或集成分析。
    输入：分析请求参数(JSON格式)
    输出：分析结果和报告路径
    """
    
    def __init__(self):
        super().__init__()
        self.analyzer = AIAgentAnalyzer()
    
    def _run(
        self, 
        analysis_params: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """运行性能分析"""
        try:
            # 解析分析参数
            params = json.loads(analysis_params)
            
            # 构建提示词
            prompt = self._build_prompt_from_params(params)
            
            # 执行分析（这里简化为模拟）
            result = {
                "status": "success",
                "analysis_type": params.get("analysis_type", "auto"),
                "model_name": params.get("model_name", "unknown"),
                "message": "性能分析已启动",
                "output_dir": f"analysis_{params.get('model_name', 'model')}_{datetime.now().strftime('%H%M%S')}",
                "estimated_time": "5-10分钟"
            }
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"性能分析启动失败: {str(e)}"
            }, ensure_ascii=False)
    
    def _build_prompt_from_params(self, params: Dict) -> str:
        """从参数构建提示词"""
        model_name = params.get("model_name", "unknown")
        analysis_type = params.get("analysis_type", "auto")
        batch_size = params.get("batch_size", [8])
        
        if isinstance(batch_size, list):
            batch_str = ",".join(map(str, batch_size))
        else:
            batch_str = str(batch_size)
        
        return f"分析模型 {model_name}，使用 {analysis_type} 分析，batch_size: {batch_str}"

class OptimizationAdvisorTool(BaseTool):
    """性能优化建议工具"""
    
    name = "optimization_advisor"
    description = """
    基于分析结果提供性能优化建议和最佳实践。
    输入：分析结果数据(JSON格式)
    输出：详细的优化建议和操作步骤
    """
    
    def _run(
        self, 
        analysis_results: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """生成优化建议"""
        try:
            results = json.loads(analysis_results)
            
            # 生成基于结果的建议
            suggestions = self._generate_optimization_suggestions(results)
            
            result = {
                "status": "success",
                "optimization_suggestions": suggestions,
                "priority_actions": [
                    "检查GPU内存使用率",
                    "优化batch_size设置",
                    "考虑使用混合精度训练",
                    "分析kernel执行效率"
                ]
            }
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"生成优化建议失败: {str(e)}"
            }, ensure_ascii=False)
    
    def _generate_optimization_suggestions(self, results: Dict) -> List[str]:
        """基于结果生成建议"""
        suggestions = []
        
        model_name = results.get("model_name", "")
        analysis_type = results.get("analysis_type", "")
        
        # 基于模型类型的建议
        if "7b" in model_name.lower():
            suggestions.append("🎯 7B模型优化：推荐batch_size=8-16，使用FP16精度")
        elif "13b" in model_name.lower():
            suggestions.append("🎯 13B模型优化：考虑tensor并行，batch_size=4-8")
        
        # 基于分析类型的建议
        if analysis_type == "nsys":
            suggestions.append("📊 NSys分析建议：关注timeline中的空隙，优化kernel启动间隔")
        elif analysis_type == "ncu":
            suggestions.append("🔬 NCU分析建议：检查SM效率和内存带宽利用率")
        
        # 通用建议
        suggestions.extend([
            "💾 内存优化：启用gradient checkpointing减少内存使用",
            "⚡ 计算优化：使用FlashAttention加速注意力计算",
            "🔄 Pipeline优化：考虑使用流水线并行提高吞吐量"
        ])
        
        return suggestions

class LangChainAgent:
    """基于LangChain的AI Agent，支持可选FAISS向量检索增强 (RAG)。

    参数:
        use_openai: 是否使用 OpenAI Chat 模型
        api_key: OpenAI API Key
        enable_faiss: 是否启用 FAISS 检索增强
        faiss_embedding_model: 使用的嵌入模型名称
        faiss_index_dir: 已保存的向量索引目录 (为空则需运行构建)
    """
    
    def __init__(self, use_openai: bool = False, api_key: Optional[str] = None,
                 enable_faiss: bool = False,
                 faiss_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 faiss_index_dir: Optional[str] = None):
        self.use_openai = use_openai and HAS_OPENAI
        self.api_key = api_key
        self.enable_faiss = enable_faiss
        self.faiss_embedding_model = faiss_embedding_model
        self.faiss_index_dir = Path(faiss_index_dir) if faiss_index_dir else None
        self._faiss_store = None
        self._faiss_available = False
        
        # 初始化LLM
        if self.use_openai and api_key:
            self.llm = ChatOpenAI(
                temperature=0.1,
                openai_api_key=api_key,
                model_name="gpt-3.5-turbo"
            )
        else:
            self.llm = MockLLM()
        
        # 初始化工具
        self.tools = [
            PromptAnalysisTool(),
            ConfigAnalysisTool(),
            PerformanceAnalysisTool(),
            OptimizationAdvisorTool()
        ]
        
        # 初始化记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 创建Agent
        if self.use_openai:
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True
            )
        else:
            # 使用简化的工具调用逻辑
            self.agent = None
        
        # 创建专门的提示模板
    self.system_prompt = """你是一个专业的LLM性能分析助手。你可以：

1. 分析用户的自然语言请求，提取分析需求
2. 解析配置文件，提供优化建议
3. 执行性能分析任务
4. 基于结果提供专业的优化建议

请根据用户输入选择合适的工具来完成任务。始终保持专业、友好的语调。

可用工具：
- prompt_analyzer: 分析用户的自然语言请求
- config_analyzer: 分析配置文件
- performance_analyzer: 执行性能分析
- optimization_advisor: 提供优化建议
"""
    
    async def process_message(self, message: str, context: Dict = None) -> Dict:
        """处理用户消息"""
        try:
            # 生成增强报告触发: "完整报告" / "增强报告" / "enriched report"
            lower_msg = message.lower()
            if any(k in lower_msg for k in ["完整报告", "增强报告", "enriched report"]):
                try:
                    from backend.report_generator import generate_enriched_report
                    # 默认使用最近一次分析目录: 通过配置或上下文传入 output_dir
                    # 如果 context 指定 output_dir, 使用它; 否则尝试 config 中的 last_analysis_dir
                    target_dir = None
                    if context and isinstance(context, dict):
                        target_dir = context.get('output_dir')
                    if not target_dir and hasattr(self, 'last_analysis_dir'):
                        target_dir = getattr(self, 'last_analysis_dir')
                    if not target_dir:
                        # 回退默认主目录
                        target_dir = "/workspace/Agent/AI_Agent_Complete"
                    from pathlib import Path
                    enriched_path = generate_enriched_report(Path(target_dir))
                    response_text = f"📘 已生成增强版性能报告\n路径: {enriched_path}\n您可以查看该 Markdown 文件以获取详细的理论支撑与优化建议。"
                    return {
                        "status": "success",
                        "response": response_text,
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as rep_e:
                    return {
                        "status": "error",
                        "response": f"生成增强报告失败: {rep_e}",
                        "timestamp": datetime.now().isoformat()
                    }
            # 如果启用了 FAISS，尝试加载或构建索引
                try:
                    # 模型信息提取与理论预取 (只做一次快速解析，不阻塞主流程)
                    from backend.model_intel import extract_model_info, prefetch_theory_snippets
                    model_info = extract_model_info(message)
                    theory_prefetch = None
                    if model_info.get('detected') and self.enable_faiss:
                        theory_prefetch = prefetch_theory_snippets(model_info)
                except Exception:
                    model_info = {"detected": False}
                    theory_prefetch = None

                try:
                    if self.enable_faiss:
                        self._init_faiss_store()

            if self.agent:
                # 使用LangChain Agent
                response = await self._process_with_langchain(message, context)
            else:
                # 使用简化逻辑
                response = await self._process_with_simple_logic(message, context)
            
            return {
                "status": "success",
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "response": f"处理消息时出错: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _process_with_langchain(self, message: str, context: Dict = None) -> str:
        """使用LangChain处理消息"""
        try:
            # 添加上下文到消息
            if context:
                message = f"上下文: {json.dumps(context, ensure_ascii=False)}\n\n用户消息: {message}"
            
            # 调用Agent
            result = self.agent.run(message)
            return result
            
        except Exception as e:
            return f"LangChain处理出错: {str(e)}"
    
    async def _process_with_simple_logic(self, message: str, context: Dict = None) -> str:
        """使用简化逻辑处理消息"""
        message_lower = message.lower()
        
        # 判断意图并调用相应工具
        # 自动性能分析意图: 包含"分析"并检测到模型
        if ("分析" in message or "analyze" in message_lower) and 'model_path' in message_lower or model_info.get('detected'):
            # 尝试提取 batch/input/output 参数
            batch_match = re.search(r"batch[_ ]?size[=：: ]?(\d+)", message_lower)
            input_match = re.search(r"input[_ ]?len[gth]*[=：: ]?(\d+)", message_lower)
            output_match = re.search(r"output[_ ]?len[gth]*[=：: ]?(\d+)", message_lower)
            batch_size = int(batch_match.group(1)) if batch_match else 8
            input_len = int(input_match.group(1)) if input_match else 2048
            output_len = int(output_match.group(1)) if output_match else 1024
            # 近似提取模型路径 (含 /workspace/models 或 含 qwen/llama 字段)
            model_path = None
            path_candidates = re.findall(r"/workspace/models[\w/\-_.]+", message)
            if path_candidates:
                model_path = path_candidates[0]
            if not model_path and model_info.get('raw_match'):
                # 用户可能只写了 qwen3-32b，拼接默认前缀 (假设路径规则)
                family = model_info.get('family')
                size = int(model_info.get('size_b') or 0)
                if family and size:
                    # 简化映射: /workspace/models/<Family>/<Family><Size>B
                    model_path = f"/workspace/models/{family.capitalize()}/{family}{size}B"
            # 若仍无路径则提示
            if not model_path:
                return "请提供模型路径 (例如 /workspace/models/Qwen/Qwen3-32B) 以启动自动分析。"
            # 调用一体化性能分析工具
            try:
                from backend.tools.perf_tools import RunIntegratedPerfAnalysisTool
                tool = RunIntegratedPerfAnalysisTool()
                payload = json.dumps({
                    "model_path": model_path,
                    "batch_size": batch_size,
                    "input_len": input_len,
                    "output_len": output_len,
                    "max_kernels": 3
                }, ensure_ascii=False)
                result_json = tool._run(payload)
                return result_json
            except Exception as e:
                return f"自动分析触发失败: {e}"

        if "分析" in message or "analyze" in message_lower:
            # 使用提示词分析工具
            tool = self.tools[0]  # PromptAnalysisTool
            result = tool._run(message)
            result_data = json.loads(result)
            
            if result_data["status"] == "success":
                return f"""✅ 我已分析您的请求：

🤖 **模型**: {result_data['model_name']}
🔬 **分析类型**: {result_data['analysis_type']}
📊 **批次大小**: {result_data['batch_size']}
📏 **输入长度**: {result_data['input_len']}

接下来我将为您配置分析参数并开始执行。您需要我立即开始分析吗？"""
            else:
                return f"❌ 分析请求解析失败: {result_data['message']}"
        
        elif "配置" in message or "config" in message_lower:
            if context and "config_content" in context:
                # 使用配置分析工具
                tool = self.tools[1]  # ConfigAnalysisTool
                result = tool._run(context["config_content"])
                result_data = json.loads(result)
                
                if result_data["status"] == "success":
                    suggestions = result_data["suggestions"]
                    return f"""📄 **配置文件分析完成**

🤖 **模型信息**: 已识别模型参数
⚙️ **分析参数**: 已提取配置
💡 **智能建议**:
{chr(10).join(f'• {s}' for s in suggestions[:5])}

基于您的配置，我推荐进行集成性能分析。要开始吗？"""
                else:
                    return f"❌ 配置文件分析失败: {result_data['message']}"
        
        elif "建议" in message or "优化" in message or "recommend" in message_lower:
            # 使用优化建议工具
            tool = self.tools[3]  # OptimizationAdvisorTool
            mock_results = json.dumps({"model_name": "general", "analysis_type": "auto"})
            result = tool._run(mock_results)
            result_data = json.loads(result)
            
            if result_data["status"] == "success":
                suggestions = result_data["optimization_suggestions"]
                return f"""💡 **性能优化建议**

{chr(10).join(suggestions)}

🎯 **优先行动**:
{chr(10).join(f'• {a}' for a in result_data['priority_actions'])}

需要我详细解释任何一项建议吗？"""
        
        # 如果启用 FAISS 且存在“检索”或“查询”意图
        if self.enable_faiss and any(k in message_lower for k in ["检索", "查询", "search", "retrieve", "向量", "知识库"]):
            if not self._faiss_available:
                return "⚠️ 向量知识库尚未初始化，先提供原始文本或指定索引目录。"
            results = self.query_faiss(message)
            if not results:
                return "🔍 未在向量知识库中找到相关内容。"
            formatted = "\n".join([
                f"• (score={r['score']:.4f}) {r['content'][:120]}" for r in results
            ])
            theory_extra = ""
            if model_info.get('detected') and theory_prefetch and theory_prefetch.get('snippets'):
                merged = []
                for q, snips in theory_prefetch['snippets'].items():
                    if snips:
                        merged.append(f"Q: {q}\n  - " + "\n  - ".join(s[:100] for s in snips[:2]))
                if merged:
                    theory_extra = "\n\n📎 预取模型相关理论:\n" + "\n".join(merged[:3])
            return f"📚 检索到以下相关片段:\n{formatted}{theory_extra}\n\n请继续提问或要求生成总结。"

        # 默认响应
        return f"""👋 您好！我是LLM性能分析助手。

我可以帮您：
🔍 **分析请求** - 解析您的性能分析需求
📁 **配置解析** - 分析上传的配置文件
⚡ **性能分析** - 执行nsys/ncu分析
💡 **优化建议** - 提供专业的性能优化建议

请告诉我您需要什么帮助，或者直接说出分析需求，比如：
"分析llama-7b模型的性能"
"对qwen-14b进行ncu深度分析"
"""
    
    def add_uploaded_file(self, file_content: str, filename: str) -> Dict:
        """添加上传的文件到上下文"""
        try:
            # 解析文件内容
            tool = self.tools[1]  # ConfigAnalysisTool
            result = tool._run(file_content)
            result_data = json.loads(result)
            
            if result_data["status"] == "success":
                return {
                    "status": "success",
                    "message": f"已成功解析文件 {filename}",
                    "suggestions": result_data["suggestions"]
                }
            else:
                return {
                    "status": "error",
                    "message": f"文件解析失败: {result_data['message']}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"处理文件时出错: {str(e)}"
            }
    
    def get_memory_summary(self) -> str:
        """获取对话记忆摘要"""
        if hasattr(self.memory, 'buffer'):
            messages = self.memory.buffer[-5:]  # 最近5条消息
            return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
        return "无对话历史"

    # ================= FAISS 集成相关 =================
    def _init_faiss_store(self) -> None:
        """初始化或加载 FAISS 向量库。"""
        try:
            from backend.knowledge_bases.faiss_store import load_index, build_index  # 动态导入避免无依赖时失败
        except ImportError:
            print("⚠️ 未找到 faiss_store 模块或相关依赖，向量检索功能禁用。")
            self._faiss_available = False
            return
        try:
            if self.faiss_index_dir and self.faiss_index_dir.exists():
                self._faiss_store = load_index(self.faiss_index_dir, model_name=self.faiss_embedding_model)
                self._faiss_available = True
                print(f"✅ 已加载 FAISS 索引: {self.faiss_index_dir}")
            else:
                # 延迟构建一个空索引（需要后续添加文本）
                self._faiss_store = None
                self._faiss_available = False
        except Exception as e:
            print(f"⚠️ 加载 FAISS 索引失败: {e}")
            self._faiss_available = False

    def add_texts_to_faiss(self, texts: List[str]) -> str:
        """向 FAISS 知识库添加原始文本并重建索引。"""
        if not self.enable_faiss:
            return "❌ 未启用 FAISS 检索功能"
        try:
            from backend.knowledge_bases.faiss_store import build_index, save_index
            self._faiss_store = build_index(texts, model_name=self.faiss_embedding_model)
            # 自动保存到默认目录 (若未指定则创建临时目录)
            target_dir = self.faiss_index_dir or Path("/workspace/Agent/AI_Agent_Complete/faiss_index")
            save_index(self._faiss_store, target_dir)
            self.faiss_index_dir = target_dir
            self._faiss_available = True
            return f"✅ 已构建并保存向量索引到: {target_dir}"
        except Exception as e:
            return f"❌ 构建 FAISS 索引失败: {e}"

    def query_faiss(self, question: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """在 FAISS 向量库中执行检索。"""
        if not self._faiss_available or self._faiss_store is None:
            return []
        try:
            from backend.knowledge_bases.faiss_store import query
            return query(self._faiss_store, question, top_k=top_k)
        except Exception:
            return []

    def reload_faiss_index(self) -> str:
        """重新加载磁盘上的 FAISS 索引 (用于前端上传新知识后刷新)。"""
        if not self.enable_faiss:
            return "❌ 未启用 FAISS 检索功能"
        if not self.faiss_index_dir or not self.faiss_index_dir.exists():
            return "❌ 索引目录不存在，无法加载"
        try:
            from backend.knowledge_bases.faiss_store import load_index
            self._faiss_store = load_index(self.faiss_index_dir, model_name=self.faiss_embedding_model)
            self._faiss_available = True
            return f"✅ 已刷新索引: {self.faiss_index_dir}"
        except Exception as e:
            return f"❌ 刷新索引失败: {e}"

# 使用示例函数
async def test_langchain_agent():
    """测试LangChain Agent"""
    print("🧪 测试LangChain Agent...")
    
    # 创建Agent
    agent = LangChainAgent(use_openai=False)
    
    # 测试消息
    test_messages = [
        "分析 llama-7b 模型，batch_size=8,16",
        "我需要性能优化建议",
        "如何提高GPU利用率？"
    ]
    
    for message in test_messages:
        print(f"\n👤 用户: {message}")
        result = await agent.process_message(message)
        print(f"🤖 AI Agent: {result['response']}")

if __name__ == "__main__":
    asyncio.run(test_langchain_agent())


