#!/usr/bin/env python3
"""
LangChain工作流程链

定义复杂的LLM性能分析工作流程，使用LangChain的链式调用
"""

from typing import Dict, List, Any, Optional
import json
from datetime import datetime

# LangChain imports
from langchain.chains.base import Chain
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.callbacks.manager import CallbackManagerForChainRun
from pydantic import BaseModel, Field

class AnalysisWorkflowInput(BaseModel):
    """分析工作流输入"""
    user_request: str = Field(description="用户请求")
    config_data: Optional[Dict] = Field(default=None, description="配置数据")
    context: Optional[Dict] = Field(default=None, description="上下文信息")

class AnalysisWorkflowOutput(BaseModel):
    """分析工作流输出"""
    analysis_plan: Dict = Field(description="分析计划")
    execution_steps: List[Dict] = Field(description="执行步骤")
    recommendations: List[str] = Field(description="建议")
    estimated_time: str = Field(description="预估时间")

class ModelAnalysisChain(Chain):
    """模型分析链 - 负责分析模型特性和需求"""
    
    input_key: str = "user_request"
    output_key: str = "model_analysis"
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """执行模型分析"""
        
        user_request = inputs[self.input_key]
        
        # 分析用户请求中的模型信息
        analysis = self._analyze_model_requirements(user_request)
        
        return {self.output_key: analysis}
    
    def _analyze_model_requirements(self, request: str) -> Dict:
        """分析模型需求"""
        
        # 提取模型信息
        model_info = {
            "detected_model": None,
            "model_size": None,
            "analysis_type": "auto",
            "priority_metrics": []
        }
        
        request_lower = request.lower()
        
        # 检测模型名称
        model_patterns = {
            "llama-7b": {"size": "7B", "type": "decoder", "memory": "~13GB"},
            "llama-13b": {"size": "13B", "type": "decoder", "memory": "~26GB"},
            "qwen-7b": {"size": "7B", "type": "decoder", "memory": "~13GB"},
            "qwen-14b": {"size": "14B", "type": "decoder", "memory": "~28GB"},
            "chatglm-6b": {"size": "6B", "type": "encoder-decoder", "memory": "~12GB"},
            "baichuan-7b": {"size": "7B", "type": "decoder", "memory": "~13GB"},
            "baichuan-13b": {"size": "13B", "type": "decoder", "memory": "~26GB"}
        }
        
        for model_name, info in model_patterns.items():
            if model_name.replace("-", "") in request_lower.replace("-", ""):
                model_info["detected_model"] = model_name
                model_info["model_size"] = info["size"]
                model_info["architecture"] = info["type"]
                model_info["estimated_memory"] = info["memory"]
                break
        
        # 检测分析类型
        if "nsys" in request_lower:
            model_info["analysis_type"] = "nsys"
            model_info["priority_metrics"] = ["timeline", "kernel_distribution", "memory_transfers"]
        elif "ncu" in request_lower:
            model_info["analysis_type"] = "ncu"
            model_info["priority_metrics"] = ["sm_efficiency", "occupancy", "memory_bandwidth"]
        elif "深度" in request_lower or "kernel" in request_lower:
            model_info["analysis_type"] = "ncu"
            model_info["priority_metrics"] = ["kernel_analysis", "bottleneck_detection"]
        elif "全局" in request_lower or "timeline" in request_lower:
            model_info["analysis_type"] = "nsys"
            model_info["priority_metrics"] = ["global_timeline", "api_analysis"]
        else:
            model_info["analysis_type"] = "auto"
            model_info["priority_metrics"] = ["comprehensive_analysis"]
        
        return model_info
    
    @property
    def _chain_type(self) -> str:
        return "model_analysis"

class ConfigOptimizationChain(Chain):
    """配置优化链 - 基于模型分析优化配置"""
    
    input_key: str = "model_analysis"
    output_key: str = "optimized_config"
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """执行配置优化"""
        
        model_analysis = inputs[self.input_key]
        user_config = inputs.get("config_data", {})
        
        # 基于模型分析优化配置
        optimized = self._optimize_config(model_analysis, user_config)
        
        return {self.output_key: optimized}
    
    def _optimize_config(self, model_analysis: Dict, user_config: Dict) -> Dict:
        """优化配置参数"""
        
        optimized_config = {
            "batch_size": [1, 4, 8],
            "input_len": [512, 1024],
            "output_len": [64, 128],
            "analysis_params": {}
        }
        
        detected_model = model_analysis.get("detected_model", "")
        model_size = model_analysis.get("model_size", "")
        
        # 基于模型大小优化batch_size
        if "7B" in model_size or "6B" in model_size:
            optimized_config["batch_size"] = [1, 4, 8, 16]
            optimized_config["recommended_batch"] = 8
        elif "13B" in model_size or "14B" in model_size:
            optimized_config["batch_size"] = [1, 2, 4, 8]
            optimized_config["recommended_batch"] = 4
        elif "70B" in model_size:
            optimized_config["batch_size"] = [1, 2]
            optimized_config["recommended_batch"] = 1
        
        # 基于分析类型优化
        analysis_type = model_analysis.get("analysis_type", "auto")
        if analysis_type == "ncu":
            optimized_config["analysis_params"] = {
                "profile_steps": 5,
                "detailed_metrics": True,
                "kernel_filter": "top_10"
            }
        elif analysis_type == "nsys":
            optimized_config["analysis_params"] = {
                "trace_apis": ["cuda", "nvtx", "osrt"],
                "memory_usage": True,
                "timeline_detail": "high"
            }
        
        # 合并用户配置
        if user_config:
            for key, value in user_config.items():
                if key in optimized_config:
                    optimized_config[key] = value
        
        return optimized_config
    
    @property
    def _chain_type(self) -> str:
        return "config_optimization"

class ExecutionPlanChain(Chain):
    """执行计划链 - 生成详细的执行步骤"""
    
    input_key: str = "optimized_config"
    output_key: str = "execution_plan"
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """生成执行计划"""
        
        optimized_config = inputs[self.input_key]
        model_analysis = inputs.get("model_analysis", {})
        
        # 生成执行计划
        plan = self._generate_execution_plan(model_analysis, optimized_config)
        
        return {self.output_key: plan}
    
    def _generate_execution_plan(self, model_analysis: Dict, config: Dict) -> Dict:
        """生成详细执行计划"""
        
        analysis_type = model_analysis.get("analysis_type", "auto")
        detected_model = model_analysis.get("detected_model", "unknown")
        
        base_steps = [
            {
                "step": 1,
                "name": "环境检查",
                "description": "检查GPU环境和工具可用性",
                "estimated_time": "30秒",
                "command": "nvidia-smi && nsys --version"
            },
            {
                "step": 2,
                "name": "模型配置",
                "description": f"配置{detected_model}模型参数",
                "estimated_time": "1分钟",
                "parameters": config
            }
        ]
        
        # 根据分析类型添加步骤
        if analysis_type == "nsys":
            base_steps.extend([
                {
                    "step": 3,
                    "name": "NSys全局分析",
                    "description": "运行Nsight Systems进行全局性能分析",
                    "estimated_time": "3-5分钟",
                    "command": f"nsys profile -o {detected_model}_profile"
                },
                {
                    "step": 4,
                    "name": "结果解析",
                    "description": "解析nsys输出并生成报告",
                    "estimated_time": "1-2分钟"
                }
            ])
        elif analysis_type == "ncu":
            base_steps.extend([
                {
                    "step": 3,
                    "name": "热点识别",
                    "description": "先用nsys识别热点kernels",
                    "estimated_time": "2-3分钟"
                },
                {
                    "step": 4,
                    "name": "NCU深度分析",
                    "description": "使用Nsight Compute分析热点kernels",
                    "estimated_time": "5-10分钟",
                    "command": "ncu --set full -o kernel_analysis"
                },
                {
                    "step": 5,
                    "name": "瓶颈分析",
                    "description": "分析kernel性能瓶颈",
                    "estimated_time": "1-2分钟"
                }
            ])
        else:  # auto
            base_steps.extend([
                {
                    "step": 3,
                    "name": "集成分析第一阶段",
                    "description": "NSys全局分析识别热点",
                    "estimated_time": "3-5分钟"
                },
                {
                    "step": 4,
                    "name": "集成分析第二阶段", 
                    "description": "NCU深度分析热点kernels",
                    "estimated_time": "5-8分钟"
                },
                {
                    "step": 5,
                    "name": "综合报告生成",
                    "description": "生成综合性能分析报告",
                    "estimated_time": "1-2分钟"
                }
            ])
        
        # 添加最终步骤
        base_steps.append({
            "step": len(base_steps) + 1,
            "name": "报告生成",
            "description": "生成可视化图表和优化建议",
            "estimated_time": "30秒"
        })
        
        total_time = sum([self._parse_time(step.get("estimated_time", "1分钟")) 
                         for step in base_steps])
        
        return {
            "steps": base_steps,
            "total_steps": len(base_steps),
            "estimated_total_time": f"{total_time}分钟",
            "analysis_type": analysis_type,
            "priority": "normal" if total_time <= 10 else "high"
        }
    
    def _parse_time(self, time_str: str) -> float:
        """解析时间字符串为分钟"""
        if "秒" in time_str:
            return float(time_str.replace("秒", "")) / 60
        elif "分钟" in time_str:
            parts = time_str.replace("分钟", "").split("-")
            if len(parts) == 2:
                return (float(parts[0]) + float(parts[1])) / 2
            else:
                return float(parts[0])
        return 1.0
    
    @property
    def _chain_type(self) -> str:
        return "execution_plan"

class RecommendationChain(Chain):
    """建议生成链 - 基于分析计划生成建议"""
    
    input_key: str = "execution_plan"
    output_key: str = "recommendations"
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """生成建议"""
        
        execution_plan = inputs[self.input_key]
        model_analysis = inputs.get("model_analysis", {})
        
        # 生成建议
        recommendations = self._generate_recommendations(model_analysis, execution_plan)
        
        return {self.output_key: recommendations}
    
    def _generate_recommendations(self, model_analysis: Dict, execution_plan: Dict) -> List[str]:
        """生成智能建议"""
        
        recommendations = []
        
        detected_model = model_analysis.get("detected_model", "")
        analysis_type = execution_plan.get("analysis_type", "auto")
        total_time = execution_plan.get("estimated_total_time", "")
        
        # 基于模型的建议
        if "7b" in detected_model.lower():
            recommendations.extend([
                "🎯 7B模型优化建议：推荐使用FP16精度以减少内存占用",
                "📊 batch_size建议：8-16为最佳范围，平衡吞吐量和延迟",
                "🚀 加速建议：考虑使用FlashAttention优化注意力计算"
            ])
        elif "13b" in detected_model.lower() or "14b" in detected_model.lower():
            recommendations.extend([
                "🎯 13B/14B模型优化建议：建议使用Tensor并行提高性能",
                "📊 batch_size建议：4-8为推荐范围，避免内存溢出",
                "💾 内存优化：启用gradient checkpointing节省内存"
            ])
        
        # 基于分析类型的建议
        if analysis_type == "nsys":
            recommendations.extend([
                "📈 NSys分析重点：关注kernel执行timeline和并行度",
                "🔍 优化方向：识别kernel间的空隙时间，提高GPU利用率",
                "📊 监控指标：重点关注CUDA API调用开销"
            ])
        elif analysis_type == "ncu":
            recommendations.extend([
                "🔬 NCU分析重点：深度分析SM效率和内存带宽",
                "⚡ 优化方向：提高occupancy和减少warp停顿",
                "🎯 关键指标：SM efficiency, Memory bandwidth, Tensor Core利用率"
            ])
        
        # 基于执行时间的建议
        if "10" in total_time or "15" in total_time:
            recommendations.append("⏰ 分析时间较长，建议在非高峰时间运行")
        
        # 通用建议
        recommendations.extend([
            "💡 建议先运行快速分析验证环境配置",
            "📝 分析期间保持GPU负载稳定，避免其他任务干扰",
            "🔄 建议对比多个batch_size的性能表现",
            "📊 关注分析结果中的性能瓶颈识别和优化建议"
        ])
        
        return recommendations
    
    @property
    def _chain_type(self) -> str:
        return "recommendation"

class PerformanceAnalysisWorkflow:
    """性能分析工作流程管理器"""
    
    def __init__(self):
        # 创建工作流程链
        self.model_chain = ModelAnalysisChain()
        self.config_chain = ConfigOptimizationChain()
        self.execution_chain = ExecutionPlanChain()
        self.recommendation_chain = RecommendationChain()
        
        # 组合成顺序链
        self.workflow_chain = SequentialChain(
            chains=[
                self.model_chain,
                self.config_chain, 
                self.execution_chain,
                self.recommendation_chain
            ],
            input_variables=["user_request", "config_data"],
            output_variables=["model_analysis", "optimized_config", "execution_plan", "recommendations"],
            verbose=True
        )
    
    async def run_workflow(self, user_request: str, config_data: Dict = None) -> Dict:
        """运行完整工作流程"""
        
        try:
            inputs = {
                "user_request": user_request,
                "config_data": config_data or {}
            }
            
            # 执行工作流程
            results = self.workflow_chain(inputs)
            
            # 整理输出
            workflow_output = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "workflow_results": {
                    "model_analysis": results["model_analysis"],
                    "optimized_config": results["optimized_config"],
                    "execution_plan": results["execution_plan"],
                    "recommendations": results["recommendations"]
                },
                "summary": self._generate_summary(results)
            }
            
            return workflow_output
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"工作流程执行失败: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_summary(self, results: Dict) -> Dict:
        """生成工作流程摘要"""
        
        model_analysis = results.get("model_analysis", {})
        execution_plan = results.get("execution_plan", {})
        recommendations = results.get("recommendations", [])
        
        return {
            "detected_model": model_analysis.get("detected_model", "unknown"),
            "analysis_type": model_analysis.get("analysis_type", "auto"),
            "total_steps": execution_plan.get("total_steps", 0),
            "estimated_time": execution_plan.get("estimated_total_time", "未知"),
            "recommendations_count": len(recommendations),
            "workflow_complexity": "simple" if execution_plan.get("total_steps", 0) <= 4 else "complex"
        }

# 使用示例
async def test_workflow():
    """测试工作流程"""
    
    print("🧪 测试LangChain工作流程...")
    
    workflow = PerformanceAnalysisWorkflow()
    
    test_requests = [
        "分析 llama-7b 模型，使用 nsys 进行全局性能分析",
        "对 qwen-14b 进行 ncu 深度kernel分析，batch_size=4,8", 
        "综合分析 chatglm-6b 的性能瓶颈"
    ]
    
    for request in test_requests:
        print(f"\n👤 请求: {request}")
        
        result = await workflow.run_workflow(request)
        
        if result["status"] == "success":
            summary = result["summary"]
            print(f"🤖 检测模型: {summary['detected_model']}")
            print(f"📊 分析类型: {summary['analysis_type']}")
            print(f"⏱️ 预估时间: {summary['estimated_time']}")
            print(f"💡 建议数量: {summary['recommendations_count']}")
        else:
            print(f"❌ 错误: {result['message']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_workflow())


