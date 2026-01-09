#!/usr/bin/env python3
"""
AI Agent for Automatic LLM Performance Analysis

这个AI Agent能够：
1. 解析用户提示词，自动配置SGlang脚本参数
2. 根据需求运行nsys/ncu性能分析
3. 调用分析脚本生成详细报告

作者: AI助手
版本: 1.0
"""

import os
import sys
import re
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import shlex

# 导入分析工具
tools_dir = Path(__file__).parent.parent.parent / "TOOLS" / "Auto_Anlyze_tool"
if tools_dir.exists():
    sys.path.append(str(tools_dir))
    try:
        from nsys_parser import NsysParser, NsysAnalyzer, NsysVisualizer, NsysReporter
        from ncu_parser import NCUParser, NCUAnalyzer, NCUVisualizer, NCUReporter  
        from nsys_to_ncu_analyzer import NSysToNCUAnalyzer
    except ImportError as e:
        print(f"⚠️  警告: 无法导入分析工具: {e}")
        print("请确保 TOOLS/Auto_Anlyze_tool/ 目录存在且包含相关脚本")
else:
    print(f"⚠️  警告: 分析工具目录不存在: {tools_dir}")
    # 创建占位符类，避免运行时错误
    class MockAnalyzer:
        def __init__(self, *args, **kwargs): pass
        def parse(self): pass
        def analyze(self): return {}
        def create_visualizations(self): pass
        def generate_report(self): pass
    
    NsysParser = NsysAnalyzer = NsysVisualizer = NsysReporter = MockAnalyzer
    NCUParser = NCUAnalyzer = NCUVisualizer = NCUReporter = MockAnalyzer
    NSysToNCUAnalyzer = MockAnalyzer

@dataclass
class AnalysisRequest:
    """分析请求的数据结构"""
    # 基本信息
    model_name: str
    script_type: str = "bench_one_batch_server"  # bench_one_batch_server, launch_server等
    analysis_type: str = "auto"  # nsys, ncu, auto(集成分析)
    
    # 脚本参数
    batch_size: List[int] = None
    input_len: List[int] = None 
    output_len: List[int] = None
    temperature: float = 0.0
    trust_device: bool = True
    
    # 分析参数
    profile_steps: int = 3
    profile_by_stage: bool = False
    max_ncu_kernels: int = 5
    output_dir: str = None
    
    # 高级参数
    tp_size: int = 1
    host: str = "127.0.0.1"
    port: int = 30000
    
    def __post_init__(self):
        if self.batch_size is None:
            self.batch_size = [1, 8, 16]
        if self.input_len is None:
            self.input_len = [512, 1024]
        if self.output_len is None:
            self.output_len = [64, 128]
        if self.output_dir is None:
            safe_model_name = re.sub(r'[^\w\-_]', '_', self.model_name)
            self.output_dir = f"analysis_{safe_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

class PromptParser:
    """提示词解析器"""
    
    def __init__(self):
        # 模型名称模式
        self.model_patterns = [
            r'--model[=\s]+([^\s]+)',
            r'模型[：:]\s*([^\s,，]+)',
            r'model[：:]\s*([^\s,，]+)',
        ]
        
        # 脚本类型模式
        self.script_patterns = {
            'bench_one_batch_server': [
                r'bench_one_batch_server',
                r'batch.*server', 
                r'单批次.*服务器',
                r'benchmarking?'
            ],
            'launch_server': [
                r'launch_server',
                r'启动.*服务器',
                r'server.*launch'
            ]
        }
        
        # 分析类型模式
        self.analysis_patterns = {
            'nsys': [
                r'nsys',
                r'nsight.*systems?',
                r'全局.*分析',
                r'timeline.*分析'
            ],
            'ncu': [
                r'ncu', 
                r'nsight.*compute',
                r'kernel.*分析',
                r'算子.*分析',
                r'深度.*分析'
            ],
            'auto': [
                r'集成.*分析',
                r'综合.*分析', 
                r'auto.*analy',
                r'完整.*分析'
            ]
        }
    
    def parse_prompt(self, prompt: str) -> AnalysisRequest:
        """解析用户提示词"""
        print(f"🔍 正在解析提示词: {prompt}")
        
        # 提取模型名称
        model_name = self._extract_model_name(prompt)
        if not model_name:
            raise ValueError("未能从提示词中提取模型名称，请明确指定模型")
        
        # 提取脚本类型
        script_type = self._extract_script_type(prompt)
        
        # 提取分析类型  
        analysis_type = self._extract_analysis_type(prompt)
        
        # 提取参数
        params = self._extract_parameters(prompt)
        
        request = AnalysisRequest(
            model_name=model_name,
            script_type=script_type,
            analysis_type=analysis_type,
            **params
        )
        
        print(f"✅ 解析结果:")
        print(f"  - 模型: {request.model_name}")
        print(f"  - 脚本类型: {request.script_type}")  
        print(f"  - 分析类型: {request.analysis_type}")
        print(f"  - 批次大小: {request.batch_size}")
        print(f"  - 输入长度: {request.input_len}")
        print(f"  - 输出长度: {request.output_len}")
        
        return request
    
    def _extract_model_name(self, prompt: str) -> Optional[str]:
        """提取模型名称"""
        for pattern in self.model_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # 尝试提取常见模型名称
        common_models = [
            r'llama[^/]*-?\d*[^/]*-?\d+[bB]?',
            r'qwen[^/]*-?\d*[^/]*-?\d+[bB]?',
            r'chatglm[^/]*-?\d+[bB]?',
            r'baichuan[^/]*-?\d+[bB]?',
            r'vicuna[^/]*-?\d+[bB]?'
        ]
        
        for pattern in common_models:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _extract_script_type(self, prompt: str) -> str:
        """提取脚本类型"""
        for script_type, patterns in self.script_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    return script_type
        return "bench_one_batch_server"  # 默认
    
    def _extract_analysis_type(self, prompt: str) -> str:
        """提取分析类型"""
        for analysis_type, patterns in self.analysis_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    return analysis_type
        return "auto"  # 默认集成分析
    
    def _extract_parameters(self, prompt: str) -> Dict:
        """提取参数"""
        params = {}
        
        # 提取批次大小
        batch_match = re.search(r'batch[-_\s]*size?[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)', prompt, re.IGNORECASE)
        if batch_match:
            batch_sizes = [int(x.strip()) for x in re.split(r'[,，\s]+', batch_match.group(1))]
            params['batch_size'] = batch_sizes
        
        # 提取输入长度
        input_match = re.search(r'input[-_\s]*len[gth]*[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)', prompt, re.IGNORECASE)
        if input_match:
            input_lens = [int(x.strip()) for x in re.split(r'[,，\s]+', input_match.group(1))]
            params['input_len'] = input_lens
        
        # 提取输出长度
        output_match = re.search(r'output[-_\s]*len[gth]*[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)', prompt, re.IGNORECASE)
        if output_match:
            output_lens = [int(x.strip()) for x in re.split(r'[,，\s]+', output_match.group(1))]
            params['output_len'] = output_lens
        
        # 提取温度
        temp_match = re.search(r'temperature[：:\s=]*([0-9.]+)', prompt, re.IGNORECASE)
        if temp_match:
            params['temperature'] = float(temp_match.group(1))
        
        # 提取tensor并行度
        tp_match = re.search(r'tp[-_\s]*size[：:\s=]*(\d+)', prompt, re.IGNORECASE)
        if tp_match:
            params['tp_size'] = int(tp_match.group(1))
        
        return params

class ConfigGenerator:
    """参数配置生成器"""
    
    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root)
        self.models_dir = self.workspace_root / "workspace" / "models"
        
    def generate_sglang_config(self, request: AnalysisRequest) -> Dict:
        """生成SGlang脚本配置"""
        
        # 查找模型路径
        model_path = self._resolve_model_path(request.model_name)
        
        config = {
            # 服务器参数
            'model_path': model_path,
            'host': request.host,
            'port': request.port,
            'tp_size': request.tp_size,
            'trust_remote_code': request.trust_device,
            
            # 基准测试参数
            'batch_size': request.batch_size,
            'input_len': request.input_len, 
            'output_len': request.output_len,
            'temperature': request.temperature,
            
            # 分析参数
            'profile': True,
            'profile_steps': request.profile_steps,
            'profile_by_stage': request.profile_by_stage,
            
            # 输出配置
            'show_report': True,
            'result_filename': f"{request.output_dir}/benchmark_results.jsonl"
        }
        
        print(f"📋 生成的配置:")
        print(f"  - 模型路径: {model_path}")
        print(f"  - TP大小: {request.tp_size}")
        print(f"  - 批次大小: {request.batch_size}")
        
        return config
    
    def _resolve_model_path(self, model_name: str) -> str:
        """解析模型路径"""
        
        # 如果是绝对路径，直接返回
        if Path(model_name).is_absolute():
            return model_name
        
        # 在workspace/models下查找
        possible_paths = [
            self.models_dir / model_name,
            self.models_dir / model_name.replace('/', '_'),
            self.models_dir / model_name.split('/')[-1],
        ]
        
        for path in possible_paths:
            if path.exists():
                print(f"✅ 找到模型路径: {path}")
                return str(path)
        
        # 如果本地找不到，假设是HuggingFace模型ID
        print(f"⚠️  本地未找到模型，使用HuggingFace ID: {model_name}")
        return model_name
    
    def build_command(self, request: AnalysisRequest, config: Dict) -> List[str]:
        """构建SGlang执行命令"""
        
        if request.script_type == "bench_one_batch_server":
            cmd = [
                'python', '-m', 'sglang.bench_one_batch_server',
                '--model', config['model_path'],
                '--host', config['host'],
                '--port', str(config['port']),
                '--tp-size', str(config['tp_size']),
                '--temperature', str(config['temperature']),
                '--batch-size'] + [str(bs) for bs in config['batch_size']] + [
                '--input-len'] + [str(il) for il in config['input_len']] + [
                '--output-len'] + [str(ol) for ol in config['output_len']] + [
                '--result-filename', config['result_filename'],
                '--show-report'
            ]
            
            if config.get('trust_remote_code'):
                cmd.extend(['--trust-remote-code'])
            
            if config.get('profile'):
                cmd.extend(['--profile', '--profile-steps', str(config['profile_steps'])])
                
                if config.get('profile_by_stage'):
                    cmd.extend(['--profile-by-stage'])
        
        elif request.script_type == "launch_server":
            cmd = [
                'python', '-m', 'sglang.launch_server',
                '--model-path', config['model_path'],
                '--host', config['host'],
                '--port', str(config['port']),
                '--tp-size', str(config['tp_size'])
            ]
            
            if config.get('trust_remote_code'):
                cmd.extend(['--trust-remote-code'])
        
        return cmd

class AnalysisOrchestrator:
    """分析编排器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_analysis(self, request: AnalysisRequest, sglang_command: List[str]) -> Dict:
        """根据请求类型运行相应的分析"""
        
        print(f"🚀 开始执行 {request.analysis_type} 分析...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'request': asdict(request),
            'command': sglang_command,
            'analysis_results': {}
        }
        
        if request.analysis_type == "nsys":
            results['analysis_results'] = self._run_nsys_analysis(sglang_command, request)
            
        elif request.analysis_type == "ncu":
            results['analysis_results'] = self._run_ncu_analysis(sglang_command, request)
            
        elif request.analysis_type == "auto":
            results['analysis_results'] = self._run_integrated_analysis(sglang_command, request)
        
        else:
            raise ValueError(f"不支持的分析类型: {request.analysis_type}")
        
        # 保存结果
        results_file = self.output_dir / "analysis_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📋 分析结果已保存: {results_file}")
        return results
    
    def _run_nsys_analysis(self, sglang_command: List[str], request: AnalysisRequest) -> Dict:
        """运行nsys分析"""
        
        nsys_file = self.output_dir / "profile.nsys-rep"
        
        # 构建nsys命令
        nsys_cmd = [
            'nsys', 'profile',
            '-o', str(nsys_file.with_suffix('')),
            '-t', 'cuda,nvtx,osrt',
            '--cuda-memory-usage=true',
            '--force-overwrite=true'
        ] + sglang_command
        
        print(f"🔄 执行nsys命令: {' '.join(nsys_cmd)}")
        
        try:
            # 运行nsys profiling
            result = subprocess.run(nsys_cmd, capture_output=True, text=True, check=True,
                                  cwd='SGlang')
            
            print(f"✅ nsys分析完成: {nsys_file}")
            
            # 使用nsys解析器分析结果
            parser = NsysParser(str(nsys_file))
            parser.parse()
            
            analyzer = NsysAnalyzer(parser)
            stats = analyzer.analyze()
            
            # 生成可视化
            visualizer = NsysVisualizer(parser, analyzer)
            visualizer.output_dir = self.output_dir / "nsys_visualization"
            visualizer.create_visualizations()
            
            # 生成报告
            reporter = NsysReporter(parser, analyzer)
            reporter.output_dir = self.output_dir / "nsys_reports" 
            reporter.generate_report()
            
            return {
                'nsys_file': str(nsys_file),
                'stats': stats,
                'kernels_count': len(parser.kernels),
                'memory_transfers_count': len(parser.memory_transfers),
                'visualization_dir': str(visualizer.output_dir),
                'reports_dir': str(reporter.output_dir)
            }
            
        except subprocess.CalledProcessError as e:
            error_msg = f"nsys分析失败: {e.stderr}"
            print(f"❌ {error_msg}")
            return {'error': error_msg}
    
    def _run_ncu_analysis(self, sglang_command: List[str], request: AnalysisRequest) -> Dict:
        """运行ncu分析"""
        
        # 先运行nsys获取热点kernels
        print("🔍 首先运行nsys识别热点kernels...")
        nsys_result = self._run_nsys_analysis(sglang_command, request)
        
        if 'error' in nsys_result:
            return nsys_result
        
        # 提取热点kernels (这里简化处理)
        ncu_file = self.output_dir / "ncu_profile.ncu-rep"
        
        # 构建ncu命令 (分析所有kernels)
        ncu_cmd = [
            'ncu',
            '--set', 'full',
            '-o', str(ncu_file.with_suffix('')),
            '--force-overwrite'
        ] + sglang_command
        
        print(f"🔄 执行ncu命令: {' '.join(ncu_cmd)}")
        
        try:
            # 运行ncu profiling
            result = subprocess.run(ncu_cmd, capture_output=True, text=True, 
                                  check=True, timeout=600, cwd='SGlang')
            
            print(f"✅ ncu分析完成: {ncu_file}")
            
            # 导出为CSV
            csv_file = ncu_file.with_suffix('.csv')
            export_cmd = ['ncu', '--csv', '--log-file', str(csv_file), 
                         '--import', str(ncu_file)]
            subprocess.run(export_cmd, check=True)
            
            # 使用ncu解析器分析结果
            parser = NCUParser(str(csv_file))
            parser.parse()
            
            analyzer = NCUAnalyzer(parser)
            stats = analyzer.analyze()
            
            # 生成可视化
            visualizer = NCUVisualizer(parser, analyzer)
            visualizer.output_dir = self.output_dir / "ncu_visualization"
            visualizer.create_visualizations()
            
            # 生成报告
            reporter = NCUReporter(parser, analyzer)
            reporter.output_dir = self.output_dir / "ncu_reports"
            reporter.generate_report()
            
            return {
                'ncu_file': str(ncu_file),
                'csv_file': str(csv_file),
                'stats': stats,
                'kernels_count': len(parser.kernels),
                'bottlenecks_count': len(analyzer.bottlenecks),
                'visualization_dir': str(visualizer.output_dir),
                'reports_dir': str(reporter.output_dir),
                'nsys_result': nsys_result
            }
            
        except subprocess.CalledProcessError as e:
            error_msg = f"ncu分析失败: {e.stderr}"
            print(f"❌ {error_msg}")
            return {'error': error_msg}
        except subprocess.TimeoutExpired:
            error_msg = "ncu分析超时"
            print(f"⏰ {error_msg}")
            return {'error': error_msg}
    
    def _run_integrated_analysis(self, sglang_command: List[str], request: AnalysisRequest) -> Dict:
        """运行集成分析"""
        
        # 使用集成分析器
        analyzer = NSysToNCUAnalyzer(str(self.output_dir / "integrated"))
        
        try:
            # 步骤1: nsys全局分析
            nsys_file = analyzer.step1_nsys_analysis(sglang_command, "sglang_overview")
            
            # 步骤2: 提取热点kernels
            hot_kernels = analyzer.step2_extract_hot_kernels(nsys_file, top_k=10)
            
            if not hot_kernels:
                return {'error': '未发现热点kernels'}
            
            # 步骤3: ncu深度分析
            ncu_files = analyzer.step3_ncu_targeted_analysis(
                sglang_command, hot_kernels, request.max_ncu_kernels
            )
            
            # 步骤4: 综合分析
            comprehensive_results = analyzer.step4_comprehensive_analysis(ncu_files)
            
            # 生成最终报告
            report_file = analyzer.generate_final_report(comprehensive_results)
            
            return {
                'nsys_file': nsys_file,
                'hot_kernels': hot_kernels,
                'ncu_files': ncu_files,
                'comprehensive_results': comprehensive_results,
                'final_report': report_file,
                'analysis_dir': str(analyzer.output_dir)
            }
            
        except Exception as e:
            error_msg = f"集成分析失败: {str(e)}"
            print(f"❌ {error_msg}")
            return {'error': error_msg}

class AIAgentAnalyzer:
    """AI性能分析Agent主类"""
    
    def __init__(self, workspace_root: str = "."):
        self.workspace_root = workspace_root
        self.parser = PromptParser()
        self.config_generator = ConfigGenerator(workspace_root)
    
    def analyze_from_prompt(self, prompt: str) -> Dict:
        """从用户提示词开始完整的分析流程"""
        
        print("🤖 AI Agent性能分析器启动")
        print("=" * 60)
        
        try:
            # 1. 解析提示词
            print("\n📝 步骤1: 解析用户提示词")
            request = self.parser.parse_prompt(prompt)
            
            # 2. 生成配置
            print("\n⚙️  步骤2: 生成脚本配置")
            config = self.config_generator.generate_sglang_config(request)
            sglang_command = self.config_generator.build_command(request, config)
            
            # 3. 创建输出目录
            output_dir = Path(request.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 4. 运行分析
            print(f"\n🔬 步骤3: 执行{request.analysis_type}分析")
            orchestrator = AnalysisOrchestrator(request.output_dir)
            results = orchestrator.run_analysis(request, sglang_command)
            
            print(f"\n🎉 分析完成!")
            print(f"📁 结果目录: {request.output_dir}")
            
            return results
            
        except Exception as e:
            error_msg = f"AI Agent分析失败: {str(e)}"
            print(f"❌ {error_msg}")
            return {'error': error_msg}
    
    def analyze_existing_files(self, file_path: str, analysis_type: str = "auto") -> Dict:
        """分析已有的profile文件"""
        
        file_path = Path(file_path)
        if not file_path.exists():
            return {'error': f'文件不存在: {file_path}'}
        
        output_dir = file_path.parent / f"analysis_{file_path.stem}_{datetime.now().strftime('%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 分析已有文件: {file_path}")
        
        try:
            if file_path.suffix.lower() == '.nsys-rep' or 'nsys' in analysis_type:
                # nsys文件分析
                parser = NsysParser(str(file_path))
                parser.parse()
                
                analyzer = NsysAnalyzer(parser)
                stats = analyzer.analyze()
                
                visualizer = NsysVisualizer(parser, analyzer)
                visualizer.output_dir = output_dir / "visualization"
                visualizer.create_visualizations()
                
                reporter = NsysReporter(parser, analyzer)
                reporter.output_dir = output_dir / "reports"
                reporter.generate_report()
                
                return {
                    'file_type': 'nsys',
                    'stats': stats,
                    'visualization_dir': str(visualizer.output_dir),
                    'reports_dir': str(reporter.output_dir)
                }
            
            elif file_path.suffix.lower() in ['.ncu-rep', '.csv'] or 'ncu' in analysis_type:
                # ncu文件分析
                parser = NCUParser(str(file_path))
                parser.parse()
                
                analyzer = NCUAnalyzer(parser)
                stats = analyzer.analyze()
                
                visualizer = NCUVisualizer(parser, analyzer) 
                visualizer.output_dir = output_dir / "visualization"
                visualizer.create_visualizations()
                
                reporter = NCUReporter(parser, analyzer)
                reporter.output_dir = output_dir / "reports"
                reporter.generate_report()
                
                return {
                    'file_type': 'ncu',
                    'stats': stats,
                    'bottlenecks_count': len(analyzer.bottlenecks),
                    'visualization_dir': str(visualizer.output_dir),
                    'reports_dir': str(reporter.output_dir)
                }
            
            else:
                return {'error': f'不支持的文件类型: {file_path.suffix}'}
                
        except Exception as e:
            return {'error': f'分析文件失败: {str(e)}'}

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI Agent自动LLM性能分析器')
    
    subparsers = parser.add_subparsers(dest='command', help='命令类型')
    
    # 从提示词分析
    prompt_parser = subparsers.add_parser('prompt', help='从提示词开始分析')
    prompt_parser.add_argument('prompt', help='用户提示词')
    prompt_parser.add_argument('--workspace', default='.', help='工作空间根目录')
    
    # 分析已有文件  
    file_parser = subparsers.add_parser('file', help='分析已有profile文件')
    file_parser.add_argument('file_path', help='profile文件路径')
    file_parser.add_argument('--analysis-type', choices=['nsys', 'ncu', 'auto'], 
                            default='auto', help='分析类型')
    
    # 交互式模式
    interactive_parser = subparsers.add_parser('interactive', help='交互式模式')
    interactive_parser.add_argument('--workspace', default='.', help='工作空间根目录')
    
    args = parser.parse_args()
    
    if args.command == 'prompt':
        agent = AIAgentAnalyzer(args.workspace)
        results = agent.analyze_from_prompt(args.prompt)
        
        if 'error' not in results:
            print(f"\n✅ 分析成功完成")
        else:
            print(f"\n❌ 分析失败: {results['error']}")
    
    elif args.command == 'file':
        agent = AIAgentAnalyzer()
        results = agent.analyze_existing_files(args.file_path, args.analysis_type)
        
        if 'error' not in results:
            print(f"\n✅ 文件分析完成")
            print(f"📊 分析类型: {results['file_type']}")
        else:
            print(f"\n❌ 文件分析失败: {results['error']}")
    
    elif args.command == 'interactive':
        agent = AIAgentAnalyzer(args.workspace)
        
        print("🤖 AI Agent交互式模式")
        print("输入'quit'或'exit'退出")
        print("=" * 40)
        
        while True:
            try:
                prompt = input("\n💬 请输入分析需求: ").strip()
                
                if prompt.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见!")
                    break
                
                if not prompt:
                    continue
                
                results = agent.analyze_from_prompt(prompt)
                
                if 'error' not in results:
                    print(f"✅ 分析完成")
                else:
                    print(f"❌ 分析失败: {results['error']}")
                    
            except KeyboardInterrupt:
                print("\n👋 再见!")
                break
            except Exception as e:
                print(f"❌ 意外错误: {e}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
