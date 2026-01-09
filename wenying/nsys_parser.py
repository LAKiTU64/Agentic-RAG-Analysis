#!/usr/bin/env python3
"""
NVIDIA Nsight Systems (nsys) 输出文件自动化解析工具

支持解析多种 nsys 输出格式：
- SQLite 数据库文件 (.sqlite)
- CSV 导出文件
- JSON 导出文件
- 自动调用 nsys 导出工具

作者: AI助手
版本: 1.0
"""

import os
import sys
import sqlite3
import json
import csv
import subprocess
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class KernelInfo:
    """CUDA Kernel 执行信息"""
    name: str
    start_time: float
    duration: float
    grid_size: Optional[Tuple[int, int, int]] = None
    block_size: Optional[Tuple[int, int, int]] = None
    registers_per_thread: Optional[int] = None
    shared_memory: Optional[int] = None

@dataclass
class MemoryTransfer:
    """内存传输信息"""
    kind: str  # H2D, D2H, D2D
    size: int
    start_time: float
    duration: float
    bandwidth: Optional[float] = None

@dataclass
class APICall:
    """API 调用信息"""
    name: str
    start_time: float
    duration: float
    thread_id: int

class NsysParser:
    """Nsys 输出文件解析器"""
    
    def __init__(self, input_file: str):
        self.input_file = Path(input_file)
        self.kernels: List[KernelInfo] = []
        self.memory_transfers: List[MemoryTransfer] = []
        self.api_calls: List[APICall] = []
        self.metadata: Dict = {}
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    def parse(self) -> None:
        """解析输入文件"""
        suffix = self.input_file.suffix.lower()
        
        if suffix == '.nsys-rep':
            self._parse_nsys_rep()
        elif suffix in ['.db', '.sqlite', '.sqlite3']:
            self._parse_sqlite()
        elif suffix == '.csv':
            self._parse_csv()
        elif suffix == '.json':
            self._parse_json()
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    def _parse_nsys_rep(self) -> None:
        """解析 .nsys-rep 文件（需要先导出为SQLite）"""
        print("📋 检测到 .nsys-rep 文件，正在导出为SQLite格式...")
        
        # 生成输出文件名
        sqlite_file = self.input_file.with_suffix('.sqlite')
        
        # 调用 nsys 导出命令
        cmd = [
            'nsys', 'export', 
            '--type=sqlite',
            '--output', str(sqlite_file),
            str(self.input_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ 导出成功: {sqlite_file}")
            
            # 解析导出的SQLite文件
            self._parse_sqlite(sqlite_file)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ nsys导出失败: {e.stderr}")
            print("请确保 nsys 工具已正确安装并在PATH中")
            raise
        except FileNotFoundError:
            print("❌ 未找到 nsys 命令")
            print("请安装 NVIDIA Nsight Systems 并确保 nsys 在PATH中")
            raise
    
    def _parse_sqlite(self, sqlite_file: Optional[Path] = None) -> None:
        """解析 SQLite 数据库文件"""
        db_file = sqlite_file or self.input_file
        
        print(f"📊 正在解析SQLite文件: {db_file}")
        
        conn = sqlite3.connect(db_file)
        
        try:
            # 获取表信息
            tables = self._get_table_names(conn)
            print(f"🔍 发现表: {', '.join(tables)}")
            
            # 解析CUDA kernels
            if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
                self._parse_cuda_kernels(conn)
            
            # 解析内存传输
            if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
                self._parse_memory_transfers(conn)
            
            # 解析API调用
            if 'CUPTI_ACTIVITY_KIND_RUNTIME' in tables:
                self._parse_api_calls(conn)
            
            # 获取元数据
            self._parse_metadata(conn)
            
        finally:
            conn.close()
    
    def _get_table_names(self, conn: sqlite3.Connection) -> List[str]:
        """获取数据库中的所有表名"""
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cursor.fetchall()]
    
    def _parse_cuda_kernels(self, conn: sqlite3.Connection) -> None:
        """解析CUDA kernel信息"""
        query = """
        SELECT 
            demangledName,
            start,
            end,
            gridX, gridY, gridZ,
            blockX, blockY, blockZ,
            registersPerThread,
            sharedMemoryExecuted
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        ORDER BY start
        """
        
        cursor = conn.cursor()
        cursor.execute(query)
        
        for row in cursor.fetchall():
            kernel = KernelInfo(
                name=row[0] or "Unknown Kernel",
                start_time=row[1] / 1e9,  # 转换为秒
                duration=(row[2] - row[1]) / 1e9,  # 转换为秒
                grid_size=(row[3], row[4], row[5]) if row[3] else None,
                block_size=(row[6], row[7], row[8]) if row[6] else None,
                registers_per_thread=row[9],
                shared_memory=row[10]
            )
            self.kernels.append(kernel)
        
        print(f"🔥 解析到 {len(self.kernels)} 个CUDA kernels")
    
    def _parse_memory_transfers(self, conn: sqlite3.Connection) -> None:
        """解析内存传输信息"""
        query = """
        SELECT 
            copyKind,
            bytes,
            start,
            end
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        ORDER BY start
        """
        
        cursor = conn.cursor()
        cursor.execute(query)
        
        kind_map = {
            1: "H2D",  # Host to Device
            2: "D2H",  # Device to Host
            3: "D2D",  # Device to Device
        }
        
        for row in cursor.fetchall():
            duration_ns = row[3] - row[2]
            duration_s = duration_ns / 1e9
            bandwidth = (row[1] / (1024**3)) / duration_s if duration_s > 0 else 0  # GB/s
            
            transfer = MemoryTransfer(
                kind=kind_map.get(row[0], f"Kind_{row[0]}"),
                size=row[1],
                start_time=row[2] / 1e9,
                duration=duration_s,
                bandwidth=bandwidth
            )
            self.memory_transfers.append(transfer)
        
        print(f"💾 解析到 {len(self.memory_transfers)} 个内存传输")
    
    def _parse_api_calls(self, conn: sqlite3.Connection) -> None:
        """解析API调用信息"""
        query = """
        SELECT 
            nameId,
            start,
            end,
            threadId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        ORDER BY start
        LIMIT 10000  -- 限制数量避免过多数据
        """
        
        cursor = conn.cursor()
        cursor.execute(query)
        
        for row in cursor.fetchall():
            api_call = APICall(
                name=f"API_{row[0]}",
                start_time=row[1] / 1e9,
                duration=(row[2] - row[1]) / 1e9,
                thread_id=row[3]
            )
            self.api_calls.append(api_call)
        
        print(f"🔧 解析到 {len(self.api_calls)} 个API调用")
    
    def _parse_metadata(self, conn: sqlite3.Connection) -> None:
        """解析元数据信息"""
        self.metadata = {
            'total_kernels': len(self.kernels),
            'total_memory_transfers': len(self.memory_transfers),
            'total_api_calls': len(self.api_calls),
            'parse_time': datetime.now().isoformat()
        }
        
        if self.kernels:
            total_time = max(k.start_time + k.duration for k in self.kernels) - min(k.start_time for k in self.kernels)
            self.metadata['total_execution_time'] = total_time
    
    def _parse_csv(self) -> None:
        """解析CSV文件"""
        print(f"📋 正在解析CSV文件: {self.input_file}")
        # CSV解析逻辑（根据具体CSV格式实现）
        pass
    
    def _parse_json(self) -> None:
        """解析JSON文件"""
        print(f"📋 正在解析JSON文件: {self.input_file}")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # JSON解析逻辑（根据具体JSON格式实现）
        pass

class NsysAnalyzer:
    """Nsys 数据分析器"""
    
    def __init__(self, parser: NsysParser):
        self.parser = parser
        self.stats = {}
    
    def analyze(self) -> Dict:
        """执行完整分析"""
        print("🔍 开始性能分析...")
        
        self.stats = {
            'kernel_analysis': self._analyze_kernels(),
            'memory_analysis': self._analyze_memory(),
            'timeline_analysis': self._analyze_timeline(),
            'bottleneck_analysis': self._analyze_bottlenecks()
        }
        
        return self.stats
    
    def _analyze_kernels(self) -> Dict:
        """分析CUDA kernel性能"""
        if not self.parser.kernels:
            return {}
        
        kernels_df = pd.DataFrame([
            {
                'name': k.name,
                'duration': k.duration * 1000,  # 转换为毫秒
                'start_time': k.start_time
            }
            for k in self.parser.kernels
        ])
        
        kernel_stats = kernels_df.groupby('name').agg({
            'duration': ['count', 'mean', 'std', 'min', 'max', 'sum']
        }).round(4)
        
        return {
            'total_kernels': len(self.parser.kernels),
            'unique_kernels': kernels_df['name'].nunique(),
            'total_kernel_time': kernels_df['duration'].sum(),
            'avg_kernel_time': kernels_df['duration'].mean(),
            'top_kernels': kernel_stats.sort_values(('duration', 'sum'), ascending=False).head(10),
            'kernel_distribution': kernels_df.groupby('name').size().sort_values(ascending=False)
        }
    
    def _analyze_memory(self) -> Dict:
        """分析内存传输"""
        if not self.parser.memory_transfers:
            return {}
        
        memory_df = pd.DataFrame([
            {
                'kind': m.kind,
                'size_mb': m.size / (1024 * 1024),
                'duration': m.duration * 1000,
                'bandwidth': m.bandwidth
            }
            for m in self.parser.memory_transfers
        ])
        
        return {
            'total_transfers': len(self.parser.memory_transfers),
            'total_data_mb': memory_df['size_mb'].sum(),
            'avg_bandwidth': memory_df['bandwidth'].mean(),
            'transfer_breakdown': memory_df.groupby('kind').agg({
                'size_mb': ['count', 'sum', 'mean'],
                'bandwidth': 'mean'
            }).round(4)
        }
    
    def _analyze_timeline(self) -> Dict:
        """分析时间线"""
        all_events = []
        
        for k in self.parser.kernels:
            all_events.append(('kernel', k.start_time, k.duration))
        
        for m in self.parser.memory_transfers:
            all_events.append(('memory', m.start_time, m.duration))
        
        if not all_events:
            return {}
        
        all_events.sort(key=lambda x: x[1])  # 按开始时间排序
        
        return {
            'total_events': len(all_events),
            'execution_span': max(e[1] + e[2] for e in all_events) - min(e[1] for e in all_events),
            'first_event_time': min(e[1] for e in all_events),
            'last_event_time': max(e[1] + e[2] for e in all_events)
        }
    
    def _analyze_bottlenecks(self) -> Dict:
        """识别性能瓶颈"""
        bottlenecks = []
        
        # 分析longest running kernels
        if self.parser.kernels:
            kernel_times = [(k.name, k.duration) for k in self.parser.kernels]
            kernel_times.sort(key=lambda x: x[1], reverse=True)
            bottlenecks.append({
                'type': 'longest_kernels',
                'data': kernel_times[:5]
            })
        
        # 分析内存带宽利用率
        if self.parser.memory_transfers:
            low_bandwidth = [
                (m.kind, m.bandwidth, m.size / (1024*1024))
                for m in self.parser.memory_transfers 
                if m.bandwidth and m.bandwidth < 100  # < 100 GB/s
            ]
            if low_bandwidth:
                bottlenecks.append({
                    'type': 'low_bandwidth_transfers',
                    'data': low_bandwidth
                })
        
        return {'identified_bottlenecks': bottlenecks}

class NsysVisualizer:
    """Nsys 数据可视化"""
    
    def __init__(self, parser: NsysParser, analyzer: NsysAnalyzer):
        self.parser = parser
        self.analyzer = analyzer
        self.output_dir = Path("nsys_analysis_output")
        self.output_dir.mkdir(exist_ok=True)
    
    def create_visualizations(self) -> None:
        """创建所有可视化图表"""
        print("📊 生成可视化图表...")
        
        if self.parser.kernels:
            self._plot_kernel_timeline()
            self._plot_kernel_duration_distribution()
            self._plot_top_kernels()
        
        if self.parser.memory_transfers:
            self._plot_memory_transfers()
            self._plot_bandwidth_analysis()
        
        print(f"📁 图表已保存到: {self.output_dir}")
    
    def _plot_kernel_timeline(self) -> None:
        """绘制kernel执行时间线"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        for i, kernel in enumerate(self.parser.kernels[:50]):  # 限制显示数量
            ax.barh(i, kernel.duration * 1000, left=kernel.start_time * 1000, height=0.8)
        
        ax.set_xlabel('时间 (毫秒)')
        ax.set_ylabel('Kernel 索引')
        ax.set_title('CUDA Kernel 执行时间线')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'kernel_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_kernel_duration_distribution(self) -> None:
        """绘制kernel执行时间分布"""
        durations = [k.duration * 1000 for k in self.parser.kernels]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 直方图
        ax1.hist(durations, bins=50, alpha=0.7, color='skyblue')
        ax1.set_xlabel('执行时间 (毫秒)')
        ax1.set_ylabel('频次')
        ax1.set_title('Kernel 执行时间分布')
        ax1.grid(True, alpha=0.3)
        
        # 箱线图
        ax2.boxplot(durations)
        ax2.set_ylabel('执行时间 (毫秒)')
        ax2.set_title('Kernel 执行时间箱线图')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'kernel_duration_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_top_kernels(self) -> None:
        """绘制耗时最长的kernels"""
        kernel_stats = {}
        for kernel in self.parser.kernels:
            if kernel.name not in kernel_stats:
                kernel_stats[kernel.name] = {'count': 0, 'total_time': 0}
            kernel_stats[kernel.name]['count'] += 1
            kernel_stats[kernel.name]['total_time'] += kernel.duration * 1000
        
        # 按总执行时间排序
        sorted_kernels = sorted(kernel_stats.items(), 
                              key=lambda x: x[1]['total_time'], reverse=True)[:10]
        
        names = [item[0][:30] + '...' if len(item[0]) > 30 else item[0] for item, _ in sorted_kernels]
        times = [stats['total_time'] for _, stats in sorted_kernels]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(names, times, color='lightcoral')
        
        ax.set_xlabel('总执行时间 (毫秒)')
        ax.set_title('耗时最长的 Top 10 Kernels')
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}ms', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_kernels.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_transfers(self) -> None:
        """绘制内存传输分析"""
        if not self.parser.memory_transfers:
            return
        
        transfer_data = {}
        for transfer in self.parser.memory_transfers:
            if transfer.kind not in transfer_data:
                transfer_data[transfer.kind] = {'count': 0, 'total_size': 0, 'total_time': 0}
            transfer_data[transfer.kind]['count'] += 1
            transfer_data[transfer.kind]['total_size'] += transfer.size / (1024 * 1024)  # MB
            transfer_data[transfer.kind]['total_time'] += transfer.duration * 1000  # ms
        
        kinds = list(transfer_data.keys())
        sizes = [transfer_data[kind]['total_size'] for kind in kinds]
        times = [transfer_data[kind]['total_time'] for kind in kinds]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 传输数据量
        ax1.pie(sizes, labels=kinds, autopct='%1.1f%%', startangle=90)
        ax1.set_title('内存传输数据量分布 (MB)')
        
        # 传输时间
        ax2.bar(kinds, times, color=['#ff9999', '#66b3ff', '#99ff99'])
        ax2.set_ylabel('总传输时间 (毫秒)')
        ax2.set_title('内存传输时间统计')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'memory_transfers.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_bandwidth_analysis(self) -> None:
        """绘制带宽分析"""
        if not self.parser.memory_transfers:
            return
        
        bandwidths = [m.bandwidth for m in self.parser.memory_transfers if m.bandwidth]
        if not bandwidths:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(bandwidths, bins=30, alpha=0.7, color='lightgreen')
        ax.set_xlabel('带宽 (GB/s)')
        ax.set_ylabel('频次')
        ax.set_title('内存传输带宽分布')
        ax.axvline(x=sum(bandwidths)/len(bandwidths), color='red', linestyle='--', 
                  label=f'平均带宽: {sum(bandwidths)/len(bandwidths):.2f} GB/s')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bandwidth_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

class NsysReporter:
    """Nsys 分析报告生成器"""
    
    def __init__(self, parser: NsysParser, analyzer: NsysAnalyzer):
        self.parser = parser
        self.analyzer = analyzer
        self.output_dir = Path("nsys_analysis_output")
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_report(self) -> None:
        """生成分析报告"""
        print("📄 生成分析报告...")
        
        report_path = self.output_dir / "analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_header())
            f.write(self._generate_summary())
            f.write(self._generate_kernel_analysis())
            f.write(self._generate_memory_analysis())
            f.write(self._generate_bottleneck_analysis())
            f.write(self._generate_recommendations())
        
        print(f"📋 报告已生成: {report_path}")
        
        # 同时生成JSON格式的详细数据
        json_path = self.output_dir / "analysis_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.analyzer.stats, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📊 详细数据已保存: {json_path}")
    
    def _generate_header(self) -> str:
        """生成报告头部"""
        return f"""
{'='*80}
NVIDIA Nsight Systems 性能分析报告
{'='*80}
分析文件: {self.parser.input_file}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

"""
    
    def _generate_summary(self) -> str:
        """生成摘要"""
        return f"""
📊 性能摘要
{'-'*40}
• 总 CUDA Kernels: {len(self.parser.kernels)}
• 总内存传输: {len(self.parser.memory_transfers)}
• 总 API 调用: {len(self.parser.api_calls)}

"""
    
    def _generate_kernel_analysis(self) -> str:
        """生成kernel分析"""
        if not self.parser.kernels:
            return "🔥 CUDA Kernel 分析\n" + "-"*40 + "\n无 kernel 数据\n\n"
        
        stats = self.analyzer.stats.get('kernel_analysis', {})
        
        return f"""
🔥 CUDA Kernel 分析
{'-'*40}
• 总执行时间: {stats.get('total_kernel_time', 0):.2f} ms
• 平均kernel时间: {stats.get('avg_kernel_time', 0):.2f} ms
• 唯一kernel数量: {stats.get('unique_kernels', 0)}

耗时最长的 Kernels:
"""
    
    def _generate_memory_analysis(self) -> str:
        """生成内存分析"""
        if not self.parser.memory_transfers:
            return "💾 内存传输分析\n" + "-"*40 + "\n无内存传输数据\n\n"
        
        stats = self.analyzer.stats.get('memory_analysis', {})
        
        return f"""
💾 内存传输分析
{'-'*40}
• 总数据传输: {stats.get('total_data_mb', 0):.2f} MB
• 平均带宽: {stats.get('avg_bandwidth', 0):.2f} GB/s
• 传输次数: {stats.get('total_transfers', 0)}

"""
    
    def _generate_bottleneck_analysis(self) -> str:
        """生成瓶颈分析"""
        bottlenecks = self.analyzer.stats.get('bottleneck_analysis', {}).get('identified_bottlenecks', [])
        
        if not bottlenecks:
            return "🚫 性能瓶颈分析\n" + "-"*40 + "\n未发现明显瓶颈\n\n"
        
        result = f"""
🚫 性能瓶颈分析
{'-'*40}
"""
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'longest_kernels':
                result += "• 耗时最长的 kernels 可能是瓶颈\n"
            elif bottleneck['type'] == 'low_bandwidth_transfers':
                result += "• 检测到低带宽内存传输\n"
        
        return result + "\n"
    
    def _generate_recommendations(self) -> str:
        """生成优化建议"""
        return f"""
💡 优化建议
{'-'*40}
• 分析耗时最长的 kernels，考虑算法优化
• 检查内存传输效率，减少不必要的传输
• 考虑使用异步传输和计算重叠
• 优化内存访问模式以提高带宽利用率

{'='*80}
"""

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='NVIDIA Nsight Systems 输出文件自动化解析工具')
    parser.add_argument('input_file', help='输入文件路径 (.nsys-rep, .sqlite, .csv, .json)')
    parser.add_argument('--no-viz', action='store_true', help='不生成可视化图表')
    parser.add_argument('--no-report', action='store_true', help='不生成分析报告')
    parser.add_argument('--output-dir', default='nsys_analysis_output', help='输出目录')
    
    args = parser.parse_args()
    
    try:
        # 解析文件
        print(f"🚀 开始解析文件: {args.input_file}")
        nsys_parser = NsysParser(args.input_file)
        nsys_parser.parse()
        
        # 分析数据
        analyzer = NsysAnalyzer(nsys_parser)
        analyzer.analyze()
        
        # 生成可视化
        if not args.no_viz:
            visualizer = NsysVisualizer(nsys_parser, analyzer)
            visualizer.output_dir = Path(args.output_dir)
            visualizer.create_visualizations()
        
        # 生成报告
        if not args.no_report:
            reporter = NsysReporter(nsys_parser, analyzer)
            reporter.output_dir = Path(args.output_dir)
            reporter.generate_report()
        
        print(f"\n✅ 分析完成! 结果保存在: {args.output_dir}")
        print(f"📊 解析了 {len(nsys_parser.kernels)} 个kernels, {len(nsys_parser.memory_transfers)} 个内存传输")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


