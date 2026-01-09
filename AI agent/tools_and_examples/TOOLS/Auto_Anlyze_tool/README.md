# NVIDIA 性能分析工具集 🚀

一套功能完整的 NVIDIA GPU 性能分析工具，包含：
- **Nsight Systems (nsys)** 分析工具：全局性能分析、timeline 视图
- **Nsight Compute (ncu)** 分析工具：CUDA kernel 详细性能分析

支持多种文件格式，提供详细的性能分析报告和专业可视化图表。

## ✨ 主要功能

### 🔥 Nsight Systems (nsys) 分析
- 📊 **多格式支持**: `.nsys-rep`, `.sqlite`, `.csv`, `.json`
- 🔍 **智能解析**: 自动识别并解析 CUDA kernels、内存传输、API 调用
- 📈 **数据分析**: 性能统计、瓶颈识别、时间线分析
- 📊 **可视化**: Timeline、分布图、带宽分析图

### ⚡ Nsight Compute (ncu) 分析  
- 🎯 **Kernel 深度分析**: SM 效率、占用率、Warp 执行效率
- 💾 **内存性能**: DRAM 带宽、L1/L2 缓存命中率
- 🧮 **计算分析**: Tensor Core 使用率、流水线利用率
- 🚫 **瓶颈识别**: 自动识别计算、内存、延迟瓶颈
- 📊 **专业可视化**: 雷达图、占用率分析、瓶颈分布

### 🔧 通用功能
- 📄 **报告生成**: 详细的性能分析报告  

























- 🔧 **批量处理**: 支持批量分析多个文件
- 🎯 **易于使用**: 命令行工具 + Python API

## 🛠️ 安装

### 环境要求
- Python 3.7+
- NVIDIA Nsight Systems (可选，用于 .nsys-rep 文件转换)
- NVIDIA Nsight Compute (可选，用于 .ncu-rep 文件转换)

### 依赖安装
```bash
pip install -r requirements.txt
```

### 核心依赖
- `pandas`: 数据处理和分析
- `matplotlib`: 图表生成
- `seaborn`: 高级可视化
- `numpy`: 数值计算

## 🚀 快速开始

### 命令行使用

#### Nsys 分析
```bash
# 分析 nsys-rep 文件
python nsys_parser.py profile.nsys-rep

# 分析 SQLite 导出文件
python nsys_parser.py profile.sqlite

# 只分析，不生成图表
python nsys_parser.py profile.nsys-rep --no-viz

# 自定义输出目录
python nsys_parser.py profile.nsys-rep --output-dir my_analysis
```

#### NCU 分析
```bash
# 分析 ncu-rep 文件
python ncu_parser.py kernel_profile.ncu-rep

# 分析 CSV 导出文件
python ncu_parser.py metrics.csv

# 只分析，不生成图表
python ncu_parser.py kernel_profile.ncu-rep --no-viz

# 自定义输出目录
python ncu_parser.py kernel_profile.ncu-rep --output-dir ncu_analysis
```

### Python API 使用

#### Nsys 分析
```python
from nsys_parser import NsysParser, NsysAnalyzer, NsysVisualizer

# 解析文件
parser = NsysParser("profile.nsys-rep")
parser.parse()

# 分析数据
analyzer = NsysAnalyzer(parser)
stats = analyzer.analyze()

# 生成可视化
visualizer = NsysVisualizer(parser, analyzer)
visualizer.create_visualizations()

print(f"解析了 {len(parser.kernels)} 个 CUDA kernels")
```

#### NCU 分析
```python
from ncu_parser import NCUParser, NCUAnalyzer, NCUVisualizer

# 解析 NCU 文件
parser = NCUParser("kernel_profile.ncu-rep")
parser.parse()

# 深度分析 kernel 性能
analyzer = NCUAnalyzer(parser)
stats = analyzer.analyze()

# 生成可视化
visualizer = NCUVisualizer(parser, analyzer)
visualizer.create_visualizations()

print(f"解析了 {len(parser.kernels)} 个 kernels")
print(f"识别了 {len(analyzer.bottlenecks)} 个性能瓶颈")
```

## 📊 支持的文件格式

### 1. `.nsys-rep` 文件 (推荐)
NVIDIA Nsight Systems 生成的二进制报告文件，包含最完整的性能数据。

**生成方法:**
```bash
# 基本 profiling
nsys profile -o my_profile ./your_cuda_program

# Python 程序 (如 PyTorch)
nsys profile -o torch_profile python train.py

# SGLang 服务
nsys profile -o sglang_profile python -m sglang.launch_server ...

# 详细分析（推荐）
nsys profile -o detailed_profile -t cuda,nvtx,osrt,cudnn,cublas ./program
```

### 2. `.sqlite` 文件
从 .nsys-rep 导出的 SQLite 数据库格式。

**导出方法:**
```bash
nsys export --type=sqlite --output=profile.sqlite profile.nsys-rep
```

### 3. `.csv` 和 `.json` 文件
其他格式的 nsys 导出文件（部分支持）。

### 4. `.ncu-rep` 文件 (NCU 专用)
NVIDIA Nsight Compute 生成的 kernel 分析报告文件。

**生成方法:**
```bash
# 基本 kernel 分析
ncu -o kernel_profile ./your_cuda_program

# 深度分析（推荐）
ncu --set full -o detailed_kernel_profile ./your_cuda_program

# Python/PyTorch 程序
ncu -o torch_kernels python train.py

# 分析特定 kernel
ncu --kernel-name "your_kernel_name" -o specific_kernel ./program

# 收集所有指标
ncu --metrics all -o complete_metrics ./program
```

## 📈 分析功能详解

### 🔥 CUDA Kernel 分析
- **执行时间统计**: 总时间、平均时间、最大/最小时间
- **Kernel 分布**: 每个 kernel 的调用次数和时间占比
- **性能瓶颈识别**: 找出耗时最长的 kernels
- **时间线分析**: kernel 执行的时间序列

### 💾 内存传输分析
- **传输类型统计**: Host↔Device, Device↔Device
- **带宽分析**: 实际带宽 vs 理论带宽
- **传输效率**: 大小 vs 耗时关系
- **瓶颈识别**: 低效的内存传输

### 🔧 API 调用分析
- **CUDA Runtime API** 调用统计
- **调用频率和耗时**
- **线程并发分析**

### ⚡ NCU Kernel 深度分析

#### 🎯 GPU 利用率分析
- **SM 效率 (SM Efficiency)**: 流多处理器利用率
- **占用率 (Occupancy)**: 理论 vs 实际占用率
- **资源限制**: 寄存器、共享内存限制分析

#### 💾 内存系统分析
- **DRAM 带宽利用率**: 内存子系统性能
- **L1/L2 缓存**: 命中率和访问模式分析
- **内存访问效率**: 合并访问、bank conflicts

#### 🧮 计算单元分析
- **Tensor Core 利用率**: AI 工作负载加速分析
- **FP32/FP16 流水线**: 浮点运算效率
- **指令吞吐量**: 不同类型指令的执行效率

#### 🚫 性能瓶颈自动识别
- **计算瓶颈**: SM 效率低、算法复杂度问题
- **内存瓶颈**: 带宽限制、缓存未命中
- **延迟瓶颈**: 占用率低、资源争用

#### 🔧 Warp 执行分析
- **Warp 执行效率**: 分支分歧、线程利用率
- **停顿分析**: 内存依赖、长记分板停顿
- **指令级并行**: ILP 分析

## 📊 生成的可视化图表

工具会自动生成以下专业图表：

1. **`kernel_timeline.png`**: CUDA Kernel 执行时间线
2. **`kernel_duration_distribution.png`**: Kernel 执行时间分布
3. **`top_kernels.png`**: 耗时最长的 Top 10 Kernels
4. **`memory_transfers.png`**: 内存传输分析
5. **`bandwidth_analysis.png`**: 内存带宽分析

#### NCU 专用图表
6. **`gpu_utilization.png`**: SM 效率和利用率分析
7. **`memory_performance.png`**: DRAM 带宽和缓存性能
8. **`occupancy_analysis.png`**: 占用率效率对比
9. **`bottleneck_analysis.png`**: 性能瓶颈类型分布
10. **`kernel_comparison.png`**: Kernel 性能对比雷达图

### 示例图表说明

#### Kernel 时间线图
```
显示所有 kernel 的执行时间线，帮助识别:
- 并行度不足的时间段
- Kernel 启动的间隙
- 执行时间异常的 kernels
```

#### 带宽分析图
```
内存传输带宽分布直方图，帮助发现:
- 低带宽传输（可能的瓶颈）
- 带宽利用率统计
- 不同传输类型的效率对比
```

## 📄 分析报告

工具生成两种格式的报告：

### 1. 文本报告 (`analysis_report.txt`)
```
================================================================================
NVIDIA Nsight Systems 性能分析报告
================================================================================

📊 性能摘要
• 总 CUDA Kernels: 1,234
• 总内存传输: 56
• 总 API 调用: 890

🔥 CUDA Kernel 分析
• 总执行时间: 145.67 ms
• 平均kernel时间: 0.118 ms
• 唯一kernel数量: 23

💾 内存传输分析
• 总数据传输: 512.34 MB
• 平均带宽: 234.56 GB/s
• 传输次数: 56

🚫 性能瓶颈分析
• 识别的瓶颈点...

💡 优化建议
• 具体的优化建议...
```

### 2. JSON 数据 (`analysis_data.json`)
包含所有详细的分析数据，便于进一步处理。

## 💡 使用场景

### 1. CUDA 程序性能调优
```python
# 分析自定义 CUDA kernel
parser = NsysParser("my_kernels.nsys-rep")
parser.parse()

# 找出最慢的 kernels
slow_kernels = [k for k in parser.kernels if k.duration > 0.001]  # > 1ms
for kernel in sorted(slow_kernels, key=lambda k: k.duration, reverse=True)[:5]:
    print(f"{kernel.name}: {kernel.duration*1000:.2f} ms")
```

### 2. 深度学习模型分析
```python
# 分析 PyTorch 训练过程
# nsys profile -o training.nsys-rep python train.py

parser = NsysParser("training.nsys-rep")
parser.parse()

# 分析不同类型的操作
conv_kernels = [k for k in parser.kernels if 'conv' in k.name.lower()]
matmul_kernels = [k for k in parser.kernels if 'gemm' in k.name.lower()]

print(f"卷积操作时间: {sum(k.duration for k in conv_kernels)*1000:.2f} ms")
print(f"矩阵乘法时间: {sum(k.duration for k in matmul_kernels)*1000:.2f} ms")
```

### 3. SGLang 性能优化
```python
# 分析 SGLang 推理性能
parser = NsysParser("sglang_inference.nsys-rep")
parser.parse()

# 找出 attention 相关的 kernels
attention_kernels = [k for k in parser.kernels if 'attention' in k.name.lower()]
kv_cache_ops = [k for k in parser.kernels if 'cache' in k.name.lower()]
```

## 🔧 高级功能

### 批量分析
```python
import os
from nsys_parser import NsysParser, NsysAnalyzer

# 分析目录中的所有文件
profile_dir = "experiments/"
results = {}

for filename in os.listdir(profile_dir):
    if filename.endswith('.nsys-rep'):
        parser = NsysParser(os.path.join(profile_dir, filename))
        parser.parse()
        
        analyzer = NsysAnalyzer(parser)
        stats = analyzer.analyze()
        
        results[filename] = {
            'total_kernel_time': stats['kernel_analysis']['total_kernel_time'],
            'memory_bandwidth': stats['memory_analysis']['avg_bandwidth']
        }

# 对比不同实验的结果
for exp, metrics in results.items():
    print(f"{exp}: {metrics['total_kernel_time']:.2f}ms, {metrics['memory_bandwidth']:.2f}GB/s")
```

### 自定义分析
```python
class CustomAnalyzer(NsysAnalyzer):
    def analyze_custom_pattern(self):
        """自定义分析逻辑"""
        # 分析特定的 kernel 模式
        pattern_kernels = []
        for kernel in self.parser.kernels:
            if self._matches_pattern(kernel.name):
                pattern_kernels.append(kernel)
        
        return {
            'pattern_count': len(pattern_kernels),
            'pattern_time': sum(k.duration for k in pattern_kernels)
        }
    
    def _matches_pattern(self, kernel_name):
        # 自定义模式匹配逻辑
        return 'my_pattern' in kernel_name.lower()
```

## 🚨 常见问题

### Q1: "nsys 命令未找到"
**A:** 确保已安装 NVIDIA Nsight Systems：
```bash
# Ubuntu/Debian
sudo apt-get install nsight-systems

# 或从 NVIDIA 官网下载安装包
```

### Q2: .nsys-rep 文件无法解析
**A:** 工具会自动调用 nsys 导出为 SQLite 格式。确保：
- nsys 在 PATH 中
- 有足够的磁盘空间
- nsys 版本兼容

### Q3: 内存不足
**A:** 对于大文件：
```python
# 限制解析的事件数量
parser = NsysParser("large_file.nsys-rep")
parser.parse_limit = 100000  # 限制解析事件数
```

### Q4: 图表中文显示问题
**A:** 确保系统有中文字体：
```python
# 在代码开头添加
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Linux/macOS
```

## 📚 相关资源

- [NVIDIA Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/)
- [CUDA Profiler 最佳实践](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [SGLang 性能优化指南](https://github.com/sgl-project/sglang)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 开发设置
```bash
git clone <repo>
cd nsys-parser
pip install -r requirements.txt

# 运行测试
python example_usage.py
```

## 📄 许可证

MIT License

## 🔗 更新日志

### v1.0.0
- ✅ 支持 .nsys-rep、.sqlite 文件解析
- ✅ CUDA kernel、内存传输、API 调用分析
- ✅ 自动生成可视化图表
- ✅ 性能瓶颈识别
- ✅ 详细分析报告生成
- ✅ 命令行工具和 Python API

---

**🎯 立即开始分析您的 CUDA 程序性能！**
