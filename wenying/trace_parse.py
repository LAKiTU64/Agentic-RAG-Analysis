#!/usr/bin/env python3  # 指定用系统的 python3 解释器执行
import json              # 处理 JSON 文件读取/解析
import sys               # 访问命令行参数、退出等系统功能
import statistics        # 统计函数（均值、标准差、中位数等）
import os                # 操作系统相关（当前未使用，可留作扩展）
import re                # 正则表达式匹配（提取 batch size 等）
from pathlib import Path # 文件路径对象封装（当前未直接使用 Path 操作）
GPU_TRACE_CAT = ["kernel", "gpu_memcpy", "gpu_memset"]  # 需要保留的 GPU 事件类别
EMBEDDING_PREFIX = "void rtp_llm::embedding_lookup_kernel"  # 嵌入查找 kernel 前缀，用于识别前向开始
LAYER_START_PREFIX = "void rtp_llm::addBiasResidual"         # 每层起始 kernel 前缀（判定 layer 分界）

PRE_GEMM_MATCHES = ["dynamic_per_token_scaled_quant","Memset","computeFP8Quantize128Kernel"]  # GEMM 前的准备算子特征
GEMM_MATCHES = ["nvjet_", "deep_gemm", "Cijk_Ailk", "Cijk_Alik", "_gemm_"]                    # GEMM 算子匹配前缀/片段
POST_GEMM_MATCHES = ["Cijk_SB_BiasS", "cublasLt::splitKreduce_kernel"]                       # GEMM 后续算子特征

GENERAL_NORM_MATCHES = ["generalRmsNorm", "Rmsnorm2dFwd"]  # 通用归一化算子
ATTENTION_MATCHES = ["FmhaFwdKernel", "aiter::pa_", "flash_attention", "aiter::fmha_fwd_hd128_bf16_causal",
                     "BatchPrefillWithPagedKVCacheKernel", 
                     "PersistentVariableLengthMergeStatesKernel",
                     "BatchDecodeWithPagedKVCacheKernel",
                     "paged_attention","xqa_kernel", "at::native::unrolled_elementwise_kernel"]  # 注意力相关算子列表

SILU_MATCHES = ["silu_kernel", "Silu"]                 # SiLU 激活匹配
ALL_REDUCE_END_MATCHES = ["cross_device_reduce", "AllReduce"]  # AllReduce 结束算子匹配
NVIDIA_MATCHES = ["deep_gemm", "nvjet_"]               # NVIDIA 特征算子集合（当前未单独使用）

def is_embedding(name):  # 判断是否是 embedding kernel
    return name.startswith(EMBEDDING_PREFIX)

def is_layer_start(name):  # 判断是否是 layer 起始 kernel
    return name.startswith(LAYER_START_PREFIX)

def get_gpu_trace(trace):  # 从原始 traceEvents 过滤出 GPU 相关事件
    gpu_trace = list()
    for event in trace:
        try:
            if event["cat"] in GPU_TRACE_CAT:  # 只保留关心的类别
                gpu_trace.append(event)
        except:
            pass  # 忽略无效事件
    return gpu_trace

def get_one_forward_trace(gpu_trace, start_index = 0):  # 获取一次前向的事件序列（embedding 到下一次 embedding）
    forward_trace = list()
    started = False
    next_start_index = None
    for index, event in enumerate(gpu_trace[start_index:]):
        if not started and is_embedding(event['name']):  # 第一次遇到 embedding，标记进入
            started = True
            continue
        if started:
            if not is_embedding(event['name']):          # 前向中：持续收集非 embedding 事件
                forward_trace.append(event)
                continue
            else:                                        # 遇到下一个 embedding 说明本次前向结束
                started = False
                next_start_index = index
                break
    return forward_trace, next_start_index  # 返回当前前向trace以及下次继续的起点偏移

def get_layer_traces(forward_trace):  # 按 layer_start 标记切分前向 trace 为多个 layer
    one_layer_trace = []
    layer_traces = []
    started = False
    for event in forward_trace:
        if not started and is_layer_start(event["name"]):  # 第一次进入一层
            started = True
            one_layer_trace.append([event["name"], event["dur"], event["ts"]])
            continue
        if started:
            if not is_layer_start(event["name"]):          # 层内部事件收集
                one_layer_trace.append([event["name"], event["dur"], event["ts"]])
            else:                                          # 遇到下一层起点，封闭上一层
                kernels_dur = sum([t[1] for t in one_layer_trace])  # 所有 kernel dur 总和
                trace_dur = one_layer_trace[-1][2] + one_layer_trace[-1][1] - one_layer_trace[0][2]  # 层时间跨度
                layer_traces.append((trace_dur, kernels_dur, one_layer_trace))  # 保存层 (层总时间, 内核时间和, 详细列表)
                one_layer_trace = []
                one_layer_trace.append([event["name"], event["dur"], event["ts"]])  # 开始新层
    return layer_traces  # 返回所有层

def match(name, match_list):  # 判断名称是否包含匹配列表中的任一子串
    if not isinstance(name, str):
        name = name[0]
    if isinstance(match_list, str):
        match_list = [match_list]
    for m in match_list:
        if m in name:
            return True
    return False

def get_match_indexes(trace, match_list):  # 获取 trace 中所有匹配某类子串的索引
    if isinstance(match_list, str):
        match_list = [match_list]
    indexes = list()
    for index, (n, t, ts) in enumerate(trace):
        for m in match_list:
            if m in n:
                indexes.append(index)
                break
    return indexes

def tune_trace(trace):  # 对 layer 内 trace 做模式合并（减少碎片化算子）
    new_trace = list()
    str_0 = "void flashinfer::"
    str_1 =  "elementwise"
    for index, (n, t, ts) in enumerate(trace):
        try:
            if match(n, str_0) and match(trace[index+1], str_1):  # flashinfer 主算子与后面 elementwise 合并
                t = t + trace[index+1][1]
            if match(n, str_1) and match(trace[index-1], str_0):  # 已被合并的 elementwise 跳过
                continue
        except:
            pass
        new_trace.append((n, t, ts))
    trace = new_trace

    # 合并注意力+Memcpy 模式
    new_trace = list()
    str_0 = ATTENTION_MATCHES
    str_1 =  "Memcpy"
    for index, (n, t, ts) in enumerate(trace):
        try:
            if match(n, str_0) and match(trace[index-1], str_1):  # 注意力核与前驱 Memcpy 合并
                t = t + trace[index+1][1]
            if match(n, str_1) and match(trace[index-1], str_0):  # 已合并的 Memcpy 跳过
                continue
        except:
            pass
        new_trace.append((n, t, ts))
    trace = new_trace

    # 合并 quantize + gemm + Memcpy 模式（典型 o_proj 序列）
    new_trace = list()
    str_0 = ATTENTION_MATCHES
    str_1 = "computeFP8Quantize128Kernel"
    str_2 = "deep_gemm"
    str_3 = "Memcpy"
    for index, (n, t, ts) in enumerate(trace):
        try:
            if match(n, str_2) and match(trace[index-1][0], str_1) and match(trace[index+1][0], str_3) and match(trace[index-2][0], str_0):
                t = t + trace[index+1][1]  # gemm 吞并后续 Memcpy
            if match(n, str_3) and match(trace[index-1][0], str_2) and match(trace[index-2],str_1) and match(trace[index-3], str_0):
                continue                  # 已被吞并的 Memcpy 跳过
        except:
            pass
        new_trace.append((n, t, ts))
    trace = new_trace

    return trace  # 返回合并后的 trace

def padding(obj, target):  # 不足指定长度用占位补齐，方便列对齐
    for _ in range(target - len(obj)):
        obj.append(("NA", 0, 0))

def get_details(layer_trace):  # 将单层 trace 分类拆分为各功能段并统计
    trace_dur, kernels_dur, trace = layer_trace
    trace = tune_trace(trace)              # 先合并模式
    bubble_dur = trace_dur - kernels_dur   # 计算气泡时间（调度/等待）

    # 预设各模块期望的条目数量（用于 padding）
    rms_norm_num = 3
    qkv_proj_num = 2
    qk_norm_num = 1
    silu_num = 1
    after_silu_gemm_num=3
    rms_norm = list()
    qk_norm = list()
    qkv_proj = list()
    rotary_emb = list()
    rotary_emb_num = 4
    mha = list()
    mha_num = 2
    o_proj = list()
    o_proj_num = 2
    mha_all_reduce = list()
    mha_all_reduce_num = 2
    post_norm = list()
    post_norm_num = 1
    before_silu_gemm = list()
    before_silu_gemm_num = 6
    silu = list()
    after_silu_gemm = list()
    gemm_all_reduce = list()
    gemm_all_reduce_num = 2
    bubbles = [("bubbles", bubble_dur, 0)]  # 气泡占位

    # rms_norm（假设前3个）
    rms_norm.extend(trace[:3])

    # qkv_proj 与 qk_norm（支持 fusedQkRmsNorm）
    qk_norm_indexes = get_match_indexes(trace, "fusedQkRmsNorm")
    has_qk_norm = len(qk_norm_indexes)>0
    if has_qk_norm:
        qk_norm_index = qk_norm_indexes[0]
    if has_qk_norm:
        qk_norm.append(trace[qk_norm_index])          # 单条 qk_norm
        qkv_proj.extend(trace[3:qk_norm_index])       # 前面剩余属于 qkv 投影
        qkv_proj_end_index = qk_norm_index-1
    else:
        qkv_proj_indexes = get_match_indexes(trace[:5], PRE_GEMM_MATCHES + GEMM_MATCHES + POST_GEMM_MATCHES)
        qkv_proj_end_index = qkv_proj_indexes[-1]
        for i in qkv_proj_indexes:
            qkv_proj.append(trace[i])

    # mha 主体（可能两条）
    mha_index = get_match_indexes(trace, ATTENTION_MATCHES)[0]
    mha.append(trace[mha_index])
    if match(trace[mha_index+1], ATTENTION_MATCHES):
        mha_end_index = mha_index+1
        mha.append(trace[mha_index+1])
    else:
        mha_end_index = mha_index

    # rotary embedding 范围
    if has_qk_norm:
        rotary_emb.extend(trace[qk_norm_index+1: mha_index])
    else:
        rotary_emb.extend(trace[qkv_proj_end_index+1: mha_index])

    # post norm（最后一个归一化）
    post_norm_index = get_match_indexes(trace, GENERAL_NORM_MATCHES)[-1]
    post_norm.append(trace[post_norm_index])

    # o_proj 与 mha_all_reduce 分离
    all_reduce_end_indexes = get_match_indexes(trace, ALL_REDUCE_END_MATCHES)
    has_all_reduce = len(all_reduce_end_indexes)>0
    if not has_all_reduce:
        o_proj.extend(trace[mha_end_index+1: post_norm_index])
    else:
        mha_all_reduce_end_index = all_reduce_end_indexes[0]
        if match(trace[mha_all_reduce_end_index-1],"Memcpy"):
            mha_all_reduce_index = mha_all_reduce_end_index-1
        else:
            mha_all_reduce_index = mha_all_reduce_end_index
        mha_all_reduce.extend(trace[mha_all_reduce_index: mha_all_reduce_end_index+1])
        o_proj.extend(trace[mha_end_index+1: mha_all_reduce_index])

    # silu 激活
    silu_index = get_match_indexes(trace, SILU_MATCHES)[0]
    silu.append(trace[silu_index])

    # silu 前的 gemm 及其它（post_norm 后到 silu 前）
    before_silu_gemm.extend(trace[post_norm_index+1:silu_index])

    # silu 后的算子 + 可能的 all_reduce
    if has_all_reduce:
        gemm_all_reduce_end_index = all_reduce_end_indexes[-1]
        if match(trace[gemm_all_reduce_end_index-1],"Memcpy"):
            gemm_all_reduce_index = gemm_all_reduce_end_index-1
        else:
            gemm_all_reduce_index = gemm_all_reduce_end_index
        gemm_all_reduce.extend(trace[gemm_all_reduce_index: gemm_all_reduce_end_index+1])
        after_silu_gemm.extend(trace[silu_index+1: gemm_all_reduce_index])
    else:
        after_silu_gemm.extend(trace[silu_index+1:])

    # padding 各模块到固定长度
    padding(rms_norm, rms_norm_num)
    padding(qkv_proj, qkv_proj_num)
    padding(qk_norm, qk_norm_num)
    padding(rotary_emb, rotary_emb_num)
    padding(mha, mha_num)
    padding(o_proj, o_proj_num)
    padding(mha_all_reduce, mha_all_reduce_num)
    padding(post_norm, post_norm_num)
    padding(before_silu_gemm, before_silu_gemm_num)
    padding(silu, silu_num)
    padding(after_silu_gemm, after_silu_gemm_num)
    padding(gemm_all_reduce, gemm_all_reduce_num)

    # 聚合各段顺序输出
    details = [rms_norm, qkv_proj, qk_norm, rotary_emb, mha, o_proj, 
              mha_all_reduce, post_norm, before_silu_gemm, silu, 
              after_silu_gemm, gemm_all_reduce, bubbles]
    return details  # 返回分类后的层细节

def print_details(details):  # 打印层内各算子及时间统计
    kernels_dur, trace_dur = 0, 0
    # 计算所有算子时间总和（不含气泡）
    for item in details[:-1]:
        for n, t, ts in item:
            kernels_dur += t
    bubble_t = details[-1][0][1]        # 气泡时间
    trace_dur = kernels_dur + bubble_t  # 层总时间 = 算子 + 气泡

    # 输出每个条目（截断名称）
    for item in details:
        for n, t, ts in item:
            print(f'"{n[:60]}",  {t:.3f}')
    print(f'=> kernel total = {kernels_dur:.3f}')
    print(f'=> layer total = {trace_dur:.3f}')

def save_to_csv(data):  # 将处理后的多 batch 层结构统计写成 CSV（含汇总与详细）
    '''
              prefill_bs_1     rms_norm                mha
    data = [ (title, [ [(n,t,ts),(n,t,ts)], [(n,t,ts),(n,t,ts)] ]), 
             (title, [ [(n,t,ts),(n,t,ts)], [(n,t,ts),(n,t,ts)] ])]
    
              prefill_bs_1  
    data_sum = [ (title,       [ 1, 3, 5 ]), 
                 (title,       [ 1, 3, 5 ])]

                 prefill_bs_1     
    data_flat = [ (title, [ (n,t,ts),(n,t,ts), (n,t,ts),(n,t,ts)] ), 
                  (title, [ (n,t,ts),(n,t,ts), (n,t,ts),(n,t,ts)] )
    '''
    rows = []

    # 汇总图（每段总时间）
    data_sum = list()
    for title, details in data:
        detail_sum = list()
        for i, part in enumerate(details):
            part_sum = sum([k[1] for k in part])  # 该功能段总时间
            detail_sum.append(part_sum)
            if i == 5:  # i==5 时附加一个累计项（1~5 段之和，业务自定义）
                detail_sum.append(sum(detail_sum[1:6]))
        data_sum.append((title, detail_sum))
    
    rows.append(",".join([d[0] for d in data_sum]))  # 写标题行
    rows_len = len(data_sum[0][1])
    for row_index in range(rows_len):
        row_str = ""
        for title, details_sum in data_sum:
            t = details_sum[row_index]
            row_str += f"{t:.3f},"
        rows.append(row_str.rstrip(","))

    # 详细图（展开所有算子时间）
    rows.extend([""]*3)  # 空行分隔
    data_flat = list()
    for title, details in data:
        details_flat = list()
        for d in details:
            details_flat.extend(d)  # 拼接所有段条目
        data_flat.append((title, details_flat))

    rows.append(",".join([d[0] for d in data_flat]))  # 标题行（batch 名）
    rows_len = len(data_flat[0][1])
    for row_index in range(rows_len):
        row_str = ""
        for title, details_flat in data_flat:
            n,t,ts = details_flat[row_index]
            row_str += f"{t:.3f},"
        rows.append(row_str.rstrip(","))

    assert rows  # 保护：必须有内容
    out_name = f"kernel.csv"
    with open(out_name, "w") as f:
        for row in rows:
            f.write(row + "\n")
    print(f" ==> Saved to {out_name}")  # 输出保存提示
    
def coefficient_of_variation(data):  # 变异系数：标准差 / 均值
    mean = statistics.mean(data)
    std_dev = statistics.stdev(data) 
    cv = std_dev / mean
    return cv

def print_layer_traces(layer_traces):  # 打印最小/中位/最大层耗时及方差情况，并返回最小层的细节
    min_layer_trace = min(layer_traces)  # 最短层
    max_layer_trace = max(layer_traces)  # 最长层
    median_layer_trace = statistics.median(layer_traces)  # 中位层
    cv = coefficient_of_variation([t[0] for t in layer_traces])  # 计算 layer 时间的变异系数
    print("=======================================================================")
    print("[min details]")
    details = get_details(min_layer_trace)  # 获取最短层的分类细节
    print_details(details)                  # 打印该层的详细算子时间
    print(f"\n[min] kernel total = {min_layer_trace[1]:.3f}, layer total = {min_layer_trace[0]:.3f}")
    print(f"[median] kernel total = {median_layer_trace[1]:.3f}, layer total = {median_layer_trace[0]:.3f}")
    print(f"[max] kernel total = {max_layer_trace[1]:.3f}, layer total = {max_layer_trace[0]:.3f}")
    print(f'[variance] coefficient of variation = {cv:.3f} {" !!!This is large!!!" if cv >0.1 else "" }')
    print("=======================================================================\n")
    return details  # 返回最短层细节（后续用于 prefill/decode 判定）

def process_json_file(json_file):  # 处理单个 nsys 导出的 JSON trace 文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    gpu_trace = get_gpu_trace(data['traceEvents'])  # 过滤 GPU 相关事件
    forward_trace, start_index = get_one_forward_trace(gpu_trace)  # 获取第一次前向
    # WAR：如果没找到 embedding 分界，就退化为整个 gpu_trace
    if not forward_trace:
        forward_trace = gpu_trace
    layer_traces = get_layer_traces(forward_trace)  # 拆层
    details = print_layer_traces(layer_traces)      # 打印层统计详情
    # aggressive decision：以首段 rms_norm 时间粗判是否是 prefill（>50 视为 prefill）
    is_prefill = details[0][0][1] > 50
    prefill_details, decode_details = [], []

    # 如果不是 prefill，则直接当成 decode 返回
    if not is_prefill:
        decode_details = details
        return [], decode_details
    # 是 prefill：记录 prefill，并尝试继续寻找 decode（第二次前向）
    else:
        prefill_details = details 
        if start_index is not None:
            forward_trace, start_index = get_one_forward_trace(gpu_trace, start_index=start_index)
            layer_traces = get_layer_traces(forward_trace)
            decode_details = print_layer_traces(layer_traces)
    return prefill_details, decode_details  # 返回两类细节（可能某类为空）

if __name__ == '__main__':  # 脚本入口
    json_file_list = sys.argv[1:]  # 获取命令行传入的多个 JSON 文件路径
    json_file_list = [ (int(re.search(r'_b(\d+)_', json_file).group(1)), json_file) for json_file in json_file_list ]  # 解析文件名中 batch size（_b{num}_）
    json_file_list.sort()  # 按 batch size 排序
    data = list()
    for bs, json_file in json_file_list:
        print(f"==== batch size [{bs}]==")
        prefill_details, decode_details = process_json_file(json_file)  # 处理单文件
        if bs == 1 and prefill_details:            # 仅在 batch=1 时记录 prefill（策略约束）
            data.append((f"Prefill_BS_{bs}", prefill_details))
        if decode_details:                         # 若有 decode 结果则加入
            data.append((f"Decode_BS_{bs}", decode_details))

    save_to_csv(data)  # 按收集的多组数据输出 CSV
