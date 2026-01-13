集成性能分析报告

生成时间: 2026年01月12日

一、Roofline 预测与实测指标
 硬件: nvidia_H800_SXM5_80G
 精度: W16 / A16 / KV16
 Prefill 阶段: 强度 540.79 OPs/Byte, 受限类型 compute, 性能 989.50 TOPS, 预计时长 0.708 ms, 内存访问 1.29 GB
 Decode 单token: 强度 1.00 OPs/Byte, 受限类型 memory, 性能 3.35 TOPS, 单token时长 0.204 ms, 总时长(@输出1 token) 0.204 ms, 总内存 0.68 GB
 总体估计: 强度 354.18 OPs/Byte, 受限类型 compute, 性能 989.50 TOPS, 预计总时长 0.912 ms, 总体OPs 0.70 TOPs

 实测均值: SM效率 81.1%, 内存带宽 278.1 GB/s, 算术强度 2886.61 OPs/Byte, 计算吞吐 802.84 TOPS, 推断受限 compute
 利用率: 计算 81.1%, 内存 8.3%
 差异: 性能差 -18.9%, 强度差 715.0%, 边界判断 match

二、Nsys 全局性能概览
 总kernels数量: 13
 总kernel执行时间: 0.79 ms
 Layer[0]#Run[2] 范围持续时间: 1.38 ms, 空泡率 42.59 %

1. void flashinfer::norm::RMSNormKernel<(unsigned int)8, __half>(T2 *, T2 *, T2 *, unsigned int, unsigned int, unsigned int, float, float)
   - 执行时间: 0.007 ms
   - 时间占比: 0.89%

2. nvjet_hsh_128x256_64x4_2x1_v_bz_coopA_TNN
   - 执行时间: 0.162 ms
   - 时间占比: 20.51%

3. void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel<(bool)0, (unsigned int)128, (unsigned int)8, (unsigned int)16, __half, long>(T5 *, T5 *, T5 *, T5 *, float *, T6 *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long)
   - 执行时间: 0.013 ms
   - 时间占比: 1.65%

4. void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(int)2>>(at::TensorIterator &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3)
   - 执行时间: 0.024 ms
   - 时间占比: 3.04%

5. void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(int)2>>(at::TensorIterator &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3)
   - 执行时间: 0.025 ms
   - 时间占比: 3.16%

6. void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 10)]::operator ()() const::[lambda(c10::Half) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
   - 执行时间: 0.015 ms
   - 时间占比: 1.90%

7. flash::prepare_varlen_num_blocks_kernel(int, int, int, const int *, const int *, const int *, const int *, const int *, const int *, int, int, int, int, int, cutlass::FastDivmod, cutlass::FastDivmod, int *, int *, bool)
   - 执行时间: 0.003 ms
   - 时间占比: 0.38%

8. void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<(int)2, cute::tuple<cute::C<(int)1>, cute::C<(int)1>, cute::C<(int)1>>, cute::tuple<cute::C<(int)128>, cute::C<(int)128>, cute::C<(int)128>>, (int)128, cutlass::half_t, float, cutlass::arch::Sm90, (bool)1, (bool)0, (bool)0, (bool)1, (bool)1, (bool)0, (bool)0, (bool)1, (bool)1, (bool)1, (bool)0, (bool)0, cutlass::bfloat16_t>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<(int)128>, cute::C<(int)128>, cute::C<(int)128>>, cute::tuple<cute::C<(int)1>, cute::C<(int)1>, cute::C<(int)1>>, cutlass::half_t, cutlass::arch::Sm90, (int)256, (bool)1, (bool)1, (bool)0, (bool)0>, flash::VarlenDynamicPersistentTileScheduler<(int)128, (int)256, (int)128, (bool)0, (bool)1, (bool)1>>>>(T1::Params)
   - 执行时间: 0.046 ms
   - 时间占比: 5.82%

9. nvjet_hsh_128x256_64x4_2x1_v_bz_coopA_TNN
   - 执行时间: 0.054 ms
   - 时间占比: 6.84%

10. void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __half>(T2 *, T2 *, T2 *, unsigned int, unsigned int, unsigned int, float, float)
   - 执行时间: 0.011 ms
   - 时间占比: 1.39%

11. nvjet_hsh_192x128_64x5_1x2_h_bz_coopB_TNT
   - 执行时间: 0.277 ms
   - 时间占比: 35.06%

12. void flashinfer::activation::act_and_mul_kernel<__half, &silu<float>>(T1 *, const T1 *, int)
   - 执行时间: 0.024 ms
   - 时间占比: 3.04%

13. nvjet_hsh_128x256_64x4_2x1_v_bz_coopA_TNN
   - 执行时间: 0.129 ms
   - 时间占比: 16.33%

三、 NCU 深度分析结果

1. ncu_kernel_0_void_flashinfer__norm__RMSNormKernel__unsigned_int

   - 识别瓶颈数: 1
   - 平均SM效率: 60.905
   - 最高SM效率: 60.905
   - 最低SM效率: 60.905
   - 低于50%数量: 0 / 1
   - 平均带宽: 864.19 GB/s
   - 最高带宽: 864.19 GB/s
   - 最低带宽: 864.19 GB/s
   - 平均L2命中率: 53.565
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 46.5
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - L2缓存命中率低 (53.6%) (medium)

2. ncu_kernel_1_nvjet_hsh_128x256_64x4_2x1_v_bz_coopA_TNN

   - 识别瓶颈数: 1
   - 平均SM效率: 93.73166666666667
   - 最高SM效率: 93.73166666666667
   - 最低SM效率: 93.73166666666667
   - 低于50%数量: 0 / 1
   - 平均带宽: 867.38 GB/s
   - 最高带宽: 867.38 GB/s
   - 最低带宽: 867.38 GB/s
   - 平均L2命中率: 69.49166666666666
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 0.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - L2缓存命中率低 (69.5%) (medium)

3. ncu_kernel_2_void_flashinfer__BatchQKApplyRotaryPosIdsCosSinCac

   - 识别瓶颈数: 2
   - 平均SM效率: 50.615
   - 最高SM效率: 50.615
   - 最低SM效率: 50.615
   - 低于50%数量: 0 / 1
   - 平均带宽: 1.3050000000000002 GB/s
   - 最高带宽: 1.3050000000000002 GB/s
   - 最低带宽: 1.3050000000000002 GB/s
   - 平均L2命中率: 66.45
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 61.78
   - 低L1命中率kernel数: 0
   - 主要瓶颈:
     - 内存带宽利用率低 (1.3 GB/s) (medium)
     - L2缓存命中率低 (66.5%) (medium)

4. ncu_kernel_3_void_at__native__index_elementwise_kernel__int_128

   - 识别瓶颈数: 1
   - 平均SM效率: 66.9625
   - 最高SM效率: 66.9625
   - 最低SM效率: 66.9625
   - 低于50%数量: 0 / 1
   - 平均带宽: 317.495 GB/s
   - 最高带宽: 317.495 GB/s
   - 最低带宽: 317.495 GB/s
   - 平均L2命中率: 54.102500000000006
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 36.985
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - L2缓存命中率低 (54.1%) (medium)

5. ncu_kernel_4_void_at__native__elementwise_kernel__int_128___int

   - 识别瓶颈数: 1
   - 平均SM效率: 58.44
   - 最高SM效率: 58.44
   - 最低SM效率: 58.44
   - 低于50%数量: 0 / 1
   - 平均带宽: 586.41 GB/s
   - 最高带宽: 586.41 GB/s
   - 最低带宽: 586.41 GB/s
   - 平均L2命中率: 53.535
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 22.02
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - L2缓存命中率低 (53.5%) (medium)

6. ncu_kernel_5_flash__prepare_varlen_num_blocks_kernel_int__int__

   - 识别瓶颈数: 3
   - 平均SM效率: 22.345
   - 最高SM效率: 22.345
   - 最低SM效率: 22.345
   - 低于50%数量: 1 / 1
   - 平均带宽: 1.145 GB/s
   - 最高带宽: 1.145 GB/s
   - 最低带宽: 1.145 GB/s
   - 平均L2命中率: 99.015
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 25.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - SM效率过低 (22.3%) (medium)
     - 内存带宽利用率低 (1.1 GB/s) (medium)
     - 占用率效率低 (46.1%) (medium)

7. ncu_kernel_6_void_cutlass__device_kernel_flash__enable_sm90_or_

   - 识别瓶颈数: 1
   - 平均SM效率: 37.615
   - 最高SM效率: 37.615
   - 最低SM效率: 37.615
   - 低于50%数量: 1 / 1
   - 平均带宽: 549.725 GB/s
   - 最高带宽: 549.725 GB/s
   - 最低带宽: 549.725 GB/s
   - 平均L2命中率: 64.515
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 1.47
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - L2缓存命中率低 (64.5%) (medium)

8. ncu_kernel_7_void_flashinfer__norm__FusedAddRMSNormKernel__unsi

   - 识别瓶颈数: 2
   - 平均SM效率: 47.019999999999996
   - 最高SM效率: 47.019999999999996
   - 最低SM效率: 47.019999999999996
   - 低于50%数量: 1 / 1
   - 平均带宽: 1.1925 GB/s
   - 最高带宽: 1.1925 GB/s
   - 最低带宽: 1.1925 GB/s
   - 平均L2命中率: 52.0975
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 54.45
   - 低L1命中率kernel数: 0
   - 主要瓶颈:
     - 内存带宽利用率低 (1.2 GB/s) (medium)
     - L2缓存命中率低 (52.1%) (medium)

9. ncu_kernel_8_nvjet_hsh_192x128_64x5_1x2_h_bz_coopB_TNT

   - 识别瓶颈数: 2
   - 平均SM效率: 94.045
   - 最高SM效率: 94.045
   - 最低SM效率: 94.045
   - 低于50%数量: 0 / 1
   - 平均带宽: 1.2650000000000001 GB/s
   - 最高带宽: 1.2650000000000001 GB/s
   - 最低带宽: 1.2650000000000001 GB/s
   - 平均L2命中率: 67.34
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 0.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - 内存带宽利用率低 (1.3 GB/s) (medium)
     - L2缓存命中率低 (67.3%) (medium)

10. ncu_kernel_9_void_flashinfer__activation__act_and_mul_kernel___

   - 识别瓶颈数: 2
   - 平均SM效率: 48.269999999999996
   - 最高SM效率: 48.269999999999996
   - 最低SM效率: 48.269999999999996
   - 低于50%数量: 1 / 1
   - 平均带宽: 2.2350000000000003 GB/s
   - 最高带宽: 2.2350000000000003 GB/s
   - 最低带宽: 2.2350000000000003 GB/s
   - 平均L2命中率: 39.96
   - 低L2命中率kernel数: 1
   - 平均L1命中率: 21.744999999999997
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - 内存带宽利用率低 (2.2 GB/s) (medium)
     - L2缓存命中率低 (40.0%) (medium)

