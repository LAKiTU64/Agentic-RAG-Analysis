# 集成性能分析报告

- 生成时间: 2026年01月14日
- 硬件: nvidia_H800_SXM5_80G
- batch_size: 1
- input_len: 128
- output_len: 1

# 一、Roofline 预测与实测指标
 硬件: nvidia_H800_SXM5_80G
 精度: W16 / A16 / KV16
 Prefill 阶段: 强度 540.79 OPs/Byte, 受限类型 compute, 性能 989.50 TOPS, 预计时长 0.708 ms, 内存访问 1.29 GB
 Decode 单token: 强度 1.00 OPs/Byte, 受限类型 memory, 性能 3.35 TOPS, 单token时长 0.204 ms, 总时长(@输出1 token) 0.204 ms, 总内存 0.68 GB
 总体估计: 强度 354.18 OPs/Byte, 受限类型 compute, 性能 989.50 TOPS, 预计总时长 0.912 ms, 总体OPs 0.70 TOPs

 实测均值: SM效率 81.3%, 内存带宽 280.2 GB/s, 算术强度 2869.67 OPs/Byte, 计算吞吐 804.12 TOPS, 推断受限 compute
 利用率: 计算 81.3%, 内存 8.4%
 差异: 性能差 -18.7%, 强度差 710.2%, 边界判断 match

# 二、Nsys 全局性能概览
 总kernels数量: 13
 总kernel执行时间: 0.79 ms
 Layer[0]#Run[1] 范围持续时间: 76.02 ms, 空泡率 98.96 %

1. **kernel name**: void flashinfer::norm::RMSNormKernel<(unsigned int)8, __half>(T2 *, T2 *, T2 *, unsigned int, unsigned int, unsigned int, float, float)
   - 执行时间: 0.007 ms
   - 时间占比: 0.88%

2. **kernel name**: nvjet_hsh_128x256_64x4_2x1_v_bz_coopA_TNN
   - 执行时间: 0.163 ms
   - 时间占比: 20.55%

3. **kernel name**: void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel<(bool)0, (unsigned int)128, (unsigned int)8, (unsigned int)16, __half, long>(T5 *, T5 *, T5 *, T5 *, float *, T6 *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long)
   - 执行时间: 0.013 ms
   - 时间占比: 1.64%

4. **kernel name**: void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(int)2>>(at::TensorIterator &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3)
   - 执行时间: 0.023 ms
   - 时间占比: 2.90%

5. **kernel name**: void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(int)2>>(at::TensorIterator &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3)
   - 执行时间: 0.025 ms
   - 时间占比: 3.15%

6. **kernel name**: void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 10)]::operator ()() const::[lambda(c10::Half) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
   - 执行时间: 0.015 ms
   - 时间占比: 1.89%

7. **kernel name**: flash::prepare_varlen_num_blocks_kernel(int, int, int, const int *, const int *, const int *, const int *, const int *, const int *, int, int, int, int, int, cutlass::FastDivmod, cutlass::FastDivmod, int *, int *, bool)
   - 执行时间: 0.004 ms
   - 时间占比: 0.50%

8. **kernel name**: void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<(int)2, cute::tuple<cute::C<(int)1>, cute::C<(int)1>, cute::C<(int)1>>, cute::tuple<cute::C<(int)128>, cute::C<(int)128>, cute::C<(int)128>>, (int)128, cutlass::half_t, float, cutlass::arch::Sm90, (bool)1, (bool)0, (bool)0, (bool)1, (bool)1, (bool)0, (bool)0, (bool)1, (bool)1, (bool)1, (bool)0, (bool)0, cutlass::bfloat16_t>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<(int)128>, cute::C<(int)128>, cute::C<(int)128>>, cute::tuple<cute::C<(int)1>, cute::C<(int)1>, cute::C<(int)1>>, cutlass::half_t, cutlass::arch::Sm90, (int)256, (bool)1, (bool)1, (bool)0, (bool)0>, flash::VarlenDynamicPersistentTileScheduler<(int)128, (int)256, (int)128, (bool)0, (bool)1, (bool)1>>>>(T1::Params)
   - 执行时间: 0.045 ms
   - 时间占比: 5.67%

9. **kernel name**: nvjet_hsh_128x256_64x4_2x1_v_bz_coopA_TNN
   - 执行时间: 0.053 ms
   - 时间占比: 6.68%

10. **kernel name**: void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __half>(T2 *, T2 *, T2 *, unsigned int, unsigned int, unsigned int, float, float)
   - 执行时间: 0.010 ms
   - 时间占比: 1.26%

11. **kernel name**: nvjet_hsh_192x128_64x5_1x2_h_bz_coopB_TNT
   - 执行时间: 0.279 ms
   - 时间占比: 35.18%

12. **kernel name**: void flashinfer::activation::act_and_mul_kernel<__half, &silu<float>>(T1 *, const T1 *, int)
   - 执行时间: 0.024 ms
   - 时间占比: 3.03%

13. **kernel name**: nvjet_hsh_128x256_64x4_2x1_v_bz_coopA_TNN
   - 执行时间: 0.132 ms
   - 时间占比: 16.65%

# 三、 NCU 深度分析结果

1. **kernel name**: ncu_kernel_0_void_flashinfer__norm__RMSNormKernel__unsigned_int

   - 识别瓶颈数: 1
   - 平均SM效率: 61.3
   - 最高SM效率: 61.3
   - 最低SM效率: 61.3
   - 低于50%数量: 0 / 1
   - 平均带宽: 893.88 GB/s
   - 最高带宽: 893.88 GB/s
   - 最低带宽: 893.88 GB/s
   - 平均L2命中率: 53.2
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 46.58
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - L2缓存命中率低 (53.2%) (medium)

2. **kernel name**: ncu_kernel_1_nvjet_hsh_128x256_64x4_2x1_v_bz_coopA_TNN

   - 识别瓶颈数: 0
   - 平均SM效率: 93.71999999999998
   - 最高SM效率: 93.71999999999998
   - 最低SM效率: 93.71999999999998
   - 低于50%数量: 0 / 1
   - 平均带宽: 866.9933333333333 GB/s
   - 最高带宽: 866.9933333333333 GB/s
   - 最低带宽: 866.9933333333333 GB/s
   - 平均L2命中率: 71.26333333333334
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 0.0
   - 低L1命中率kernel数: 1

3. **kernel name**: ncu_kernel_2_void_flashinfer__BatchQKApplyRotaryPosIdsCosSinCac

   - 识别瓶颈数: 2
   - 平均SM效率: 50.72
   - 最高SM效率: 50.72
   - 最低SM效率: 50.72
   - 低于50%数量: 0 / 1
   - 平均带宽: 1.29 GB/s
   - 最高带宽: 1.29 GB/s
   - 最低带宽: 1.29 GB/s
   - 平均L2命中率: 66.67
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 61.69
   - 低L1命中率kernel数: 0
   - 主要瓶颈:
     - 内存带宽利用率低 (1.3 GB/s) (medium)
     - L2缓存命中率低 (66.7%) (medium)

4. **kernel name**: ncu_kernel_3_void_at__native__index_elementwise_kernel__int_128

   - 识别瓶颈数: 1
   - 平均SM效率: 67.315
   - 最高SM效率: 67.315
   - 最低SM效率: 67.315
   - 低于50%数量: 0 / 1
   - 平均带宽: 319.20000000000005 GB/s
   - 最高带宽: 319.20000000000005 GB/s
   - 最低带宽: 319.20000000000005 GB/s
   - 平均L2命中率: 53.91
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 37.03
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - L2缓存命中率低 (53.9%) (medium)

5. **kernel name**: ncu_kernel_4_void_at__native__elementwise_kernel__int_128___int

   - 识别瓶颈数: 1
   - 平均SM效率: 58.65
   - 最高SM效率: 58.65
   - 最低SM效率: 58.65
   - 低于50%数量: 0 / 1
   - 平均带宽: 601.99 GB/s
   - 最高带宽: 601.99 GB/s
   - 最低带宽: 601.99 GB/s
   - 平均L2命中率: 53.88
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 22.06
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - L2缓存命中率低 (53.9%) (medium)

6. **kernel name**: ncu_kernel_5_flash__prepare_varlen_num_blocks_kernel_int__int__

   - 识别瓶颈数: 3
   - 平均SM效率: 24.51
   - 最高SM效率: 24.51
   - 最低SM效率: 24.51
   - 低于50%数量: 1 / 1
   - 平均带宽: 1.24 GB/s
   - 最高带宽: 1.24 GB/s
   - 最低带宽: 1.24 GB/s
   - 平均L2命中率: 99.16
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 25.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - SM效率过低 (24.5%) (medium)
     - 内存带宽利用率低 (1.2 GB/s) (medium)
     - 占用率效率低 (45.7%) (medium)

7. **kernel name**: ncu_kernel_6_void_cutlass__device_kernel_flash__enable_sm90_or_

   - 识别瓶颈数: 1
   - 平均SM效率: 37.33
   - 最高SM效率: 37.33
   - 最低SM效率: 37.33
   - 低于50%数量: 1 / 1
   - 平均带宽: 557.43 GB/s
   - 最高带宽: 557.43 GB/s
   - 最低带宽: 557.43 GB/s
   - 平均L2命中率: 64.43
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 1.47
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - L2缓存命中率低 (64.4%) (medium)

8. **kernel name**: ncu_kernel_7_void_flashinfer__norm__FusedAddRMSNormKernel__unsi

   - 识别瓶颈数: 2
   - 平均SM效率: 46.94
   - 最高SM效率: 46.94
   - 最低SM效率: 46.94
   - 低于50%数量: 1 / 1
   - 平均带宽: 1.26 GB/s
   - 最高带宽: 1.26 GB/s
   - 最低带宽: 1.26 GB/s
   - 平均L2命中率: 52.07
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 53.815
   - 低L1命中率kernel数: 0
   - 主要瓶颈:
     - 内存带宽利用率低 (1.3 GB/s) (medium)
     - L2缓存命中率低 (52.1%) (medium)

9. **kernel name**: ncu_kernel_8_nvjet_hsh_192x128_64x5_1x2_h_bz_coopB_TNT

   - 识别瓶颈数: 2
   - 平均SM效率: 93.99
   - 最高SM效率: 93.99
   - 最低SM效率: 93.99
   - 低于50%数量: 0 / 1
   - 平均带宽: 1.27 GB/s
   - 最高带宽: 1.27 GB/s
   - 最低带宽: 1.27 GB/s
   - 平均L2命中率: 69.47
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 0.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - 内存带宽利用率低 (1.3 GB/s) (medium)
     - L2缓存命中率低 (69.5%) (medium)

10. **kernel name**: ncu_kernel_9_void_flashinfer__activation__act_and_mul_kernel___

   - 识别瓶颈数: 2
   - 平均SM效率: 47.91
   - 最高SM效率: 47.91
   - 最低SM效率: 47.91
   - 低于50%数量: 1 / 1
   - 平均带宽: 2.26 GB/s
   - 最高带宽: 2.26 GB/s
   - 最低带宽: 2.26 GB/s
   - 平均L2命中率: 39.98
   - 低L2命中率kernel数: 1
   - 平均L1命中率: 21.73
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - 内存带宽利用率低 (2.3 GB/s) (medium)
     - L2缓存命中率低 (40.0%) (medium)

