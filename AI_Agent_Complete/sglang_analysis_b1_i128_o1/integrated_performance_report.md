集成性能分析报告

生成时间: 2025年12月17日

一、Nsys 全局性能概览
 总kernels数量: 13
 总kernel执行时间: 0.22 ms

1. void flashinfer::norm::RMSNormKernel<(unsigned int)8, __half>(T2 *, T2 *, T2 *, unsigned int, unsigned int, unsigned int, float, float)
   - 执行时间: 0.003 ms
   - 时间占比: 1.35%

2. nvjet_hsh_96x128_64x7_2x1_v_bz_TNN
   - 执行时间: 0.046 ms
   - 时间占比: 20.72%

3. void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel<(bool)0, (unsigned int)128, (unsigned int)8, (unsigned int)16, __half, long>(T5 *, T5 *, T5 *, T5 *, float *, T6 *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long)
   - 执行时间: 0.004 ms
   - 时间占比: 1.80%

4. void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(int)2>>(at::TensorIterator &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3)
   - 执行时间: 0.006 ms
   - 时间占比: 2.70%

5. void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(int)2>>(at::TensorIterator &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3)
   - 执行时间: 0.006 ms
   - 时间占比: 2.70%

6. void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 10)]::operator ()() const::[lambda(c10::Half) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
   - 执行时间: 0.004 ms
   - 时间占比: 1.80%

7. flash::prepare_varlen_num_blocks_kernel(int, int, int, const int *, const int *, const int *, const int *, const int *, const int *, int, int, int, int, int, cutlass::FastDivmod, cutlass::FastDivmod, int *, int *, bool)
   - 执行时间: 0.003 ms
   - 时间占比: 1.35%

8. void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<(int)2, cute::tuple<cute::C<(int)1>, cute::C<(int)1>, cute::C<(int)1>>, cute::tuple<cute::C<(int)128>, cute::C<(int)128>, cute::C<(int)128>>, (int)128, cutlass::half_t, float, cutlass::arch::Sm90, (bool)1, (bool)0, (bool)0, (bool)1, (bool)1, (bool)0, (bool)0, (bool)1, (bool)1, (bool)1, (bool)0, (bool)0, cutlass::bfloat16_t>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<(int)128>, cute::C<(int)128>, cute::C<(int)128>>, cute::tuple<cute::C<(int)1>, cute::C<(int)1>, cute::C<(int)1>>, cutlass::half_t, cutlass::arch::Sm90, (int)256, (bool)1, (bool)1, (bool)0, (bool)0>, flash::VarlenDynamicPersistentTileScheduler<(int)128, (int)256, (int)128, (bool)0, (bool)1, (bool)1>>>>(T1::Params)
   - 执行时间: 0.011 ms
   - 时间占比: 4.95%

9. nvjet_hsh_64x64_64x13_1x2_h_bz_TNT
   - 执行时间: 0.017 ms
   - 时间占比: 7.66%

10. void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __half>(T2 *, T2 *, T2 *, unsigned int, unsigned int, unsigned int, float, float)
   - 执行时间: 0.004 ms
   - 时间占比: 1.80%

11. nvjet_hsh_168x128_64x5_2x1_v_bz_TNN
   - 执行时间: 0.074 ms
   - 时间占比: 33.33%

12. void flashinfer::activation::act_and_mul_kernel<__half, &silu<float>>(T1 *, const T1 *, int)
   - 执行时间: 0.004 ms
   - 时间占比: 1.80%

13. nvjet_hsh_64x64_64x13_1x2_h_bz_TNT
   - 执行时间: 0.040 ms
   - 时间占比: 18.02%

二、 NCU 深度分析结果

1. ncu_kernel_0_void_flashinfer__norm__RMSNormKernel__unsigned_int

   - 识别瓶颈数: 3
   - 平均SM效率: 23.54
   - 最高SM效率: 23.54
   - 最低SM效率: 23.54
   - 低于50%数量: 1 / 1
   - 平均带宽: 204.89 GB/s
   - 最高带宽: 204.89 GB/s
   - 最低带宽: 204.89 GB/s
   - 平均L2命中率: 67.67
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 25.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - SM效率过低 (23.5%) (medium)
     - L2缓存命中率低 (67.7%) (medium)
     - 占用率效率低 (24.3%) (medium)

2. ncu_kernel_1_nvjet_hsh_96x128_64x7_2x1_v_bz_TNN

   - 识别瓶颈数: 2
   - 平均SM效率: 50.505
   - 最高SM效率: 50.505
   - 最低SM效率: 50.505
   - 低于50%数量: 0 / 1
   - 平均带宽: 2.565 GB/s
   - 最高带宽: 2.565 GB/s
   - 最低带宽: 2.565 GB/s
   - 平均L2命中率: 31.96
   - 低L2命中率kernel数: 1
   - 平均L1命中率: 0.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - 内存带宽利用率低 (2.6 GB/s) (medium)
     - L2缓存命中率低 (32.0%) (medium)

3. ncu_kernel_2_void_flashinfer__BatchQKApplyRotaryPosIdsCosSinCac

   - 识别瓶颈数: 3
   - 平均SM效率: 25.59
   - 最高SM效率: 25.59
   - 最低SM效率: 25.59
   - 低于50%数量: 1 / 1
   - 平均带宽: 387.20000000000005 GB/s
   - 最高带宽: 387.20000000000005 GB/s
   - 最低带宽: 387.20000000000005 GB/s
   - 平均L2命中率: 60.745000000000005
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 71.38
   - 低L1命中率kernel数: 0
   - 主要瓶颈:
     - SM效率过低 (25.6%) (medium)
     - L2缓存命中率低 (60.7%) (medium)
     - 占用率效率低 (58.5%) (medium)

4. ncu_kernel_3_void_at__native__index_elementwise_kernel__int_128

   - 识别瓶颈数: 2
   - 平均SM效率: 38.0325
   - 最高SM效率: 38.0325
   - 最低SM效率: 38.0325
   - 低于50%数量: 1 / 1
   - 平均带宽: 141.28249999999997 GB/s
   - 最高带宽: 141.28249999999997 GB/s
   - 最低带宽: 141.28249999999997 GB/s
   - 平均L2命中率: 67.11250000000001
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 36.3025
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - L2缓存命中率低 (67.1%) (medium)
     - 占用率效率低 (60.5%) (medium)

5. ncu_kernel_4_void_at__native__index_elementwise_kernel__int_128

   - 识别瓶颈数: 2
   - 平均SM效率: 38.0125
   - 最高SM效率: 38.0125
   - 最低SM效率: 38.0125
   - 低于50%数量: 1 / 1
   - 平均带宽: 139.6125 GB/s
   - 最高带宽: 139.6125 GB/s
   - 最低带宽: 139.6125 GB/s
   - 平均L2命中率: 67.03
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 36.3425
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - L2缓存命中率低 (67.0%) (medium)
     - 占用率效率低 (60.2%) (medium)

6. ncu_kernel_5_void_at__native__elementwise_kernel__int_128___int

   - 识别瓶颈数: 3
   - 平均SM效率: 24.275
   - 最高SM效率: 24.275
   - 最低SM效率: 24.275
   - 低于50%数量: 1 / 1
   - 平均带宽: 173.51999999999998 GB/s
   - 最高带宽: 173.51999999999998 GB/s
   - 最低带宽: 173.51999999999998 GB/s
   - 平均L2命中率: 66.27000000000001
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 22.005
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - SM效率过低 (24.3%) (medium)
     - L2缓存命中率低 (66.3%) (medium)
     - 占用率效率低 (43.6%) (medium)

7. ncu_kernel_6_flash__prepare_varlen_num_blocks_kernel_int__int__

   - 识别瓶颈数: 3
   - 平均SM效率: 21.82
   - 最高SM效率: 21.82
   - 最低SM效率: 21.82
   - 低于50%数量: 1 / 1
   - 平均带宽: 1.195 GB/s
   - 最高带宽: 1.195 GB/s
   - 最低带宽: 1.195 GB/s
   - 平均L2命中率: 100.355
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 25.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - SM效率过低 (21.8%) (medium)
     - 内存带宽利用率低 (1.2 GB/s) (medium)
     - 占用率效率低 (46.2%) (medium)

8. ncu_kernel_7_void_cutlass__device_kernel_flash__enable_sm90_or_

   - 识别瓶颈数: 2
   - 平均SM效率: 15.190000000000001
   - 最高SM效率: 15.190000000000001
   - 最低SM效率: 15.190000000000001
   - 低于50%数量: 1 / 1
   - 平均带宽: 232.24 GB/s
   - 最高带宽: 232.24 GB/s
   - 最低带宽: 232.24 GB/s
   - 平均L2命中率: 42.745
   - 低L2命中率kernel数: 1
   - 平均L1命中率: 2.0999999999999996
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - SM效率过低 (15.2%) (medium)
     - L2缓存命中率低 (42.7%) (medium)

9. ncu_kernel_8_nvjet_hsh_64x64_64x13_1x2_h_bz_TNT

   - 识别瓶颈数: 2
   - 平均SM效率: 42.9625
   - 最高SM效率: 42.9625
   - 最低SM效率: 42.9625
   - 低于50%数量: 1 / 1
   - 平均带宽: 2.265 GB/s
   - 最高带宽: 2.265 GB/s
   - 最低带宽: 2.265 GB/s
   - 平均L2命中率: 39.1375
   - 低L2命中率kernel数: 1
   - 平均L1命中率: 0.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - 内存带宽利用率低 (2.3 GB/s) (medium)
     - L2缓存命中率低 (39.1%) (medium)

10. ncu_kernel_9_void_flashinfer__norm__FusedAddRMSNormKernel__unsi

   - 识别瓶颈数: 3
   - 平均SM效率: 20.52
   - 最高SM效率: 20.52
   - 最低SM效率: 20.52
   - 低于50%数量: 1 / 1
   - 平均带宽: 333.86 GB/s
   - 最高带宽: 333.86 GB/s
   - 最低带宽: 333.86 GB/s
   - 平均L2命中率: 61.377500000000005
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 40.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - SM效率过低 (20.5%) (medium)
     - L2缓存命中率低 (61.4%) (medium)
     - 占用率效率低 (24.3%) (medium)

11. ncu_kernel_10_nvjet_hsh_168x128_64x5_2x1_v_bz_TNN

   - 识别瓶颈数: 2
   - 平均SM效率: 49.91
   - 最高SM效率: 49.91
   - 最低SM效率: 49.91
   - 低于50%数量: 1 / 1
   - 平均带宽: 2.56 GB/s
   - 最高带宽: 2.56 GB/s
   - 最低带宽: 2.56 GB/s
   - 平均L2命中率: 22.57
   - 低L2命中率kernel数: 1
   - 平均L1命中率: 0.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - 内存带宽利用率低 (2.6 GB/s) (medium)
     - L2缓存命中率低 (22.6%) (medium)

12. ncu_kernel_11_void_flashinfer__activation__act_and_mul_kernel___

   - 识别瓶颈数: 2
   - 平均SM效率: 34.644999999999996
   - 最高SM效率: 34.644999999999996
   - 最低SM效率: 34.644999999999996
   - 低于50%数量: 1 / 1
   - 平均带宽: 814.815 GB/s
   - 最高带宽: 814.815 GB/s
   - 最低带宽: 814.815 GB/s
   - 平均L2命中率: 44.165
   - 低L2命中率kernel数: 1
   - 平均L1命中率: 21.64
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - L2缓存命中率低 (44.2%) (medium)
     - 占用率效率低 (43.8%) (medium)

13. ncu_kernel_12_nvjet_hsh_64x64_64x13_1x2_h_bz_TNT

   - 识别瓶颈数: 2
   - 平均SM效率: 43.1975
   - 最高SM效率: 43.1975
   - 最低SM效率: 43.1975
   - 低于50%数量: 1 / 1
   - 平均带宽: 2.2475 GB/s
   - 最高带宽: 2.2475 GB/s
   - 最低带宽: 2.2475 GB/s
   - 平均L2命中率: 37.662499999999994
   - 低L2命中率kernel数: 1
   - 平均L1命中率: 0.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - 内存带宽利用率低 (2.2 GB/s) (medium)
     - L2缓存命中率低 (37.7%) (medium)

