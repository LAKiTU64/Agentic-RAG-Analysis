集成性能分析报告

生成时间: 2025年12月19日

一、Nsys 全局性能概览
 总kernels数量: 13
 总kernel执行时间: 0.22 ms

1. void flashinfer::norm::RMSNormKernel<(unsigned int)8, __half>(T2 *, T2 *, T2 *, unsigned int, unsigned int, unsigned int, float, float)
   - 执行时间: 0.003 ms
   - 时间占比: 1.36%

2. nvjet_hsh_96x128_64x7_2x1_v_bz_TNN
   - 执行时间: 0.046 ms
   - 时间占比: 20.81%

3. void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel<(bool)0, (unsigned int)128, (unsigned int)8, (unsigned int)16, __half, long>(T5 *, T5 *, T5 *, T5 *, float *, T6 *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long)
   - 执行时间: 0.004 ms
   - 时间占比: 1.81%

4. void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(int)2>>(at::TensorIterator &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3)
   - 执行时间: 0.006 ms
   - 时间占比: 2.71%

5. void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(int)2>>(at::TensorIterator &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &, bool)::[lambda(int) (instance 1)]>(long, T3)
   - 执行时间: 0.006 ms
   - 时间占比: 2.71%

6. void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 10)]::operator ()() const::[lambda(c10::Half) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
   - 执行时间: 0.004 ms
   - 时间占比: 1.81%

7. flash::prepare_varlen_num_blocks_kernel(int, int, int, const int *, const int *, const int *, const int *, const int *, const int *, int, int, int, int, int, cutlass::FastDivmod, cutlass::FastDivmod, int *, int *, bool)
   - 执行时间: 0.003 ms
   - 时间占比: 1.36%

8. void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<(int)2, cute::tuple<cute::C<(int)1>, cute::C<(int)1>, cute::C<(int)1>>, cute::tuple<cute::C<(int)128>, cute::C<(int)128>, cute::C<(int)128>>, (int)128, cutlass::half_t, float, cutlass::arch::Sm90, (bool)1, (bool)0, (bool)0, (bool)1, (bool)1, (bool)0, (bool)0, (bool)1, (bool)1, (bool)1, (bool)0, (bool)0, cutlass::bfloat16_t>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<(int)128>, cute::C<(int)128>, cute::C<(int)128>>, cute::tuple<cute::C<(int)1>, cute::C<(int)1>, cute::C<(int)1>>, cutlass::half_t, cutlass::arch::Sm90, (int)256, (bool)1, (bool)1, (bool)0, (bool)0>, flash::VarlenDynamicPersistentTileScheduler<(int)128, (int)256, (int)128, (bool)0, (bool)1, (bool)1>>>>(T1::Params)
   - 执行时间: 0.012 ms
   - 时间占比: 5.43%

9. nvjet_hsh_64x64_64x13_1x2_h_bz_TNT
   - 执行时间: 0.016 ms
   - 时间占比: 7.24%

10. void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __half>(T2 *, T2 *, T2 *, unsigned int, unsigned int, unsigned int, float, float)
   - 执行时间: 0.004 ms
   - 时间占比: 1.81%

11. nvjet_hsh_168x128_64x5_2x1_v_bz_TNN
   - 执行时间: 0.073 ms
   - 时间占比: 33.03%

12. void flashinfer::activation::act_and_mul_kernel<__half, &silu<float>>(T1 *, const T1 *, int)
   - 执行时间: 0.004 ms
   - 时间占比: 1.81%

13. nvjet_hsh_64x64_64x13_1x2_h_bz_TNT
   - 执行时间: 0.040 ms
   - 时间占比: 18.10%

二、 NCU 深度分析结果

1. ncu_kernel_0_void_flashinfer__norm__RMSNormKernel__unsigned_int

   - 识别瓶颈数: 3
   - 平均SM效率: 22.33
   - 最高SM效率: 22.33
   - 最低SM效率: 22.33
   - 低于50%数量: 1 / 1
   - 平均带宽: 201.69 GB/s
   - 最高带宽: 201.69 GB/s
   - 最低带宽: 201.69 GB/s
   - 平均L2命中率: 67.69
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 25.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - SM效率过低 (22.3%) (medium)
     - L2缓存命中率低 (67.7%) (medium)
     - 占用率效率低 (24.1%) (medium)

2. ncu_kernel_1_nvjet_hsh_96x128_64x7_2x1_v_bz_TNN

   - 识别瓶颈数: 2
   - 平均SM效率: 50.33
   - 最高SM效率: 50.33
   - 最低SM效率: 50.33
   - 低于50%数量: 0 / 1
   - 平均带宽: 2.545 GB/s
   - 最高带宽: 2.545 GB/s
   - 最低带宽: 2.545 GB/s
   - 平均L2命中率: 31.96
   - 低L2命中率kernel数: 1
   - 平均L1命中率: 0.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - 内存带宽利用率低 (2.5 GB/s) (medium)
     - L2缓存命中率低 (32.0%) (medium)

3. ncu_kernel_2_void_flashinfer__BatchQKApplyRotaryPosIdsCosSinCac

   - 识别瓶颈数: 3
   - 平均SM效率: 25.905
   - 最高SM效率: 25.905
   - 最低SM效率: 25.905
   - 低于50%数量: 1 / 1
   - 平均带宽: 392.57000000000005 GB/s
   - 最高带宽: 392.57000000000005 GB/s
   - 最低带宽: 392.57000000000005 GB/s
   - 平均L2命中率: 61.205
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 71.38
   - 低L1命中率kernel数: 0
   - 主要瓶颈:
     - SM效率过低 (25.9%) (medium)
     - L2缓存命中率低 (61.2%) (medium)
     - 占用率效率低 (58.4%) (medium)

4. ncu_kernel_3_void_at__native__index_elementwise_kernel__int_128

   - 识别瓶颈数: 2
   - 平均SM效率: 38.5775
   - 最高SM效率: 38.5775
   - 最低SM效率: 38.5775
   - 低于50%数量: 1 / 1
   - 平均带宽: 140.475 GB/s
   - 最高带宽: 140.475 GB/s
   - 最低带宽: 140.475 GB/s
   - 平均L2命中率: 67.15
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 36.4375
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - L2缓存命中率低 (67.2%) (medium)
     - 占用率效率低 (60.3%) (medium)

5. ncu_kernel_4_void_at__native__index_elementwise_kernel__int_128

   - 识别瓶颈数: 2
   - 平均SM效率: 38.1625
   - 最高SM效率: 38.1625
   - 最低SM效率: 38.1625
   - 低于50%数量: 1 / 1
   - 平均带宽: 141.72749999999996 GB/s
   - 最高带宽: 141.72749999999996 GB/s
   - 最低带宽: 141.72749999999996 GB/s
   - 平均L2命中率: 67.0575
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 36.365
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - L2缓存命中率低 (67.1%) (medium)
     - 占用率效率低 (60.3%) (medium)

6. ncu_kernel_5_void_at__native__elementwise_kernel__int_128___int

   - 识别瓶颈数: 3
   - 平均SM效率: 25.340000000000003
   - 最高SM效率: 25.340000000000003
   - 最低SM效率: 25.340000000000003
   - 低于50%数量: 1 / 1
   - 平均带宽: 179.775 GB/s
   - 最高带宽: 179.775 GB/s
   - 最低带宽: 179.775 GB/s
   - 平均L2命中率: 66.28999999999999
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 22.17
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - SM效率过低 (25.3%) (medium)
     - L2缓存命中率低 (66.3%) (medium)
     - 占用率效率低 (43.5%) (medium)

7. ncu_kernel_6_flash__prepare_varlen_num_blocks_kernel_int__int__

   - 识别瓶颈数: 3
   - 平均SM效率: 23.235
   - 最高SM效率: 23.235
   - 最低SM效率: 23.235
   - 低于50%数量: 1 / 1
   - 平均带宽: 1.15 GB/s
   - 最高带宽: 1.15 GB/s
   - 最低带宽: 1.15 GB/s
   - 平均L2命中率: 100.65
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 25.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - SM效率过低 (23.2%) (medium)
     - 内存带宽利用率低 (1.1 GB/s) (medium)
     - 占用率效率低 (45.9%) (medium)

8. ncu_kernel_7_void_cutlass__device_kernel_flash__enable_sm90_or_

   - 识别瓶颈数: 2
   - 平均SM效率: 15.375
   - 最高SM效率: 15.375
   - 最低SM效率: 15.375
   - 低于50%数量: 1 / 1
   - 平均带宽: 232.485 GB/s
   - 最高带宽: 232.485 GB/s
   - 最低带宽: 232.485 GB/s
   - 平均L2命中率: 42.375
   - 低L2命中率kernel数: 1
   - 平均L1命中率: 2.0949999999999998
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - SM效率过低 (15.4%) (medium)
     - L2缓存命中率低 (42.4%) (medium)

9. ncu_kernel_8_nvjet_hsh_64x64_64x13_1x2_h_bz_TNT

   - 识别瓶颈数: 2
   - 平均SM效率: 42.3875
   - 最高SM效率: 42.3875
   - 最低SM效率: 42.3875
   - 低于50%数量: 1 / 1
   - 平均带宽: 2.2024999999999997 GB/s
   - 最高带宽: 2.2024999999999997 GB/s
   - 最低带宽: 2.2024999999999997 GB/s
   - 平均L2命中率: 38.455
   - 低L2命中率kernel数: 1
   - 平均L1命中率: 0.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - 内存带宽利用率低 (2.2 GB/s) (medium)
     - L2缓存命中率低 (38.5%) (medium)

10. ncu_kernel_9_void_flashinfer__norm__FusedAddRMSNormKernel__unsi

   - 识别瓶颈数: 3
   - 平均SM效率: 20.7625
   - 最高SM效率: 20.7625
   - 最低SM效率: 20.7625
   - 低于50%数量: 1 / 1
   - 平均带宽: 341.3975 GB/s
   - 最高带宽: 341.3975 GB/s
   - 最低带宽: 341.3975 GB/s
   - 平均L2命中率: 61.08
   - 低L2命中率kernel数: 0
   - 平均L1命中率: 40.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - SM效率过低 (20.8%) (medium)
     - L2缓存命中率低 (61.1%) (medium)
     - 占用率效率低 (24.3%) (medium)

11. ncu_kernel_10_nvjet_hsh_168x128_64x5_2x1_v_bz_TNN

   - 识别瓶颈数: 2
   - 平均SM效率: 50.535
   - 最高SM效率: 50.535
   - 最低SM效率: 50.535
   - 低于50%数量: 0 / 1
   - 平均带宽: 2.52 GB/s
   - 最高带宽: 2.52 GB/s
   - 最低带宽: 2.52 GB/s
   - 平均L2命中率: 22.525
   - 低L2命中率kernel数: 1
   - 平均L1命中率: 0.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - 内存带宽利用率低 (2.5 GB/s) (medium)
     - L2缓存命中率低 (22.5%) (medium)

12. ncu_kernel_11_void_flashinfer__activation__act_and_mul_kernel___

   - 识别瓶颈数: 2
   - 平均SM效率: 34.38
   - 最高SM效率: 34.38
   - 最低SM效率: 34.38
   - 低于50%数量: 1 / 1
   - 平均带宽: 785.29 GB/s
   - 最高带宽: 785.29 GB/s
   - 最低带宽: 785.29 GB/s
   - 平均L2命中率: 44.2
   - 低L2命中率kernel数: 1
   - 平均L1命中率: 21.564999999999998
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - L2缓存命中率低 (44.2%) (medium)
     - 占用率效率低 (43.8%) (medium)

13. ncu_kernel_12_nvjet_hsh_64x64_64x13_1x2_h_bz_TNT

   - 识别瓶颈数: 2
   - 平均SM效率: 42.385
   - 最高SM效率: 42.385
   - 最低SM效率: 42.385
   - 低于50%数量: 1 / 1
   - 平均带宽: 2.2375 GB/s
   - 最高带宽: 2.2375 GB/s
   - 最低带宽: 2.2375 GB/s
   - 平均L2命中率: 38.582499999999996
   - 低L2命中率kernel数: 1
   - 平均L1命中率: 0.0
   - 低L1命中率kernel数: 1
   - 主要瓶颈:
     - 内存带宽利用率低 (2.2 GB/s) (medium)
     - L2缓存命中率低 (38.6%) (medium)

