; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_kernel void @matmul_kernel_with_block_pointers_0d1d2d3d4d5d(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5) local_unnamed_addr !intel_reqd_sub_group_size !6 !max_work_group_size !7 {
  %7 = tail call i64 @_Z12get_local_idj(i32 0)
  %8 = trunc i64 %7 to i32
  %9 = tail call i64 @_Z12get_local_idj(i32 1)
  %10 = trunc i64 %9 to i32
  %11 = tail call i64 @_Z12get_local_idj(i32 2)
  %12 = trunc i64 %11 to i32
  %13 = tail call i64 @_Z14get_local_sizej(i32 0)
  %14 = trunc i64 %13 to i32
  %15 = tail call i64 @_Z14get_local_sizej(i32 1)
  %16 = trunc i64 %15 to i32
  %17 = mul i32 %16, %12
  %18 = add i32 %17, %10
  %19 = mul i32 %18, %14
  %20 = add i32 %19, %8
  %21 = lshr i32 %20, 4
  %22 = tail call i64 @_Z12get_group_idj(i32 0)
  %23 = trunc i64 %22 to i32
  %24 = sdiv i32 %23, 64
  %25 = shl nsw i32 %24, 2
  %26 = sub nsw i32 16, %25
  %27 = tail call i32 @llvm.smin.i32(i32 %26, i32 4)
  %28 = srem i32 %23, %27
  %29 = add nsw i32 %25, %28
  %30 = and i32 %23, 63
  %31 = sdiv i32 %30, %27
  %32 = shl i32 %29, 8
  %33 = shl nuw nsw i32 %21, 3
  %34 = add i32 %33, %32
  %35 = ptrtoint ptr addrspace(1) %0 to i64
  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %35, i32 8191, i32 4095, i32 8191, i32 0, i32 %34, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %35, i32 8191, i32 4095, i32 8191, i32 16, i32 %34, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %35, i32 8191, i32 4095, i32 8191, i32 32, i32 %34, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %35, i32 8191, i32 4095, i32 8191, i32 48, i32 %34, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %35, i32 8191, i32 4095, i32 8191, i32 64, i32 %34, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %35, i32 8191, i32 4095, i32 8191, i32 80, i32 %34, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
  %36 = insertelement <2 x i32> <i32 96, i32 poison>, i32 %34, i64 1
  %37 = insertelement <2 x i32> <i32 112, i32 poison>, i32 %34, i64 1
  %38 = lshr i32 %20, 1
  %39 = and i32 %38, 224
  %40 = or disjoint i32 %39, %32
  %41 = insertelement <2 x i32> <i32 0, i32 poison>, i32 %40, i64 1
  %42 = shl nsw i32 %31, 8
  %43 = and i32 %21, 24
  %44 = shl i32 %21, 5
  %45 = and i32 %44, 224
  %46 = or disjoint i32 %45, %42
  %47 = insertelement <2 x i32> poison, i32 %46, i64 0
  %48 = or disjoint i32 %46, 16
  %49 = insertelement <2 x i32> poison, i32 %48, i64 0
  %50 = ptrtoint ptr addrspace(1) %1 to i64
  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %50, i32 8191, i32 4095, i32 8191, i32 %46, i32 %43, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %50, i32 8191, i32 4095, i32 8191, i32 %48, i32 %43, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
  %51 = or disjoint i32 %43, 32
  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %50, i32 8191, i32 4095, i32 8191, i32 %46, i32 %51, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %50, i32 8191, i32 4095, i32 8191, i32 %48, i32 %51, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
  %52 = or disjoint i32 %43, 64
  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %50, i32 8191, i32 4095, i32 8191, i32 %46, i32 %52, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %50, i32 8191, i32 4095, i32 8191, i32 %48, i32 %52, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
  %53 = or disjoint i32 %43, 96
  %54 = insertelement <2 x i32> %47, i32 %53, i64 1
  %55 = insertelement <2 x i32> %49, i32 %53, i64 1
  %56 = shl i32 %21, 6
  %57 = and i32 %56, 192
  %58 = or disjoint i32 %57, %42
  %59 = insertelement <2 x i32> <i32 poison, i32 0>, i32 %58, i64 0
  %60 = or disjoint i32 %58, 32
  %61 = insertelement <2 x i32> <i32 poison, i32 0>, i32 %60, i64 0
  br label %62

62:                                               ; preds = %6, %90
  %63 = phi <2 x i32> [ %55, %6 ], [ %165, %90 ]
  %64 = phi <2 x i32> [ %54, %6 ], [ %163, %90 ]
  %65 = phi <2 x i32> [ %37, %6 ], [ %155, %90 ]
  %66 = phi <2 x i32> [ %36, %6 ], [ %153, %90 ]
  %67 = phi <2 x i32> [ %61, %6 ], [ %169, %90 ]
  %68 = phi <2 x i32> [ %59, %6 ], [ %167, %90 ]
  %69 = phi <2 x i32> [ %41, %6 ], [ %157, %90 ]
  %70 = phi <8 x float> [ zeroinitializer, %6 ], [ %147, %90 ]
  %71 = phi <8 x float> [ zeroinitializer, %6 ], [ %145, %90 ]
  %72 = phi <8 x float> [ zeroinitializer, %6 ], [ %143, %90 ]
  %73 = phi <8 x float> [ zeroinitializer, %6 ], [ %141, %90 ]
  %74 = phi <8 x float> [ zeroinitializer, %6 ], [ %137, %90 ]
  %75 = phi <8 x float> [ zeroinitializer, %6 ], [ %135, %90 ]
  %76 = phi <8 x float> [ zeroinitializer, %6 ], [ %133, %90 ]
  %77 = phi <8 x float> [ zeroinitializer, %6 ], [ %131, %90 ]
  %78 = phi <8 x float> [ zeroinitializer, %6 ], [ %127, %90 ]
  %79 = phi <8 x float> [ zeroinitializer, %6 ], [ %125, %90 ]
  %80 = phi <8 x float> [ zeroinitializer, %6 ], [ %123, %90 ]
  %81 = phi <8 x float> [ zeroinitializer, %6 ], [ %121, %90 ]
  %82 = phi <8 x float> [ zeroinitializer, %6 ], [ %117, %90 ]
  %83 = phi <8 x float> [ zeroinitializer, %6 ], [ %113, %90 ]
  %84 = phi <8 x float> [ zeroinitializer, %6 ], [ %109, %90 ]
  %85 = phi <8 x float> [ zeroinitializer, %6 ], [ %105, %90 ]
  %86 = phi i32 [ 0, %6 ], [ %170, %90 ]
  %87 = and i32 %86, 224
  %88 = icmp eq i32 %87, 0
  br i1 %88, label %89, label %90

89:                                               ; preds = %62
  ;tail call void @_Z7barrierj(i32 1) #1
  ;tail call void @llvm.genx.GenISA.threadgroupbarrier()
  br label %90

90:                                               ; preds = %89, %62
  %91 = extractelement <2 x i32> %69, i64 0
  %92 = extractelement <2 x i32> %69, i64 1
  %93 = tail call <64 x i16> @llvm.genx.GenISA.LSC2DBlockRead.v64i16(i64 %35, i32 8191, i32 4095, i32 8191, i32 %91, i32 %92, i32 16, i32 16, i32 32, i32 2, i1 false, i1 false, i32 0)
  %94 = extractelement <2 x i32> %68, i64 0
  %95 = extractelement <2 x i32> %68, i64 1
  %96 = tail call <32 x i32> @llvm.genx.GenISA.LSC2DBlockRead.v32i32(i64 %50, i32 8191, i32 4095, i32 8191, i32 %94, i32 %95, i32 16, i32 16, i32 32, i32 2, i1 false, i1 true, i32 0)
  %97 = extractelement <2 x i32> %67, i64 0
  %98 = extractelement <2 x i32> %67, i64 1
  %99 = tail call <32 x i32> @llvm.genx.GenISA.LSC2DBlockRead.v32i32(i64 %50, i32 8191, i32 4095, i32 8191, i32 %97, i32 %98, i32 16, i32 16, i32 32, i32 2, i1 false, i1 true, i32 0)
  %100 = shufflevector <64 x i16> %93, <64 x i16> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %101 = shufflevector <32 x i32> %96, <32 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %102 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %85, <8 x i16> %100, <8 x i32> %101, i32 12, i32 12, i32 8, i32 8, i1 false)
  %103 = shufflevector <64 x i16> %93, <64 x i16> poison, <8 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39>
  %104 = shufflevector <32 x i32> %96, <32 x i32> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %105 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %102, <8 x i16> %103, <8 x i32> %104, i32 12, i32 12, i32 8, i32 8, i1 false)
  %106 = shufflevector <64 x i16> %93, <64 x i16> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %107 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %84, <8 x i16> %106, <8 x i32> %101, i32 12, i32 12, i32 8, i32 8, i1 false)
  %108 = shufflevector <64 x i16> %93, <64 x i16> poison, <8 x i32> <i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47>
  %109 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %107, <8 x i16> %108, <8 x i32> %104, i32 12, i32 12, i32 8, i32 8, i1 false)
  %110 = shufflevector <64 x i16> %93, <64 x i16> poison, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  %111 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %83, <8 x i16> %110, <8 x i32> %101, i32 12, i32 12, i32 8, i32 8, i1 false)
  %112 = shufflevector <64 x i16> %93, <64 x i16> poison, <8 x i32> <i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55>
  %113 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %111, <8 x i16> %112, <8 x i32> %104, i32 12, i32 12, i32 8, i32 8, i1 false)
  %114 = shufflevector <64 x i16> %93, <64 x i16> poison, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %115 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %82, <8 x i16> %114, <8 x i32> %101, i32 12, i32 12, i32 8, i32 8, i1 false)
  %116 = shufflevector <64 x i16> %93, <64 x i16> poison, <8 x i32> <i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %117 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %115, <8 x i16> %116, <8 x i32> %104, i32 12, i32 12, i32 8, i32 8, i1 false)
  %118 = shufflevector <32 x i32> %96, <32 x i32> poison, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  %119 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %81, <8 x i16> %100, <8 x i32> %118, i32 12, i32 12, i32 8, i32 8, i1 false)
  %120 = shufflevector <32 x i32> %96, <32 x i32> poison, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %121 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %119, <8 x i16> %103, <8 x i32> %120, i32 12, i32 12, i32 8, i32 8, i1 false)
  %122 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %80, <8 x i16> %106, <8 x i32> %118, i32 12, i32 12, i32 8, i32 8, i1 false)
  %123 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %122, <8 x i16> %108, <8 x i32> %120, i32 12, i32 12, i32 8, i32 8, i1 false)
  %124 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %79, <8 x i16> %110, <8 x i32> %118, i32 12, i32 12, i32 8, i32 8, i1 false)
  %125 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %124, <8 x i16> %112, <8 x i32> %120, i32 12, i32 12, i32 8, i32 8, i1 false)
  %126 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %78, <8 x i16> %114, <8 x i32> %118, i32 12, i32 12, i32 8, i32 8, i1 false)
  %127 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %126, <8 x i16> %116, <8 x i32> %120, i32 12, i32 12, i32 8, i32 8, i1 false)
  %128 = shufflevector <32 x i32> %99, <32 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %129 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %77, <8 x i16> %100, <8 x i32> %128, i32 12, i32 12, i32 8, i32 8, i1 false)
  %130 = shufflevector <32 x i32> %99, <32 x i32> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %131 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %129, <8 x i16> %103, <8 x i32> %130, i32 12, i32 12, i32 8, i32 8, i1 false)
  %132 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %76, <8 x i16> %106, <8 x i32> %128, i32 12, i32 12, i32 8, i32 8, i1 false)
  %133 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %132, <8 x i16> %108, <8 x i32> %130, i32 12, i32 12, i32 8, i32 8, i1 false)
  %134 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %75, <8 x i16> %110, <8 x i32> %128, i32 12, i32 12, i32 8, i32 8, i1 false)
  %135 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %134, <8 x i16> %112, <8 x i32> %130, i32 12, i32 12, i32 8, i32 8, i1 false)
  %136 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %74, <8 x i16> %114, <8 x i32> %128, i32 12, i32 12, i32 8, i32 8, i1 false)
  %137 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %136, <8 x i16> %116, <8 x i32> %130, i32 12, i32 12, i32 8, i32 8, i1 false)
  %138 = shufflevector <32 x i32> %99, <32 x i32> poison, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  %139 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %73, <8 x i16> %100, <8 x i32> %138, i32 12, i32 12, i32 8, i32 8, i1 false)
  %140 = shufflevector <32 x i32> %99, <32 x i32> poison, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %141 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %139, <8 x i16> %103, <8 x i32> %140, i32 12, i32 12, i32 8, i32 8, i1 false)
  %142 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %72, <8 x i16> %106, <8 x i32> %138, i32 12, i32 12, i32 8, i32 8, i1 false)
  %143 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %142, <8 x i16> %108, <8 x i32> %140, i32 12, i32 12, i32 8, i32 8, i1 false)
  %144 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %71, <8 x i16> %110, <8 x i32> %138, i32 12, i32 12, i32 8, i32 8, i1 false)
  %145 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %144, <8 x i16> %112, <8 x i32> %140, i32 12, i32 12, i32 8, i32 8, i1 false)
  %146 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %70, <8 x i16> %114, <8 x i32> %138, i32 12, i32 12, i32 8, i32 8, i1 false)
  %147 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %146, <8 x i16> %116, <8 x i32> %140, i32 12, i32 12, i32 8, i32 8, i1 false)
  %148 = extractelement <2 x i32> %66, i64 0
  %149 = extractelement <2 x i32> %66, i64 1
  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %35, i32 8191, i32 4095, i32 8191, i32 %148, i32 %149, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
  %150 = extractelement <2 x i32> %65, i64 0
  %151 = extractelement <2 x i32> %65, i64 1
  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %35, i32 8191, i32 4095, i32 8191, i32 %150, i32 %151, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
  %152 = add i32 %148, 32
  %153 = insertelement <2 x i32> %66, i32 %152, i64 0
  %154 = add i32 %150, 32
  %155 = insertelement <2 x i32> %65, i32 %154, i64 0
  %156 = add i32 %91, 32
  %157 = insertelement <2 x i32> %69, i32 %156, i64 0
  %158 = extractelement <2 x i32> %64, i64 0
  %159 = extractelement <2 x i32> %64, i64 1
  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %50, i32 8191, i32 4095, i32 8191, i32 %158, i32 %159, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
  %160 = extractelement <2 x i32> %63, i64 0
  %161 = extractelement <2 x i32> %63, i64 1
  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %50, i32 8191, i32 4095, i32 8191, i32 %160, i32 %161, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
  %162 = add i32 %159, 32
  %163 = insertelement <2 x i32> %64, i32 %162, i64 1
  %164 = add i32 %161, 32
  %165 = insertelement <2 x i32> %63, i32 %164, i64 1
  %166 = add i32 %95, 32
  %167 = insertelement <2 x i32> %68, i32 %166, i64 1
  %168 = add i32 %98, 32
  %169 = insertelement <2 x i32> %67, i32 %168, i64 1
  %170 = add nuw nsw i32 %86, 32
  %171 = icmp ult i32 %86, 4064
  br i1 %171, label %62, label %172

172:                                              ; preds = %90
  %173 = or disjoint i32 %40, 8
  %174 = or disjoint i32 %40, 16
  %175 = or disjoint i32 %40, 24
  %176 = or disjoint i32 %58, 16
  %177 = or disjoint i32 %58, 48
  %178 = bitcast <8 x float> %105 to <8 x i32>
  %179 = ptrtoint ptr addrspace(1) %2 to i64
  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %179, i32 16383, i32 4095, i32 16383, i32 %58, i32 %40, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %178)
  %180 = bitcast <8 x float> %109 to <8 x i32>
  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %179, i32 16383, i32 4095, i32 16383, i32 %58, i32 %173, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %180)
  %181 = bitcast <8 x float> %113 to <8 x i32>
  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %179, i32 16383, i32 4095, i32 16383, i32 %58, i32 %174, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %181)
  %182 = bitcast <8 x float> %117 to <8 x i32>
  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %179, i32 16383, i32 4095, i32 16383, i32 %58, i32 %175, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %182)
  %183 = bitcast <8 x float> %121 to <8 x i32>
  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %179, i32 16383, i32 4095, i32 16383, i32 %176, i32 %40, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %183)
  %184 = bitcast <8 x float> %123 to <8 x i32>
  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %179, i32 16383, i32 4095, i32 16383, i32 %176, i32 %173, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %184)
  %185 = bitcast <8 x float> %125 to <8 x i32>
  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %179, i32 16383, i32 4095, i32 16383, i32 %176, i32 %174, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %185)
  %186 = bitcast <8 x float> %127 to <8 x i32>
  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %179, i32 16383, i32 4095, i32 16383, i32 %176, i32 %175, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %186)
  %187 = bitcast <8 x float> %131 to <8 x i32>
  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %179, i32 16383, i32 4095, i32 16383, i32 %60, i32 %40, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %187)
  %188 = bitcast <8 x float> %133 to <8 x i32>
  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %179, i32 16383, i32 4095, i32 16383, i32 %60, i32 %173, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %188)
  %189 = bitcast <8 x float> %135 to <8 x i32>
  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %179, i32 16383, i32 4095, i32 16383, i32 %60, i32 %174, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %189)
  %190 = bitcast <8 x float> %137 to <8 x i32>
  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %179, i32 16383, i32 4095, i32 16383, i32 %60, i32 %175, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %190)
  %191 = bitcast <8 x float> %141 to <8 x i32>
  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %179, i32 16383, i32 4095, i32 16383, i32 %177, i32 %40, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %191)
  %192 = bitcast <8 x float> %143 to <8 x i32>
  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %179, i32 16383, i32 4095, i32 16383, i32 %177, i32 %173, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %192)
  %193 = bitcast <8 x float> %145 to <8 x i32>
  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %179, i32 16383, i32 4095, i32 16383, i32 %177, i32 %174, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %193)
  %194 = bitcast <8 x float> %147 to <8 x i32>
  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %179, i32 16383, i32 4095, i32 16383, i32 %177, i32 %175, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %194)
  ret void
}

declare spir_func i64 @_Z12get_local_idj(i32) local_unnamed_addr

declare spir_func i64 @_Z14get_local_sizej(i32) local_unnamed_addr

declare spir_func i64 @_Z12get_group_idj(i32) local_unnamed_addr

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #0

declare void @llvm.genx.GenISA.threadgroupbarrier() #1

; Function Attrs: convergent
declare spir_func void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) #1

; Function Attrs: nounwind
declare void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32, <8 x i32>) #2

; Function Attrs: convergent
declare spir_func void @_Z7barrierj(i32) local_unnamed_addr #1

; Function Attrs: nounwind
declare <64 x i16> @llvm.genx.GenISA.LSC2DBlockRead.v64i16(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) #2

; Function Attrs: nounwind
declare <32 x i32> @llvm.genx.GenISA.LSC2DBlockRead.v32i32(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) #2

; Function Attrs: convergent nounwind
declare <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float>, <8 x i16>, <8 x i32>, i32, i32, i32, i32, i1) #3

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent }
attributes #2 = { nounwind }
attributes #3 = { convergent nounwind }

!opencl.spir.version = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!spirv.Source = !{!1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1}
!opencl.compiler.options = !{!2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2}
!llvm.ident = !{!3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}
!llvm.module.flags = !{!4, !5}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{}
!3 = !{!"Intel(R) oneAPI DPC++/C++ Compiler 2023.2.0 (2023.2.0.20230622)"}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{i64 16}
!7 = !{i64 512, i64 1, i64 1}



















;;;; ModuleID = 'LLVMDialectModule'
;;;source_filename = "LLVMDialectModule"
;;;target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
;;;target triple = "spir64-unknown-unknown"
;;;
;;;define spir_kernel void @matmul_kernel_with_block_pointers_0d1d2d3d4d5d(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5) local_unnamed_addr !intel_reqd_sub_group_size !6 !max_work_group_size !7 {
;;;  %7 = tail call i64 @_Z12get_local_idj(i32 0)
;;;  %8 = trunc i64 %7 to i32
;;;  %9 = tail call i64 @_Z12get_local_idj(i32 1)
;;;  %10 = trunc i64 %9 to i32
;;;  %11 = tail call i64 @_Z12get_local_idj(i32 2)
;;;  %12 = trunc i64 %11 to i32
;;;  %13 = tail call i64 @_Z14get_local_sizej(i32 0)
;;;  %14 = trunc i64 %13 to i32
;;;  %15 = tail call i64 @_Z14get_local_sizej(i32 1)
;;;  %16 = trunc i64 %15 to i32
;;;  %17 = mul i32 %16, %12
;;;  %18 = add i32 %17, %10
;;;  %19 = mul i32 %18, %14
;;;  %20 = add i32 %19, %8
;;;  %21 = lshr i32 %20, 4
;;;  %22 = tail call i64 @_Z12get_group_idj(i32 0)
;;;  %23 = trunc i64 %22 to i32
;;;  %24 = sdiv i32 %23, 64
;;;  %25 = shl nsw i32 %24, 2
;;;  %26 = sub nsw i32 16, %25
;;;  %27 = tail call i32 @llvm.smin.i32(i32 %26, i32 4)
;;;  %28 = srem i32 %23, %27
;;;  %29 = add nsw i32 %25, %28
;;;  %30 = and i32 %23, 63
;;;  %31 = sdiv i32 %30, %27
;;;  %32 = shl i32 %29, 8
;;;  %33 = shl nuw nsw i32 %21, 3
;;;  %34 = add i32 %33, %32
;;;  %35 = ptrtoint ptr addrspace(1) %0 to i64
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %35, i32 8191, i32 4095, i32 8191, i32 0, i32 %34, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %35, i32 8191, i32 4095, i32 8191, i32 16, i32 %34, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %35, i32 8191, i32 4095, i32 8191, i32 32, i32 %34, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %35, i32 8191, i32 4095, i32 8191, i32 48, i32 %34, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %35, i32 8191, i32 4095, i32 8191, i32 64, i32 %34, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %35, i32 8191, i32 4095, i32 8191, i32 80, i32 %34, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
;;;  %36 = insertelement <2 x i32> <i32 96, i32 poison>, i32 %34, i64 1
;;;  %37 = insertelement <2 x i32> <i32 112, i32 poison>, i32 %34, i64 1
;;;  %38 = lshr i32 %20, 1
;;;  %39 = and i32 %38, 224
;;;  %40 = or disjoint i32 %39, %32
;;;  %41 = insertelement <2 x i32> <i32 0, i32 poison>, i32 %40, i64 1
;;;  %42 = shl nsw i32 %31, 8
;;;  %43 = and i32 %21, 24
;;;  %44 = shl i32 %21, 5
;;;  %45 = and i32 %44, 224
;;;  %46 = or disjoint i32 %45, %42
;;;  %47 = insertelement <2 x i32> poison, i32 %46, i64 0
;;;  %48 = or disjoint i32 %46, 16
;;;  %49 = insertelement <2 x i32> poison, i32 %48, i64 0
;;;  %50 = ptrtoint ptr addrspace(1) %1 to i64
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %50, i32 8191, i32 4095, i32 8191, i32 %46, i32 %43, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %50, i32 8191, i32 4095, i32 8191, i32 %48, i32 %43, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
;;;  %51 = or disjoint i32 %43, 32
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %50, i32 8191, i32 4095, i32 8191, i32 %46, i32 %51, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %50, i32 8191, i32 4095, i32 8191, i32 %48, i32 %51, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
;;;  %52 = or disjoint i32 %43, 64
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %50, i32 8191, i32 4095, i32 8191, i32 %46, i32 %52, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %50, i32 8191, i32 4095, i32 8191, i32 %48, i32 %52, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
;;;  %53 = or disjoint i32 %43, 96
;;;  %54 = insertelement <2 x i32> %47, i32 %53, i64 1
;;;  %55 = insertelement <2 x i32> %49, i32 %53, i64 1
;;;  %56 = shl i32 %21, 6
;;;  %57 = and i32 %56, 192
;;;  %58 = or disjoint i32 %57, %42
;;;  %59 = insertelement <2 x i32> <i32 poison, i32 0>, i32 %58, i64 0
;;;  %60 = or disjoint i32 %58, 32
;;;  %61 = insertelement <2 x i32> <i32 poison, i32 0>, i32 %60, i64 0
;;;  br label %62
;;;
;;;62:                                               ; preds = %6, %62
;;;  %63 = phi <2 x i32> [ %55, %6 ], [ %161, %62 ]
;;;  %64 = phi <2 x i32> [ %54, %6 ], [ %159, %62 ]
;;;  %65 = phi <2 x i32> [ %37, %6 ], [ %151, %62 ]
;;;  %66 = phi <2 x i32> [ %36, %6 ], [ %149, %62 ]
;;;  %67 = phi <2 x i32> [ %61, %6 ], [ %165, %62 ]
;;;  %68 = phi <2 x i32> [ %59, %6 ], [ %163, %62 ]
;;;  %69 = phi <2 x i32> [ %41, %6 ], [ %153, %62 ]
;;;  %70 = phi <8 x float> [ zeroinitializer, %6 ], [ %143, %62 ]
;;;  %71 = phi <8 x float> [ zeroinitializer, %6 ], [ %141, %62 ]
;;;  %72 = phi <8 x float> [ zeroinitializer, %6 ], [ %139, %62 ]
;;;  %73 = phi <8 x float> [ zeroinitializer, %6 ], [ %137, %62 ]
;;;  %74 = phi <8 x float> [ zeroinitializer, %6 ], [ %133, %62 ]
;;;  %75 = phi <8 x float> [ zeroinitializer, %6 ], [ %131, %62 ]
;;;  %76 = phi <8 x float> [ zeroinitializer, %6 ], [ %129, %62 ]
;;;  %77 = phi <8 x float> [ zeroinitializer, %6 ], [ %127, %62 ]
;;;  %78 = phi <8 x float> [ zeroinitializer, %6 ], [ %123, %62 ]
;;;  %79 = phi <8 x float> [ zeroinitializer, %6 ], [ %121, %62 ]
;;;  %80 = phi <8 x float> [ zeroinitializer, %6 ], [ %119, %62 ]
;;;  %81 = phi <8 x float> [ zeroinitializer, %6 ], [ %117, %62 ]
;;;  %82 = phi <8 x float> [ zeroinitializer, %6 ], [ %113, %62 ]
;;;  %83 = phi <8 x float> [ zeroinitializer, %6 ], [ %109, %62 ]
;;;  %84 = phi <8 x float> [ zeroinitializer, %6 ], [ %105, %62 ]
;;;  %85 = phi <8 x float> [ zeroinitializer, %6 ], [ %101, %62 ]
;;;  %86 = phi i32 [ 0, %6 ], [ %166, %62 ]
;;;  ;tail call void @_Z7barrierj(i32 1) #1
;;;  tail call void @llvm.genx.GenISA.threadgroupbarrier()
;;;  %87 = extractelement <2 x i32> %69, i64 0
;;;  %88 = extractelement <2 x i32> %69, i64 1
;;;  %89 = tail call <64 x i16> @llvm.genx.GenISA.LSC2DBlockRead.v64i16(i64 %35, i32 8191, i32 4095, i32 8191, i32 %87, i32 %88, i32 16, i32 16, i32 32, i32 2, i1 false, i1 false, i32 0)
;;;  %90 = extractelement <2 x i32> %68, i64 0
;;;  %91 = extractelement <2 x i32> %68, i64 1
;;;  %92 = tail call <32 x i32> @llvm.genx.GenISA.LSC2DBlockRead.v32i32(i64 %50, i32 8191, i32 4095, i32 8191, i32 %90, i32 %91, i32 16, i32 16, i32 32, i32 2, i1 false, i1 true, i32 0)
;;;  %93 = extractelement <2 x i32> %67, i64 0
;;;  %94 = extractelement <2 x i32> %67, i64 1
;;;  %95 = tail call <32 x i32> @llvm.genx.GenISA.LSC2DBlockRead.v32i32(i64 %50, i32 8191, i32 4095, i32 8191, i32 %93, i32 %94, i32 16, i32 16, i32 32, i32 2, i1 false, i1 true, i32 0)
;;;  %96 = shufflevector <64 x i16> %89, <64 x i16> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
;;;  %97 = shufflevector <32 x i32> %92, <32 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
;;;  %98 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %85, <8 x i16> %96, <8 x i32> %97, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %99 = shufflevector <64 x i16> %89, <64 x i16> poison, <8 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39>
;;;  %100 = shufflevector <32 x i32> %92, <32 x i32> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
;;;  %101 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %98, <8 x i16> %99, <8 x i32> %100, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %102 = shufflevector <64 x i16> %89, <64 x i16> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
;;;  %103 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %84, <8 x i16> %102, <8 x i32> %97, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %104 = shufflevector <64 x i16> %89, <64 x i16> poison, <8 x i32> <i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47>
;;;  %105 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %103, <8 x i16> %104, <8 x i32> %100, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %106 = shufflevector <64 x i16> %89, <64 x i16> poison, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
;;;  %107 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %83, <8 x i16> %106, <8 x i32> %97, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %108 = shufflevector <64 x i16> %89, <64 x i16> poison, <8 x i32> <i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55>
;;;  %109 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %107, <8 x i16> %108, <8 x i32> %100, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %110 = shufflevector <64 x i16> %89, <64 x i16> poison, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
;;;  %111 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %82, <8 x i16> %110, <8 x i32> %97, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %112 = shufflevector <64 x i16> %89, <64 x i16> poison, <8 x i32> <i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
;;;  %113 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %111, <8 x i16> %112, <8 x i32> %100, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %114 = shufflevector <32 x i32> %92, <32 x i32> poison, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
;;;  %115 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %81, <8 x i16> %96, <8 x i32> %114, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %116 = shufflevector <32 x i32> %92, <32 x i32> poison, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
;;;  %117 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %115, <8 x i16> %99, <8 x i32> %116, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %118 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %80, <8 x i16> %102, <8 x i32> %114, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %119 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %118, <8 x i16> %104, <8 x i32> %116, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %120 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %79, <8 x i16> %106, <8 x i32> %114, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %121 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %120, <8 x i16> %108, <8 x i32> %116, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %122 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %78, <8 x i16> %110, <8 x i32> %114, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %123 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %122, <8 x i16> %112, <8 x i32> %116, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %124 = shufflevector <32 x i32> %95, <32 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
;;;  %125 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %77, <8 x i16> %96, <8 x i32> %124, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %126 = shufflevector <32 x i32> %95, <32 x i32> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
;;;  %127 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %125, <8 x i16> %99, <8 x i32> %126, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %128 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %76, <8 x i16> %102, <8 x i32> %124, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %129 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %128, <8 x i16> %104, <8 x i32> %126, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %130 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %75, <8 x i16> %106, <8 x i32> %124, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %131 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %130, <8 x i16> %108, <8 x i32> %126, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %132 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %74, <8 x i16> %110, <8 x i32> %124, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %133 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %132, <8 x i16> %112, <8 x i32> %126, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %134 = shufflevector <32 x i32> %95, <32 x i32> poison, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
;;;  %135 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %73, <8 x i16> %96, <8 x i32> %134, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %136 = shufflevector <32 x i32> %95, <32 x i32> poison, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
;;;  %137 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %135, <8 x i16> %99, <8 x i32> %136, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %138 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %72, <8 x i16> %102, <8 x i32> %134, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %139 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %138, <8 x i16> %104, <8 x i32> %136, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %140 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %71, <8 x i16> %106, <8 x i32> %134, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %141 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %140, <8 x i16> %108, <8 x i32> %136, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %142 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %70, <8 x i16> %110, <8 x i32> %134, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %143 = tail call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %142, <8 x i16> %112, <8 x i32> %136, i32 12, i32 12, i32 8, i32 8, i1 false)
;;;  %144 = extractelement <2 x i32> %66, i64 0
;;;  %145 = extractelement <2 x i32> %66, i64 1
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %35, i32 8191, i32 4095, i32 8191, i32 %144, i32 %145, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
;;;  %146 = extractelement <2 x i32> %65, i64 0
;;;  %147 = extractelement <2 x i32> %65, i64 1
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %35, i32 8191, i32 4095, i32 8191, i32 %146, i32 %147, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
;;;  %148 = add i32 %144, 32
;;;  %149 = insertelement <2 x i32> %66, i32 %148, i64 0
;;;  %150 = add i32 %146, 32
;;;  %151 = insertelement <2 x i32> %65, i32 %150, i64 0
;;;  %152 = add i32 %87, 32
;;;  %153 = insertelement <2 x i32> %69, i32 %152, i64 0
;;;  %154 = extractelement <2 x i32> %64, i64 0
;;;  %155 = extractelement <2 x i32> %64, i64 1
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %50, i32 8191, i32 4095, i32 8191, i32 %154, i32 %155, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
;;;  %156 = extractelement <2 x i32> %63, i64 0
;;;  %157 = extractelement <2 x i32> %63, i64 1
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %50, i32 8191, i32 4095, i32 8191, i32 %156, i32 %157, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4) #1
;;;  %158 = add i32 %155, 32
;;;  %159 = insertelement <2 x i32> %64, i32 %158, i64 1
;;;  %160 = add i32 %157, 32
;;;  %161 = insertelement <2 x i32> %63, i32 %160, i64 1
;;;  %162 = add i32 %91, 32
;;;  %163 = insertelement <2 x i32> %68, i32 %162, i64 1
;;;  %164 = add i32 %94, 32
;;;  %165 = insertelement <2 x i32> %67, i32 %164, i64 1
;;;  %166 = add nuw nsw i32 %86, 32
;;;  %167 = icmp ult i32 %86, 4064
;;;  br i1 %167, label %62, label %168
;;;
;;;168:                                              ; preds = %62
;;;  %169 = or disjoint i32 %40, 8
;;;  %170 = or disjoint i32 %40, 16
;;;  %171 = or disjoint i32 %40, 24
;;;  %172 = or disjoint i32 %58, 16
;;;  %173 = or disjoint i32 %58, 48
;;;  %174 = bitcast <8 x float> %101 to <8 x i32>
;;;  %175 = ptrtoint ptr addrspace(1) %2 to i64
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %175, i32 16383, i32 4095, i32 16383, i32 %58, i32 %40, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %174)
;;;  %176 = bitcast <8 x float> %105 to <8 x i32>
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %175, i32 16383, i32 4095, i32 16383, i32 %58, i32 %169, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %176)
;;;  %177 = bitcast <8 x float> %109 to <8 x i32>
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %175, i32 16383, i32 4095, i32 16383, i32 %58, i32 %170, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %177)
;;;  %178 = bitcast <8 x float> %113 to <8 x i32>
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %175, i32 16383, i32 4095, i32 16383, i32 %58, i32 %171, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %178)
;;;  %179 = bitcast <8 x float> %117 to <8 x i32>
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %175, i32 16383, i32 4095, i32 16383, i32 %172, i32 %40, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %179)
;;;  %180 = bitcast <8 x float> %119 to <8 x i32>
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %175, i32 16383, i32 4095, i32 16383, i32 %172, i32 %169, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %180)
;;;  %181 = bitcast <8 x float> %121 to <8 x i32>
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %175, i32 16383, i32 4095, i32 16383, i32 %172, i32 %170, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %181)
;;;  %182 = bitcast <8 x float> %123 to <8 x i32>
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %175, i32 16383, i32 4095, i32 16383, i32 %172, i32 %171, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %182)
;;;  %183 = bitcast <8 x float> %127 to <8 x i32>
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %175, i32 16383, i32 4095, i32 16383, i32 %60, i32 %40, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %183)
;;;  %184 = bitcast <8 x float> %129 to <8 x i32>
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %175, i32 16383, i32 4095, i32 16383, i32 %60, i32 %169, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %184)
;;;  %185 = bitcast <8 x float> %131 to <8 x i32>
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %175, i32 16383, i32 4095, i32 16383, i32 %60, i32 %170, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %185)
;;;  %186 = bitcast <8 x float> %133 to <8 x i32>
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %175, i32 16383, i32 4095, i32 16383, i32 %60, i32 %171, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %186)
;;;  %187 = bitcast <8 x float> %137 to <8 x i32>
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %175, i32 16383, i32 4095, i32 16383, i32 %173, i32 %40, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %187)
;;;  %188 = bitcast <8 x float> %139 to <8 x i32>
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %175, i32 16383, i32 4095, i32 16383, i32 %173, i32 %169, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %188)
;;;  %189 = bitcast <8 x float> %141 to <8 x i32>
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %175, i32 16383, i32 4095, i32 16383, i32 %173, i32 %170, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %189)
;;;  %190 = bitcast <8 x float> %143 to <8 x i32>
;;;  tail call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %175, i32 16383, i32 4095, i32 16383, i32 %173, i32 %171, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %190)
;;;  ret void
;;;}
;;;
;;;declare spir_func i64 @_Z12get_local_idj(i32) local_unnamed_addr
;;;
;;;declare spir_func i64 @_Z14get_local_sizej(i32) local_unnamed_addr
;;;
;;;declare spir_func i64 @_Z12get_group_idj(i32) local_unnamed_addr
;;;
;;;; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
;;;declare i32 @llvm.smin.i32(i32, i32) #0
;;;
;;;declare void @llvm.genx.GenISA.threadgroupbarrier() #1
;;;
;;;; Function Attrs: convergent
;;;declare spir_func void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) #1
;;;
;;;; Function Attrs: nounwind
;;;declare void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32, <8 x i32>) #2
;;;
;;;; Function Attrs: convergent
;;;declare spir_func void @_Z7barrierj(i32) local_unnamed_addr #1
;;;
;;;; Function Attrs: nounwind
;;;declare <64 x i16> @llvm.genx.GenISA.LSC2DBlockRead.v64i16(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) #2
;;;
;;;; Function Attrs: nounwind
;;;declare <32 x i32> @llvm.genx.GenISA.LSC2DBlockRead.v32i32(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) #2
;;;
;;;; Function Attrs: convergent nounwind
;;;declare <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float>, <8 x i16>, <8 x i32>, i32, i32, i32, i32, i1) #3
;;;
;;;attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
;;;attributes #1 = { convergent }
;;;attributes #2 = { nounwind }
;;;attributes #3 = { convergent nounwind }
;;;
;;;!opencl.spir.version = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
;;;!spirv.Source = !{!1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1}
;;;!opencl.compiler.options = !{!2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2}
;;;!llvm.ident = !{!3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}
;;;!llvm.module.flags = !{!4, !5}
;;;
;;;!0 = !{i32 1, i32 2}
;;;!1 = !{i32 4, i32 100000}
;;;!2 = !{}
;;;!3 = !{!"Intel(R) oneAPI DPC++/C++ Compiler 2023.2.0 (2023.2.0.20230622)"}
;;;!4 = !{i32 1, !"wchar_size", i32 4}
;;;!5 = !{i32 7, !"frame-pointer", i32 2}
;;;!6 = !{i64 16}
;;;!7 = !{i64 512, i64 1, i64 1}
