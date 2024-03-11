; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"
;target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"

; Function Desc: LSC 2d block prefetch
; Output: nothing is returned
; Arg 0: flat image base offset
; Arg 1: flat image base width
; Arg 2: flat image base height
; Arg 3: flat image base pitch
; Arg 4: offset x
; Arg 5: offset y
; Arg 6: elemSize
; Arg 7: tile width
; Arg 8: tile height
; Arg 9: V - num blocks (2 for simple 2d block read)
; Arg 10: transpose
; Arg 11: vnni transform (for transpose+transform use transpose only and elemSize 32)
; Arg 12: cache controls options (LSC_CACHE_OPTS)
declare void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32)

; Function Desc: LSC 2d block read
; Output: 
; Arg 0: flat image base offset
; Arg 1: flat image base width
; Arg 2: flat image base height
; Arg 3: flat image base pitch
; Arg 4: offset x
; Arg 5: offset y
; Arg 6: elemSize
; Arg 7: tile width
; Arg 8: tile height
; Arg 9: V - num blocks (2 for simple 2d block read)
; Arg 10: transpose
; Arg 11: vnni transform (for transpose+transform use transpose only and elemSize 32)
; Arg 12: cache controls options (LSC_CACHE_OPTS)
declare <64 x i16> @llvm.genx.GenISA.LSC2DBlockRead.v64i16(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32)

; Function Desc: LSC 2d block read
; Output: 
; Arg 0: flat image base offset
; Arg 1: flat image base width
; Arg 2: flat image base height
; Arg 3: flat image base pitch
; Arg 4: offset x
; Arg 5: offset y
; Arg 6: elemSize
; Arg 7: tile width
; Arg 8: tile height
; Arg 9: V - num blocks (2 for simple 2d block read)
; Arg 10: transpose
; Arg 11: vnni transform (for transpose+transform use transpose only and elemSize 32)
; Arg 12: cache controls options (LSC_CACHE_OPTS)
declare <32 x i32> @llvm.genx.GenISA.LSC2DBlockRead.v32i32(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32)

; Function Desc: XeHP SDV: dot product accumulate systolic
; Output: dst
; Arg 0: src0(acc)
; Arg 1: src1
; Arg 2: src2
; Arg 3: src1's precision
; Arg 4: src2's precision
; Arg 5: systolic depth
; Arg 6: repeat count
; Arg 7: isDpasw
declare <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float>, <8 x i16>, <8 x i32>, i32, i32, i32, i32, i1)

; Function Desc: LSC 2d block write
; Output: nothing is returned
; Arg 0: flat image base offset
; Arg 1: flat image base width
; Arg 2: flat image base height
; Arg 3: flat image base pitch
; Arg 4: offset x
; Arg 5: offset y
; Arg 6: elemSize
; Arg 7: tile width
; Arg 8: tile height
; Arg 9: V - num blocks (2 for simple 2d block read)
; Arg 10: transpose
; Arg 11: vnni transform (for transpose+transform use transpose only and elemSize 32)
; Arg 12: cache controls options (LSC_CACHE_OPTS)
; Arg 13: stored value
declare void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32, <8 x i32>)

; Function Attrs: nounwind
define spir_kernel void @matmul_kernel_with_block_pointers_0d1d2d3d4d5d(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, i32 %x, i32 %y, i32 %z) local_unnamed_addr !intel_reqd_sub_group_size !297 {
;define spir_kernel void @test_kernel([16777216 x half] addrspace(1)* %0, [16777216 x half] addrspace(1)* %1, [16777216 x float] addrspace(1)* %2) #0 !kernel_arg_addr_space !293 !kernel_arg_access_qual !294 !kernel_arg_type !295 !kernel_arg_type_qual !296 !kernel_arg_base_type !295 !kernel_arg_name !296 !intel_reqd_sub_group_size !297 {
  %4 = alloca <2 x i32>, align 8
  %5 = alloca <2 x i32>, align 8
  %6 = alloca <2 x i32>, align 8
  %7 = alloca <8 x float>, align 32
  %8 = alloca <8 x float>, align 32
  %9 = alloca <8 x float>, align 32
  %10 = alloca <8 x float>, align 32
  %11 = alloca <8 x float>, align 32
  %12 = alloca <8 x float>, align 32
  %13 = alloca <8 x float>, align 32
  %14 = alloca <8 x float>, align 32
  %15 = alloca <8 x float>, align 32
  %16 = alloca <8 x float>, align 32
  %17 = alloca <8 x float>, align 32
  %18 = alloca <8 x float>, align 32
  %19 = alloca <8 x float>, align 32
  %20 = alloca <8 x float>, align 32
  %21 = alloca <8 x float>, align 32
  %22 = alloca <8 x float>, align 32
  %23 = alloca <8 x float>, align 32
  %24 = alloca <8 x float>, align 32
  %25 = call spir_func i32 @_Z25__spirv_BuiltInSubgroupIdv() #2
  %26 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 0) #2
  %27 = insertelement <3 x i64> undef, i64 %26, i32 0
  %28 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 1) #2
  %29 = insertelement <3 x i64> %27, i64 %28, i32 1
  %30 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 2) #2
  %31 = insertelement <3 x i64> %29, i64 %30, i32 2
  %32 = extractelement <3 x i64> %31, i32 0
  %33 = trunc i64 %32 to i32
  %34 = sdiv i32 %33, 64
  %35 = shl nsw i32 %34, 2, !spirv.Decorations !298
  %36 = sub nsw i32 16, %35, !spirv.Decorations !298
  %37 = icmp slt i32 %36, 4
  %38 = select i1 %37, i32 %36, i32 4
  %39 = srem i32 %33, %38
  %40 = add nsw i32 %35, %39, !spirv.Decorations !298
  %41 = and i32 %33, 63
  %42 = sdiv i32 %41, %38
  %43 = shl i32 %40, 8
  %44 = udiv i32 %33, 16
  %45 = mul i32 %44, 256
  %46 = urem i32 %33, 16
  %47 = mul i32 %46, 256
  %48 = udiv i32 %25, 4
  %49 = mul i32 %48, 32
  %50 = add i32 %45, %49
  %51 = urem i32 %25, 4
  %52 = mul i32 %51, 64
  %53 = add i32 %47, %52
  %54 = add i32 %50, 0
  %55 = insertelement <2 x i32> zeroinitializer, i32 %50, i32 1
  %56 = add i32 %53, 0
  %57 = insertelement <2 x i32> zeroinitializer, i32 %53, i32 0
  %58 = add i32 %53, 32
  %59 = insertelement <2 x i32> zeroinitializer, i32 %58, i32 0
  %60 = ptrtoint ptr addrspace(1) %0 to i64
  %61 = ptrtoint ptr addrspace(1) %1 to i64
  %62 = udiv i32 %33, 16
  %63 = mul i32 %62, 256
  %64 = mul i32 %25, 8
  %65 = add i32 %64, %63
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %60, i32 8191, i32 4095, i32 8191, i32 0, i32 %65, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4)
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %60, i32 8191, i32 4095, i32 8191, i32 16, i32 %65, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4)
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %60, i32 8191, i32 4095, i32 8191, i32 32, i32 %65, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4)
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %60, i32 8191, i32 4095, i32 8191, i32 48, i32 %65, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4)
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %60, i32 8191, i32 4095, i32 8191, i32 64, i32 %65, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4)
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %60, i32 8191, i32 4095, i32 8191, i32 80, i32 %65, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4)
  %66 = add i32 %65, 0
  %67 = add i32 %65, 0
  %68 = udiv i32 %25, 8
  %69 = and i32 %68, 3
  %70 = mul i32 %69, 8
  %71 = and i32 %33, 15
  %72 = mul i32 %71, 256
  %73 = and i32 %25, 7
  %74 = mul i32 %73, 32
  %75 = add i32 %74, %72
  %76 = add i32 %75, 16
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %61, i32 8191, i32 4095, i32 8191, i32 %75, i32 %70, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4)
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %61, i32 8191, i32 4095, i32 8191, i32 %76, i32 %70, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4)
  %77 = add i32 %70, 32
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %61, i32 8191, i32 4095, i32 8191, i32 %75, i32 %77, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4)
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %61, i32 8191, i32 4095, i32 8191, i32 %76, i32 %77, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4)
  %78 = add i32 %77, 32
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %61, i32 8191, i32 4095, i32 8191, i32 %75, i32 %78, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4)
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %61, i32 8191, i32 4095, i32 8191, i32 %76, i32 %78, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4)
  %79 = add i32 %75, 0
  %80 = add i32 %76, 0
  %81 = add i32 %78, 32
  %82 = add i32 %78, 32
  br label %83

83:                                               ; preds = %118, %3
  %84 = phi i32 [ %190, %118 ], [ 0, %3 ]
  %85 = phi <2 x i32> [ %181, %118 ], [ %59, %3 ]
  %86 = phi <2 x i32> [ %179, %118 ], [ %57, %3 ]
  %87 = phi <2 x i32> [ %177, %118 ], [ %55, %3 ]
  %88 = phi <8 x float> [ %175, %118 ], [ zeroinitializer, %3 ]
  %89 = phi <8 x float> [ %173, %118 ], [ zeroinitializer, %3 ]
  %90 = phi <8 x float> [ %171, %118 ], [ zeroinitializer, %3 ]
  %91 = phi <8 x float> [ %169, %118 ], [ zeroinitializer, %3 ]
  %92 = phi <8 x float> [ %165, %118 ], [ zeroinitializer, %3 ]
  %93 = phi <8 x float> [ %163, %118 ], [ zeroinitializer, %3 ]
  %94 = phi <8 x float> [ %161, %118 ], [ zeroinitializer, %3 ]
  %95 = phi <8 x float> [ %159, %118 ], [ zeroinitializer, %3 ]
  %96 = phi <8 x float> [ %155, %118 ], [ zeroinitializer, %3 ]
  %97 = phi <8 x float> [ %153, %118 ], [ zeroinitializer, %3 ]
  %98 = phi <8 x float> [ %151, %118 ], [ zeroinitializer, %3 ]
  %99 = phi <8 x float> [ %149, %118 ], [ zeroinitializer, %3 ]
  %100 = phi <8 x float> [ %145, %118 ], [ zeroinitializer, %3 ]
  %101 = phi <8 x float> [ %141, %118 ], [ zeroinitializer, %3 ]
  %102 = phi <8 x float> [ %137, %118 ], [ zeroinitializer, %3 ]
  %103 = phi <8 x float> [ %133, %118 ], [ zeroinitializer, %3 ]
  %104 = phi i32 [ %182, %118 ], [ 96, %3 ]
  %105 = phi i32 [ %183, %118 ], [ 112, %3 ]
  %106 = phi i32 [ %184, %118 ], [ %66, %3 ]
  %107 = phi i32 [ %185, %118 ], [ %67, %3 ]
  %108 = phi i32 [ %186, %118 ], [ %79, %3 ]
  %109 = phi i32 [ %187, %118 ], [ %80, %3 ]
  %110 = phi i32 [ %188, %118 ], [ %81, %3 ]
  %111 = phi i32 [ %189, %118 ], [ %82, %3 ]
  %112 = icmp slt i32 %84, 4096
  br i1 %112, label %113, label %191

113:                                              ; preds = %83
  %114 = and i32 %84, 255
  %115 = icmp eq i32 %114, 0
  br label %116

116:                                              ; preds = %113
  br i1 %115, label %117, label %118

117:                                              ; preds = %116
  call spir_func void @_Z22__spirv_ControlBarrieriii(i32 2, i32 2, i32 264) #1
  br label %118

118:                                              ; preds = %117, %116
  %119 = extractelement <2 x i32> %87, i32 0
  %120 = extractelement <2 x i32> %87, i32 1
  %121 = call <64 x i16> @llvm.genx.GenISA.LSC2DBlockRead.v64i16(i64 %60, i32 8191, i32 4095, i32 8191, i32 %119, i32 %120, i32 16, i32 16, i32 32, i32 2, i1 false, i1 false, i32 0)
  %122 = extractelement <2 x i32> %86, i32 0
  %123 = extractelement <2 x i32> %86, i32 1
  %124 = call <32 x i32> @llvm.genx.GenISA.LSC2DBlockRead.v32i32(i64 %61, i32 8191, i32 4095, i32 8191, i32 %122, i32 %123, i32 16, i32 16, i32 32, i32 2, i1 false, i1 true, i32 0)
  %125 = extractelement <2 x i32> %85, i32 0
  %126 = extractelement <2 x i32> %85, i32 1
  %127 = call <32 x i32> @llvm.genx.GenISA.LSC2DBlockRead.v32i32(i64 %61, i32 8191, i32 4095, i32 8191, i32 %125, i32 %126, i32 16, i32 16, i32 32, i32 2, i1 false, i1 true, i32 0)
  %128 = shufflevector <64 x i16> %121, <64 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %129 = shufflevector <32 x i32> %124, <32 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %130 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %103, <8 x i16> %128, <8 x i32> %129, i32 12, i32 12, i32 8, i32 8, i1 false)
  %131 = shufflevector <64 x i16> %121, <64 x i16> undef, <8 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39>
  %132 = shufflevector <32 x i32> %124, <32 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %133 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %130, <8 x i16> %131, <8 x i32> %132, i32 12, i32 12, i32 8, i32 8, i1 false)
  %134 = shufflevector <64 x i16> %121, <64 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %135 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %102, <8 x i16> %134, <8 x i32> %129, i32 12, i32 12, i32 8, i32 8, i1 false)
  %136 = shufflevector <64 x i16> %121, <64 x i16> undef, <8 x i32> <i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47>
  %137 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %135, <8 x i16> %136, <8 x i32> %132, i32 12, i32 12, i32 8, i32 8, i1 false)
  %138 = shufflevector <64 x i16> %121, <64 x i16> undef, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  %139 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %101, <8 x i16> %138, <8 x i32> %129, i32 12, i32 12, i32 8, i32 8, i1 false)
  %140 = shufflevector <64 x i16> %121, <64 x i16> undef, <8 x i32> <i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55>
  %141 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %139, <8 x i16> %140, <8 x i32> %132, i32 12, i32 12, i32 8, i32 8, i1 false)
  %142 = shufflevector <64 x i16> %121, <64 x i16> undef, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %143 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %100, <8 x i16> %142, <8 x i32> %129, i32 12, i32 12, i32 8, i32 8, i1 false)
  %144 = shufflevector <64 x i16> %121, <64 x i16> undef, <8 x i32> <i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %145 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %143, <8 x i16> %144, <8 x i32> %132, i32 12, i32 12, i32 8, i32 8, i1 false)
  %146 = shufflevector <32 x i32> %124, <32 x i32> undef, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  %147 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %99, <8 x i16> %128, <8 x i32> %146, i32 12, i32 12, i32 8, i32 8, i1 false)
  %148 = shufflevector <32 x i32> %124, <32 x i32> undef, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %149 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %147, <8 x i16> %131, <8 x i32> %148, i32 12, i32 12, i32 8, i32 8, i1 false)
  %150 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %98, <8 x i16> %134, <8 x i32> %146, i32 12, i32 12, i32 8, i32 8, i1 false)
  %151 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %150, <8 x i16> %136, <8 x i32> %148, i32 12, i32 12, i32 8, i32 8, i1 false)
  %152 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %97, <8 x i16> %138, <8 x i32> %146, i32 12, i32 12, i32 8, i32 8, i1 false)
  %153 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %152, <8 x i16> %140, <8 x i32> %148, i32 12, i32 12, i32 8, i32 8, i1 false)
  %154 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %96, <8 x i16> %142, <8 x i32> %146, i32 12, i32 12, i32 8, i32 8, i1 false)
  %155 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %154, <8 x i16> %144, <8 x i32> %148, i32 12, i32 12, i32 8, i32 8, i1 false)
  %156 = shufflevector <32 x i32> %127, <32 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %157 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %95, <8 x i16> %128, <8 x i32> %156, i32 12, i32 12, i32 8, i32 8, i1 false)
  %158 = shufflevector <32 x i32> %127, <32 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %159 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %157, <8 x i16> %131, <8 x i32> %158, i32 12, i32 12, i32 8, i32 8, i1 false)
  %160 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %94, <8 x i16> %134, <8 x i32> %156, i32 12, i32 12, i32 8, i32 8, i1 false)
  %161 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %160, <8 x i16> %136, <8 x i32> %158, i32 12, i32 12, i32 8, i32 8, i1 false)
  %162 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %93, <8 x i16> %138, <8 x i32> %156, i32 12, i32 12, i32 8, i32 8, i1 false)
  %163 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %162, <8 x i16> %140, <8 x i32> %158, i32 12, i32 12, i32 8, i32 8, i1 false)
  %164 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %92, <8 x i16> %142, <8 x i32> %156, i32 12, i32 12, i32 8, i32 8, i1 false)
  %165 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %164, <8 x i16> %144, <8 x i32> %158, i32 12, i32 12, i32 8, i32 8, i1 false)
  %166 = shufflevector <32 x i32> %127, <32 x i32> undef, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  %167 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %91, <8 x i16> %128, <8 x i32> %166, i32 12, i32 12, i32 8, i32 8, i1 false)
  %168 = shufflevector <32 x i32> %127, <32 x i32> undef, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %169 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %167, <8 x i16> %131, <8 x i32> %168, i32 12, i32 12, i32 8, i32 8, i1 false)
  %170 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %90, <8 x i16> %134, <8 x i32> %166, i32 12, i32 12, i32 8, i32 8, i1 false)
  %171 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %170, <8 x i16> %136, <8 x i32> %168, i32 12, i32 12, i32 8, i32 8, i1 false)
  %172 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %89, <8 x i16> %138, <8 x i32> %166, i32 12, i32 12, i32 8, i32 8, i1 false)
  %173 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %172, <8 x i16> %140, <8 x i32> %168, i32 12, i32 12, i32 8, i32 8, i1 false)
  %174 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %88, <8 x i16> %142, <8 x i32> %166, i32 12, i32 12, i32 8, i32 8, i1 false)
  %175 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %174, <8 x i16> %144, <8 x i32> %168, i32 12, i32 12, i32 8, i32 8, i1 false)
  %176 = add i32 %119, 32
  %177 = insertelement <2 x i32> %87, i32 %176, i32 0
  %178 = add i32 %123, 32
  %179 = insertelement <2 x i32> %86, i32 %178, i32 1
  %180 = add i32 %126, 32
  %181 = insertelement <2 x i32> %85, i32 %180, i32 1
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %60, i32 8191, i32 4095, i32 8191, i32 %104, i32 %106, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4)
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %60, i32 8191, i32 4095, i32 8191, i32 %105, i32 %107, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4)
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %61, i32 8191, i32 4095, i32 8191, i32 %108, i32 %110, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4)
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %61, i32 8191, i32 4095, i32 8191, i32 %109, i32 %111, i32 16, i32 16, i32 8, i32 1, i1 false, i1 false, i32 4)
  %182 = add i32 %104, 32
  %183 = add i32 %105, 32
  %184 = add i32 %106, 0
  %185 = add i32 %107, 0
  %186 = add i32 %108, 0
  %187 = add i32 %109, 0
  %188 = add i32 %110, 32
  %189 = add i32 %111, 32
  store <2 x i32> %181, <2 x i32>* %4, align 8
  store <2 x i32> %179, <2 x i32>* %5, align 8
  store <2 x i32> %177, <2 x i32>* %6, align 8
  store <8 x float> %175, <8 x float>* %7, align 32
  store <8 x float> %173, <8 x float>* %8, align 32
  store <8 x float> %171, <8 x float>* %9, align 32
  store <8 x float> %169, <8 x float>* %10, align 32
  store <8 x float> %165, <8 x float>* %11, align 32
  store <8 x float> %163, <8 x float>* %12, align 32
  store <8 x float> %161, <8 x float>* %13, align 32
  store <8 x float> %159, <8 x float>* %14, align 32
  store <8 x float> %155, <8 x float>* %15, align 32
  store <8 x float> %153, <8 x float>* %16, align 32
  store <8 x float> %151, <8 x float>* %17, align 32
  store <8 x float> %149, <8 x float>* %18, align 32
  store <8 x float> %145, <8 x float>* %19, align 32
  store <8 x float> %141, <8 x float>* %20, align 32
  store <8 x float> %137, <8 x float>* %21, align 32
  store <8 x float> %133, <8 x float>* %22, align 32
  %190 = add i32 %84, 32
  br label %83, !llvm.loop !300

191:                                              ; preds = %83
  %192 = load <8 x float>, <8 x float>* %7, align 32
  %193 = load <8 x float>, <8 x float>* %8, align 32
  %194 = load <8 x float>, <8 x float>* %9, align 32
  %195 = load <8 x float>, <8 x float>* %10, align 32
  %196 = load <8 x float>, <8 x float>* %11, align 32
  %197 = load <8 x float>, <8 x float>* %12, align 32
  %198 = load <8 x float>, <8 x float>* %13, align 32
  %199 = load <8 x float>, <8 x float>* %14, align 32
  %200 = load <8 x float>, <8 x float>* %15, align 32
  %201 = load <8 x float>, <8 x float>* %16, align 32
  %202 = load <8 x float>, <8 x float>* %17, align 32
  %203 = load <8 x float>, <8 x float>* %18, align 32
  %204 = load <8 x float>, <8 x float>* %19, align 32
  %205 = load <8 x float>, <8 x float>* %20, align 32
  %206 = load <8 x float>, <8 x float>* %21, align 32
  %207 = load <8 x float>, <8 x float>* %22, align 32
  %208 = add i32 %54, 8
  %209 = add i32 %54, 16
  %210 = add i32 %54, 24
  %211 = add i32 %56, 16
  %212 = add i32 %56, 48
  %213 = bitcast <8 x float> %207 to <8 x i32>
  %214 = ptrtoint ptr addrspace(1) %2 to i64
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %214, i32 16383, i32 4095, i32 16383, i32 %56, i32 %54, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %213)
  %215 = bitcast <8 x float> %206 to <8 x i32>
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %214, i32 16383, i32 4095, i32 16383, i32 %56, i32 %208, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %215)
  %216 = bitcast <8 x float> %205 to <8 x i32>
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %214, i32 16383, i32 4095, i32 16383, i32 %56, i32 %209, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %216)
  %217 = bitcast <8 x float> %204 to <8 x i32>
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %214, i32 16383, i32 4095, i32 16383, i32 %56, i32 %210, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %217)
  %218 = bitcast <8 x float> %203 to <8 x i32>
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %214, i32 16383, i32 4095, i32 16383, i32 %211, i32 %54, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %218)
  %219 = bitcast <8 x float> %202 to <8 x i32>
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %214, i32 16383, i32 4095, i32 16383, i32 %211, i32 %208, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %219)
  %220 = bitcast <8 x float> %201 to <8 x i32>
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %214, i32 16383, i32 4095, i32 16383, i32 %211, i32 %209, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %220)
  %221 = bitcast <8 x float> %200 to <8 x i32>
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %214, i32 16383, i32 4095, i32 16383, i32 %211, i32 %210, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %221)
  %222 = bitcast <8 x float> %199 to <8 x i32>
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %214, i32 16383, i32 4095, i32 16383, i32 %58, i32 %54, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %222)
  %223 = bitcast <8 x float> %198 to <8 x i32>
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %214, i32 16383, i32 4095, i32 16383, i32 %58, i32 %208, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %223)
  %224 = bitcast <8 x float> %197 to <8 x i32>
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %214, i32 16383, i32 4095, i32 16383, i32 %58, i32 %209, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %224)
  %225 = bitcast <8 x float> %196 to <8 x i32>
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %214, i32 16383, i32 4095, i32 16383, i32 %58, i32 %210, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %225)
  %226 = bitcast <8 x float> %195 to <8 x i32>
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %214, i32 16383, i32 4095, i32 16383, i32 %212, i32 %54, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %226)
  %227 = bitcast <8 x float> %194 to <8 x i32>
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %214, i32 16383, i32 4095, i32 16383, i32 %212, i32 %208, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %227)
  %228 = bitcast <8 x float> %193 to <8 x i32>
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %214, i32 16383, i32 4095, i32 16383, i32 %212, i32 %209, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %228)
  %229 = bitcast <8 x float> %192 to <8 x i32>
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %214, i32 16383, i32 4095, i32 16383, i32 %212, i32 %210, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %229)
  ret void
}

; Function Attrs: convergent nounwind
declare spir_func void @_Z22__spirv_ControlBarrieriii(i32, i32, i32) #1

; Function Attrs: nounwind readnone willreturn
declare spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32) #2

; Function Attrs: nounwind readnone willreturn
declare spir_func i32 @_Z25__spirv_BuiltInSubgroupIdv() #2

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
attributes #2 = { nounwind readnone willreturn }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!2}
!opencl.used.extensions = !{!3}
!opencl.used.optional.core.features = !{!3}
!spirv.Generator = !{!4}
!opencl.compiler.options = !{!3}
!igc.functions = !{}
!IGCMetadata = !{!5}

!0 = !{i32 2, i32 2}
!1 = !{i32 3, i32 102000}
!2 = !{i32 1, i32 2}
!3 = !{}
!4 = !{i16 0, i16 22}
!5 = !{!"ModuleMD", !6, !7, !98, !99, !130, !131, !135, !138, !139, !140, !173, !196, !209, !210, !211, !224, !225, !226, !227, !228, !229, !230, !231, !232, !233, !237, !238, !245, !246, !247, !248, !249, !250, !251, !252, !253, !254, !255, !256, !258, !262, !263, !264, !265, !266, !267, !268, !269, !270, !271, !272, !273, !274, !275, !276, !277, !278, !279, !280, !281, !282, !283, !284, !286, !289, !290, !291}
!6 = !{!"isPrecise", i1 false}
!7 = !{!"compOpt", !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97}
!8 = !{!"DenormsAreZero", i1 false}
!9 = !{!"BFTFDenormsAreZero", i1 false}
!10 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!11 = !{!"OptDisable", i1 false}
!12 = !{!"MadEnable", i1 false}
!13 = !{!"NoSignedZeros", i1 false}
!14 = !{!"NoNaNs", i1 false}
!15 = !{!"FloatRoundingMode", i32 0}
!16 = !{!"FloatCvtIntRoundingMode", i32 3}
!17 = !{!"LoadCacheDefault", i32 -1}
!18 = !{!"StoreCacheDefault", i32 -1}
!19 = !{!"VISAPreSchedRPThreshold", i32 0}
!20 = !{!"SetLoopUnrollThreshold", i32 0}
!21 = !{!"UnsafeMathOptimizations", i1 false}
!22 = !{!"disableCustomUnsafeOpts", i1 false}
!23 = !{!"disableReducePow", i1 false}
!24 = !{!"disableSqrtOpt", i1 false}
!25 = !{!"FiniteMathOnly", i1 false}
!26 = !{!"FastRelaxedMath", i1 false}
!27 = !{!"DashGSpecified", i1 false}
!28 = !{!"FastCompilation", i1 false}
!29 = !{!"UseScratchSpacePrivateMemory", i1 true}
!30 = !{!"RelaxedBuiltins", i1 false}
!31 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!32 = !{!"GreaterThan2GBBufferRequired", i1 true}
!33 = !{!"GreaterThan4GBBufferRequired", i1 true}
!34 = !{!"DisableA64WA", i1 false}
!35 = !{!"ForceEnableA64WA", i1 false}
!36 = !{!"PushConstantsEnable", i1 true}
!37 = !{!"HasPositivePointerOffset", i1 false}
!38 = !{!"HasBufferOffsetArg", i1 false}
!39 = !{!"BufferOffsetArgOptional", i1 true}
!40 = !{!"replaceGlobalOffsetsByZero", i1 false}
!41 = !{!"forcePixelShaderSIMDMode", i32 0}
!42 = !{!"ForceGeomFFShaderSIMDMode", i32 0}
!43 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!44 = !{!"UniformWGS", i1 false}
!45 = !{!"disableVertexComponentPacking", i1 false}
!46 = !{!"disablePartialVertexComponentPacking", i1 false}
!47 = !{!"PreferBindlessImages", i1 false}
!48 = !{!"UseBindlessMode", i1 false}
!49 = !{!"UseLegacyBindlessMode", i1 true}
!50 = !{!"disableMathRefactoring", i1 false}
!51 = !{!"atomicBranch", i1 false}
!52 = !{!"spillCompression", i1 false}
!53 = !{!"ForceInt32DivRemEmu", i1 false}
!54 = !{!"ForceInt32DivRemEmuSP", i1 false}
!55 = !{!"WaveIntrinsicUsed", i1 false}
!56 = !{!"DisableMultiPolyPS", i1 false}
!57 = !{!"NeedTexture3DLODWA", i1 false}
!58 = !{!"DisableFastestSingleCSSIMD", i1 false}
!59 = !{!"DisableFastestLinearScan", i1 false}
!60 = !{!"UseStatelessforPrivateMemory", i1 false}
!61 = !{!"EnableTakeGlobalAddress", i1 false}
!62 = !{!"IsLibraryCompilation", i1 false}
!63 = !{!"LibraryCompileSIMDSize", i32 0}
!64 = !{!"FastVISACompile", i1 false}
!65 = !{!"MatchSinCosPi", i1 false}
!66 = !{!"ExcludeIRFromZEBinary", i1 false}
!67 = !{!"EmitZeBinVISASections", i1 false}
!68 = !{!"FP64GenEmulationEnabled", i1 false}
!69 = !{!"allowDisableRematforCS", i1 false}
!70 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!71 = !{!"DisableCPSOmaskWA", i1 false}
!72 = !{!"DisableFastestGopt", i1 false}
!73 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!74 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!75 = !{!"DisableConstantCoalescing", i1 false}
!76 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!77 = !{!"WaEnableALTModeVisaWA", i1 false}
!78 = !{!"WaEnableAtomicWaveFusion", i1 false}
!79 = !{!"WaEnableAtomicWaveFusionNonNullResource", i1 false}
!80 = !{!"WaEnableAtomicWaveFusionStateless", i1 false}
!81 = !{!"WaEnableAtomicWaveFusionTyped", i1 false}
!82 = !{!"ForceCBThroughSampler3D", i1 false}
!83 = !{!"WaStoreRawVectorToTypedWrite", i1 false}
!84 = !{!"WaLoadRawVectorToTypedRead", i1 false}
!85 = !{!"WaZeroSLMBeforeUse", i1 false}
!86 = !{!"NewSpillCostFunction", i1 false}
!87 = !{!"EnableVRT", i1 false}
!88 = !{!"ForceLargeGRFNum4RQ", i1 false}
!89 = !{!"Enable2xGRFRetry", i1 false}
!90 = !{!"Detect2xGRFCandidate", i1 false}
!91 = !{!"EnableURBWritesMerging", i1 true}
!92 = !{!"DisableEUFusion", i1 false}
!93 = !{!"DisableFDivToFMulInvOpt", i1 false}
!94 = !{!"initializePhiSampleSourceWA", i1 false}
!95 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!96 = !{!"DisableLoosenSimd32Occu", i1 false}
!97 = !{!"FastestS1Options", i32 0}
!98 = !{!"FuncMD"}
!99 = !{!"pushInfo", !100, !101, !102, !106, !107, !108, !109, !110, !111, !112, !113, !126, !127, !128, !129}
!100 = !{!"pushableAddresses"}
!101 = !{!"bindlessPushInfo"}
!102 = !{!"dynamicBufferInfo", !103, !104, !105}
!103 = !{!"firstIndex", i32 0}
!104 = !{!"numOffsets", i32 0}
!105 = !{!"forceDisabled", i1 false}
!106 = !{!"MaxNumberOfPushedBuffers", i32 0}
!107 = !{!"inlineConstantBufferSlot", i32 -1}
!108 = !{!"inlineConstantBufferOffset", i32 -1}
!109 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!110 = !{!"constants"}
!111 = !{!"inputs"}
!112 = !{!"constantReg"}
!113 = !{!"simplePushInfoArr", !114, !123, !124, !125}
!114 = !{!"simplePushInfoArrVec[0]", !115, !116, !117, !118, !119, !120, !121, !122}
!115 = !{!"cbIdx", i32 0}
!116 = !{!"pushableAddressGrfOffset", i32 -1}
!117 = !{!"pushableOffsetGrfOffset", i32 -1}
!118 = !{!"offset", i32 0}
!119 = !{!"size", i32 0}
!120 = !{!"isStateless", i1 false}
!121 = !{!"isBindless", i1 false}
!122 = !{!"simplePushLoads"}
!123 = !{!"simplePushInfoArrVec[1]", !115, !116, !117, !118, !119, !120, !121, !122}
!124 = !{!"simplePushInfoArrVec[2]", !115, !116, !117, !118, !119, !120, !121, !122}
!125 = !{!"simplePushInfoArrVec[3]", !115, !116, !117, !118, !119, !120, !121, !122}
!126 = !{!"simplePushBufferUsed", i32 0}
!127 = !{!"pushAnalysisWIInfos"}
!128 = !{!"inlineRTGlobalPtrOffset", i32 0}
!129 = !{!"rtSyncSurfPtrOffset", i32 0}
!130 = !{!"WaEnableICBPromotion", i1 false}
!131 = !{!"vsInfo", !132, !133, !134}
!132 = !{!"DrawIndirectBufferIndex", i32 -1}
!133 = !{!"vertexReordering", i32 -1}
!134 = !{!"MaxNumOfOutputs", i32 0}
!135 = !{!"hsInfo", !136, !137}
!136 = !{!"numPatchAttributesPatchBaseName", !""}
!137 = !{!"numVertexAttributesPatchBaseName", !""}
!138 = !{!"dsInfo", !134}
!139 = !{!"gsInfo", !134}
!140 = !{!"psInfo", !141, !142, !143, !144, !145, !146, !147, !148, !149, !150, !151, !152, !153, !154, !155, !156, !157, !158, !159, !160, !161, !162, !163, !164, !165, !166, !167, !168, !169, !170, !171, !172}
!141 = !{!"BlendStateDisabledMask", i8 0}
!142 = !{!"SkipSrc0Alpha", i1 false}
!143 = !{!"DualSourceBlendingDisabled", i1 false}
!144 = !{!"ForceEnableSimd32", i1 false}
!145 = !{!"outputDepth", i1 false}
!146 = !{!"outputStencil", i1 false}
!147 = !{!"outputMask", i1 false}
!148 = !{!"blendToFillEnabled", i1 false}
!149 = !{!"forceEarlyZ", i1 false}
!150 = !{!"hasVersionedLoop", i1 false}
!151 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!152 = !{!"requestCPSizeRelevant", i1 false}
!153 = !{!"requestCPSize", i1 false}
!154 = !{!"texelMaskFastClearMode", !"Disabled"}
!155 = !{!"NumSamples", i8 0}
!156 = !{!"blendOptimizationMode"}
!157 = !{!"colorOutputMask"}
!158 = !{!"ProvokingVertexModeNosIndex", i32 0}
!159 = !{!"ProvokingVertexModeNosPatch", !""}
!160 = !{!"ProvokingVertexModeLast", !"Negative"}
!161 = !{!"VertexAttributesBypass", i1 false}
!162 = !{!"LegacyBaryAssignmentDisableLinear", i1 false}
!163 = !{!"LegacyBaryAssignmentDisableLinearNoPerspective", i1 false}
!164 = !{!"LegacyBaryAssignmentDisableLinearCentroid", i1 false}
!165 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveCentroid", i1 false}
!166 = !{!"LegacyBaryAssignmentDisableLinearSample", i1 false}
!167 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveSample", i1 false}
!168 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", !"Negative"}
!169 = !{!"meshShaderWAPerPrimitiveUserDataEnablePatchName", !""}
!170 = !{!"generatePatchesForRTWriteSends", i1 false}
!171 = !{!"forceVMask", i1 false}
!172 = !{!"WaDisableVRS", i1 false}
!173 = !{!"csInfo", !174, !175, !176, !177, !178, !19, !20, !179, !180, !181, !182, !183, !184, !185, !186, !187, !188, !189, !190, !51, !52, !191, !192, !193, !194, !195}
!174 = !{!"maxWorkGroupSize", i32 0}
!175 = !{!"waveSize", i32 0}
!176 = !{!"ComputeShaderSecondCompile"}
!177 = !{!"forcedSIMDSize", i8 0}
!178 = !{!"forceTotalGRFNum", i32 0}
!179 = !{!"forceSpillCompression", i1 false}
!180 = !{!"allowLowerSimd", i1 false}
!181 = !{!"disableSimd32Slicing", i1 false}
!182 = !{!"disableSplitOnSpill", i1 false}
!183 = !{!"enableNewSpillCostFunction", i1 false}
!184 = !{!"forcedVISAPreRAScheduler", i1 false}
!185 = !{!"forceUniformBuffer", i1 false}
!186 = !{!"forceUniformSurfaceSampler", i1 false}
!187 = !{!"disableLocalIdOrderOptimizations", i1 false}
!188 = !{!"disableDispatchAlongY", i1 false}
!189 = !{!"neededThreadIdLayout", i1* null}
!190 = !{!"forceTileYWalk", i1 false}
!191 = !{!"walkOrderEnabled", i1 false}
!192 = !{!"walkOrderOverride", i32 0}
!193 = !{!"ResForHfPacking"}
!194 = !{!"hasWaveMatrix", i1 false}
!195 = !{!"constantFoldSimdSize", i1 false}
!196 = !{!"msInfo", !197, !198, !199, !200, !201, !202, !203, !204, !205, !206, !207, !160, !158, !208}
!197 = !{!"PrimitiveTopology", i32 3}
!198 = !{!"MaxNumOfPrimitives", i32 0}
!199 = !{!"MaxNumOfVertices", i32 0}
!200 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!201 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!202 = !{!"WorkGroupSize", i32 0}
!203 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!204 = !{!"IndexFormat", i32 6}
!205 = !{!"SubgroupSize", i32 0}
!206 = !{!"VPandRTAIndexAutostripEnable", i1 false}
!207 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", i1 false}
!208 = !{!"numPrimitiveAttributesPatchBaseName", !""}
!209 = !{!"taskInfo", !134, !202, !203, !205}
!210 = !{!"NBarrierCnt", i32 0}
!211 = !{!"rtInfo", !212, !213, !214, !215, !216, !217, !218, !219, !220, !221, !222, !223}
!212 = !{!"RayQueryAllocSizeInBytes", i32 0}
!213 = !{!"NumContinuations", i32 0}
!214 = !{!"RTAsyncStackAddrspace", i32 -1}
!215 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!216 = !{!"SWHotZoneAddrspace", i32 -1}
!217 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!218 = !{!"SWStackAddrspace", i32 -1}
!219 = !{!"SWStackSurfaceStateOffset", i1* null}
!220 = !{!"RTSyncStackAddrspace", i32 -1}
!221 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!222 = !{!"doSyncDispatchRays", i1 false}
!223 = !{!"MemStyle", !"Xe"}
!224 = !{!"EnableTextureIndirection", i1 false}
!225 = !{!"EnableSamplerIndirection", i1 false}
!226 = !{!"samplerStateStride", i32 0}
!227 = !{!"samplerStateOffset", i32 0}
!228 = !{!"textureStateStride", i32 0}
!229 = !{!"textureStateOffset", i32 0}
!230 = !{!"CurUniqueIndirectIdx", i32 0}
!231 = !{!"inlineDynTextures"}
!232 = !{!"inlineResInfoData"}
!233 = !{!"immConstant", !234, !235, !236}
!234 = !{!"data"}
!235 = !{!"sizes"}
!236 = !{!"zeroIdxs"}
!237 = !{!"stringConstants"}
!238 = !{!"inlineBuffers", !239, !243, !244}
!239 = !{!"inlineBuffersVec[0]", !240, !241, !242}
!240 = !{!"alignment", i32 0}
!241 = !{!"allocSize", i64 0}
!242 = !{!"Buffer"}
!243 = !{!"inlineBuffersVec[1]", !240, !241, !242}
!244 = !{!"inlineBuffersVec[2]", !240, !241, !242}
!245 = !{!"GlobalPointerProgramBinaryInfos"}
!246 = !{!"ConstantPointerProgramBinaryInfos"}
!247 = !{!"GlobalBufferAddressRelocInfo"}
!248 = !{!"ConstantBufferAddressRelocInfo"}
!249 = !{!"forceLscCacheList"}
!250 = !{!"SrvMap"}
!251 = !{!"RootConstantBufferOffsetInBytes"}
!252 = !{!"RasterizerOrderedByteAddressBuffer"}
!253 = !{!"RasterizerOrderedViews"}
!254 = !{!"MinNOSPushConstantSize", i32 0}
!255 = !{!"inlineProgramScopeOffsets"}
!256 = !{!"shaderData", !257}
!257 = !{!"numReplicas", i32 0}
!258 = !{!"URBInfo", !259, !260, !261}
!259 = !{!"has64BVertexHeaderInput", i1 false}
!260 = !{!"has64BVertexHeaderOutput", i1 false}
!261 = !{!"hasVertexHeader", i1 true}
!262 = !{!"m_ForcePullModel", i1 false}
!263 = !{!"UseBindlessImage", i1 false}
!264 = !{!"enableRangeReduce", i1 false}
!265 = !{!"disableNewTrigFuncRangeReduction", i1 false}
!266 = !{!"enableFRemToSRemOpt", i1 false}
!267 = !{!"enableSampleptrToLdmsptrSample0", i1 false}
!268 = !{!"enableSampleLptrToLdmsptrSample0", i1 false}
!269 = !{!"WaForceSIMD32MicropolyRasterize", i1 false}
!270 = !{!"WaEnableFastestForAllWaveIntrinsicsCS", i1 false}
!271 = !{!"WaEnableFastestForAllWaveIntrinsicsPS", i1 false}
!272 = !{!"allowMatchMadOptimizationforVS", i1 false}
!273 = !{!"disableMatchMadOptimizationForCS", i1 false}
!274 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!275 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!276 = !{!"statefulResourcesNotAliased", i1 false}
!277 = !{!"disableMixMode", i1 false}
!278 = !{!"genericAccessesResolved", i1 false}
!279 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!280 = !{!"disableSeparateScratchWA", i1 false}
!281 = !{!"privateMemoryPerWI", i32 0}
!282 = !{!"PrivateMemoryPerFG"}
!283 = !{!"m_OptsToDisable"}
!284 = !{!"capabilities", !285}
!285 = !{!"globalVariableDecorationsINTEL", i1 false}
!286 = !{!"m_ShaderResourceViewMcsMask", !287, !288}
!287 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!288 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!289 = !{!"computedDepthMode", i32 0}
!290 = !{!"isHDCFastClearShader", i1 false}
!291 = !{!"argRegisterReservations", !292}
!292 = !{!"argRegisterReservationsVec[0]", i32 0}
!293 = !{i32 1, i32 1, i32 1}
!294 = !{!"none", !"none", !"none"}
!295 = !{!"array*", !"array*", !"array*"}
!296 = !{!"", !"", !""}
!297 = !{i32 16}
!298 = !{!299}
!299 = !{i32 4469}
!300 = distinct !{!300}
