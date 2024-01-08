from triton.common.backend import BaseBackend
from dataclasses import dataclass
from ..._C.libtriton.translation import ClusterInfo, get_num_warps, TMAInfos, translate_triton_gpu_to_llvmir, get_shared_memory_size, add_external_libs, translate_llvmir_to_spirv
from ...common.backend import get_cuda_version_key
from ..._C.libtriton import ir, passes
import functools
from typing import Any
from ..utils import get_ids_of_tensormaps, parse_tma_info
from ..make_launcher import make_stub
import hashlib


def get_ir_kernel_name(src: str, pattern: str) -> str:
    '''
    Get kernel name from the input source file.
    This Kernel name is required when launching the kernel.
    '''
    # This is the original kernel names in Triton IR for SPIRV target.
    assert src
    for line in src.split('\n'):
        line = line.strip()
        if line.startswith(pattern):
            return line.split('@', 1)[-1].split('(', 1)[0]


@functools.lru_cache()
def ptx_get_version(cuda_version) -> int:
    '''
    Get the highest PTX version supported by the current CUDA driver.
    '''
    assert isinstance(cuda_version, str)
    major, minor = map(int, cuda_version.split('.'))
    if major == 12:
        return 80 + minor
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor
    raise RuntimeError("Triton only support CUDA 10.0 or higher")


@dataclass(frozen=True)
class XPUOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 2
    cluster_dims: tuple = (1, 1, 1)
    ptx_version: int = None
    enable_warp_specialization: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False

    def __post_init__(self):
        # TODO: change API
        if isinstance(self.extern_libs, dict):
            extern_libs = tuple([(k, v) for k, v in self.extern_libs.items() if v])
            object.__setattr__(self, 'extern_libs', extern_libs)
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class XPUBackend(BaseBackend):

    def __init__(self, device_type: tuple) -> None:
        super().__init__(device_type)
        self.capability = 0
        assert isinstance(self.capability, int)

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in XPUOptions.__dataclass_fields__.keys() if k in opts}
        args["allow_fp8e4nv"] = True
        args["max_num_imprecise_acc_default"] = 2**30 if self.capability == 90 else 0
        return XPUOptions(**args)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, capability):
        cluster_info = ClusterInfo()
        if opt.cluster_dims is not None:
            cluster_info.clusterDimX = opt.cluster_dims[0]
            cluster_info.clusterDimY = opt.cluster_dims[1]
            cluster_info.clusterDimZ = opt.cluster_dims[2]
        # TTIR -> TTGIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, opt.num_warps, 32, opt.num_ctas, capability)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
        passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
        passes.ttnvgpuir.add_rewrite_tensor_pointer(pm, capability)
        passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_accelerate_matmul(pm, capability)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        if opt.optimize_epilogue:
            passes.ttgpuir.add_optimize_epilogue(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm)
        passes.common.add_cse(pm)
        # `num_warps` does not mean the total number of warps of a CTA when
        # warp specialization is enabled.
        # it's the responsibility of the compiler to figure out the exact
        # `num_warps` to use.
        # TODO: support the case where `num_warps` from user is not 4.
        ws_enabled = False
        if capability // 10 >= 9 and opt.enable_warp_specialization and opt.num_warps == 4:
            passes.ttnvgpuir.add_wsfeasibility_checking(pm, capability)
            pm.run(mod)
            ws_enabled = passes.ttnvgpuir.is_ws_supported(mod)
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
        if ws_enabled:
            passes.ttnvgpuir.add_wsdecomposing(pm, capability)
            passes.ttnvgpuir.add_wspipeline(pm, opt.num_stages, opt.num_warps, capability)
            passes.ttnvgpuir.add_wsmutex(pm, capability)
            passes.ttnvgpuir.add_wsmaterialization(pm, capability)
            passes.common.add_licm(pm)
            passes.common.add_cse(pm)
        else:
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, opt.num_warps, opt.num_ctas, capability)
        passes.ttnvgpuir.add_materialize_load_store(pm, opt.num_warps, capability)
        if capability // 10 <= 8:
            passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_decompose_conversions(pm)
        passes.ttnvgpuir.add_wsfixup_missing_attrs(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if capability // 10 >= 9:
            passes.ttnvgpuir.add_fence_insertion(pm)
        passes.ttnvgpuir.add_wsfixup_missing_attrs(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
        return mod

    @staticmethod
    def make_llir(src, metadata, options, capability):
        metadata["enable_warp_specialization"] = passes.ttnvgpuir.is_ws_supported(src)
        metadata["num_warps"] = get_num_warps(src)
        tma_infos = TMAInfos()
        # link libraries
        if options.extern_libs:
            names = [lib[0] for lib in options.extern_libs]
            paths = [lib[1] for lib in options.extern_libs]
            add_external_libs(src, names, paths)
        # TritonGPU -> LLVM-IR
        ret = translate_triton_gpu_to_llvmir(src, capability, tma_infos)
        if len(tma_infos) > 0:
            metadata["tensormaps_info"] = parse_tma_info(tma_infos, metadata["ids_of_folded_args"])
            for i, _ in enumerate(metadata["tensormaps_info"]):
                metadata["tensormaps_info"][i].ids_of_folded_args = metadata["ids_of_folded_args"]
        metadata["ids_of_tensormaps"] = get_ids_of_tensormaps(metadata.get("tensormaps_info", None))
        metadata["shared"] = get_shared_memory_size(src)
        return ret

    @staticmethod
    def make_spv(src, metadata):
        metadata["name"] = get_ir_kernel_name(src, pattern='define spir_kernel void')
        return translate_llvmir_to_spirv(src)

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.capability)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, self.capability)
        stages["spv"] = lambda src, metadata: self.make_spv(src, metadata)

    def hash(self):
        return f'{get_cuda_version_key()}-{self.capability}'

    def make_launcher_stub(self, src, metadata):
        ids = {
            "ids_of_tensormaps": metadata.get("ids_of_tensormaps", tuple()), "ids_of_folded_args":
            metadata.get("ids_of_folded_args",
                         tuple()), "ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()
        }
        constants = src.constants if hasattr(src, "constants") else dict()
        enable_warp_specialization = False

        # set constant
        return make_stub(src.name, src.signature, constants, ids, enable_warp_specialization=enable_warp_specialization)

    @classmethod
    def create_backend(cls, device_type: str):
        return cls(device_type)