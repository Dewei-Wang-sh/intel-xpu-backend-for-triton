# Xe4 User Guide


## setup docker
[docker setup](https://github.com/intel-sandbox/containers.docker.gpu.compute.sim/blob/main/README.md)


## setup simulator
[xesim](https://axeweb.intel.com/axe/software/ci/64/1/versions)

[llc](https://ubit-gfx.intel.com/build/20569811/artifacts)

[zesim](https://gfx-assets.fm.intel.com/artifactory/gfx-gca-assets-fm/zesim)

currently I'm using the below version to make it work
```bash
https://gfx-assets.fm.intel.com/artifactory/gfx-cobalt-assets-fm/XeSim/Linux/XE3P_V2/91530/XE3P_V2-91530-Linux.zip
gta@pvc125074:/home/gta/deweiwang/xe4/XE3P_V2-91530-Linux.zip
gta@pvc125074:/home/gta/deweiwang/xe4/IGC_20_Linux_internal.tzst
gta@pvc125074:/home/gta/deweiwang/xe4/archive.zip
```





## setup llvm & triton
[llvm](https://github.com/Dewei-Wang-sh/drivers.gpu.compiler.llvm-pisa/tree/xe4_draft)

[triton](https://github.com/Dewei-Wang-sh/intel-xpu-backend-for-triton/tree/xe4_draft)

## run test
```bash
cp -r /root/neo/usr/include/level_zero /usr/local/include
source /opt/intel/oneapi.2025.0/setvars.sh
#simulator env
export ZESIM_ROOT=/root/zesim/debug-sys20/zesim
export LD_LIBRARY_PATH=$ZESIM_ROOT:$LD_LIBRARY_PATH
export L0SIM_DEVICE_KIND=Xe4
export L0SIM_GRITS_PATH=/root/XE3P_V2
export L0SIM_SELECT_DEVICES=GRITS
#triton env
export TRITON_INTEL_ENABLE_XE4=1
export TRITON_XPU_GEN_NATIVE_CODE=1
python intel-xpu-backend-for-triton/python/tutorials/01-vector-add.py
```
