#!/bin/bash
set -e
seq_len=${1:-512}

echo -e "================seq_len: $seq_len=========="

## basekit
#source /opt/intel/oneapi/setvars.sh --force
#
# store result
rm -rf result.csv result.txt

# update shape size in driver.py and paged-attention.forward.py
sed -i "s/x_vals=.*/x_vals=[$seq_len],/g" paged-attention-simplified.py
sed -i "s/float M = .*/float M = 1, N_CTX = $seq_len, D_HEAD = 64, num_seqs = 16, num_heads = 16;/g" /home/gta/deweiwang/xpu/intel-xpu-backend-for-triton/third_party/intel/backend/driver.py
#
## clean Triton cache
#rm -rf ./tt_cache
#export TRITON_CACHE_DIR=./tt_cache
## clean IGC cache
#export NEO_CACHE_PERSISTENT=0
#
#TRITON_INTEL_ENABLE_BLOCK_PTR=1 \
#TRITON_DISABLE_LINE_INFO=1 \
#TRITON_INTEL_ENABLE_ADDRESS_PAYLOAD_OPT=1 \
#IGC_VISAOptions=" -TotalGRFNum 256 -enableBCR -nolocalra -printregusage -DPASTokenReduction -enableHalfLSC" \
#IGC_ForcePrefetchToL1Cache=1 \
#IGC_VATemp=1 \
#UR_L0_IN_ORDER_BARRIER_BY_SIGNAL=0 \
#IGC_DisableLoopUnroll=1 \
#NEO_CACHE_PERSISTENT=0 \
python paged-attention-simplified.py 2>&1 | tee result.txt

if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    exit 1
fi

Triton_GB_max=`grep "Triton Peak GB" result.txt | awk '{print $NF}' |  tail -n10 | awk 'BEGIN{max=0} {if ($1>max) max=$1} END{print max}'`
Triton_GB_min=`grep "Triton Peak GB" result.txt | awk '{print $NF}'  | tail -n10 | awk 'BEGIN{min=9999} {if ($1<min) min=$1} END{print min}'`
Triton_GB_avg=$(grep "Triton Peak GB" result.txt | awk '{print $NF}'  | tail -n10 | awk -v max="$Triton_GB_max" -v min="$Triton_GB_min" '{sum+=$1} END{print (sum-max-min)/(NR-2)}')

echo -e "=================================== Result ========================================"
echo "seq_len, avg_GB, max_GB, min_GB" | tee result.csv
echo $seq_len, $Triton_GB_avg, $Triton_GB_max, $Triton_GB_min | tee -a result.csv
