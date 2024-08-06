import torch
import intel_extension_for_pytorch

import triton
import triton.language as tl

@triton.jit
def paged_attention_v2_first_kernel(
    output_ptr,  # [num_seqs, num_query_heads, head_size]
    query_ptr,  # [num_seqs, num_query_heads, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size, block_size]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    NUM_HEADS: tl.constexpr,  # int
    CACHE_BLOCK_STRIDE: tl.constexpr,  # int
    PARTITION_SIZE: tl.constexpr, #int
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    MAX_NUM_BLOCKS_PER_SEQ: tl.constexpr,  # int, must be power of 2
):
    seq_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    partion_idx = tl.program_id(2).to(tl.int64)

    # Load queries: [1, HEAD_SIZE]
    slm_block_ptr = tl.alloc(shape=(1, HEAD_SIZE), dtype=tl.float16)
    if tl.warp_id() == 0:
        q_offset = seq_idx * (NUM_HEADS * HEAD_SIZE) + head_idx * HEAD_SIZE
        Q_block_ptr = tl.make_block_ptr(base=query_ptr + q_offset, shape=(1, HEAD_SIZE), strides=(HEAD_SIZE, 1),
                                        offsets=(0, 0), block_shape=(1, HEAD_SIZE), order=(1, 0))
        q = tl.load(Q_block_ptr)
        tl.store(slm_block_ptr, q)
    tl.barrier()
    q = tl.load(slm_block_ptr)

    # Load keys: [HEAD_SIZE, BLOCK_SIZE]
    num_prev_blocks = partion_idx * (PARTITION_SIZE // BLOCK_SIZE)
    block_idx = num_prev_blocks + tl.warp_id()
    physical_block_idx = tl.load(block_tables_ptr + seq_idx * MAX_NUM_BLOCKS_PER_SEQ + block_idx)
    kv_block_offset = physical_block_idx *CACHE_BLOCK_STRIDE + head_idx * HEAD_SIZE * BLOCK_SIZE
    K_block_ptr = tl.make_block_ptr(base=key_cache_ptr + kv_block_offset, shape=(HEAD_SIZE, BLOCK_SIZE), strides=(HEAD_SIZE, 1),
                                    offsets=(0, 0), block_shape=(HEAD_SIZE, BLOCK_SIZE), order=(1, 0),)
    k = tl.load(K_block_ptr)

    # compute qk: [1, BLOCK_SIZE]
    qk = tl.zeros([1, BLOCK_SIZE], dtype=tl.float32)
    qk += tl.dot(q, k)

    # softmax
    m_i = tl.max(qk, axis=1)
    m_i_final = tl.max(m_i, cross_warp = True)
    p = tl.math.exp((qk - m_i_final[:, None]))
    l_i = tl.sum(p, axis = 1)
    l_i_final = tl.sum(l_i, cross_warp = True)
    p /= l_i_final[:, None]

    # load values: [BLOCK_SIZE, HEAD_SIZE]
    V_block_ptr = tl.make_block_ptr(base=value_cache_ptr + kv_block_offset, shape=(BLOCK_SIZE, HEAD_SIZE), strides=(1, HEAD_SIZE),
                                    offsets=(0, 0), block_shape=(BLOCK_SIZE, HEAD_SIZE), order=(0, 1),)
    v = tl.load(V_block_ptr)

    # compute and store output: [1, HEAD_SIZE]
    o = tl.dot(p.to(tl.float16), v, tiling="horizontal")
    output = tl.sum(o, cross_warp = True, dst_warps=(0))
    if tl.warp_id() == 0:
        output_offset = seq_idx * (NUM_HEADS * HEAD_SIZE) + head_idx * HEAD_SIZE
        O_block_ptr = tl.make_block_ptr(base=output_ptr + output_offset, shape=(1, HEAD_SIZE), strides=(HEAD_SIZE, 1),
                                        offsets=(0, 0), block_shape=(1, HEAD_SIZE), order=(1, 0))
        tl.store(O_block_ptr, output)

def forward(q, k, v, block_tables, seq_len, max_num_blocks_per_seq):
    Lq, Lk, Lv = q.shape[-1], k.shape[-2], v.shape[-2]
    assert Lq == Lk and Lk == Lv
    assert Lk == 64
    HEAD_SIZE = Lk

    Hq, Hk, Hv = q.shape[1], k.shape[1], v.shape[1]
    assert Hq == Hk and Hk == Hv
    assert Hq == 16
    NUM_HEADS = Hq

    BLOCK_SIZE = k.shape[-1]
    assert BLOCK_SIZE == 32

    CACHE_BLOCK_STRIDE = NUM_HEADS * HEAD_SIZE * BLOCK_SIZE
    PARTITION_SIZE = 512
    num_warps = PARTITION_SIZE // BLOCK_SIZE

    o = torch.empty_like(q, dtype=torch.float32)

    grid = (q.shape[0], q.shape[1], seq_len//PARTITION_SIZE)
    paged_attention_v2_first_kernel[grid](
        o, q, k, v, block_tables,
        NUM_HEADS=NUM_HEADS,
        CACHE_BLOCK_STRIDE=CACHE_BLOCK_STRIDE,
        PARTITION_SIZE=PARTITION_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_SIZE=HEAD_SIZE,
        MAX_NUM_BLOCKS_PER_SEQ=max_num_blocks_per_seq,
        num_warps=num_warps,
        threads_per_warp=16,
        warp_level=True
    )
    return o

@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['seq_len'],
        x_vals=[512, 1024],
        #x_vals=[2048, 4096, 8192, 16384],
        # x_vals=[[512, 1024, 2048, 4096, 8192, 16384]],
        line_arg='provider',
        # possible values for `line_arg``
        #line_vals=['triton', 'onednn'],
        line_vals=['triton'],
        # label name for the lines
        #line_names=["Triton", "oneDNN"],
        line_names=["Triton"],
        ylabel="TFLOPS",
        plot_name="attn-performance",
        args={},
    ))
def benchmark(seq_len, provider):
    num_seqs = 16
    num_heads = 16
    head_size = 64
    block_size = 32
    # suppose max sequence length: 16K
    max_num_blocks_per_seq = 16 * 1024 // block_size
    num_blocks = num_seqs * max_num_blocks_per_seq
    dtype = torch.float16
    q = torch.randn((num_seqs, num_heads, head_size), device='xpu', dtype=dtype)
    k = torch.randn((num_blocks, num_heads, head_size, block_size), device='xpu', dtype=dtype)
    v = torch.randn((num_blocks, num_heads, head_size, block_size), device='xpu', dtype=dtype)
    # in order increase to simulate use case
    block_tables = torch.randint(0, 1024, (num_seqs, max_num_blocks_per_seq,), device='xpu', dtype=torch.int32)
    for i in range(num_seqs):
        for j in range(max_num_blocks_per_seq):
            block_tables[i][j] = i * max_num_blocks_per_seq + j

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'onednn':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.paged_attention_simplified(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False), rep=100, quantiles=quantiles,
                                                     fast_flush=False)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: forward(q, k, v, block_tables, seq_len, max_num_blocks_per_seq), rep=100, quantiles=quantiles,
                                                     fast_flush=False)

    def bandwidth(ms):
        # GB/s
        return  num_seqs * num_heads * (1*head_size*2 + 2*head_size*seq_len*2 + 1*head_size*4) * 1e-9 / (ms * 1e-3)

    return bandwidth(ms), bandwidth(max_ms), bandwidth(min_ms)
benchmark.run(show_plots=False, print_data=True)