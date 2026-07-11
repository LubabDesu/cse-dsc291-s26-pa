
KERNEL_CONFIGS = [
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "num_warps": 8, "num_stages": 4},
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "num_warps": 8, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "num_warps": 8, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 4},
]


@triton.jit
def matmul_add_relu_kernel_fp16(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_dm,
    stride_dn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # Step 1: Tile: Assignment
    #
    # Each kernel instance is mapped to a tile in the output matrix C.
    # Compute the starting indices (m_start, n_start) for this tile.
    # -------------------------------------------------------------------------
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)


    # -------------------------------------------------------------------------
    # Step 2: Register Tiling
    # -------------------------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -------------------------------------------------------------------------
    # Step 3: Shared Memory Tiling & Cooperative Fetching.
    # Compute pointers to the sub-tiles of A and B that are needed to compute
    # the current C tile. The offsets here serve to load BLOCK_M x BLOCK_K
    # and BLOCK_K x BLOCK_N blocks from A and B respectively.
    # -------------------------------------------------------------------------
    # for each slice of K : choose current k indicies, load A : rows offs_m and cols offs_k
    # load B : rows offs_k and cols offs_n, then multiply A_tile @ B tile and add into accumulator
    for k in range(0, K, BLOCK_K) : 
        offs_k = k + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(
          a_ptrs,
          mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
          other=0.0,
      )
        b = tl.load(
          b_ptrs,
          mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
          other=0.0,
      )
        acc += tl.dot(a, b)
    # -------------------------------------------------------------------------
    # Step 4: Add C and Apply ReLU to the accumulator
    # -------------------------------------------------------------------------
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c = tl.load(c_ptrs,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), 
                other=0.0,)
    acc += c
    # relu
    acc = tl.maximum(acc, 0.0)
    # -------------------------------------------------------------------------
    # Step 5: Write Cache / Epilogue Fusion: Write the computed tile to D.
    # -------------------------------------------------------------------------
    d_ptrs = d_ptr + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn


    tl.store(
        d_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )
