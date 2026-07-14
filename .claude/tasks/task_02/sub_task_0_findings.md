# task_02 sub-task 0 — 契约核对 findings（2026-07-13）

## FI FP8 entry 精确 contract（`flashinfer/grouped_mm/cute_sm120_fp8_groupwise/core.py:77`）

| 参数 | 精确要求 |
|---|---|
| `a` | `(cum_m, k)` fp8 row-major，exact token-packed（无 per-expert M pad） |
| `a_scale` | **float32 `(k_blocks, m_padded)` MN-major contiguous**；`m_padded = (cum_m + E*3)//4*4`；expert i 的列起点 = `(m_indptr[i] + 3*i)//4*4`（**per-expert 4 列对齐**）；padding 列 zero-fill；数据指针 16B 对齐 |
| `b` | `(E, n, k)` fp8 col-major |
| `b_scale` | float32 **`(E, k_blocks, n_blocks)`** contiguous（注意 K 在前！） |
| `m_indptr` | `(E+1,)` int32 CSR |
| granularity | 固定 `(1,128,128)`，MN mode，输出 bf16 |

## vLLM 侧对接事实

- `ep_scatter(align_m=...)` 是显式参数（`deep_gemm_utils.py:531+`）；stage-1 kernel 的 round-up 在
  `ALIGN_M=1` 时退化为精确 cumsum（`deep_gemm_utils.py:131`），**无隐藏对齐假设** →
  D1(a) 可行：experts 类直接调 `ep_scatter(align_m=1)`，`M_sum = topk_ids.numel()`（python int，无 GPU→CPU sync），
  `m_indptr = [0, cumsum(expert_num_tokens)]`（GPU torch.cumsum，无 sync）
- float32 scale 的 scatter 输出 = row-major `(M_sum, k_blocks)`（`deep_gemm_utils.py:508`）
- 中间激活融合 kernel `silu_mul_per_token_group_quant_fp8_colmajor` 输出 scale 天然 col-major
  （存储 `(k_blocks, M)` contiguous）
- checkpoint weight scale = `(E, n_blocks, k_blocks)`（`fp8.py create_weights`）——与 FI 的 `(E, k_blocks, n_blocks)` **转置关系**

## D2–D4 定案提议

| # | 定案 | 实现 |
|---|---|---|
| D2 | a_scale 需一个 **repack helper**：row-major/col-major `(·, k_blocks)` → FI `(k_blocks, m_padded)`（转置 + per-expert 4 列对齐 offset copy + zero-fill） | 精度阶段用 torch 实现（E 次窄条 copy）；P5 前评估并入 scatter kernel |
| D3 | b_scale load 期一次性 `transpose(-1,-2).contiguous()` | 放 experts 的 `process_weights_after_loading`（或 oracle convert 分支） |
| D4 | GEMM2 中间激活复用 vLLM colmajor 融合 kernel（输出已是 MN-major 语义）+ 同一 repack helper | 无需 FI 侧 quant |

## 风险更新

- repack helper 每层 2 次（GEMM1/GEMM2 各一）×E 窄条 copy——精度阶段可接受，perf 阶段是已知优化点
- `a_scale` 16B 指针对齐：torch.empty float32 天然 64B 对齐，满足
