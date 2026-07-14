# task_02 — cute FP8 moe_gemm 接入 vLLM（路径 2）

对应母 plan sub-task 2.1–2.3。分支 `cute_sm120_precision_internal`。状态：**plan 已 review（D1 lock=(a)，D2–D4 待 sub-task 0 定案），sub-task 0 放行**。

## 目标

把 FI `moe_gemm_fp8_nt_groupwise`（fork 分支 `sm120_moe_gemm_fp8_internal` @ `5358413`，PR #3891 内容）
注册为 vLLM 的 `--moe-backend=cute_sm120_fp8`，Qwen3.5-35B 单卡 GSM8K/MMLU 全量对齐 triton baseline。

## 两侧 contract（已知部分）

**FI entry**（zero-padding 模式）：
- `a (cum_m, k)` fp8 row-major token-packed，**M 不做 per-expert padding**
- `a_scale` float32 (1,128,128) groupwise（精确 shape/layout 在 sub-task 0 从 FI core.py 确认）
- `b (E, n, k)` fp8 col-major；`b_scale` float32
- `m_indptr (E+1,)` int32 CSR cumsum
- 输出 `(cum_m, n)` bf16

**vLLM DeepGEMM experts 路径产出**（Q3 已查）：
- `deepgemm_moe_permute` → `aq (M_sum, K)`，M_sum = **per-expert round-up 到 align 的 padded 总数**，
  `expert_ids (M_sum,)`（padding 行 = -1）+ 内部 `expert_start_loc`（padded CSR）+ `inv_perm`
- act scale：fp32 路径 = col-major (M, K//128)
- 中间 SiLU+quant：`silu_mul_per_token_group_quant_fp8_colmajor`
- unpermute 带 weighted reduce

## 核心设计问题（sub-task 0 解决后 lock）

| # | 问题 | 候选 |
|---|---|---|
| D1 | M padding 语义差异 | **已 lock (xiy 2026-07-13)：方案 (a)** —— permute 产 exact packed（align=1）+ 自建 m_indptr；无 NaN 风险 |
| D2 | act scale layout | **定案（sub-task 0）**：需 repack helper → FI `(k_blocks, m_padded)` MN-major + per-expert 4 列对齐 + zero-fill；精度阶段 torch 实现，见 sub_task_0_findings.md |
| D3 | b_scale layout | **定案（sub-task 0）**：FI 要 `(E, k_blocks, n_blocks)`（K 在前），load 期一次 transpose+contiguous |
| D4 | 中间激活 quant | **定案（sub-task 0）**：复用 vLLM colmajor 融合 kernel + D2 同一 repack helper |

## Sub-tasks

| # | 内容 | Gate |
|---|---|---|
| 0 | **契约核对**：读 FI `flashinfer/grouped_mm/cute_sm120_fp8_groupwise/core.py` + thop 校验 + vLLM `deep_gemm_utils.py` permute 细节；D1–D4 定案写回本 plan | xiy review 定案 |
| 1 | 环境：FI fork 分支 editable 装进同一 venv；节点上 FI smoke（单独跑 `moe_gemm_fp8_nt_groupwise` 小 case，JIT 编译过 + 数值 sanity） | FI kernel 在该 venv 可调 |
| 2 | `CuteFp8Experts`（新文件 `experts/cute_sm120_moe.py`，仿 `DeepGemmExperts` 8 个抽象方法 + apply 按 D1–D4 marshaling） | AST/import OK |
| 3 | 单层对拍：改写 `tests/kernels/moe/test_deepgemm.py::run_single_case` 为 `CuteFp8Experts` vs `TritonExperts`，扫 E/m_pe/N/K cells | calc_diff < 1e-3 全 cells |
| 4 | oracle 注册：`Fp8MoeBackend.CUTE_SM120_FP8` + `_AVAILABLE_BACKENDS`（排 TRITON 前、DEEPGEMM 后）+ `backend_to_kernel_cls` + `map_fp8_backend` + `MoEBackend` Literal + quant_config 透传核对 | `--moe-backend=cute_sm120_fp8` serve 起来（oracle log 证据行） |
| 5 | e2e：GSM8K 冒烟 100（工程验证）→ 全量 1319 + MMLU 全量 | vs triton 全量差 ≤ 1pp（同 float-scale 语义预期对齐）；数字落表 |
| 6 | findings 沉淀 + Results 汇总 | 表全 |

## Risks

| 风险 | 说明/预案 |
|---|---|
| padding 行 NaN 污染 | D1 选 (b) 时垃圾 fp8 字节可解码为 NaN → 必须 zero-fill；选 (a) 则无此问题 |
| scale 转换引入 copy 开销 | 精度阶段可接受，perf 阶段（P5）前审视是否可 zero-copy |
| FI JIT 首次编译慢 / 与 vLLM torch 版本冲突 | venv torch cu130 vs FI 要求核对；JIT cache 放 scratch |
| cudagraph / torch.compile 兼容 | FI entry 内部有 Python 分支/分配；首验用 `--enforce-eager`，通过后再开 graph 对比 |
| chunking（vLLM 对大 batch 分 chunk 调 experts） | workspace_shapes 第一维按 token 数，遵守即可 |

## Results

见 [result.md](result.md)。
