# task_02 Result — cute_sm120_fp8 接入（路径 2）

日期：2026-07-14。状态：**完成，待 xiy 收口确认**。

## 结论

1. **接入正确性成立**：单层对拍 24/24（calc_diff 5.8e-4~7.6e-4，含 M=1/7 exact-packing
   极端 cell 与 E=128 空 expert）；e2e GSM8K 全量与 triton 差 0.7pp（< 1pp gate，
   stderr≈1.1pp 内 = 同分布）；MMLU 全量三 backend 同分布。
2. **精度故事闭环**：cute_sm120_fp8 = float-scale 零转换语义，精度对齐 triton、
   高于 dg-UE8M0 1.8pp（GSM8K）——sm120 上除 triton 兜底外唯一 float-scale 实现。
3. **zero-padding 红线全程守住**：FI 两次 GEMM 输入输出 exact `cum_m` 行
   （`ep_scatter(align_m=1)`）；pad 仅存在于 scale 平面（4 列对齐）与 quant 临时 buffer。

## 测试条件

同 task_01 条件表（`../task_01/result.md`），差异项：
`--moe-backend=cute_sm120_fp8`；FI = fork 分支 `sm120_moe_gemm_fp8_internal` @ `5358413`
editable（`FLASHINFER_DISABLE_VERSION_CHECK=1`，JIT deps: nvidia-cutlass-dsl 4.6 + ninja +
cuda-13.3）；vLLM 含本分支接入 diff（experts 类 + oracle 注册，基 `dcf4072`）。

## Recipe

cute_sm120_fp8 = checkpoint 原样 float-scale FP8（act 1×128 / weight 128×128，零转换），
与 triton 列同 recipe；kernel = FI `moe_gemm_fp8_nt_groupwise`（zero-padding CSR contract）。

## 单层对拍（gate：calc_diff < 1e-3）

24/24 PASS：M∈{1,7,33,128,1024,4096} × topk∈{6,8} × E∈{32,128}，
diff 区间 5.783e-4 ~ 7.618e-4。证据：`results/cute_fp8_vs_triton.csv`。
Reference = Triton `fused_experts` 同权重同输入。

## e2e 精度（全量 only；Qwen3.5-35B 单卡三 backend 矩阵）

| backend | recipe | GSM8K 1319 | MMLU 14042 | 证据 |
|---|---|---|---|---|
| triton（task_01） | float-scale | 0.797 | 0.8465 (±0.0029) | task_01/results |
| deep_gemm（task_01） | UE8M0 GranK=128 | 0.772 | 0.8478 (±0.0029) | task_01/results |
| **cute_sm120_fp8** | float-scale | **0.790**（invalid 0.0008） | **0.8480 (±0.0029)** | `results/gsm8k_cute_sm120_fp8_n1319.json`、`results/mmlu_cute_sm120_fp8/` |

Backend 证据行：`Using CUTE_FP8 Fp8 MoE backend`（冒烟期）/ `results/evidence_cute_sm120_fp8_run_mmlu.txt`。
注：GSM8K 全量 run 在 backend 改名前执行（当时名 `cute_fp8`），代码与改名后 bit 级相同（纯字符串重命名）。

## 实现要点（review 两轮 + 复核全 PASS）

- `experts/cute_sm120_moe.py::CuteFp8Experts`：`ep_scatter(align_m=1)` exact 打包、
  `_repack_a_scale_for_fi`（4 列对齐 MN-major）、w scale 惰性转置缓存、
  quant 阶段 128-pad zero-fill 视图（B1 修复）
- 注册 5 处 + Literal：`--moe-backend=cute_sm120_fp8`；AUTO 模式在 deep_gemm 可用时
  仍落 DEEPGEMM（不干扰 baseline），仅显式选择启用
