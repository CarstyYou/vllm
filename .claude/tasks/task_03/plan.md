# task_03 — cute MXFP8 moe_gemm 接入（3a GranK=128 / 3b GranK=32）

对应母 plan 第二轮（Q5 已 lock）。分支 `cute_sm120_precision_internal`。
状态：**plan 待 xiy review；sub-task 0 可并行进行**。

## 目标

FI `moe_gemm_mxfp8_nt_groupwise` 注册为 `--moe-backend=cute_sm120_mxfp8_128`（3a）/
`cute_sm120_mxfp8_32`（3b），Qwen3.5-35B 单卡 GSM8K+MMLU 全量，与
triton / dg-UE8M0 / cute_sm120_fp8 组成五列矩阵。

## 已 lock 的设计（Q5 + 方案 A）

- **3a 链**（weight，load 期）：checkpoint float(128,128) → `requant_weight_ue8m0_inplace`
  同语义 requant（粒度不变）→ N 方向 broadcast 成 per-token (1,128) → 4 UE8M0/int32 pack
- **3b 链**（weight，load 期）：dequant → per-row 1×32 requant（DG per_token_cast 风格）→ pack
- act 侧 runtime：vLLM ue8m0 packed quant（group=128/32 参数化）+
  `ep_scatter(pack_ue8m0=True, block_size=granK)` 现成
- 切换：两个薄子类（`_GRAN_K` 类属性）+ 两个枚举，无 env 开关
- DG 不跑 3b（sm120 gate 不存在，精度冗余）

## Sub-tasks

| # | 内容 | Gate |
|---|---|---|
| 0 | **MXFP8 契约核对（已完成，见下节）** | ✓ |
| 1 | requant helpers：oracle convert 加 `CUTE_SM120_MXFP8_{128,32}` 分支（load 期 weight 转换 + pack）；单测 vs FI upstream test 的 eager 打包器 bit-equal | bit-equal PASS |
| 2 | `CuteMxfp8Experts` 基类 + `_128`/`_32` 薄子类（进 `cute_sm120_moe.py`，复用 permute/unpermute；repack helper 的 mxfp8 packed 版） | py_compile + review |
| 3 | 注册：2 枚举 + priority + kernel_cls + map + convert 白名单 + Literal + docstring | `map_fp8_backend` 双名 OK |
| 4 | 单层对拍：**reference = triton 喂同一份 requant 后权重**（UE8M0 scale 是合法 float scale）——diff 隔离 kernel，不混 requant 损失；另留一列 vs 原始权重 triton 的 diff 作 requant 损失量化 | kernel-diff < 1e-3 全 cells |
| 5 | e2e：3a/3b 各 GSM8K 1319 + MMLU 全量 | 数字落表；3a 期望 ≈ dg-UE8M0（同 recipe）；3b−3a = 粒度效应 |
| 6 | result.md（测试条件+recipe+五列矩阵）+ findings + 收口 | xiy 收口确认 → commit |

## Sub-task 0 结论（契约，ground truth = upstream test 打包器）

- **a_scale**：物理存储 `(k_align, m_padded)` int32 contiguous（MN-major，与 FP8 同向），
  entry 接收的是它的 `.transpose(0,1)` 视图（docstring 的 `(m_padded, k_align)` 即此逻辑 shape）
- **4 列对齐公式与 FP8 完全相同**：`(offset + 3i)//4*4`（PACK_NSF=4）；zero-fill 同
- **int32 打包沿 K**：4 个连续 K 块的 UE8M0 → 1 int32；`k_align = ⌈K/(4·granK)⌉`
  （granK=128 → 每 int32 覆盖 512 K；granK=32 → 128 K）
- **b_scale**：`(E, n, k_align)` int32 per-token（N 广播后同款打包），无 padding 列语义
- 打包 helper 直接复用 DG 的 `per_token_cast_to_fp8(use_ue8m0)` + `pack_ue8m0_to_int`
  （upstream test 打包器本身就这么写，与 [[feedback-reuse-dg-helpers]] 一致）
- `ep_scatter(pack_ue8m0=True)` 输出 `(M_sum, packed_sf_k)` strided `(1, round_up(M_sum,4))`
  ——正是 MN-major 存储 + 转置视图的同款语义，映射到 FI 布局只差 4 列对齐 repack（复用 FP8 helper 思路）

## Risks

| 风险 | 预案 |
|---|---|
| a_scale 方向/4 行 pad 语义与 FP8 不同 | sub-task 0 以 upstream test 打包器为 ground truth |
| requant 后 fp8 值域越界/NaN | 复用 vLLM 函数自带 clamp；bit-equal 单测把关 |
| load 期 requant 峰值显存（dequant fp32 副本） | per-expert 逐个转（E 循环），峰值 1 个 expert 的 fp32 |
| 3b 的 act group=32 kernel 实测（vLLM 参数化是否真支持 32） | sub-task 0 顺带验证（DG mxfp8 路径先例在，风险低） |

## Results（待填）
