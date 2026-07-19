# task_07 — dg vs cute 小 M MoE GEMM 实现差异调查

分支 `cute_sm120_precision_internal`。承接 [task_06]（实测 cc=1 decode cute MoE 67μs/call vs
dg 28μs/call，慢 2.4×；cc=128 齐平）。状态：**plan 待 xiy review**。

## 目标

调查 cute（FI `mxfp8_cute_sm120`）与 dg（`sm120_fp8_fp4_gemm_1d1d`）在**小 M（decode，
m_per_expert≈1）** MoE GEMM 上的实现差异，定位 cute 每次调用固定开销比 dg 高 39μs 的**具体来源**，
产出可优化的结构差异清单。

**纪律（xiy 硬规则）**：所有机制/差异判断必须先读到代码 file:line 或 profiling 数据；
fact 与 hypothesis 严格分栏；对比双方源码都要实际读过。不允许无证据推测。

## 已确认事实（task_06 + 已读代码）

- cute MoE 小 M 走 swapAB path（`runner.cu:297` m_per_expert≤12 → `KT_SWAPAB_N8`
  = `SM120BlockScaledBuilder<128, 8, 128, 4, GranK, kGT, SwapAB=true>`）——非 dispatch 漏
- 两侧均为 persistent + warp-specialized（TMA producer / Math consumer）：
  cute `kernel_impl.cuh:45-47,427` scheduler + while(get_next_block)；
  dg `sm120_fp8_fp4_gemm_1d1d.cuh:205,223` sched::Scheduler + while(get_next_block)
- 骨架同构 → 差异在更细参数/实现，**目前无证据指向具体哪一处**（待本 exp 查）

## 前置核对（G3/G4，读 5 轴前必须先 pin，否则读错对象 = F2 artifact）

- **cute 侧对象确认**：nsys task_06 trace 里核对小 M cute MoE kernel 的 **grid dim + mangled
  模板实参**，确认确实是 `KT_SWAPAB_N8`（tile N=8）而非 KT_M32/M64；确认 GemmType =
  `MGroupedContiguousWithZeroPadding`（vLLM 调 `moe_gemm_mxfp8_nt_groupwise`，
  `cute_sm120_moe.py:50,479` → runner `:296`）
- **dg 侧对象确认**：vLLM dg 走 `m_grouped_fp8_gemm_nt_contiguous`（`deep_gemm_moe.py:358,375`）；
  pin 传入的**实际 M（contiguous padded aggregate ≈ E×alignment，不是 1）+ num_groups**，
  据此读 `sm120.hpp` heuristic 的**正确分支**（按真实 M 不是按 1）

## 调查轴（每轴产出 file:line 证据，不下结论直到证据齐）

| 轴 | cute 读什么 | dg 读什么 | 目的 |
|---|---|---|---|
| **F. 分组/padding 方案（G1，疑主导）** | `MGroupedContiguousWithZeroPadding`：m_per_expert≈1 时 padded 有效 M = E×tile_m？scheduler 迭代 E 个 group 的 per-group prologue | dg contiguous 的 padding alignment、tile 数、per-group setup | **有效 padded-M / launched tiles / wasted-tile 比 / per-group prologue 次数**——这是 5 轴测不到、可能最大的固定成本 |
| A. tile 形状 | KT_SWAPAB_N8 TileM/N/K 实参 + swapAB 后有效 M/N 映射 | DG heuristic 在真实 M 选的 BLOCK_M/N/K | grid CTA 数、每 CTA 有效工作量 |
| B. pipeline stages | Builder 第 4 参 `4` = AB_Stages；swapAB smem 布局 | dg `kNumStages` 真实 M 取值 | pipeline 填充/排空成本占比 |
| C. grid / persistent 粒度 | `get_grid_shape(num_sms)` 实际 grid；scheduler block 划分 | dg grid = f(num_sms)；scheduler tile 数 | CTA 数 × 每 CTA 工作，launch/tail |
| D. SF (scale) 加载 | swapAB SFA/SFB TMA load（`sf_mxfp8_tma_load.cuh`） | dg SF load（1d1d kNumSFAStagesPerLoad） | scale 是否引额外固定 stage |
| E. epilogue + setup | cute 写回 + smem/barrier init / scheduler 构造 | dg epilogue + barrier init | 固定 epilogue + prologue setup |

## Sub-tasks

| # | 内容 | Gate |
|---|---|---|
| 0 | 读双方源码，A-E 五轴逐轴记 file:line 证据到 findings（纯代码，零 GPU） | 每轴 fact 表 + 明确标出「已确认差异」vs「需 NCU 验证」 |
| 1 | NCU profile cc=1 decode 的 cute vs dg MoE kernel（grid/occupancy/stall reason/warp cycles/smem） | 拿到 kernel-internal 数据，验证/反驳 sub-task 0 的差异假设 |
| 2 | 汇总 result.md：确认的实现差异清单 + 每条的证据（code + NCU）+ 可优化方向 | xiy 收口 |

## 固定成本二分（G2 硬约束）

39μs 固定开销拆两类，**不同工具、不 gate 于完全对上宏观**：
- **kernel-internal**（NCU 可测）：grid tail / occupancy / pipeline 填充 / padded-tile 浪费 / smem-barrier
- **launch·host·cudagraph**（NCU 测不到，NCU 会 serialize+flush cache 抹掉）：需 nsys **per-launch gap**
  （相邻 MoE kernel launch 间隔）量。**结论按两类分别给证据**，不要求 NCU 单值 = 67/28μs。

## 方法

- sub-task 0（零 GPU）：先做「前置核对」pin 双方真实 kernel 对象 + 传入 M/num_groups；再 Read
  两侧源码逐轴（F,A-E）对比，证据 = file:line。**不读到不写结论；fact/hypothesis 分栏。**
- sub-task 1（NCU）：走 ncu-profile-analysis skill。**先验证锁定正确 kernel**（G3）：
  抓到的 launch 的 grid dim + mangled 实参必须匹配 KT_SWAPAB_N8（N=8）/ dg 真实 BLOCK，
  核对是 decode（小有效 M）非 prefill；确认后再读 grid/occupancy/stall/warp-cycles/smem。
  overhead 大，只 profile 目标 2 kernel
- sub-task 1b（nsys per-launch gap）：从 task_06 cc=1 trace 量 cute vs dg 的 MoE kernel
  相邻 launch 间隔，捕获 NCU 测不到的 host/launch 固定成本
- 对照锚点：kernel-internal Δ + launch Δ 合计**趋近**（非等于）宏观 39μs，两类各自有证据即可

## Risks

| 风险 | 预案 |
|---|---|
| DG kernel JIT 生成、小 M heuristic 选型需运行时确认 | sub-task 0 读 heuristic 逻辑 + sub-task 1 NCU 实测确认实际 launch 的 kernel 参数 |
| NCU 在 MoE grouped kernel 上 launch 过滤复杂 | 用 kernel name regex + launch-count；先 smoke 一个 kernel 验证可锁定 |
| cute swapAB 的有效 M/N 语义（swap 后谁是 M） | sub-task 0 A 轴明确 swapAB 后 tile 到 problem 的映射，避免误读 |

## Results
（待填）

## Plan Review

**Date**: 2026-07-16
**Reviewer**: subagent

**Verdict**: ⚠️ Gaps（4 条，已全部一次性修入，不 re-review）

**Gaps + suggested fix**:
- G1（severe）zero-padding/grouped-contiguous 膨胀可能是主导固定成本，5 轴测不到 → 加 **轴 F**（padding 方案：有效 padded-M / launched tiles / wasted-tile 比 / per-group prologue）✓ 已修
- G2（severe）NCU 测不到 launch/host/cudagraph 开销，"NCU 对上 67/28μs 宏观"可能无法满足 → 固定成本二分（kernel-internal NCU vs launch/host nsys per-launch gap），不 gate 于完全对上 ✓ 已修
- G3（moderate）无机制确认 NCU 抓到的是 decode KT_SWAPAB_N8(N=8) 非 prefill（F2 artifact 类）→ 前置核对 grid dim + mangled 实参 ✓ 已修
- G4（moderate）dg 侧 M 是 contiguous padded aggregate(≈E×align) 非 1 → 先 pin 真实 M+num_groups 再读对 heuristic 分支 ✓ 已修
