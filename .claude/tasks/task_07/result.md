# task_07 Result — dg vs cute 小 M MoE 实现差异（根因 = ZeroPadding scheduler O(E) token_offset 线性扫描）

日期 2026-07-17。分支 `cute_sm120_precision_internal`。承接 task_06（cc=1 decode cute MoE 67μs/call
vs dg 28μs）。方法：真实 35B MoE shape 的 decode microbench（UT）+ NCU（ncu 2026.2.1，`--set full`，
kernel-name regex 锁 MoE kernel）+ veloq 分析。**全程代码 file:line + NCU 数据支撑，fact/hypothesis 分栏。**

## 结论

**cute 小 M（decode）的 2× 慢 = ZeroPadding scheduler 的 token_offset 串行线性扫描 O(num_experts)。**
`get_next_block`（`scheduler.cuh:177-190`）每 CTA ~10 个 warp **各自串行扫过全部 E 个 group**，每轮 2 个
相互依赖的 `LDG.E`（cumsum 对 `grouped_layout[i]/[i+1]`），推进依赖其结果 → LSU global-ld sectors 随 E
**严格线性**（963,840，6KD exp_30 SASS 实测），warp 50% 时间卡在这条依赖 load 链（long_scoreboard）。
**E=8 时 zeropad ≈ contiguous（14.99 vs 14.73μs）→ ZeroPadding GemmType 本身零代价，全部超额 duration
来自随 E 增长的扫描；固定成本 ≈ num_experts × ~60ns。** dg 的 MGroupedContiguous 无此扫描（O(1)/block）→
不卡。**与 work 量、CTA 数、pipeline 深度、L2 命中率均无关**（L2 是分母 artifact，见下）。

## 真实 shape（config.json 实锤）

hidden=2048 / E=256 / topk=8 / inter=512。decode m_per_expert≈1（M=1 token × topk=8 over 256 expert）。
GEMM1(gate+up) N=1024 K=2048；GEMM2(down) N=2048 K=512。

## kernel 对象确认（G3，NCU demangled + DG_PRINT_CONFIGS 运行时）

- cute = `SM120BlockScaledBuilder<128,8,128,4,32,GemmType=5(ZeroPadding),SwapAB=1>`，grid=[188,1,1]（persistent num_sms），regs/thread=55
- dg = MGroupedContiguous（swap_ab=0），8 real token **padding 到 M=512**（block_m=64 对齐）；
  GEMM1 block 64×64 stages=5；GEMM2 block 64×128 stages=3；num_waves=1；regs/thread=168

## NCU 数据（launch:0=GEMM1, launch:1=GEMM2；两 GEMM 一致）

| metric | cute GEMM1 | cute GEMM2 | dg GEMM1 | dg GEMM2 |
|---|---|---|---|---|
| duration | 33.2 μs | 30.1 μs | 15.5 μs | 9.6 μs |
| GPC cycles | 76163 | — | 37321 | — |
| **long_scoreboard 占比** | **50.6%** | **52%** | **12.3%** | ~12% |
| wait | 27.2% | 27% | 12.2% | 11% |
| sleeping | 3.2% | — | 48.7% | 55% |
| **L2 hit rate** | **10.2%** | — | **46.0%** | — |
| SM 吞吐 / L1tex 吞吐 | 20.8% / 8.0% | 23.1% / 8.3% | 15.5% / 16.3% | 12.1% / 11.8% |
| achieved occupancy | 20.4% | — | 18.7% | — |

**宏观对上**：cute 33.2+30.1=63μs ≈ nsys 67μs；dg 15.5+9.6=25μs ≈ nsys 28μs → 固定成本是
kernel-internal（NCU 可见），非 host launch overhead。

## 证据链解读

1. 两者都低 occupancy(~20%)、低吞吐（计算 ~20% / 内存 ~8-16%）= 都 latency-bound（小 M 预期）。
2. **区别在等什么**：cute 50-52% long_scoreboard（等长延迟 load）；dg 12% long_scoreboard、49-55% sleeping
   （warp 干完活空闲，不等内存）。
3. **cute 等的是哪条 load**：不是 A/B/SF 数据 load，而是 scheduler 的 token_offset cumsum load ——
   每 CTA 10 warp × E group × 2 依赖 LDG，LSU sectors 随 E 线性（31K/242K/964K @ E=8/64/256，6KD exp_30
   SASS 逐条对上）。L2 hit 10% 是分母 artifact（cute stream-once 权重、总请求少），非 miss 惩罚。

## 被数据/代码推翻的假设（不再持有）

- ~~cute launch 更多 CTA~~：两侧 grid 均 [188,1,1] persistent（NCU 实测）。
- ~~cute 慢因 padding 行浪费~~：dg padding 到 512 行 > cute exact-小 M，dg 反而快 → 非 work 量。
- ~~dg 深流水(≤16)藏延迟~~：dg 实际 stages=5/3（DG_PRINT_CONFIGS），与 cute 4 差异小 → 非 stage 深度。

## 根因归纳（SASS 已闭环，6KD exp_30 lineinfo build）

- **fact**：cute 小 M 慢 = ZeroPadding scheduler token_offset 串行线性扫描 O(E)，LSU sectors 随 E 线性；
  E=8 时与 contiguous 打平 → GemmType 零代价；固定成本 ≈ num_experts × ~60ns。
- **SASS 闭环**：相邻 `LDG.E [R18.64]/[+0x4]` 三 call site，385,536×2 + 48,192×2×2 = 963,840 = LSU
  sector 数；385,536 = 48,192×8 → 每 CTA 10 个 warp 各自串行扫全部 E group（8 个 consumer warp 冗余重复）。
- **修正早期归因**：task_07 原判 "L2 局部性"、及中途 "ragged-scatter 访存 pattern" 均误判；三臂 DRAM 流量
  相同（≈compulsory 权重字节），L2 hit% 差异只是冗余 L2 请求数不同的分母 artifact，非因果、非优化目标。

## 验证（pad-8 隔离，同节点 2u2g-spr-0095）

- pad-8 + MGroupedContiguous（保 swapAB 128×8 tile 不变，仅换掉 ZeroPadding 的 O(E) 扫描）→ GEMM1
  33.2→**14.9μs（-55%）追平 dg**，long_scoreboard 49%→16%，**L2 仍 9%**。同 tile 只换 scheduler、
  L2 不变而 duration 追平 → 坐实杠杆是 O(E) 扫描、非 L2。
- pad-8/MGroupedContiguous 无扫描：`num_m_blocks=ceil(m/BlockM)` 算一次 + 每 block `grouped_layout[
  m_block*BlockM]>=0` 的 O(1) 检查（`scheduler.cuh:138-142`）。
- DG runtime identity（exp_30 对齐）= vLLM vendored `deep_gemm` 2.5.0 = deepseek-ai/DeepGEMM `nv-dev`
  `a6b593d`；MoE（m-grouped contiguous/masked）路径无 SwapAB（SwapAB 仅 dense m≤16 + BMM）。
- 详 `6KD_fp8_block_scale/.claude/moe_gemm/experiments/exp_30_zeropad_scheduler_linear_scan/result.md`。
  kernel 侧修复（SMEM 预载 token_offset）交后续（另 agent）。

## 可优化方向（针对 O(E) 扫描根因）

目标 = 消除随 E 线性增长的 token_offset 串行扫描开销（非 L2、非 pipeline 深度）：
1. **SMEM 预载 token_offset**（E+1 个 int32 ≤ 1KB）：把 per-iteration 延迟从 L2/DRAM 依赖链降到 SMEM，
   一次 coalesced 预载覆盖执行扫描的全部 warp（exp_30 sub-task 2 候选）。
2. decode 小 M 走 MGroupedContiguous（pad-8）替代 ZeroPadding：结构性消除 O(E) 扫描（本 result 已验，
   同 tile 追平 dg）——但涉及 token 侧 padding，与 zero-padding 红线的权衡由 kernel 侧决策。
> 原「更深 pipeline / 提 L2 局部性」方向基于已更正的 L2 归因，作废。

## 测试条件

| 项 | 值 |
|---|---|
| 硬件 | 1×RTX PRO 6000 Blackwell server edition（sm120，188 SM） |
| UT | `moe_microbench.py`（真实 35B dims，decode M=1，单 backend/进程，warmup 3 + steady 40） |
| NCU | bundled ncu 2026.2.1（`--set full`）+ `--kernel-name regex` 锁 MoE kernel + `--launch-skip 12 --launch-count 4`（跳 warmup 抓 steady gemm1/gemm2） |
| 分析 | veloq 0.2.2 `ncu inspect / warp-stalls --by reason` |
| 环境踩坑 | ncu/nsys 缺 OPENSSL_3.3.0 → bundled `target/linux-desktop-glibc_2_11_3-x64/ncu` + LD_LIBRARY_PATH；`--kernel-name-base` 指定名字类型不是 pattern（pattern 给 `--kernel-name`）；ts inline bash -c 引号嵌套坏 → 用脚本文件 |

## 产出
2 个 .ncu-rep + microbench/capture 脚本在 results/ 与 tests/（.ncu-rep gitignore，不入 git）。
