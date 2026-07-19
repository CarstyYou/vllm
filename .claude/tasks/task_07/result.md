# task_07 Result — dg vs cute 小 M MoE 实现差异（根因 = L2 局部性 / memory latency）

日期 2026-07-17。分支 `cute_sm120_precision_internal`。承接 task_06（cc=1 decode cute MoE 67μs/call
vs dg 28μs）。方法：真实 35B MoE shape 的 decode microbench（UT）+ NCU（ncu 2026.2.1，`--set full`，
kernel-name regex 锁 MoE kernel）+ veloq 分析。**全程代码 file:line + NCU 数据支撑，fact/hypothesis 分栏。**

## 结论

**cute swapAB kernel 在小 M（decode）是 memory-latency-bound：L2 局部性差（hit 10% vs dg 46%）→
A/B/SF load miss 到 DRAM → warp 50% 时间卡 long_scoreboard → 2× cycle。dg 不卡内存（long_scoreboard
仅 12%，主要 sleeping=idle）。根因是 cute 的访存结构/局部性，不是 work 量、不是 CTA 数、不是 pipeline 深度。**

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
3. **为何 cute 等 load**：L2 hit 10.2% vs dg 46.0% —— cute 的 load 90% miss L2 打 DRAM，全延迟，
   4 stage + 20% occupancy 藏不住 → 堆在 long_scoreboard。

## 被数据/代码推翻的假设（不再持有）

- ~~cute launch 更多 CTA~~：两侧 grid 均 [188,1,1] persistent（NCU 实测）。
- ~~cute 慢因 padding 行浪费~~：dg padding 到 512 行 > cute exact-小 M，dg 反而快 → 非 work 量。
- ~~dg 深流水(≤16)藏延迟~~：dg 实际 stages=5/3（DG_PRINT_CONFIGS），与 cute 4 差异小 → 非 stage 深度。

## 根因归纳（fact + 一条 hypothesis）

- **fact**：cute 小 M memory-latency-bound（long_scoreboard 50%+、L2 hit 10%）；dg 不是（12%、L2 46%）。
- **hypothesis（代码 axis F 支撑，未 SASS 定死）**：cute 的 zero-padding exact-packing（token 侧无 padding=
  核心优化优势，`GemmType=ZeroPadding`）在 decode 把 token 按 expert 分散 + SF 4-row 对齐 → 访存分散、
  L2 复用差；dg 的 contiguous padded 布局访存连续、L2 复用好。**即：zero-padding 省了 padded 计算，
  却牺牲了小 M 的访存局部性。** SASS/source-line 定死需 FI kernel 加 `-lineinfo` 重编（本 exp 未做，
  report 无 source 相关计数）。

## 后续验证（同节点 pad-8 复现，证实 hypothesis + 修正 L2 因果地位）

同节点（2u2g-spr-0095）三 kernel 重现 + pad-8 隔离实测（pad-8 + MGroupedContiguous，保 swapAB
128×8 tile 不变，仅 ZeroPadding→contiguous 布局）：

- pad-8 contiguous GEMM1 = **14.9 μs**，比 ZeroPadding 33.2 μs 快 **-55%**，追平 dg 15.6 μs；
  long_scoreboard **49%→16%**（翻成 dg 型 sleeping-bound）。
- **但 L2 hit 几乎没动（9.4% vs 原 10.2%，未及 dg 46%）**。两种 L2 区间（dg 46% / pad8 9%）
  达到同样 ~15 μs。
- **修正**：上 fact 里 "L2 hit 10%" 是**伴随相关量、非因果杠杆**；真正杠杆 = **逃离 long_scoreboard**
  （zero-padding ragged-scatter → per-expert 连续访存，降延迟停顿）。**根因 = 访存 pattern（hypothesis 已证实），
  非 L2 hit 率本身。** task_07 测量数据本身准确。kernel 侧 pad-8 dispatch 优化交后续。

## 可优化方向（数据支撑）

目标 = 降 cute 小 M 的 long_scoreboard（藏住 L2-miss 延迟 / 提局部性），非降 padded work（cute 已最省）：
1. 提高小 M load 的延迟隐藏：更深 pipeline stage / 更激进 prefetch（cute 当前固定 4 stage）
2. 改善访存局部性：decode 小 M 的 A/SF 布局提高 L2 复用，或对照 dg contiguous 的连续访存
3. swapAB 128×8 tile 在小 M 的访存模式复核（B 权重 load 是否 thrash L2）

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
