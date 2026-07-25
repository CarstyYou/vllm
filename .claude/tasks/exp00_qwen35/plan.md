# exp_00 Qwen3.5 vLLM 三臂 Serving Benchmark

> 状态：xiy 已确认三臂、双平台、vLLM E2E comparison boundary；plan review 已完成并按 review 修正。

## Goal

使用 Qwen3.5-35B-A3B-FP8 单卡 vLLM online serving，对比：

- CuTe FP8 `(1,128,128)`：`cute_sm120_fp8`
- CuTe MXFP8 GranK=32：`cute_sm120_mxfp8_32`
- DeepGEMM MXFP8 GranK=32：`deep_gemm_mxfp8_32`

在 RTX PRO 5000 Blackwell（5K Pro）和 RTX PRO 6000 Blackwell（6K Pro）分别回答
`ISL/OSL=8000/1000`、`cc={1,4,8,16,32,64,128}` 下三种 MoE backend 的逐点端到端吞吐关系。

## Comparison boundary

- 类型：vLLM E2E online serving，不解释为独立 MoE GEMM kernel benchmark。
- 模型：Qwen3.5-35B-A3B-FP8，TP=1；三臂使用同一 checkpoint。
- FP8 臂保留 checkpoint 的 `(1,128,128)` block-scale contract；两个 MXFP8 臂均从同一
  checkpoint 重编码为 UE8M0 GranK=32。
- 唯一允许变化的是 MoE backend 及其必需的 DeepGEMM E8M0 环境开关。模型、GPU、容器、
  vLLM revision、6KD revision、请求集、seed、serve 参数和 GDN backend 在同一平台内固定。
- 5KP 与 6KP 分开报告，不跨硬件聚合 ratio，不用旧实验数据补点。
- 不修改 production，不运行 NCU/PerfSim。

vLLM backend mapping 的 production evidence：

```python
mapping = {
    "cute_sm120_fp8": Fp8MoeBackend.CUTE_SM120_FP8,
    "cute_sm120_mxfp8_32": Fp8MoeBackend.CUTE_SM120_MXFP8_32,
    "deep_gemm_mxfp8_32": Fp8MoeBackend.DEEP_GEMM_MXFP8_32,
}
```

MXFP8 weight conversion 的 production evidence：

```python
if fp8_backend == Fp8MoeBackend.CUTE_SM120_MXFP8_32:
    w13, w13_scale = requant_weight_for_cute_mxfp8(w13, w13_scale, 32)
if fp8_backend == Fp8MoeBackend.DEEP_GEMM_MXFP8_32:
    w13, w13_scale = requant_weight_for_dg_mxfp8_32(w13, w13_scale)
```

## Setup identity

- 6KD canonical source：`/home/scratch.xiy_gpu/mega_inference/6KD_fp8_block_scale` 当前
  `mxfp8_vs_fp8@76920f3` production；使用 `git archive` 固定 source identity 和 archive SHA，再由 canonical
  `.release/scripts/sync_to_flashinfer.py` 同步到独立 FlashInfer prefix。
- vLLM runtime source 固定为 `82d415a026d37d0278956ca92b509a0c5a6bf5a2`；其 `vllm/` tree ID
  `0e29ab9f08eb3cdfe746d269a1e434339c06b395` 与本地 docs HEAD `b13fae8f8` 的 runtime tree 完全一致。
  现有 PPT 与 task docs 的 dirty state 不进入 runtime source identity，也不得修改；记录 runtime module path
  与 `vllm/` tree ID。
- FlashInfer base revision、submodule revisions、同步文件 manifest 与生成 extension 的完整 JIT workspace
  都固定并记录。CuTe extension 只构建一次；5KP/6KP 必须加载同一 immutable ELF，server maps 与
  SHA256 完全一致。
- DeepGEMM：使用该 vLLM revision vendored source；记录版本、source SHA 和运行时 module path。
- DeepGEMM 可以按平台独立 JIT，但必须固定相同 source/config，记录生成 cache、module path 与 binary SHA。
- 两平台使用同一 `nvcr.io/nvidia/pytorch:26.05-py3` image digest、CUDA/PyTorch/vLLM toolchain；
  每个平台使用独立 run ID 与 output root，不复用旧 task binary/cache。
- 每次 server cold start 记录 extension `__file__`、ELF SHA256 和 server `/proc/<pid>/maps`，
  并从 serve log 校验实际 backend dispatch。
- 模型使用完整 manifest hash，而非只记录 `config.json`；记录 model path、config 与 tokenizer identity。
- 5KP：RTX PRO 5000 Blackwell，110 SM；通过 frontend `10.6.131.67` direct SSH 到空闲 5KP，
  Python/build/benchmark 只在容器内执行；记录 hostname、GPU UUID/PCI、显存、driver、clock/power/temperature。
- 6KP：RTX PRO 6000 Blackwell server edition，188 SM；通过 6KD execution profile 的 Slurm
  allocation 和容器执行；记录相同硬件字段。

## Serving workload

- `ISL=8000`
- `OSL=1000`
- `cc={1,4,8,16,32,64,128}`
- dataset：random
- `--ignore-eos --seed 42 --temperature 0`
- `num_prompts=max(4*cc,16)`
- 每个 cc 固化 request manifest 并记录 SHA256；三 backend、三轮、同一平台复用同一 manifest。
- serve：
  - `--max-model-len 10240`
  - `--no-enable-prefix-caching`
  - `--tensor-parallel-size 1`
  - `--max-num-seqs 128`
  - `--gdn-prefill-backend=triton`
- 每臂从净化环境启动；显式 unset/set DeepGEMM E8M0/TMA/JIT 相关变量。保存 resolved engine config，
  去除 backend 名称及其必需环境项后，三臂必须逐字段一致；固定并核对 GPU memory utilization、
  chunked prefill、CUDA graph 与 max-num-batched-tokens 等自动配置。

## Measurement protocol

- 每个 backend/round 独立 cold-start server；正式测量前逐 cc warmup。
- 三轮使用平衡顺序，确保每个 backend 各处于 first/middle/last 一次：
  - r1：FP8 → CuTe MXFP8 → DeepGEMM MXFP8
  - r2：CuTe MXFP8 → DeepGEMM MXFP8 → FP8
  - r3：DeepGEMM MXFP8 → FP8 → CuTe MXFP8
- CC 顺序也按 round 轮换，三臂在同一 round 使用完全相同顺序：
  - r1：`1,4,8,16,32,64,128`
  - r2：`32,64,128,1,4,8,16`
  - r3：`128,1,4,8,16,32,64`
- 每次正式 case 前记录无 compute apps、temperature、graphics/SM clock、power；超出同平台门槛则等待。
- server ready 后先完成所有可能的 JIT/warmup；正式测量窗口出现 compile/JIT 日志则该 arm 无效。
- 每个平台、每 backend、每 cc 最终取 r1/r2/r3 median。
- 主指标：output throughput（tok/s）。
- 补充指标：total throughput、request throughput、TTFT、TPOT、ITL。
- 每个平台逐 cc 展示三臂吞吐、CuTe FP8/CuTe MXFP8/DG MXFP8 的 pairwise ratio、领先方与轮间波动；
  不用 geomean 代替逐点结果。

## Correctness and evidence gates

- 正式采集前，当前 6KD source 的 FP8/MXFP8 correctness regressions 必须通过；每个平台再用同一
  extension 完成三臂 serving smoke。任一 gate 失败则不采集该平台性能。
- 正式结果必须恰好包含 `2 platforms × 3 backends × 7 cc × 3 rounds = 126` 个 JSON；
  每个平台 63 个 raw rows、21 个 median rows。
- 唯一 key `(platform,backend,cc,round)` 必须精确等于上述笛卡尔积，无缺失、重复或额外 key；
  不能只用总行数通过门禁。
- 每个 JSON 必须满足：
  - `completed == num_prompts`
  - `failed == 0`
  - input/output token 数与 workload 一致
  - throughput/latency metrics finite
- `max_concurrency` 配置必须等于目标 cc；vLLM 的 observed `max_concurrent_requests` 允许因客户端
  overlap 大于 cc，但必须落在 `[cc,num_prompts]`，不把它误当配置值。
- serve log 必须分别命中 `CUTE_SM120_FP8`、`CUTE_SM120_MXFP8_32` 和
  `DEEP_GEMM_MXFP8_32` backend identity。
- source/archive、extension ELF、server maps、DeepGEMM module 和 dispatch evidence 必须闭环；
  只改 CLI label 不能通过 identity gate。
- 同一平台三臂三轮的 request manifest SHA256 必须逐 cc 一致。
- manifest 由固定 generator 生成 JSONL prompt/token payload；正式命令使用 custom dataset、
  `--disable-shuffle --skip-chat-template` 回放。每个 JSON 的实际 input/output token totals 必须与
  manifest 和 `num_prompts` 精确匹配。
- 每次正式采集前 GPU 必须 idle；异常占用、OOM、server crash 或 schema mismatch 只记录失败，
  不用缺失点形成性能结论。

## Artifacts

正式结果保存在本实验目录：

- `results/5kp/raw/{cute_fp8,cute_mxfp8_32,dg_mxfp8_32}/`
- `results/6kp/raw/{cute_fp8,cute_mxfp8_32,dg_mxfp8_32}/`
- `results/benchmark_rounds.csv`：126 个 raw round rows
- `results/benchmark_medians.csv`：42 个 per-platform/backend/cc median rows
- `results/benchmark.csv`：14 个 per-platform/cc matched rows，横向展示三臂吞吐和 pairwise ratio
- `results/run_meta.txt`：硬件、容器、commit、binary、backend identity 与完成门禁
- `result.md`：high-level 结论、5KP/6KP 分表、测试条件和局限

## Decision and stop condition

- 只有 126/126 正式 JSON、三臂 dispatch/binary identity、manifest、schema 和进程清理全部通过，
  才宣称实验完成。
- 若单点差异与三轮波动同量级，只报告观察值，不扩大为稳定性能差异。
- 任一平台无法获得独占目标 GPU 时，保留另一平台的有效数据但整体实验状态为 incomplete，不用单平台
  代替双平台目标。
- 完成或中止后清理本任务 server、benchmark、build 和辅助进程，不遗留 zombie/orphan。

## Plan Review

**Date**: 2026-07-23

**Reviewer**: subagent

**Verdict**: BLOCK，按同一轮 review 的最小修正实施，不扩大 case matrix。

**Blocking gaps and accepted corrections**:

- source/runtime identity 不完整：补齐 canonical archive、FlashInfer base/submodules、container/toolchain、
  vLLM/runtime/model/hardware identity；CuTe extension 一次构建并在两平台校验同一 ELF SHA。
- 三臂环境可能不一致：每次冷启净化环境，并对 resolved engine config 做 backend-excluded equality gate。
- fixed CC 顺序、JIT 和硬件状态可能污染结果：按 round 轮换 CC，正式窗口禁止 JIT，逐 case 记录
  idle/temperature/clock/power。
- 126 计数可能被重复 key 冒充：校验完整笛卡尔积唯一性，并定义 manifest 生成、hash、回放及 token gate。
- completed requests 不能替代当前 source correctness：正式采集前通过 FP8/MXFP8 regression 和逐平台 serving smoke。
