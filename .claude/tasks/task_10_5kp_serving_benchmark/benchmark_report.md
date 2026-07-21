# task_10 — 5KP E2E Serving Benchmark：CuTe MXFP8_32 vs DeepGEMM MXFP8_32（3-round median）

**类型**：端到端（E2E）vLLM online serving 吞吐/时延对比，**非** kernel microbenchmark。
**硬件**：NVIDIA RTX PRO 5000 Blackwell（5kp，客户硬件），host 10.6.142.16 GPU2（`GPU-ab3d387a`，CC 12.0）。
**模型**：Qwen3.5-35B-A3B-FP8，TP=1，quant=fp8。
**唯一变量**：MoE GEMM backend（`--moe-backend`）。模型/权重/量化/容器/GPU/请求集（seed=42）/serve 参数全部一致。
**轮次**：**3-round median（r1/r2/r3）**。轮次协议：r1 首轮 sweep（每 cc：warmup→r1）；r2/r3 随后补充 sweep（每 cc：warmup2→r2→r3），同 GPU/容器/配置；warmup/warmup2 不计入统计。
**负载**：`random` dataset，ISL=8000 / OSL=1000，`--ignore-eos --seed 42`，per-cc num_prompts=max(4·cc,16)。

## 结论（3 轮 median，全部 7 点三轮领先方一致）

1. **crossover 在 cc=16→32 且三轮稳定**：小并发（cc≤16）DeepGEMM 领先，大并发（cc≥32）CuTe 反超；每个 cc 的三轮领先方向完全一致（21 组 per-round 对比无一翻转）。
2. **低并发区间** cc=1 CuTe 落后最多 −25.8%（out tok/s 122.14 vs 164.69）；gap 随并发收窄（−25.8% → −10.4% → −6.6% → −2.6%）。
3. **高并发区间**（cc≥32）CuTe 领先 +1.4% ~ +2.0%（E2E 口径；幅度小于 task_09 kernel 层差异，可能因 attention/GDN/KV 等非 MoE 部分占比稀释——此为推断，未做归因测量）。
4. **重复性好，结论表述保守**：per-point 3 轮吞吐波动（(max−min)/median）≤1.6%，多数点 ≤0.8%。高并发段 +1.4%~+2.0% 的领先幅度与单端/轮间波动同量级，因此只宣称**稳定的 1–2% 趋势**（每点三轮方向一致），不宣称统计显著。
5. **42/42 门禁通过**：每点 completed==num_prompts、failed==0、token 总数精确、metrics finite、max_concurrent_requests∈[cc,np]；backend identity 两端实证命中。

## 主对比表（E2E output throughput，tok/s，3-round median）

| cc | CuTe med | DG med | CuTe/DG | 差异% | 领先 | 波动% C/D | per-round ratio (r1/r2/r3) | 三轮稳定 |
|----|---------|--------|---------|-------|------|-----------|---------------------------|----------|
| 1 | 122.14 | 164.69 | 0.742 | −25.8% | DG | 0.0/0.1 | 0.742 / 0.742 / 0.742 | 稳定 |
| 4 | 332.50 | 371.10 | 0.896 | −10.4% | DG | 0.6/0.3 | 0.899 / 0.897 / 0.892 | 稳定 |
| 8 | 485.70 | 519.96 | 0.934 | −6.6% | DG | 1.6/1.5 | 0.932 / 0.944 / 0.943 | 稳定 |
| 16 | 673.97 | 692.31 | 0.974 | −2.6% | DG | 1.3/0.3 | 0.983 / 0.973 / 0.974 | 稳定 |
| 32 | 859.08 | 842.08 | 1.020 | **+2.0%** | **CuTe** | 1.5/0.6 | 1.023 / 1.012 / 1.031 | 稳定 |
| 64 | 1043.73 | 1029.31 | 1.014 | **+1.4%** | **CuTe** | 1.2/0.8 | 1.006 / 1.023 / 1.018 | 稳定 |
| 128 | 1082.60 | 1063.82 | 1.018 | **+1.8%** | **CuTe** | 0.0/0.6 | 1.018 / 1.013 / 1.019 | 稳定 |

（ratio = median(CuTe)/median(DG)；差异% = ratio−1，正=CuTe 快；波动% = (max−min)/median per backend；per-round ratio = 同轮 CuTe/DG。与 r1 单轮相比结论不变，cc=16/64 的 gap 微调：−1.7%→−2.6%、+0.6%→+1.4%。）

## 总吞吐 + 时延（3-round median，per-cc）

| cc | backend | tot_tput tok/s | TTFT med (ms) | TPOT med (ms) | ITL med (ms) | max_conc(med) |
|----|---------|----------------|---------------|---------------|--------------|---------------|
| 1 | CuTe | 1099.22 | 290 | 7.90 | 7.91 | 2 |
| 1 | DG | 1482.18 | 292 | 5.78 | 5.79 | 2 |
| 4 | CuTe | 2992.53 | 849 | 11.1 | 10.9 | 8 |
| 4 | DG | 3339.88 | 849 | 9.9 | 9.7 | 8 |
| 8 | CuTe | 4371.32 | 1121 | 15.5 | 14.3 | 12 |
| 8 | DG | 4679.63 | 1130 | 14.3 | 13.2 | 14 |
| 16 | CuTe | 6065.74 | 1163 | 22.6 | 19.5 | 21 |
| 16 | DG | 6230.83 | 1174 | 22.0 | 18.9 | 22 |
| 32 | CuTe | 7731.68 | 992 | 36.8 | 27.9 | 36 |
| 32 | DG | 7578.70 | 1000 | 36.8 | 28.6 | 36 |
| 64 | CuTe | 9393.60 | 1088 | 60.7 | 41.6 | 68 |
| 64 | DG | 9263.75 | 1095 | 61.3 | 42.4 | 68 |
| 128 | CuTe | 9743.42 | 7311 | 104.5 | 62.6 | 132 |
| 128 | DG | 9574.42 | 6929 | 107.1 | 64.2 | 132 |

（每个指标独立取 3 轮 median；完整 mean/p99 与 per-round 42 行见 `benchmark.csv`。cc=128 使用 512 prompts、客户端并发 128，TTFT ≈7s 与排队现象同时出现；未做因果拆分。）

## 测试条件

| 项 | 值 |
|----|----|
| 硬件 | RTX PRO 5000 Blackwell（5kp），host 10.6.142.16（R6KD-CX8aaS-GPU-16），GPU2 `GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522`，CC 12.0，绑定 via `CUDA_VISIBLE_DEVICES=<UUID>`（容器内 visible_device_count=1，cap=(12,0)） |
| 容器 | enroot `fi-ci-cu130`（torch 2.11.0+cu130，CUDA 13.0.1）；python=`/opt/conda/envs/py312/bin/python` |
| vLLM | fork `cute_sm120_precision_internal`，v0.1.dev1+g82d415a02（r1 与 r2/r3 同版本，serve log 实证） |
| FlashInfer overlay | db7abc0a（`grouped_mm.moe_gemm_mxfp8_nt_groupwise`），PYTHONPATH 先于 venv 0.6.13 |
| DeepGEMM | a6b593d（=vLLM vendored 2.5.0）；DG arm 设 `VLLM_USE_DEEP_GEMM_E8M0=1` |
| serve | `--max-model-len 10240 --no-enable-prefix-caching --tensor-parallel-size 1 --max-num-seqs 128 --moe-backend=<BK> --gdn-prefill-backend=triton` |
| bench | `vllm bench serve --dataset-name random --random-input-len 8000 --random-output-len 1000 --num-prompts max(4cc,16) --max-concurrency <cc> --ignore-eos --seed 42 --percentile-metrics ttft,tpot,itl` |
| 轮次 | r1：warmup→r1（`bench_5kp.sh`）；r2/r3：warmup2→r2→r3（`bench_5kp_r23.sh`，同 server 会话内连跑）；两 sweep 各自冷启动 server |

## 完成门禁（3-round）

- r23 sweep 退出 rc=0（两 backend ALL_DONE rc=0）；r1 文件 md5 与 snapshot 一致（`R1_INTACT`，零覆盖）。
- **formal JSON = 42**（2 backend × 7 cc × 3 rounds，精确文件名匹配，无重复/未知/缺失）；`CLOSURE_GATE=PASS` 42/42（容器内 `validate_closure_3r.py`）。
- 每点强 schema：7 计数字段 strict int；completed==num_prompts=max(4cc,16)；failed==0；max_concurrency==cc；total_input_tokens==np·8000；total_output_tokens==np·1000；13 项 metrics finite（throughput>0）；max_concurrent_requests∈[cc,np]。
- backend identity：CuTe=`CUTE_SM120_MXFP8_32`、DG=`DEEP_GEMM_MXFP8_32`（fp8.py:368，r1 与 r23 各自 serve log 实证）。
- 聚合独立复核：PPT 脚本对 CSV 42 raw + 14 median 共 56 行逐行核对硬件/GPU/模型/配置/commit 身份后才渲染。

## 局限

- E2E serving 指标包含 attention/GDN/KV 等非 MoE 部分，MoE GEMM backend 差异被稀释为推断解释（未做归因测量）；纯 MoE GEMM kernel 收益见 task_09（3-round median，CuTe 1.07×~1.31×）。
- r2/r3 与 r1 分属两次 server 冷启动：round 间波动包含冷启动/session 差异，其单独影响未做隔离估计。
- cc≥32 的 +1.4%~+2.0% 领先与两端波动同量级，但方向三轮一致（21/21 无翻转），结论以"方向稳定 + 幅度 1~2%"表述为宜。

## Artifacts

- `benchmark.csv` — 42 per-round 行 + 14 median 行（含 tput_r_min/max、spread、ratio、crossover_stable）。
- `run_meta.txt` — 环境/绑定/nvidia-smi -L/门禁证据。
- `raw/bench_*_r{1,2,3}.json` — 42 formal JSON；formal stdout 仅 r2/r3 共 28 份（r1 无 stdout 留存）；`raw/bench_*_warmup2.{json,stdout}` — r23 warmup（各 14 份）。
- `raw/serve_*_r23.log`、`raw/evidence_*_r23.txt`、`raw/gpu_binding_*_r23.txt`、`raw/full_sweep_r23.log` — r23 证据链（r1 版本同名无后缀）。
- PPT：`vllm/.claude/tasks/cute_sm120_mxfp8.pptx` slide4（`pptx_build_3r.py` 生成，slide5 task_09 页内容校验未变）；渲染 `slides_render/page-{1..5}.png`。
