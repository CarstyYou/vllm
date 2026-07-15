# task_04 — v2 条件对齐 + AIME/MBPP + 多卡 TP 精度矩阵

对应母 plan 4.0-4.2。分支 `cute_sm120_precision_internal`。
状态：**plan 待 xiy review；sub-task 1 已排队（设计在母 plan 4.0 已 lock）**。

## 目标

1. serve config v2（--enable-prefix-caching）下重跑 triton / dg-UE8M0 / cute_sm120_fp8
   三列 GSM8K+MMLU，与 task_03 的 3a/3b 对齐条件 → 单卡 5 列可横比
2. AIME + MBPP 加入矩阵 → 单卡 5 列 × 4 评测集
3. 多卡 TP：Qwen3.5-397B-A17B-FP8（8 卡 TP=8）+ DeepSeek-V4-Flash-Base（4–8 卡），
   5 列 × 4 评测（并行策略只用 TP，xiy 2026-07-13 定）

## Sub-task 0 结论（工具链调查，2026-07-14）

- lm-eval 0.4.12 内置 task：`aime`（=aime24 alias 待确认）/ `aime24` / `aime25`、
  `mbpp` / `mbpp_plus`（另有 instruct 变体，本任务用 base completion 版）
- **aime25**：`math-ai/aime25`，30 题，0-shot，greedy，`max_gen_toks=32768`，exact_match
  → **与 v2 serve `--max-model-len 4096` 冲突**（见 Decision Q-A）
- **mbpp**：`google-research-datasets/mbpp` full，test 500 题，3-shot 内置，
  greedy，pass@1，生成代码在 lm-eval 客户端进程执行 → CLI 需 `--confirm_run_unsafe_code`
  （yaml `unsafe_code: true`）；生成短，4096 ctx 够
- 权重确认（节点实测）：Qwen3.5-397B-A17B-FP8 380G、DeepSeek-V4-Flash-Base 276G 均在
  `/home/scratch.trt_llm_data/llm-models/`
- 8 卡 6K Pro server-edition 摸底（2026-07-14 时点）：可见 6 台，0 idle
  （2 allocated / 2 mixed / 2 drain*）——申请时用 ssh-gw 现查

## Decisions（待 xiy lock）

| Q | 问题 | 候选 | 状态 |
|---|---|---|---|
| Q-A | AIME serve ctx：max_gen_toks=32768 > 4096 | **已 lock（2026-07-14）：方案 ①** AIME 专用 serve `--max-model-len 36864`，5 列统一；其余评测仍 4096 | 已 lock |
| Q-B | AIME 范围 | **已 lock（2026-07-14）：方案 ①** aime24+aime25，一次 lm_eval 双 task 同 serve 实例 | 已 lock |
| Q-C | 多卡两模型是否都跑 4 评测全矩阵（8 卡占用时长） | **默认方案 ① 全矩阵**（母 plan 4.2 "重复 4.1" 原文即全矩阵；goal 模式按此执行，xiy 可随时裁剪）；DSv4-Flash TP 取 8（整台节点已占，KV 余量大，与 397B 同 TP 减少变量） | 默认执行 |

## Sub-tasks

| # | 内容 | Gate |
|---|---|---|
| 0 | 工具链调查 | ✓ 本文 |
| 1 | v2 三列重跑：GSM8K 1319 + MMLU 全量 × triton/deep_gemm/cute_sm120_fp8（ts 42-47，排在 task_03 e2e 链后） | 6 个数字落表 + evidence 行 |
| 2 | AIME 脚本（Q-A/Q-B lock 后）+ 5 列 runs | 数字落表 |
| 3 | MBPP 脚本（--confirm_run_unsafe_code）+ 5 列 runs | 数字落表 |
| 4 | 单卡汇总：5 列 × 4 评测矩阵 → result.md | 全量数字，条件表完整 |
| 5 | 多卡：8 卡 allocation → 397B TP=8 + DSv4-Flash TP=4/8，5 列 ×（Q-C 定的评测集） | 多卡矩阵落表 |
| 6 | H20 precision baseline（母 plan 4.3）：H20 单卡 35B，vLLM 原生 FP8 × 4 评测（v2 同参数；serve log 证据行记实际 oracle 落点）；环境 = H20 节点新建 venv（vLLM 同 commit，无 FI 依赖） | H20 行落 precision.md |
| 7 | **DG granK=32 backend**（xiy 2026-07-14 撤销 Q5 "DG 不跑 3b"）：`deep_gemm_mxfp8_32` —— weight 32×32 UE8M0 requant + act 1×32 quant + `recipe=(1,1,32)`，镜像 CuteMxfp8 模式；单层 parity + 单卡 4 评测（多卡列随余量） | parity < 1e-3 + 数字落表 |
| 8 | **异常 cell 补测**（xiy 2026-07-14 定纪律：异常数据必须补测复现或实验证明成因才入 result.md）：① 397B mxfp8_32 AIME24 0.200 重跑+log_samples ② 397B MBPP 坍塌单列重跑+log_samples 看生成内容 ③ 397B triton GSM8K 稳定性重跑 ④ 35B dg_mxfp8_32 GSM8K 重跑（TP1）；35B dg 已有 v1/v2 双复现免；H20 UE8M0 参考行标注不入结论 | 补测结果落 results/verify/，结论注入 result.md |
| 9 | findings 沉淀 + 收口 | xiy 收口确认 → commit |

## Risks

| 风险 | 预案 |
|---|---|
| AIME 32k 生成 × 30 题时长（长 CoT 慢） | num_concurrent 并发跑；单 run timeout 放宽 |
| aime/aime24/aime25 数据集需 HF 下载 | mmlu 先例证明节点可达 HF；失败则预下载到 scratch |
| MBPP 代码执行在节点侧 | lm-eval 官方沙箱行为，隔离节点可接受 |
| 8 卡节点竞争（摸底 0 idle） | ssh-gw 轮询 + 拿到即跑；qos=batch 24h 整台 |
| DSv4 `deepseek_v4` config 在 base dcf4072 的解析 | serve 起不来则先单独 smoke，问题上报 xiy |
| 397B TP8 显存：380G 权重 / 768G 总显存 | KV cache 预算需实测；max-model-len 视 AIME 决策 |

## Results

（待填；数据落 `results/`，汇总进 result.md）
