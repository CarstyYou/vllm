# vLLM cute_sm120_precision_internal 任务工作流

## 适用范围

`cute_sm120_precision_internal` 分支上的全部工作：FI cute FP8/MXFP8 moe_gemm 接入 vLLM +
e2e 精度/性能对比。母 plan（目标、决策、sub-task 总表）在父 repo
`$MEGA_ROOT/.claude/work_plans/e2e_precision_plan.md`（`.claude/current_plan` 指向它），
本文只管 vLLM 侧的执行结构，原则沿用 flashinfer 集成 workflow
（`$MEGA_ROOT/flashinfer/.claude/rules/workflow.md`），差异点在此文列出。

## 任务目录结构（与 FI 一致）

```
vllm/.claude/tasks/task_NN/
├── plan.md                  # 该 task 的 scope + sub-task 表 + ## Results
├── tests/                   # 对拍脚本 / eval YAML / bench 脚本
└── results/                 # correctness / eval / bench 产出
vllm/.claude/tasks/memory/findings.md   # shared findings（客观发现，不记主观决策）
```

Task 编号与母 plan sub-task 对应：

| Task | 对应母 plan | 内容 |
|---|---|---|
| task_01 | 1.x | 环境 + Qwen3.5-35B triton/deep_gemm baseline（GSM8K+MMLU 数字落表） |
| task_02 | 2.x | `CuteFp8Experts` + oracle 注册 + 单层对拍 + 路径 2 e2e |
| task_03 | 3.x | Q5 lock 后：`CuteMxfp8Experts` + requant + 路径 3 e2e |
| task_04 | 4.x | 单卡全矩阵汇总 + 多卡（397B / DSv4-Flash-Base） |
| task_05 | 5.x | e2e perf（精度达标后） |

## 与 FI workflow 的差异点

1. **环境**：无 Docker/JIT，遵循 vLLM AGENTS.md——一律 `uv venv` + `.venv/bin/python`，
   源码装 `uv pip install -e .`（改 C++ 才需全量编译；纯 Python 改动用
   `VLLM_USE_PRECOMPILED=1`）。FI 用我们 fork 的分支 editable install 进同一 venv。
2. **GPU 资源**：走 `gpu` skill / `ssh-gw`（cluster `sc`）；单卡任务不占 8 卡节点；
   allocation 默认 24h，`--purpose` 必填。
3. **正确性 gate 分两级**：
   - 单层对拍：改写 `tests/kernels/moe/test_deepgemm.py::run_single_case` 模板，
     新 experts 类 vs Triton，`calc_diff < 1e-3`
   - e2e：`tests/evals/gsm8k`（1319 题 5-shot temp=0 seed=42）+ lm-eval MMLU；
     per-backend YAML 收进 task dir 的 `tests/`
4. **上游隔离**：不给 vLLM 提 PR、不 push upstream；AGENTS.md 的贡献政策不触发，
   但其编码风格约定（docstring/行宽/lint）在改 vLLM 源码时仍遵守，便于 diff 干净。
5. **改动面纪律**：vLLM 源码改动限接入所需（oracle 三处注册 + experts 新文件 +
   quant_config 分支）；不动既有 backend 行为；三路径切换只靠 `--moe-backend`。

## 不变的部分（照 FI workflow 执行，不复制）

- plan 先 lock + xiy review；一次一个 sub-task，gate 过了才进下一步
- 每轮 code 改动后 subagent review 门禁（Phase 3.5）
- findings 沉淀到 shared `tasks/memory/findings.md`（`## task_NN` section）
- Git mutation 规则照 `$MEGA_ROOT/flashinfer/.claude/rules/git.md` 执行，
  scope 标签用 `[task_NN]`；分支固定 `cute_sm120_precision_internal`；不主动 commit/push
- 改完默认 stop unstaged 等 xiy review

## 关键 paths

| Symbol | Path |
|---|---|
| `$MEGA_ROOT` | `/home/scratch.xiy_gpu/mega_inference` |
| `$VLLM_ROOT` | `$MEGA_ROOT/vllm`（分支 `cute_sm120_precision_internal`，基 `dcf4072`；调查期 file:line 引用基于 `93e3bc8`，行号可能微漂） |
| `$FI_ROOT` | `$MEGA_ROOT/flashinfer`（分支 `sm120_moe_gemm_fp8_internal`） |
| 母 plan | `$MEGA_ROOT/.claude/work_plans/e2e_precision_plan.md` |
| DG experts 模板 | `$VLLM_ROOT/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py` |
| oracle 注册点 | `$VLLM_ROOT/vllm/model_executor/layers/fused_moe/oracle/fp8.py` |
| 对拍模板 | `$VLLM_ROOT/tests/kernels/moe/test_deepgemm.py` |
| eval 设施 | `$VLLM_ROOT/tests/evals/gsm8k/` |
