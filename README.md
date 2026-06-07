# 复杂指令多轮对话评测系统

面向"履约数字人外呼"场景的对话模型自动评测工具：将给定的复杂任务指令自动拆解成可机器评估的结构化规格，驱动多种用户画像的模拟器与被测对话模型多轮交互，并通过"确定性规则 + 多采样 LLM Judge + 人工校准集"产出可解释、可量化的评测报告。

## 1. 主要能力

- **指令解析**：从 Excel 抽取每条任务指令，输出结构化 `InstructionSpec`（节点图 / FAQ / 硬软约束 / 终止策略），同时保留原始 Markdown 副本。
- **场景矩阵**：把指令拆成 8 类用户场景（配合、犹豫、抗拒、追问 FAQ、越权、打断、忙碌/开车、诱导违规），每个场景声明应覆盖的目标节点与目标约束。
- **双 LLM 编排**：SUT ↔ UserSim 多轮对话；trace 落盘并支持重放评测。
- **8 维评分**：GSR / PCR / BCA / KAR / HCR / SCR / TER / ROB（详见 `docs/metrics.md`）。规则维度纯代码计算，语义维度走多采样 LLM Judge + 多数投票 + 置信度。
- **可解释报告**：HTML（雷达图 + 失败归因 + 证据高亮）与 Markdown，每条扣分都附带 turn id、原文证据、规则与 rationale。
- **多入口**：CLI、Streamlit Web UI、`run_store` SQLite 元数据库 + 重放/对比 CLI。

## 2. 目录结构

```
data/
  source/               原始 Excel
  instructions/         解析后的 InstructionSpec JSON + Markdown
  calibration/          人工标注样例
src/
  core/                 解析、配置、Schema、LLM 客户端、编排、SUT、用户模拟器
  evaluators/           规则与 LLM Judge、投票、聚合、校准
  report/               Jinja2 模板与 HTML/Markdown 渲染
  cli/                  parse_instructions / eval_run / rejudge / compare / meta_eval
  webui/                Streamlit 入口
configs/                YAML 配置（模型、权重、画像）
reports/                每次 run 的报告输出
traces/                 每次 run 的对话 trace
docs/                   设计、指标、校准、使用文档
tests/                  fixtures + 单测
```

## 3. 快速开始（以 DeepSeek 为例）

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 配置 LLM（默认接 DeepSeek，详见下方第 4 节）
cp .env.example .env
# 在 .env 里填上 DEEPSEEK_API_KEY；BASE URL 与模型已预填

# 3) 验证连通性（推荐第一次先跑一次 ping）
python -m src.cli.ping

# 4) 解析指令（首次或文件变化后执行）
python -m src.cli.parse_instructions --mode llm

# 5) 跑一次完整评测（指令 × 场景矩阵）
python -m src.cli.eval_run --sut llm --user-sim llm --judge llm \
       --workers 4 --run-id run_demo

# 6) 查看报告
start reports/run_demo/report.html

# 7) 启动 Web UI
streamlit run src/webui/app.py
```

无 API key 时也可全程离线跑通：

```bash
python -m src.cli.parse_instructions --mode offline
python -m src.cli.eval_run --sut stub --user-sim stub --judge offline \
       --workers 1 --run-id stub_demo
```

stub/offline 模式用启发式规则代替 LLM，便于 CI、回归与方案演示。

## 4. 接入 DeepSeek API

DeepSeek API 与 OpenAI 协议兼容，只需配置 base URL 与 API Key，本系统全部模块开箱即用。参考 [DeepSeek 官方文档](https://api-docs.deepseek.com/zh-cn/) 与 [模型价格页](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)。

### 4.1 拿到 Key 与填 `.env`

```bash
# 控制台申请：https://platform.deepseek.com/api_keys
cp .env.example .env
```

`.env` 关键字段（已默认指向 DeepSeek）：

```
DEEPSEEK_API_KEY=sk-...          # 必填
OPENAI_BASE_URL=https://api.deepseek.com
SUT_MODEL=deepseek-v4-flash      # 便宜，跑 SUT/UserSim
USER_SIM_MODEL=deepseek-v4-flash
JUDGE_MODEL=deepseek-v4-flash    # 默认 flash 全流程；要更稳的 Judge 改 deepseek-v4-pro
```

`DEEPSEEK_API_KEY` 和 `OPENAI_API_KEY` 任意一个都行；两者都存在时优先 `OPENAI_API_KEY`。

### 4.2 模型选择

| 模型 | 适用 | 说明 |
| --- | --- | --- |
| `deepseek-v4-flash` | **默认全流程**（SUT / UserSim / Judge / 解析器） | 输入 1 元 / 输出 2 元 每 1M tokens；速度快、成本最低 |
| `deepseek-v4-pro` | Judge 升级（可选） | 输入 3 元 / 输出 6 元（2.5 折优惠期，截止 2026/05/31）；判定一致性更高，开启方式：`--judge-model deepseek-v4-pro`，可同时配合 `judge.thinking: true` 进入推理模式 |
| `deepseek-chat` / `deepseek-reasoner` | 兼容旧脚本 | 2026/07/24 弃用，不建议新建 run 使用 |

### 4.3 开启 thinking 模式（可选，仅 DeepSeek v4）

在 `configs/default.yaml` 把 `judge.thinking` 设为 `true`，Judge 会自动以 `thinking={"type":"enabled"}` 调 v4-pro 的推理模式，多采样一致性更高，代价是延迟和 token 都会上升。

```yaml
judge:
  samples: 3
  temperature: 0.2
  thinking: true            # 开启推理模式
  reasoning_effort: high    # low / medium / high
```

ping 命令也支持手动测试：

```bash
python -m src.cli.ping --model deepseek-v4-pro --thinking
```

### 4.4 成本与节流

- 评测结果会经过磁盘缓存（`data/.llm_cache`），同 prompt 命中不会再花钱；
- `--judge-samples` 控制多采样次数，3 已经够稳；
- 入门可以先跑 `--instructions 1 --scenarios cooperative,busy_driving` 控制规模；
- `--reuse-traces` 在迭代 Judge 时跳过对话生成，省 60% 以上成本。

### 4.5 切换到其它 OpenAI 兼容服务

把 `.env` 里 `OPENAI_BASE_URL` 改成对应服务的 endpoint 即可（OpenAI、Qwen DashScope、Moonshot、智谱、本地 vLLM/Ollama 均可），无需改代码。

### 4.6 并行与性能调优

评测在 case 级 / case 内 judge 维度 / 维度内多采样三层都做了线程级并行，再叠加跨 client 共享的 in-flight 信号量保护：

| 旋钮 | 默认 | CLI / 配置 | 说明 |
| --- | --- | --- | --- |
| `case_workers` | 8 | `--workers` | 同时跑的 case 数；DeepSeek-v4-pro 账户上限 500 并发，远高于此值 |
| `judge_workers` | 4 | `--judge-workers` | 单 case 内 4 个语义维度（GSR/PCR+BCA/SCR/KAR）并行 |
| `samples_workers` | auto = `min(samples, 4)` | `--samples-workers` | 单维度内多采样投票并行的 worker 数 |
| `max_in_flight` | 32 | `--max-in-flight`（0 表示关闭） | 跨 SUT/UserSim/Judge 三个 LLMClient 共享的全局并发上限，避免 case×judge×samples 同时打 API 触发 429 |
| 磁盘缓存 | 开启（`data/.llm_cache`） | `runtime.cache_dir` | 同 prompt 命中直接返回，重复 run 几乎零成本 |
| `--reuse-traces` | 关 | CLI flag | 复用已有 trace 只跑 judge，迭代 Judge 时省 60%+ 成本 |

**典型加速**（2 指令 × 8 场景 = 16 case，单次 LLM ~5s）：

- 旧默认（workers=4 / judge_workers=1 / samples 串行）：~9 分钟
- 新默认（workers=8 / judge_workers=4 / samples 自动并行）：~2.5 分钟
- 缓存命中（重复 run）：~30 秒

**调参建议**：

```bash
# 高配额 / 自有专享配额：拉到极限
python -m src.cli.eval_run --workers 16 --judge-workers 8 --max-in-flight 64

# 限频严格 / 共享配额：降并发避免 429
python -m src.cli.eval_run --workers 2 --judge-workers 1 --max-in-flight 8

# 不并行（调试用）
python -m src.cli.eval_run --workers 1 --judge-workers 1 --samples-workers 1
```

注意：
- 单 case 内 dialogue 阶段的 SUT↔User 多轮天然顺序，**无法并行**，是固有下限。
- judge 阶段（4 维 × N 采样）才是真正可压缩的部分。
- 如果你看到大量 `RateLimitError` 走 retry，把 `max_in_flight` 调小，或 `case_workers` 减半。

#### 503 / 429 风暴排查

DeepSeek 等 OpenAI 兼容服务在高峰期会主动返回 `503 Service Temporarily Unavailable` 或 `429 Rate Limit Reached` 让客户端退避——**这是服务端过载保护，不是配置错误**。我们已经做了：

- httpx / openai 的请求级 INFO 日志默认压到 WARNING，避免刷屏（详见 `src/core/llm_client.py`）
- tenacity 指数退避，初始 2s，最大 30s，3 次重试
- 全局 `max_in_flight` 信号量统一总并发上限

如果仍然观察到大量 5xx：

1. 切到「低并发」预设：`--workers 2 --judge-workers 1 --max-in-flight 8`
2. 等几分钟服务端缓解后再试
3. 在 `.env` 调高 `LLM_REQUEST_TIMEOUT`（默认 60）和 `LLM_MAX_RETRIES`（默认 3）

## 5. 关键概念

| 概念 | 说明 |
| --- | --- |
| `InstructionSpec` | 解析后的指令规格，含 flow_nodes/edges、FAQ、约束、变量占位符 |
| `ScenarioSpec` | 描述用户在某种场景下的目标和行为；声明 `target_nodes` / `target_constraints` |
| `DialogueTrace` | 一次对话的完整流水：每轮 prompt、回复、字数、模型、延迟 |
| `DimensionScore` | 单个维度评分；`details` 中每个 `ScoreDetail` 都附带 turn 引用、证据、扣分、置信度 |
| `CaseReport` / `RunReport` | 单 case 与整 run 的聚合结果，含权重、置信区间、失败归因 |

## 6. 测试与 CI

```bash
# 顺序运行
pytest

# 多核并行（pytest-xdist），CI 推荐
pytest -n auto
```

测试全部使用 `FakeJudgeClient` 等离线桩，不依赖网络与 API Key。

## 7. 文档导航

- `docs/design.md`：架构详设与数据流。
- `docs/metrics.md`：指标定义、权重与计算公式。
- `docs/calibration.md`：人工标注规范与一致率验证流程。
- `docs/usage.md`：CLI / Web UI / 配置参数详解（含 DeepSeek 接入示例）。

## 8. 验收对照

- ✅ 用户模拟器覆盖主流程、分支、FAQ、越权、打断、忙碌/开车、诱导违规等场景。
- ✅ 评测过程可解释：每条扣分携带 turn id、原文证据、违规规则、rationale。
- ✅ 评测结果可量化：8 维分数 + 加权总分 + bootstrap 置信区间。
- ✅ 评测结果可靠：LLM Judge 多采样投票 + JSON Schema 校验 + 校准集 Cohen's kappa。
- ✅ 可复现：每次 run 记录模型、温度、seed、变量、prompt 版本，trace 完整持久化。
