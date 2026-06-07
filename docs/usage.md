# 使用说明

## 1. 安装

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
cp .env.example .env  # 然后填入 DEEPSEEK_API_KEY 或 OPENAI_API_KEY
```

### 1.1 DeepSeek API（默认）

`.env.example` 已经按 DeepSeek 配置好。最少只需要填一项：

```
DEEPSEEK_API_KEY=sk-...
```

`OPENAI_BASE_URL` 默认是 `https://api.deepseek.com`（与官方文档一致）。模型默认值：

- `SUT_MODEL=deepseek-v4-flash`
- `USER_SIM_MODEL=deepseek-v4-flash`
- `JUDGE_MODEL=deepseek-v4-pro`

第一次跑之前建议执行：

```bash
python -m src.cli.ping            # 简单问候
python -m src.cli.ping --json     # 验证 JSON Output 模式
python -m src.cli.ping --thinking # 验证 v4-pro 的 thinking 模式（可选）
```

成功时会输出 base URL、模型、响应延迟、token 数和一段回复。

### 1.2 切其它 OpenAI 兼容服务

把 `.env` 中 `OPENAI_BASE_URL` 改成对应 endpoint（如 `https://api.openai.com/v1`、`https://dashscope.aliyuncs.com/compatible-mode/v1`、本地 `http://localhost:11434/v1` 等）即可，无需改代码。模型名也同步替换为该服务支持的型号。

## 2. 配置

- `configs/default.yaml`：模型、温度、最大轮数、judge 采样数、维度权重、bootstrap 等。
- `configs/personas.yaml`：内置 8 种用户画像模板及变量池。
- `configs/metrics.yaml`：指标体系定义（与 `default.yaml > metrics.weights` 配套）。
- `.env`：API key、base URL、默认模型名；本系统通过 `python-dotenv` 自动加载。

## 3. CLI 命令一览

| 命令 | 用途 |
| --- | --- |
| `python -m src.cli.ping` | 一键测试 DeepSeek / OpenAI 兼容端点连通性 |
| `python -m src.cli.parse_instructions` | 解析 Excel 为 InstructionSpec JSON |
| `python -m src.cli.eval_run` | 端到端跑评测：生成 trace + 评估 + 报告 |
| `python -m src.cli.rejudge` | 对已有 trace 文件重新跑评估器 |
| `python -m src.cli.compare` | 比较两个 run 的总分与维度差异 |
| `python -m src.cli.make_calibration_template` | 从 run_report 生成待人工填写的校准样本模板 |
| `python -m src.cli.meta_eval` | 人工标注一致率（accuracy / Cohen's kappa） |

### 3.1 `parse_instructions`

```bash
python -m src.cli.parse_instructions \
    --input data/source/命题二：外呼任务对话模型指令示例.xlsx \
    --output data/instructions \
    --mode llm  # offline 时不调用 LLM
```

### 3.2 `eval_run`

```bash
python -m src.cli.eval_run \
    --instructions all --scenarios all \
    --sut llm --user-sim llm --judge llm \
    --workers 4 --judge-samples 3 \
    --run-id run_demo
```

关键参数：

- `--sut/--user-sim/--judge`：每端可独立选 `llm` 或 `stub/offline`，便于成本控制；
- `--reuse-traces`：跳过对话生成，直接对已有 trace 重新评估；
- `--max-turns`：单 case 最大轮数；
- `--seed`：固定随机种子，保证场景与变量复现。

输出：`reports/<run_id>/{run_report.json,report.html,report.md}` 与 `traces/<run_id>/<case_id>.json`。同时写入 `reports/runs.sqlite3`。

### 3.3 `rejudge`

```bash
python -m src.cli.rejudge --trace traces/run_demo/1__cooperative.json --judge llm
```

适用于"对话已经跑完，但想换 prompt / 模型 / 权重重新评估"的情形。

### 3.4 `compare`

```bash
python -m src.cli.compare --run-a reports/run_a --run-b reports/run_b
```

打印总分差异和按维度均值的 Δ 表。

### 3.5 `meta_eval`

生成待标注模板：

```bash
python -m src.cli.make_calibration_template --run reports/run_demo --limit 20
```

```bash
python -m src.cli.meta_eval --run reports/run_demo
```

读取 `data/calibration/*.json`（人工标签），与 `run_report.json` 对照，输出每个维度的 accuracy 和 Cohen's kappa。

## 4. Web UI

```bash
streamlit run src/webui/app.py
```

页面：

1. **指令管理**：上传/解析 Excel、查看 InstructionSpec JSON。
2. **场景与画像**：查看 ScenarioSpec、覆盖矩阵、模拟器 prompt 片段。
3. **评测运行**：勾选指令×场景，配置 SUT/UserSim/Judge 模式，启动评测。
4. **报告中心**：浏览历史 run，查看汇总数据，下载 HTML/Markdown。
5. **Trace 复盘**：逐轮查看 SUT 与 USER 的回复。
6. **A/B 对比**：选两个 run，比较总分与维度均值差异。

## 5. 离线模式

当 `OPENAI_API_KEY` 未配置或不希望产生费用时：

```bash
python -m src.cli.parse_instructions --mode offline
python -m src.cli.eval_run --sut stub --user-sim stub --judge offline --workers 1
```

stub 与 offline 模式会用启发式规则替代 LLM，分数会偏低，但流程完整可运行，适合调试报告与提交演示。

## 6. 常见问题

- **解析失败 / 字段缺失**：检查 Excel 是否使用了 `# Role:`、`# Conversation Flow:`、`# Constraints:` 等支持的标题；不支持的标题会被忽略。
- **judge 输出格式不正确**：会自动重试 3 次；仍失败会回退到该 judge 的离线 fallback。
- **成本控制**：将 `--judge-samples` 设为 1，或对部分维度使用 offline，先看趋势再放量。
- **结果不稳定**：把 `--seed` 设为同一值；prompt 与模型一致时，缓存命中后输出确定。

## 7. 项目结构提示

- 所有写盘行为都集中在 `core/run_store.py`、`report/renderer.py`、`orchestrator.save_trace`、`instruction_parser.parse_workbook`，便于审计。
- 不要把 `data/.llm_cache/` 和 `traces/` 目录提交进版本控制（`.gitignore` 已忽略）。
- 校准样本可提交，作为模型质量回归基线的一部分。
