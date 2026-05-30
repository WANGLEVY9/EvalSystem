# 多轮对话评测系统 v3.0

> 针对履约数字人外呼场景的**复杂指令遵循能力**自动评估系统  
> 三大核心: **可解释**（证据三元组） · **可量化**（Self-Consistency 置信区间） · **可复算**（seed + version）

📖 **新用户从这里开始** → [QUICKSTART.md](./QUICKSTART.md)（5 分钟跑出第一份报告）  
📊 **示例报告** → [examples/](./examples/)（实际产出的 HTML / PDF / Markdown 报告）  
🏆 **技术细节** → [DELIVERABLE.md](./DELIVERABLE.md)（赛事评委摘要 + 架构图）

---

## ✨ 八大创新点

| # | 创新 | 解决什么 |
|---|---|---|
| **C1** | **混合评估架构** — 规则 (RuleEvaluator) + LLM Judge 协作 | 字数/禁词等用规则确定性判定，避免 LLM 对客观项误判和成本浪费 |
| **C2** | **证据三元组** (turn_id + 原文 quote + reasoning) | 每个评分都能精确追溯到对话原文，HTML 可点击高亮 |
| **C3** | **Self-Consistency 评估** — 多次评估取中位数 + 标准差 | 量化评估的可靠性，标准差>15 自动降置信度 |
| **C4** | **指令驱动的状态感知用户模拟器** — 12 画像 + 5 维向量 + 状态机 + 人格弧 + 红队 L1/L2/L3 | 真正模拟出多样化用户行为，逼出模型真实短板 |
| **C5** | **强制分支覆盖测试** — 自动列举所有 (节点, 条件) 二元组并并发跑会话 | 量化"评测了多少种情况"，未覆盖分支高亮 |
| **C6** | **模型短板诊断** — Top-N 失败模式聚类 + 改进建议 | 把"评测系统"变成"分析诊断工具" |
| **C7** | **校准回测** — 8 个人工标注 case + MAE / Pearson-r 自动评测 | 学术级证明 evaluator 与人工标注一致性 |
| **C8** | **Web Dashboard + PDF + 模型对比** | 评委友好的多形态交付 |

---

## 🚀 快速开始

> 完整 5 分钟教程见 [QUICKSTART.md](./QUICKSTART.md)。下面是浓缩版：

### 环境要求

- **Python ≥ 3.10**（推荐 3.12+，3.9 会因类型注解语法报错）
- 网络可访问 `api.deepseek.com`

### 一键安装

```bash
git clone https://github.com/WANGLEVY9/EvalSystem.git
cd EvalSystem
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (可选) PDF 导出依赖系统库:
brew install pango glib cairo gdk-pixbuf libffi   # macOS
sudo apt-get install libpango-1.0-0 libpangoft2-1.0-0  # Linux

# 配置 DeepSeek API Key (https://platform.deepseek.com/)
export DEEPSEEK_API_KEY="sk-..."
# 或 cp .env.example .env  并编辑

# 跑 demo (无需 API)
python main.py --demo
```

### 三种使用方式

#### 1️⃣ 命令行（最快）

```bash
export DEEPSEEK_API_KEY="sk-..."

# Demo（无需 API）
python main.py --demo

# 单任务快速评测
python main.py -i config/sample_instructions/rider_feimaotui.md \
    --personas cooperative,busy,red_team_l1 --concurrency 3 --seed 42

# 完整模式: 12 画像 × 3 + 分支覆盖 + Self-Consistency=3
python main.py -i config/sample_instructions/course_live_upgrade.md --full

# 同时导出 PDF
python main.py -i task.md --personas cooperative --pdf

# Excel 批量
# Excel 批量（如有 xlsx 指令文件）
python main.py --excel "your_instructions.xlsx"
```

#### 2️⃣ Web Dashboard（评委友好）

```bash
python -m src.web_dashboard
# 浏览器打开 http://127.0.0.1:8765/
```

Dashboard 功能：
- 📂 历史报告浏览（HTML/JSON/MD/PDF 一键打开）
- ▶ 一键发起新评测（画像可视化选择 + 实时日志流）
- 🔬 一键启动校准回测
- ⚙️ 任务状态实时刷新

#### 3️⃣ 模型对比

```bash
python -m src.model_comparison \
    -i config/sample_instructions/rider_feimaotui.md \
    --models deepseek-chat,deepseek-reasoner \
    --personas cooperative,busy,red_team_l2 --seed 42
```

#### 4️⃣ 校准回测（学术级可靠性证明）

```bash
python tests/run_calibration.py                       # SC=1
python tests/run_calibration.py --self-consistency 3  # SC=3 (推荐)
```

输出 MAE / Pearson-r / 容差内比例 + HTML 可视化报告。

---

## 🏗 项目结构

```
eval_system/
├── README.md                                # 本文件
├── QUICKSTART.md                            # ⭐ 5 分钟新人上手
├── DELIVERABLE.md                           # 技术交付摘要 (评委)
├── LICENSE                                  # MIT
├── .env.example                             # API Key 模板
├── requirements.txt
├── main.py                                  # CLI 主入口
├── config/
│   ├── default_config.yaml                  # 默认配置 (8 维 + 12 画像)
│   └── sample_instructions/
│       ├── rider_feimaotui.md               # xlsx 第一条 (已脱敏)
│       ├── rider_feimaotui_enhanced.md      # 用于对比的强化版
│       ├── course_live_upgrade.md           # xlsx 第二条 (复杂度 100, 已脱敏)
│       └── delivery_notification.json
├── examples/                                # ⭐ 实测代表性报告
│   ├── sample_report_full_mode.{html,md,pdf}
│   ├── sample_calibration_report.html
│   └── sample_comparison_report.html
├── src/
│   ├── models.py                            # ⭐ Pydantic 数据模型 (含 EvidenceQuote, RunMetadata)
│   ├── instruction_parser.py                # 4 段独立 LLM 抽取
│   ├── instruction_compiler.py              # ⭐ 复杂度 + DAG + 约束分类
│   ├── dialog_state_tracker.py              # ⭐ 状态机
│   ├── user_simulator.py                    # ⭐ 12 画像 × 5 维向量 × 红队
│   ├── dialogue_engine.py                   # 并发 + 强制分支覆盖
│   ├── target_model.py                      # 被测模型适配层
│   ├── rule_evaluator.py                    # ⭐ 确定性规则评估
│   ├── llm_judge.py                         # ⭐ LLM-as-Judge + Self-Consistency
│   ├── evaluator.py                         # 评估器总编排
│   ├── weakness_analyzer.py                 # ⭐ 短板诊断
│   ├── report_generator.py                  # HTML/JSON/MD 报告
│   ├── pdf_exporter.py                      # ⭐ PDF 导出 (SVG 替代 Canvas)
│   ├── model_comparison.py                  # 多模型对比
│   ├── web_dashboard.py                     # ⭐ FastAPI Dashboard
│   └── llm_cache.py                         # LLM 调用缓存
├── tests/
│   ├── test_integration.py                  # 集成测试 (无 API)
│   ├── calibration_set.json                 # ⭐ 8 个人工标注 case
│   └── run_calibration.py                   # ⭐ 评估器可靠性回测
└── output/                                  # ⭐ 实测产出归档 (15+ 报告, 见 output/README.md)
```

---

## 📐 评估架构

### 8 维评分

| # | 维度 | 权重 | 判定方式 | 说明 |
|---|---|---|---|---|
| 1 | 任务完成度 | 20% | LLM Judge | 检查点完成度 |
| 2 | 流程遵循度 | 20% | LLM Judge | 流程 + 分支正确性 |
| 3 | 约束遵守度 | 15% | **混合** | 规则 + 语义协作 |
| 4 | 异常处理能力 | 10% | LLM Judge | 异常画像下的应对 |
| 5 | 对话效率 | 5% | **启发式 + LLM** | 轮次 + LLM 融合 |
| 6 | 话术质量 | 5% | LLM Judge | 语言专业度 |
| 7 | 知识准确度 | 15% | LLM Judge | 知识引用 + 反幻觉 |
| 8 | 话术简洁度 | 10% | **规则** | 字数/禁词/未替换变量 |

### 12 用户画像

```
配合型 / 犹豫型 / 抗拒型 / 跑题型 / 矛盾型 / 边界型 / 忙碌型 / 困惑型 / 急躁型
红队 L1 (轻度刺探) / 红队 L2 (社工诱导) / 红队 L3 (prompt 注入)
```

每个画像由 **5 维行为向量** 驱动:
```python
PersonaVector(cooperation=0.95, patience=0.7, consistency=0.95,
              emotion=0.85, focus=0.85, knowledge_curiosity=0.4, boundary_test=0.0)
```

---

## 📊 报告产物（4 种格式）

| 文件 | 用途 |
|---|---|
| `output/report_<ts>.html` | **交互式可视化** — 雷达图 / 热力图 / 可点击对话浏览器 / quote 高亮 |
| `output/report_<ts>.pdf` | **打印友好** — A4 排版, SVG 图表, 评委友好 |
| `output/report_<ts>.json` | **完整结构化数据** — 程序化处理 |
| `output/report_<ts>.md` | **1 页摘要** — 评审快速浏览 |

每个评分都标注 **判定方式标签**（📏 规则 / 🧠 LLM / 🔀 混合 / 🔧 启发式）+ 标准差 + 置信度 + 复算元数据。

---

## 🎯 实测成果

### 飞毛腿任务（3 画像 mini）

```bash
python main.py -i config/sample_instructions/rider_feimaotui.md \
    --personas cooperative,busy,red_team_l1 --concurrency 3 --seed 42
```

| 指标 | 值 |
|---|---|
| 复杂度 | 100/100 (very_complex) |
| 总分 | 73.1 ± 10.2（置信 0.95） |
| 用时 | 87 秒 |
| 知识准确度 | 100（无幻觉） |
| 关键发现 | 模型在「说明排名机制 / 减少拒单 / 恶劣天气优势」3 个检查点 3/3 都漏说 |

### 课程升级任务（含分支覆盖）

```bash
python main.py -i config/sample_instructions/course_live_upgrade.md \
    --personas cooperative,busy --concurrency 4 --seed 42 --branch-test
```

| 指标 | 值 |
|---|---|
| 复杂度 | 100/100 (very_complex)（13 节点 / 14 分支 / 7 约束 / 8 知识） |
| 总分 | 82.0 ± 6.6（置信 0.94） |
| 会话数 | 16（2 画像 + 14 分支强制覆盖） |
| 用时 | 7.4 分钟 |
| **🌳 分支覆盖** | **14/14 = 100%** |

### 校准回测（评估器可靠性证明）

```bash
python tests/run_calibration.py
```

8 个人工标注 case 实测：
- **总分 MAE: 4.72**（目标 <10）✅
- **Pearson-r: 0.967**（目标 >0.85）✅
- **容差内比例: 100% (8/8)**（目标 ≥75%）✅
- 用时: 198 秒

→ 见 `examples/sample_calibration_report.html`

---

## ⚙️ 配置

详见 `config/default_config.yaml`，常用调优：

```yaml
evaluation:
  self_consistency: 3        # 关键维度多次评估 (推荐)
  dimensions:
    task_completion: { weight: 0.20 }
    # ...

dialogue:
  max_turns: 18
  concurrency: 8             # 并发会话数
  enable_branch_testing: true

report:
  generate_html: true
  generate_pdf: false        # 默认关, 命令行 --pdf 开启
```

---

## 🧪 测试

```bash
# 集成测试 (无 API)
python tests/test_integration.py

# 校准回测 (需 API)
python tests/run_calibration.py
```

---

## 🔬 可复算 (Reproducibility)

每个 report 都包含 `RunMetadata`:
```json
{
  "run_id": "8d4f...",
  "evaluator_version": "3.0.0",
  "target_model_id": "deepseek-chat",
  "judge_model_id": "deepseek-chat",
  "seed": 42,
  "self_consistency_n": 3,
  "concurrency": 6,
  "duration_seconds": 445
}
```

只要设定相同 `seed` + 相同模型版本 + 相同 evaluator_version, 多次评测结果可重现。

---

## 📝 License

[MIT](./LICENSE) © 2026 WANGLEVY9

---

## 🤝 贡献 / 反馈

- 仓库: https://github.com/WANGLEVY9/EvalSystem
- 提 issue / PR 都欢迎

---

## 📚 相关文档

| 文档 | 用途 |
|---|---|
| [QUICKSTART.md](./QUICKSTART.md) | 5 分钟新人上手 |
| [README.md](./README.md) | 本文件，完整说明 |
| [DELIVERABLE.md](./DELIVERABLE.md) | 技术交付摘要（评委友好） |
| [examples/README.md](./examples/README.md) | 示例报告说明 |
| `config/default_config.yaml` | 配置参数详解（注释丰富） |
