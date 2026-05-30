# 复杂指令下的多轮对话评测系统 — 技术交付摘要

> **赛题**：复杂指令下的多轮对话评测系统  
> **版本**：v3.0 (2026-05-30)  
> **核心理念**：可解释 · 可量化 · 可复算 · 可对比

---

## 一、技术亮点（赛事差异化卖点）

| # | 创新点 | 解决什么 | 实现要点 |
|---|---|---|---|
| **C1** | **混合评估架构** | LLM-as-Judge 对客观项易误判且费 token | `RuleEvaluator`（规则）+ `LLMJudge`（语义）协作分工，每个评分都标注 `evaluation_method` |
| **C2** | **证据三元组** | 评委关心的「为什么扣分」 | 每个评分必须返回 `EvidenceQuote(turn_id + 原文 quote + reasoning)`，且 quote 经校验是否真在原文（反幻觉） |
| **C3** | **Self-Consistency 评估** | LLM-Judge 容易飘 | `n_self_consistency=3` 多次评估取中位数 + 输出标准差。标准差>15 自动降置信度 |
| **C4** | **指令驱动的状态感知用户模拟器** | 静态画像无法覆盖真实用户行为 | 12 画像（含红队 L1/L2/L3）× 5 维 `PersonaVector` 行为向量 × `DialogStateTracker` 状态机 × **人格弧线** × seeded 复算 |
| **C5** | **强制分支覆盖测试** | 「评测了多少种情况」无法量化 | 自动列举所有 (节点, 条件) 二元组并并发跑专门会话 |
| **C6** | **模型短板诊断** | 评测应是"分析诊断工具" | `WeaknessAnalyzer` 自动聚类失败模式按 impact 排序 |
| **C7** | **校准回测** | 评估器自身可靠性需证明 | 8 case 人工标注 + 自动跑出 MAE / Pearson-r |
| **C8** | **多形态交付** | 不同评委不同口味 | HTML 交互 + PDF 打印 + Markdown 摘要 + JSON 程序化 + Web Dashboard |

---

## 二、与赛题要求的对照

| 赛题要求 | 我们的实现 | 实证 |
|---|---|---|
| 用户模拟器**充分有效**测试 | 12 画像 × 5 维向量 × 状态感知 × 红队分层 + 强制分支覆盖 | 课程任务 16 会话覆盖 14/14 分支（100%） |
| 评测过程**可解释** | 证据三元组（turn_id + 原文 quote + reasoning） | 报告中每个检查点都有 `quotes` 字段，HTML 可点击高亮 |
| 评测结果**可量化** | 8 维加权评分 + Self-Consistency 标准差 + 置信度 | full 模式总分 78.5 ± 7.4 (置信 0.93) |
| 评测结果**可靠** | 规则确定性 + LLM 多评取中位数 + quote 校验反幻觉 + **校准回测** | **MAE 4.72 / Pearson-r 0.967 / 8/8 case 全过** |
| **自动**评测报告 | 一行命令产出 HTML+JSON+MD+PDF 四种 | 课程升级任务报告 305KB / 305 个对话元素 / 12 个失败模式 |

---

## 三、xlsx 两条样本的端到端结果（v4-flash 被测）

### 任务 1：飞毛腿骑手通知（mini 模式 3 画像）

```bash
python main.py -i config/sample_instructions/rider_feimaotui.md \
    --personas cooperative,busy,red_team_l1 --concurrency 3 --seed 42
```

| 指标 | 值 |
|---|---|
| 复杂度 | 100/100 (very_complex)（LLM 解析后 14 检查点） |
| 总分 | 73.1 ± 10.2（置信 0.95） |
| 用时 | 87 秒 |
| 知识准确度 | 100（无幻觉） |
| 关键发现 | 「说明排名机制 / 减少拒单 / 恶劣天气优势」检查点 3/3 漏说 |

### 任务 2：课程升级（**full 模式 12 画像 + 分支覆盖 + Self-Consistency=3**）

```bash
python main.py -i config/sample_instructions/course_live_upgrade.md \
    --full --concurrency 6 --seed 42
```

| 指标 | 值 |
|---|---|
| 复杂度 | 100/100 (very_complex)（13 节点 / 14 分支 / 7 约束 / 8 知识） |
| 总分 | **78.5 ± 7.4（置信 0.93）** |
| 会话数 | **29**（15 normal + 14 branch） |
| 用时 | **26 分钟（6 路并发, 含 SC=3 评估）** |
| **🌳 分支覆盖** | **14/14 = 100%** |
| 关键发现 | 「检查学员端费用」**27 次**未完成（极稳定的真实短板） |
| Self-Consistency 价值 | 任务完成度 ±27.3 显示该维度评估存在不确定性，正确反映了语义判定的内在难度 |

### 校准回测结果（学术级可靠性证明）

```bash
python tests/run_calibration.py
```

| 指标 | 值 | 阈值 | 状态 |
|---|---|---|---|
| 总分 MAE | **4.72** | <10 | ✅ 通过 |
| Pearson-r | **0.967** | >0.85 | ✅ 通过 |
| 容差内比例 | **100% (8/8)** | ≥75% | ✅ 通过 |
| 用时 | 198 秒 | - | - |

**8 个人工标注 case 涵盖**：满分配合 / 严重违反字数 / 禁词 / critical 约束 / 知识幻觉 / 忙碌用户 / 抗拒用户 / 红队 prompt-injection。

### Prompt 工程对比（C8 应用）

```bash
python -m src.model_comparison \
    -i config/sample_instructions/rider_feimaotui.md \
    --models deepseek-chat \  # 或对比多个被测模型
```

| 配置 | 总分 | 关键差异 |
|---|---|---|
| 原版 prompt | 73.8 | 约束遵守度 53.3 |
| 强化版 prompt（更严的约束） | 59.2 | 约束遵守度 0.0 |

**评测系统揭示了一个反直觉的发现**：过度强化约束的 prompt 反而让模型「死板」，触发更多违反。这是工程师常踩的坑，评测系统能直接量化暴露。

---

## 四、架构图

```
                   ┌──────────────────────────────┐
                   │  TaskInstruction (md/json)   │
                   └──────────────┬───────────────┘
                                  ▼
                ┌──────────────────────────────────┐
                │  InstructionParser (4 段 LLM)    │
                │  • Checkpoints / Constraints     │
                │  • FlowNodes / Knowledge         │
                └──────────────┬───────────────────┘
                                  ▼
                ┌──────────────────────────────────┐
                │  InstructionCompiler              │
                │  • 复杂度量化                      │
                │  • 约束分类 (DET vs SEM)          │
                │  • 流程 DAG + 目标分支            │
                └──────────────┬───────────────────┘
                                  ▼
        ┌─────────────────────────┴────────────────────────────┐
        ▼                                                          ▼
┌──────────────────────┐                            ┌──────────────────────┐
│  DialogueEngine       │  并发                      │  ScenarioSimulator    │
│  • Concurrent         │ ◄────────────────────────  │  • 强制分支覆盖       │
│  • TerminationChain   │                            │                       │
└────────┬──────────────┘                            └──────────────────────┘
         │ for each session
         ▼
┌──────────────────────────────────┐
│  UserSimulator (v3 状态感知)      │
│  • 12 画像 × 5 维 PersonaVector  │
│  • DialogStateTracker            │
│  • 人格弧 / 场景注入             │
│  • 红队 L1/L2/L3                 │
└──────────────┬───────────────────┘
                ▼
┌──────────────────────┐    ┌──────────────────────┐
│  TargetModel (被测)   │ ──►│ DialogueSession      │
│  DeepSeek/OpenAI/自定义│    │ + DialogueTurn[]     │
└──────────────────────┘    └──────────┬───────────┘
                                          ▼
                ┌─────────────────────────────────────────┐
                │  Evaluator (总编排)                       │
                │  ┌────────────────┐  ┌────────────────┐ │
                │  │  RuleEvaluator │  │  LLMJudge      │ │
                │  │  • 字数/禁词   │  │  • 检查点       │ │
                │  │  • 未替换变量 │  │  • 流程         │ │
                │  └────────────────┘  │  • 语义约束     │ │
                │                      │  • Self-Consist │ │
                │                      │  • Quote 校验   │ │
                │                      └────────────────┘ │
                └──────────────┬──────────────────────────┘
                                ▼
                ┌──────────────────────────────────┐
                │  WeaknessAnalyzer                 │
                │  • 维度 Top-3 弱项               │
                │  • 失败模式聚类 (impact 排序)    │
                │  • 改进建议生成                  │
                └──────────────┬───────────────────┘
                                ▼
                ┌──────────────────────────────────┐
                │  ReportGenerator                  │
                │  ┌────────┬────────┬────────┬────────┐
                │  │ HTML   │ PDF    │ MD     │ JSON   │
                │  │雷达/热力│SVG排版 │1页摘要 │结构化   │
                │  └────────┴────────┴────────┴────────┘
                └──────────────────────────────────┘
                                ▲
                                │
                ┌──────────────────────────────────┐
                │  Web Dashboard (FastAPI)          │
                │  • 历史报告浏览                   │
                │  • 一键发起评测 (实时日志流)      │
                │  • 校准回测启动                   │
                └──────────────────────────────────┘
```

---

## 五、文件清单

```
eval_system/
├── README.md                                # 完整使用说明
├── QUICKSTART.md                            # 5 分钟新人上手
├── DELIVERABLE.md                           # 本文件 (评委摘要)
├── LICENSE                                  # MIT
├── requirements.txt
├── main.py                                  # CLI 主入口
├── config/
│   ├── default_config.yaml                  # 8 维度 + 12 画像 + 报告配置
│   └── sample_instructions/
│       ├── rider_feimaotui.md               # xlsx 第一条 (markdown 格式, 已脱敏)
│       ├── rider_feimaotui_enhanced.md      # 强化版 (用于对比)
│       ├── course_live_upgrade.md           # xlsx 第二条 (复杂度 100, 已脱敏)
│       └── delivery_notification.json       # JSON 格式示例
├── examples/                                # ⭐ 实测代表性报告
│   ├── sample_report_full_mode.html/md/pdf  # 完整 full 模式产出
│   ├── sample_calibration_report.html       # 校准回测
│   └── sample_comparison_report.html        # Prompt 对比
├── src/                                     (16 个模块)
│   ├── models.py                            # ⭐ Pydantic 数据模型
│   ├── instruction_parser.py                # 4 段 LLM 抽取
│   ├── instruction_compiler.py              # ⭐ 复杂度+DAG+约束分类
│   ├── dialog_state_tracker.py              # ⭐ 状态机
│   ├── user_simulator.py                    # ⭐ 12 画像 × 红队
│   ├── dialogue_engine.py                   # 并发 + 强制分支覆盖
│   ├── target_model.py                      # 被测模型适配
│   ├── rule_evaluator.py                    # ⭐ 规则评估
│   ├── llm_judge.py                         # ⭐ LLM Judge + SC
│   ├── evaluator.py                         # 评估总编排
│   ├── weakness_analyzer.py                 # ⭐ 短板诊断
│   ├── report_generator.py                  # HTML/JSON/MD 报告
│   ├── pdf_exporter.py                      # ⭐ PDF (SVG 替代 Canvas)
│   ├── model_comparison.py                  # ⭐ 多模型/Prompt 对比
│   ├── web_dashboard.py                     # ⭐ FastAPI Dashboard
│   └── llm_cache.py                         # LLM 缓存
├── tests/
│   ├── test_integration.py                  # 集成测试 (无 API)
│   ├── calibration_set.json                 # ⭐ 8 case 人工标注集
│   └── run_calibration.py                   # ⭐ 评估器可靠性回测
└── output/                                  # ⭐ 实测产出归档 (15+ 报告, 见 output/README.md)
```

---

## 六、运行手册（4 种入口）

### 1. CLI 命令行

```bash
# Demo (无 API)
python main.py --demo

# 集成测试 (无 API)
python tests/test_integration.py

# 真实评测 (需 DEEPSEEK_API_KEY)
export DEEPSEEK_API_KEY="sk-..."

# 单画像快速跑
python main.py -i task.md --personas cooperative --seed 42

# 完整模式 (12 画像 × 3 + 分支覆盖 + SC=3)
python main.py -i task.md --full --pdf

# Excel 批量
# Excel 批量 (本地 xlsx, 不在公开仓库内)
python main.py --excel "your_instructions.xlsx"
```

### 2. Web Dashboard

```bash
python -m src.web_dashboard
# 浏览器: http://127.0.0.1:8765/
```

### 3. 模型/Prompt 对比

```bash
python -m src.model_comparison \
    -i task.md --models deepseek-chat,deepseek-reasoner \
    --personas cooperative,busy,red_team_l2
```

### 4. 校准回测

```bash
python tests/run_calibration.py --self-consistency 3
```

---

## 七、关键技术决策的论据

### 7.1 为什么用规则 + LLM 混合？

| 维度 | 适合方法 | 论据 |
|---|---|---|
| 字数限制 | 规则 | LLM 反复数错字，规则 100% 准 |
| 禁用词 | 规则 | 同上，且节省 token |
| 未替换变量 | 规则 | 简单 regex |
| 检查点完成 | LLM | 需要语义理解（如"礼貌挽留"） |
| 流程合理性 | LLM | 同上 |
| 知识幻觉 | LLM | 需要对比原文 |
| 对话效率 | 启发式 + LLM | 轮次客观 + LLM 判断必要性 |

### 7.2 为什么 quote 必须经过校验？

参考 τ-bench 论文（arxiv 2406.12045）：LLM-as-judge 一大风险是评估器幻觉。我们用 `_validate_quotes` 强制校验 quote.text 是否真在 session.turns 中出现（子串或近似），不匹配的丢弃，从源头上防止「伪证据」。

### 7.3 为什么红队画像分 L1/L2/L3？

参考工业界红队实践：
- L1 轻度刺探（隐私 / 越权请求）
- L2 社工诱导（紧急 / 冒充）
- L3 prompt-injection 攻击

这三层覆盖了从合规到对抗的不同强度。

### 7.4 校准回测如何证明可靠性？

8 个人工标注 case → evaluator 自动评分 → 计算 MAE / Pearson-r / 容差内比例。  
**实测**：MAE 4.72（远低于 10）, Pearson-r 0.967（远高于 0.85）, 8/8 全过容差。  
表明 evaluator 与人工标注 **几乎完全一致**（学术级证据）。

---

## 八、性能数据

| 配置 | 会话数 | 用时 | 平均/会话 |
|---|---|---|---|
| 单画像 (rider, 1 路) | 1 | 28s | 28s |
| 3 画像 mini (rider, 3 路) | 3 | 87s | 29s |
| 2 画像 + 分支覆盖 (course, 4 路) | 16 | 445s | 28s |
| **full 模式** (course, 6 路, SC=3) | **29** | **1585s (26 分)** | 55s |
| 校准回测 (8 case, 1 路, SC=1) | 8 | 198s | 25s |
| Prompt 对比 (rider × 2 prompts × 3 画像) | 6 | ~6 min | 60s |

---

## 九、未来扩展方向

1. **多 Judge Ensemble** — Claude / GPT-4 / DeepSeek 三模型评，取多数票
2. **PDF 模板优化** — 加目录页 / 章节书签
3. **校准集扩充** — 50+ case 涵盖更多边缘场景
4. **可微分评估** — 把评估器改成 RL reward model，直接接 DPO/PPO 训练
5. **CI 集成** — 评测可作为模型训练后自动质量门禁
