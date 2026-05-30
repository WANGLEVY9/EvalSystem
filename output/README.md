# Output 报告归档

> 本目录包含项目实测产出的所有评测报告，按时间戳命名。
> 评委可以直接打开 HTML 文件查看交互式报告效果。

---

## 🌟 关键报告（推荐优先查看）

| 报告 | 命令 | 说明 |
|---|---|---|
| `report_20260530_125448.{html,json,md,pdf}` | `python main.py -i config/sample_instructions/course_live_upgrade.md --full --concurrency 6 --seed 42` | **完整 full 模式** — 12 画像 × 3 + 14 分支强制覆盖 + Self-Consistency=3，**29 会话, 26 分钟跑完**。总分 **78.5 ± 7.4 (置信 0.93)**, 分支覆盖 **14/14 = 100%** |
| `report_20260530_115449.{html,json,md}` | `python main.py -i config/sample_instructions/course_live_upgrade.md --personas cooperative,busy --concurrency 4 --seed 42 --branch-test` | **课程升级任务 + 分支覆盖**（轻量版 16 会话, 7.4 分钟）总分 82.0 ± 6.6 |
| `calibration_20260530_125848.{html,json}` | `python tests/run_calibration.py` | **校准回测** — 8 case 人工标注验证 evaluator 可靠性。**MAE 4.72 / Pearson-r 0.967 / 容差内 100%** |
| `comparison_20260530_125903.{html,json}` | `python -m src.model_comparison ...` | **Prompt 对比** — 揭示「过度强化约束的 prompt 反而总分更低」(73.8 → 59.2) |

---

## 📋 其他报告（按时间倒序）

### 课程升级任务 (course_live_upgrade)

| 时间戳 | 模式 | 说明 |
|---|---|---|
| `report_20260530_125448` | **full + SC=3** ⭐ | 29 会话, 26 分钟, 总分 78.5 |
| `report_20260530_115449` | branch-test | 16 会话, 7.4 分钟, 总分 82.0 |
| `report_20260530_114642` | mini | 早期调试 |

### 飞毛腿任务 (rider_feimaotui)

| 时间戳 | 模式 | 说明 |
|---|---|---|
| `report_20260530_114448` | 3 画像 mini | 73.1 ± 10.2, 87 秒 |
| `report_20260530_114112` | 单画像 | 82.9, 30 秒 (修复 KB 幻觉后基线) |
| `report_20260530_113933`, `113715`, `113307` | 早期调试 | 修复变量替换/KB 幻觉的中间产物 |
| `report_20260530_130445` | Dashboard 启动 | 通过 Web Dashboard 触发的端到端流验证 |

### Demo 模式 (无 API)

| `report_20260530_004014` | Demo 模式产出（MockTargetModel） |

---

## 📊 报告四种格式对照

每个 `report_<时间戳>.*` 都有：

| 后缀 | 用途 |
|---|---|
| `.html` | **交互式可视化** — 雷达图、热力图、可点击对话浏览器、quote 高亮 |
| `.json` | **结构化数据** — 完整评估细节，便于程序化处理 |
| `.md` | **1 页摘要** — 评审快速浏览 |
| `.pdf` | **打印友好** — A4 排版（仅部分报告生成） |

---

## 🔄 如何复算

只要设定相同 `seed` + 相同模型版本，理论上能复现报告（受 LLM 端温度采样波动影响，可能 ±2-3 分浮动）。

```bash
export DEEPSEEK_API_KEY="sk-..."
python main.py -i config/sample_instructions/course_live_upgrade.md \
    --full --concurrency 6 --seed 42
```

每个 JSON 报告都包含 `RunMetadata`：
```json
{
  "run_id": "8d4f...",
  "evaluator_version": "3.0.0",
  "target_model_id": "deepseek-chat",
  "judge_model_id": "deepseek-chat",
  "seed": 42,
  "self_consistency_n": 3,
  "concurrency": 6,
  "duration_seconds": 1585
}
```
