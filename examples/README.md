# 示例报告

> 本目录包含项目实际运行产出的代表性报告，便于新用户快速了解系统输出形态。

| 文件 | 来源 | 说明 |
|---|---|---|
| `sample_report_full_mode.html` | `python main.py -i config/sample_instructions/course_live_upgrade.md --full --concurrency 6 --seed 42` | **完整 full 模式报告**：12 画像 × 3 + 14 分支强制覆盖 + Self-Consistency=3，共 29 会话，用时 26 分钟。总分 78.5 ± 7.4 |
| `sample_report_full_mode.md` | 同上 | 1 页 Markdown 摘要（评委友好） |
| `sample_report_full_mode.pdf` | 同上 | PDF 导出版本（雷达/热力以 SVG 渲染） |
| `sample_calibration_report.html` | `python tests/run_calibration.py` | 校准回测报告：8 个人工标注 case 验证 evaluator 可靠性。**MAE 4.72 / Pearson-r 0.967 / 容差内 100%** |
| `sample_comparison_report.html` | `python -m src.model_comparison ...` | 多模型/Prompt 对比报告：揭示「过度强化约束的 prompt 反而更差」的反直觉发现 |

---

## 如何打开

```bash
# macOS
open examples/sample_report_full_mode.html

# Linux
xdg-open examples/sample_report_full_mode.html

# Windows
start examples/sample_report_full_mode.html
```

PDF 用任意 PDF 阅读器打开。

---

## 复算

只要你有 DeepSeek API Key，把 `--seed 42` 加上，理论上能复现相同结果（受 API 端温度采样波动影响，可能有 ±2-3 分浮动）。

```bash
export DEEPSEEK_API_KEY="sk-..."
python main.py -i config/sample_instructions/course_live_upgrade.md \
    --full --concurrency 6 --seed 42
```
