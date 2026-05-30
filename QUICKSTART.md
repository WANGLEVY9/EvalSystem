# ⚡ 5 分钟快速上手

> 目标：拿到代码后 5 分钟内跑出第一份评测报告。

---

## 步骤 1：环境要求（30 秒）

| 项 | 要求 | 验证命令 |
|---|---|---|
| Python | **≥ 3.10**（推荐 3.12+） | `python3 --version` |
| 操作系统 | macOS / Linux / Windows（WSL） | - |
| 网络 | 可访问 `api.deepseek.com` | `curl -I https://api.deepseek.com` |

> ⚠️ **Python 3.9 及以下会失败**（项目使用 PEP-604 类型语法 `str \| None`）。如系统自带是 3.9，请用 `pyenv install 3.12` 或 conda 装新版。

---

## 步骤 2：克隆 + 安装（1 分钟）

```bash
git clone https://github.com/WANGLEVY9/EvalSystem.git
cd EvalSystem

# 创建虚拟环境（推荐）
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

---

## 步骤 3：跑 Demo 验证安装（30 秒，**无需 API Key**）

```bash
python main.py --demo
```

预期输出：
```
✅ Demo 完成
HTML 报告: output/report_<时间戳>.html
```

打开 HTML 看到雷达图 + 评估结果即代表安装成功。

---

## 步骤 4：跑集成测试（30 秒，**无需 API Key**）

```bash
python tests/test_integration.py
```

会验证指令编译器、规则评估器、状态跟踪器、报告生成器等核心模块。最后一行应显示：
```
🎉 全部集成测试通过
```

---

## 步骤 5：配置 DeepSeek API Key（10 秒）

DeepSeek 注册地址：https://platform.deepseek.com/  
拿到 key 后导出环境变量：

```bash
export DEEPSEEK_API_KEY="sk-..."
```

> 也可以放到 `~/.zshrc` 或 `.env` 文件里持久化。

---

## 步骤 6：跑第一次真实评测（1-2 分钟）

```bash
python main.py -i config/sample_instructions/rider_feimaotui.md \
    --personas cooperative --concurrency 1 --seed 42
```

**预期结果**：
- 用时 ~30 秒
- 生成 `output/report_<时间戳>.{html,json,md}`
- 终端打印总分（约 80-85 分）+ 失败模式

---

## 步骤 7（可选）：完整体验

### A. 多画像 + 红队测试

```bash
python main.py -i config/sample_instructions/rider_feimaotui.md \
    --personas cooperative,busy,red_team_l1 --concurrency 3 --seed 42
```

### B. 强制分支覆盖（复杂任务）

```bash
python main.py -i config/sample_instructions/course_live_upgrade.md \
    --personas cooperative,busy --concurrency 4 --seed 42 --branch-test
```

### C. Web Dashboard（评委友好）

```bash
python -m src.web_dashboard
# 浏览器打开 http://127.0.0.1:8765/
```

### D. 校准回测（评估器可靠性证明）

```bash
python tests/run_calibration.py
```

输出 MAE / Pearson-r / 容差内比例。

### E. PDF 导出（可选，需系统库）

```bash
# macOS
brew install pango glib cairo gdk-pixbuf libffi

# Ubuntu/Debian
sudo apt-get install libpango-1.0-0 libpangoft2-1.0-0

# 然后
python main.py -i task.md --personas cooperative --pdf
```

---

## 常见问题（FAQ）

### ❓ 报错 `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`
Python 版本太低，需要 3.10+。

### ❓ 报错 `OSError: cannot load library 'libgobject-2.0-0'`（PDF 导出）
缺系统库，按上面 E 节装 pango/glib。**或不用 `--pdf` 参数，HTML 报告浏览器 Cmd+P 也能存 PDF**。

### ❓ DeepSeek API 报 429（频控）
降低 `--concurrency`，从 4 降到 2 或 1。

### ❓ 报告评分异常（如 0 分或负分）
检查指令文件是否符合 markdown 格式（参考 `config/sample_instructions/rider_feimaotui.md`）。

### ❓ Excel 解析失败
用最新版 openpyxl：`pip install -U openpyxl`。

### ❓ 想用其他 LLM（非 DeepSeek）
编辑 `config/default_config.yaml` 里的 `target_model.base_url` 和 `model`，所有兼容 OpenAI API 的模型都可（如阿里 Qwen、智谱 GLM、本地 vllm）。

---

## 主要命令速查

| 场景 | 命令 |
|---|---|
| 看 demo | `python main.py --demo` |
| 集成测试（无 API） | `python tests/test_integration.py` |
| 单画像快速跑 | `python main.py -i task.md --personas cooperative --seed 42` |
| 完整 12 画像 + 分支 + SC=3 | `python main.py -i task.md --full` |
| Excel 批量 | `python main.py --excel xxx.xlsx` |
| Web Dashboard | `python -m src.web_dashboard` |
| 校准回测 | `python tests/run_calibration.py` |
| 多模型对比 | `python -m src.model_comparison -i task.md --models deepseek-chat,deepseek-reasoner` |
| PDF 导出 | 任意命令加 `--pdf` |

---

更详细的功能说明见 [README.md](./README.md)，技术细节见 [DELIVERABLE.md](./DELIVERABLE.md)。
