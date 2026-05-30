"""
模型对比工具 (v3.0)

用例: 在同一指令下并排测试多个被测模型, 直接产出对比报告

python -m src.model_comparison \\
    --instruction config/sample_instructions/rider_feimaotui.md \\
    --models deepseek-chat,deepseek-reasoner \\
    --personas cooperative,busy,red_team_l2

产出:
- output/comparison_<ts>.html (左右并排, 每模型一栏)
- output/comparison_<ts>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dialogue_engine import DialogueEngine
from src.evaluator import Evaluator
from src.instruction_parser import InstructionParser
from src.models import EVALUATOR_VERSION, EvaluationReport, PersonaType, RunMetadata
from src.report_generator import ReportGenerator
from src.target_model import DeepSeekModel, OpenAICompatibleModel
from src.user_simulator import PersonaSelector
from src.weakness_analyzer import WeaknessAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("model_comparison")


def run_one_model(
    instruction_path: str,
    model_name: str,
    config: dict,
    personas: list[str],
    sessions_per_persona: int,
    self_consistency: int,
    concurrency: int,
    seed: Optional[int],
) -> EvaluationReport:
    from openai import OpenAI

    api_key = config["llm"].get("api_key") or os.environ.get("DEEPSEEK_API_KEY")
    base_url = config["llm"].get("base_url", "https://api.deepseek.com")
    llm_client = OpenAI(api_key=api_key, base_url=base_url)

    parser = InstructionParser(llm_client=llm_client, model=config["llm"].get("evaluator_model", "deepseek-chat"))
    instruction = parser.parse_from_file(instruction_path)

    target = DeepSeekModel(api_key=api_key, base_url=base_url, model=model_name,
                          temperature=float(config["llm"].get("temperature", 0.7)),
                          max_tokens=int(config["llm"].get("max_tokens", 1024)))

    persona_types = [PersonaType(p.strip()) for p in personas]

    engine = DialogueEngine(
        target_model=target, llm_client=llm_client,
        simulator_model=config["llm"].get("simulator_model", "deepseek-chat"),
        max_turns=int(config["dialogue"].get("max_turns", 18)),
        timeout=int(config["dialogue"].get("timeout", 300)),
        concurrency=concurrency, seed=seed,
    )

    t = time.time()
    sessions = engine.run_batch_sessions(
        instruction=instruction, persona_types=persona_types,
        sessions_per_persona=sessions_per_persona, seed=seed,
    )
    evaluator = Evaluator(
        llm_client=llm_client,
        model=config["llm"].get("evaluator_model", "deepseek-chat"),
        n_self_consistency=self_consistency,
    )
    evaluations = evaluator.evaluate_batch(sessions, instruction)
    duration = time.time() - t

    rg = ReportGenerator(output_dir=config["report"].get("output_dir", "output"))
    rep = rg._build_report(
        instruction=instruction, sessions=sessions, evaluations=evaluations,
        run_metadata=RunMetadata(
            run_id=str(uuid.uuid4()), evaluator_version=EVALUATOR_VERSION,
            target_model_id=model_name,
            simulator_model_id=config["llm"].get("simulator_model", "deepseek-chat"),
            judge_model_id=config["llm"].get("evaluator_model", "deepseek-chat"),
            seed=seed, self_consistency_n=self_consistency, concurrency=concurrency,
            duration_seconds=round(duration, 2), finished_at=datetime.now(),
        ),
    )
    return rep


def write_comparison_report(reports: list[tuple[str, EvaluationReport]], output_dir: Path):
    """生成并排对比的 HTML 与 JSON"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_data = {
        "generated_at": ts,
        "models": [
            {
                "model": name,
                "overall_score": rep.overall_score,
                "overall_score_std": rep.overall_score_std,
                "overall_confidence": rep.overall_confidence,
                "dimension_scores": [d.model_dump() for d in rep.dimension_averages],
                "persona_scores": rep.persona_scores,
                "weakness": rep.weakness_profile.model_dump() if rep.weakness_profile else None,
                "branch_coverage": rep.branch_coverage.model_dump() if rep.branch_coverage else None,
            }
            for name, rep in reports
        ],
    }
    json_path = output_dir / f"comparison_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)

    # HTML
    html_path = output_dir / f"comparison_{ts}.html"
    _gen_compare_html(reports, html_path)
    logger.info(f"对比报告: {html_path}")


def _gen_compare_html(reports: list[tuple[str, EvaluationReport]], path: Path):
    """生成模型对比 HTML"""
    rows: list[str] = []
    rows.append("<!DOCTYPE html><html><head><meta charset='UTF-8'><title>模型对比报告</title>")
    rows.append("""<style>
body{font-family:-apple-system,sans-serif;background:#f3f5f9;margin:0;padding:18px;color:#222}
.container{max-width:1400px;margin:0 auto}
.hero{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:28px 32px;border-radius:14px;margin-bottom:18px}
.hero h1{margin:0 0 8px 0}
.card{background:#fff;border-radius:12px;padding:18px;margin-bottom:14px;box-shadow:0 2px 8px rgba(0,0,0,.05)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:9px 12px;border-bottom:1px solid #eee;text-align:left}
th{background:#fafbfd;font-weight:600;color:#666}
.cell-score{font-weight:600}
.s-best{color:#52c41a}.s-mid{color:#1890ff}.s-low{color:#fa541c}.s-bad{color:#f5222d}
.bar{display:inline-block;height:14px;background:#eef0f4;border-radius:7px;width:80px;vertical-align:middle;margin-right:6px;overflow:hidden}
.bar > .fill{height:100%;border-radius:7px}
</style></head><body><div class='container'>""")
    rows.append(f"<div class='hero'><h1>📊 模型对比报告</h1><div>{len(reports)} 个被测模型 · 同一指令对比</div></div>")

    # 1. 总分对比
    rows.append("<div class='card'><h2>总分对比</h2><table><tr><th>模型</th><th>总分</th><th>std</th><th>置信度</th><th>用时(s)</th></tr>")
    scores = [(name, rep.overall_score) for name, rep in reports]
    max_score = max(s for _, s in scores) if scores else 100
    for name, rep in reports:
        cls = "s-best" if rep.overall_score >= 80 else ("s-mid" if rep.overall_score >= 60 else ("s-low" if rep.overall_score >= 40 else "s-bad"))
        bar_w = int(rep.overall_score / max_score * 100) if max_score else 0
        dur = rep.run_metadata.duration_seconds if rep.run_metadata else 0
        rows.append(f"""<tr>
            <td><strong>{name}</strong></td>
            <td><span class='bar'><span class='fill' style='width:{bar_w}%;background:{"#52c41a" if rep.overall_score>=80 else "#1890ff" if rep.overall_score>=60 else "#faad14" if rep.overall_score>=40 else "#f5222d"}'></span></span>
                <span class='cell-score {cls}'>{rep.overall_score:.1f}</span></td>
            <td>±{rep.overall_score_std:.1f}</td>
            <td>{rep.overall_confidence:.2f}</td>
            <td>{dur:.1f}</td>
        </tr>""")
    rows.append("</table></div>")

    # 2. 维度对比
    rows.append("<div class='card'><h2>维度详细对比</h2>")
    # 找一个 reference 维度列表
    ref_dims = reports[0][1].dimension_averages if reports and reports[0][1].dimension_averages else []
    if ref_dims:
        rows.append("<table><tr><th>维度</th>")
        for name, _ in reports:
            rows.append(f"<th>{name}</th>")
        rows.append("</tr>")
        for d in ref_dims:
            rows.append(f"<tr><td><strong>{d.dimension_name}</strong> (权重 {d.weight*100:.0f}%)</td>")
            for name, rep in reports:
                # 找对应维度
                m = next((x for x in rep.dimension_averages if x.dimension_key == d.dimension_key), None)
                if m:
                    cls = "s-best" if m.score >= 80 else ("s-mid" if m.score >= 60 else ("s-low" if m.score >= 40 else "s-bad"))
                    rows.append(f"<td><span class='cell-score {cls}'>{m.score:.1f}</span> <small>±{m.score_std:.1f}</small></td>")
                else:
                    rows.append("<td>-</td>")
            rows.append("</tr>")
        rows.append("</table>")
    rows.append("</div>")

    # 3. 画像对比
    rows.append("<div class='card'><h2>画像表现对比</h2>")
    persona_keys = sorted({k for _, rep in reports for k in rep.persona_scores.keys()})
    if persona_keys:
        rows.append("<table><tr><th>画像</th>")
        for name, _ in reports:
            rows.append(f"<th>{name}</th>")
        rows.append("</tr>")
        for p in persona_keys:
            rows.append(f"<tr><td>{p}</td>")
            for name, rep in reports:
                v = rep.persona_scores.get(p)
                if v is None:
                    rows.append("<td>-</td>")
                else:
                    cls = "s-best" if v >= 80 else ("s-mid" if v >= 60 else ("s-low" if v >= 40 else "s-bad"))
                    rows.append(f"<td><span class='cell-score {cls}'>{v:.1f}</span></td>")
            rows.append("</tr>")
        rows.append("</table>")
    rows.append("</div>")

    # 4. 短板差异
    rows.append("<div class='card'><h2>短板诊断对比</h2>")
    for name, rep in reports:
        if rep.weakness_profile:
            wp = rep.weakness_profile
            rows.append(f"<h3>{name}</h3>")
            rows.append(f"<p><strong>风险:</strong> {wp.risk_summary}</p>")
            rows.append("<ul>")
            for fm in wp.top_failure_modes[:5]:
                rows.append(f"<li>[{fm.category}] {fm.name} (x{fm.occurrences})</li>")
            rows.append("</ul>")
    rows.append("</div>")

    rows.append("</div></body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(rows))


def main():
    p = argparse.ArgumentParser(description="模型对比工具 v3")
    p.add_argument("-c", "--config", default="config/default_config.yaml")
    p.add_argument("-i", "--instruction", required=True)
    p.add_argument("--models", required=True, help="逗号分隔, e.g. deepseek-chat,deepseek-reasoner")
    p.add_argument("--personas", default="cooperative,busy,red_team_l1")
    p.add_argument("--sessions", type=int, default=1)
    p.add_argument("--self-consistency", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    models = [m.strip() for m in args.models.split(",")]
    personas = [s.strip() for s in args.personas.split(",")]

    reports: list[tuple[str, EvaluationReport]] = []
    for m in models:
        logger.info("=" * 60)
        logger.info(f"评测被测模型: {m}")
        logger.info("=" * 60)
        rep = run_one_model(
            instruction_path=args.instruction,
            model_name=m, config=config,
            personas=personas,
            sessions_per_persona=args.sessions,
            self_consistency=args.self_consistency,
            concurrency=args.concurrency,
            seed=args.seed,
        )
        reports.append((m, rep))

    write_comparison_report(reports, Path(config.get("report", {}).get("output_dir", "output")))


if __name__ == "__main__":
    main()
