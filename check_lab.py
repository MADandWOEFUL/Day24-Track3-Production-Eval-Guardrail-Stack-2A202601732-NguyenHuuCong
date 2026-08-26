"""Kiểm tra trước khi nộp — Lab 24: Eval + Guardrail Stack."""
from __future__ import annotations

import json
import os
import subprocess
import sys


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = "✓" if condition else "✗"
    suffix = f" → {detail}" if detail and not condition else ""
    print(f"  [{icon}] {label}{suffix}")
    return condition


def main():
    print("=" * 60)
    print("CHECK LAB 24: Eval + Guardrail Stack")
    print("=" * 60)

    passed = total = 0

    # 1. Day 18 source files
    print("\n[1] Day 18 source files (copy từ lab của bạn):")
    day18_files = [
        "src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py",
        "src/m4_eval.py",     "src/m5_enrichment.py", "src/pipeline.py",
    ]
    for f in day18_files:
        total += 1
        passed += check(f, os.path.exists(f), "missing — cp <Day18>/src/ .")

    # 2. Setup output
    print("\n[2] Setup output:")
    total += 1
    answers_ok = os.path.exists("answers_50q.json")
    passed += check("answers_50q.json", answers_ok, "run: python setup_answers.py")
    if answers_ok:
        with open("answers_50q.json", encoding="utf-8") as f:
            answers = json.load(f)
        total += 1
        passed += check("answers_50q.json has 50 entries", len(answers) == 50,
                        f"có {len(answers)} entries, cần 50")

    # 3. Phase implementations
    print("\n[3] Phase module implementations:")
    for module in ["src/phase_a_ragas.py", "src/phase_b_judge.py", "src/phase_c_guard.py"]:
        if os.path.exists(module):
            with open(module, encoding="utf-8") as f:
                content = f.read()
            todos = content.count("# TODO")
            total += 1
            passed += check(module, todos == 0, f"{todos} TODO(s) còn lại")

    # 4. Generated reports
    print("\n[4] Generated reports:")
    report_files = {
        "reports/ragas_50q.json":     "Phase A — python src/phase_a_ragas.py",
        "reports/judge_results.json": "Phase B — python src/phase_b_judge.py",
        "reports/guard_results.json": "Phase C — python src/phase_c_guard.py",
        "reports/blueprint.md":       "Task 13 — điền blueprint.md",
    }
    for path, hint in report_files.items():
        total += 1
        passed += check(path, os.path.exists(path), hint)

    if os.path.exists("reports/blueprint.md"):
        with open("reports/blueprint.md", encoding="utf-8") as f:
            bp = f.read()
        total += 1
        filled = "[Họ Tên]" not in bp and len(bp) > 500
        passed += check("reports/blueprint.md filled in", filled,
                        "Điền thông tin (xóa placeholder [Họ Tên] ...)")

    # 5. RAGAS report structure
    if os.path.exists("reports/ragas_50q.json"):
        print("\n[5] RAGAS report structure:")
        with open("reports/ragas_50q.json", encoding="utf-8") as f:
            report = json.load(f)
        for key in ("total_questions", "per_distribution", "failure_clusters", "bottom_10"):
            total += 1
            passed += check(f"ragas_50q.json['{key}']", key in report)
        if "total_questions" in report:
            total += 1
            passed += check("total_questions == 50", report["total_questions"] == 50,
                            f"có {report.get('total_questions')}, cần 50")

    # 6. Test suite
    print("\n[6] Test suite:")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--tb=short", "-q"],
        capture_output=True, text=True,
    )
    tests_ok = result.returncode == 0
    total += 1
    passed += check("pytest tests/ passes", tests_ok, "run: pytest tests/ -v để xem chi tiết")
    if not tests_ok:
        lines = (result.stdout + result.stderr).strip().split("\n")
        print("\n" + "\n".join(lines[-20:]))

    print(f"\n{'='*60}")
    print(f"Score: {passed}/{total} checks passed")
    if passed == total:
        print("✓ Sẵn sàng nộp bài!")
    else:
        print(f"✗ Còn {total - passed} vấn đề cần giải quyết.")


if __name__ == "__main__":
    main()
