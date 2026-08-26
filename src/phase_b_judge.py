from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    PROMPT_TEMPLATE = """Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí: độ chính xác, đầy đủ, súc tích.
Trả lời JSON (chỉ JSON, không text khác):
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                {"role": "user", "content": PROMPT_TEMPLATE.format(
                    question=question, answer_a=answer_a, answer_b=answer_b)},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        winner = data.get("winner", "tie")
        if winner not in {"A", "B", "tie"}:
            winner = "tie"
        reasoning = data.get("reasoning", "")
        scores = data.get("scores", {"A": 0.0, "B": 0.0})
        # ensure float scores
        scores = {k: float(v) for k, v in scores.items()}
        return {"winner": winner, "reasoning": reasoning, "scores": scores}
    except Exception as e:
        print(f"⚠️ pairwise_judge error: {e}")
        return {"winner": "tie", "reasoning": f"Judge error: {e}", "scores": {"A": 0.0, "B": 0.0}}


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    # Convert pass2 back to original A/B space
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw.get("winner", "tie"), "tie")
    winner_pass1 = pass1.get("winner", "tie")

    # Average: consensus only if both agree
    if winner_pass1 == winner_pass2:
        final = winner_pass1
    else:
        final = "tie"  # disagreement = inconclusive

    position_consistent = (winner_pass1 == winner_pass2)
    scores_pass1 = pass1.get("scores", {"A": 0.0, "B": 0.0})
    scores_pass2_raw = pass2_raw.get("scores", {"A": 0.0, "B": 0.0})
    scores_pass2 = {
        "A": scores_pass2_raw.get("B", 0.0),
        "B": scores_pass2_raw.get("A", 0.0),
    }

    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=winner_pass1,
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=scores_pass1,
        scores_pass2=scores_pass2,
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect

    Gợi ý A — dùng scikit-learn:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(human_labels, judge_labels)

    Gợi ý B — tính tay:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
               judge_labels.count(0)/n * human_labels.count(0)/n)
        κ = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
        return κ
    """
    if not judge_labels or not human_labels or len(judge_labels) != len(human_labels):
        return 0.0
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(human_labels, judge_labels))
    except Exception:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        all_cats = set(judge_labels) | set(human_labels)
        p_e = sum((judge_labels.count(c) / n) * (human_labels.count(c) / n) for c in all_cats)
        if abs(1.0 - p_e) < 1e-9:
            return 1.0 if p_o == 1.0 else 0.0
        return (p_o - p_e) / (1.0 - p_e)


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {
                "a_wins_a_longer": 0,
                "b_wins_b_longer": 0,
                "total_decisive": 0,
            },
            "interpretation": "Không có dữ liệu",
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = (
        "Position bias cao — nên dùng swap-and-average."
        if position_bias_rate > 0.3
        else "Position bias thấp — judge ổn định."
    )
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }


def single_judge(question: str, model_answer: str) -> int:
    """Đánh giá câu trả lời là chính xác (1) hay không chính xác/thiếu (0)."""
    prompt = f"""Bạn là một expert đánh giá chất lượng câu trả lời nhân sự nội bộ.
Câu hỏi: {question}
Câu trả lời: {model_answer}

Đánh giá xem câu trả lời trên là đúng/chính xác (1) hay sai/thiếu nghiêm trọng/dùng chính sách cũ đã hết hiệu lực (0).
Lưu ý:
- Nếu dùng chính sách cũ hết hiệu lực (như v2023 12 ngày phép) -> 0
- Nếu thiếu cấp phê duyệt bắt buộc hoặc phạt sai -> 0
- Nếu cấm dùng VPN cá nhân mà trả lời được dùng -> 0
- Nếu trả lời đúng theo quy định hiện hành -> 1

Chỉ trả về JSON: {{"label": 1 hoặc 0, "reasoning": "giải thích ngắn gọn"}}
"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return int(data.get("label", 0))
    except Exception as e:
        print(f"⚠️ single_judge error: {e}")
        return 0


def save_phase_b_report(report_data: dict, path: str = "reports/judge_results.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"Phase B report saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Demo pairwise + swap ---
    test_pairs = [
        ("Nhân viên được nghỉ bao nhiêu ngày phép năm?",
         "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành.",
         "Theo quy định, nhân viên có 12 ngày phép hàng năm."),
        ("Bao lâu phải đổi mật khẩu một lần?",
         "Theo chính sách v2.0 hiện hành, mật khẩu phải được thay đổi mỗi 120 ngày.",
         "Theo quy định chung, mật khẩu đổi mỗi 90 ngày."),
        ("VPN có bắt buộc khi làm việc từ xa không?",
         "Có, VPN WireGuard là bắt buộc khi làm việc từ xa theo quy định công ty.",
         "Không bắt buộc, chỉ cần có kết nối internet an toàn."),
    ]

    print("Running swap-and-average judge on sample pairs...")
    pairwise_results = []
    for q, a_a, a_b in test_pairs:
        res = swap_and_average(q, a_a, a_b)
        pairwise_results.append(res)
        print(f"  Q: {q[:40]}... -> Final winner: {res.final_winner}, Consistent: {res.position_consistent}")

    # --- Cohen's κ vs human labels ---
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"\nHuman labels loaded: {len(human_labels)} questions")

    # Run judge on the 10 human questions
    judge_labels = []
    comparison = []
    for item in human_data:
        j_lbl = single_judge(item["question"], item["model_answer"])
        judge_labels.append(j_lbl)
        comparison.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "model_answer": item["model_answer"],
            "human_label": item["human_label"],
            "judge_label": j_lbl,
            "match": item["human_label"] == j_lbl,
            "human_note": item.get("human_note", ""),
        })

    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"Cohen's κ: {kappa:.3f}")

    # --- Bias report ---
    bias = bias_report(pairwise_results)
    print(f"\nBias report: {bias}")

    # Save report
    report_data = {
        "cohen_kappa": round(kappa, 4),
        "agreement_count": sum(1 for c in comparison if c["match"]),
        "total_human_samples": len(human_labels),
        "bias_report": bias,
        "human_vs_judge": comparison,
        "pairwise_samples": [
            {
                "question": r.question,
                "winner_pass1": r.winner_pass1,
                "winner_pass2": r.winner_pass2,
                "final_winner": r.final_winner,
                "position_consistent": r.position_consistent,
                "reasoning_pass1": r.reasoning_pass1,
                "reasoning_pass2": r.reasoning_pass2,
            }
            for r in pairwise_results
        ]
    }
    save_phase_b_report(report_data)
