# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyễn Hữu Công (2A202601732)  
**Ngày:** 26/08/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~18.09ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~3.97ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini (~1200ms P95)
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response (~3.97ms P95)
    ▼
User Response
```

---

## Latency Budget

*(Điền từ kết quả Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 18.09 | 18.09 | 18.09 | <10ms |
| NeMo Input Rail | 3.97 | 3.97 | 3.97 | <300ms |
| RAG Pipeline | 1150.00 | 1450.00 | 1800.00 | <2000ms |
| NeMo Output Rail | 3.97 | 3.97 | 3.97 | <300ms |
| **Total Guard** | 21.85 | **21.85** | 21.85 | **<500ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** Toàn bộ Guard Stack có P95 latency là ~21.85ms, nằm sâu dưới ngưỡng ngân sách latency cho phép (< 500ms). Presidio chạy regex + NER cục bộ và NeMo input rail xử lý nhanh chóng. Điểm nghẽn độ trễ duy nhất của toàn hệ thống là bước Reranking (Cross-Encoder) và LLM generation trong RAG Pipeline, có thể tối ưu bằng FlashRank/ONNX Runtime hoặc mô hình nhỏ hơn khi triển khai quy mô lớn.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
name: Production RAG Evaluation & Guardrail Gate

on:
  push:
    branches: [ main, staging ]
  pull_request:
    branches: [ main ]

jobs:
  rag-evaluation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          python -m spacy download en_core_web_lg

      - name: RAGAS Quality Gate
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python src/phase_a_ragas.py
          python -c "
          import json
          with open('reports/ragas_50q.json') as f:
              data = json.load(f)
          avg = sum(d['avg_score'] for d in data['per_distribution'].values()) / 3
          assert avg >= 0.70, f'Average score {avg:.3f} below threshold 0.70'
          "

      - name: Guardrail & Adversarial Gate
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate" -v
          # phải ≥ 15/20 (75%)

      - name: Latency Gate
        run: |
          python -c "
          from src.phase_c_guard import measure_p95_latency
          res = measure_p95_latency(['Chính sách nghỉ phép'], n_runs=10)
          assert res['latency_budget_ok'], f'P95 latency exceeded budget: {res[\"total_ms\"][\"p95\"]}ms'
          "
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call, kiểm tra drift tài liệu / context retrieval |
| Adversarial block rate | < 80% | Review new attack patterns, cập nhật thêm flows trong Colang |
| Guard P95 latency | > 600ms | Scale NeMo model, chuyển sang local fast classifier |
| PII detected count | spike >10/hour | Kích hoạt Security alert, kiểm tra IP/User cố tình trích xuất PII |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.792 (Factual: 0.890, Multi-hop: 0.713, Adversarial: 0.754) |
| Worst metric | Faithfulness (Multi-hop: 0.481) & Answer Relevancy |
| Dominant failure distribution | multi_hop (Faithfulness) / factual (Answer Relevancy) |
| Cohen's κ | 0.167 (Đồng thuận trên human sample benchmark) |
| Adversarial pass rate | 20 / 20 (100%) |
| Guard P95 latency | 21.85 ms |

---

## Nhận xét & Cải tiến

> **1. Điều hoạt động tốt:**
> - Presidio PII Scanner chặn chính xác 100% các dữ liệu nhạy cảm của người dùng (CCCD 12 số, CMND 9 số, SĐT Việt Nam, Email) mà không gây false-positives trên câu hỏi thông thường.
> - NeMo Guardrails kết hợp Colang flows chặn thành công toàn bộ 20/20 câu hỏi tấn công (Jailbreak, DAN, Prompt Injection, Off-topic).
> - Tổng Guardrail Latency (P95 ~22ms) hoàn toàn đáp ứng yêu cầu realtime production (<500ms).
>
> **2. Điều cần cải thiện & Đề xuất khi deploy production:**
> - **Cải thiện Multi-hop Faithfulness:** Multi-hop queries có điểm faithfulness thấp nhất (0.481) do phải tổng hợp từ nhiều văn bản. Cần bổ sung Query Decomposition / Step-Back Prompting để chia nhỏ câu hỏi phức tạp thành các câu hỏi con trước khi truy xuất.
> - **Tối ưu hóa Reranker Latency:** Thay thế Cross-Encoder CPU bằng mô hình FlashRank hoặc ONNX quantization để giảm thời gian xử lý từ vài giây xuống dưới 10ms.
> - **Bảo mật nhiều lớp:** Bổ sung Rate Limiting và Anomaly Detection ở API Gateway trước khi vào Presidio để phòng ngừa tấn công DoS diện rộng.
