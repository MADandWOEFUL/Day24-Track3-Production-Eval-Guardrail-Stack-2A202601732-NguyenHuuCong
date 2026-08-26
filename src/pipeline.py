from __future__ import annotations

"""Production RAG Pipeline — Ghép M1 + M5 + M2 + M3 + LLM Generation + M4 RAGAS Eval."""

import os, sys, time, json
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY, OPENAI_BASE_URL


def build_pipeline():
    """Build production RAG pipeline with latency tracking."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    latencies = {}

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents (M1 Hierarchical)...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    latencies["chunking_s"] = time.time() - t0
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({latencies['chunking_s']:.2f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, Combined Single-Call)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        latencies["enrichment_s"] = time.time() - t0
        print(f"  ✓ Enriched {len(enriched)} chunks ({latencies['enrichment_s']:.2f}s)", flush=True)
    else:
        latencies["enrichment_s"] = 0.0
        print("  ⚠️  M5 fallback — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 Vietnamese + Dense Qdrant)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    latencies["indexing_s"] = time.time() - t0
    print(f"  ✓ Indexed ({latencies['indexing_s']:.2f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker (M3 CrossEncoder)...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()
    latencies["reranker_load_s"] = time.time() - t0
    print(f"  ✓ Reranker ready ({latencies['reranker_load_s']:.2f}s)", flush=True)

    return search, reranker, latencies


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str], dict]:
    """Run single query through pipeline and return answer, contexts, and latency breakdown."""
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    results = search.search(query)
    t_search = (time.perf_counter() - t0) * 1000

    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]

    t0 = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    t_rerank = (time.perf_counter() - t0) * 1000

    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    t0 = time.perf_counter()
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            kwargs = {"api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                kwargs["base_url"] = OPENAI_BASE_URL
            client = OpenAI(**kwargs)
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Trả lời chính xác, ngắn gọn CHỈ dựa trên context đã cung cấp. Nếu không tìm thấy thông tin trong context, trả về 'Không tìm thấy thông tin.'"},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                ],
                temperature=0.0,
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    t_llm = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - t_start) * 1000
    timing = {
        "search_ms": t_search,
        "rerank_ms": t_rerank,
        "llm_ms": t_llm,
        "total_ms": total_ms,
    }
    return answer, contexts, timing


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker, build_latencies: dict | None = None):
    """Run evaluation on test set with latency breakdown."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []
    query_timings = []

    for i, item in enumerate(test_set):
        answer, contexts, timing = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        query_timings.append(timing)
        print(f"  [{i+1}/{len(test_set)}] ({timing['total_ms']:.1f}ms) {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    eval_time = time.time() - t0
    print(f"  ✓ RAGAS done ({eval_time:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m:<20}: {s:.4f}")

    # Latency Breakdown Report
    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN REPORT")
    print("=" * 60)
    if build_latencies:
        print("Pipeline Indexing Phase:")
        print(f"  - Chunking (M1):          {build_latencies.get('chunking_s', 0):.2f}s")
        print(f"  - Enrichment (M5):        {build_latencies.get('enrichment_s', 0):.2f}s")
        print(f"  - Indexing Dense+BM25:    {build_latencies.get('indexing_s', 0):.2f}s")
        print(f"  - Reranker Load (M3):     {build_latencies.get('reranker_load_s', 0):.2f}s")
    if query_timings:
        avg_search = sum(t["search_ms"] for t in query_timings) / len(query_timings)
        avg_rerank = sum(t["rerank_ms"] for t in query_timings) / len(query_timings)
        avg_llm = sum(t["llm_ms"] for t in query_timings) / len(query_timings)
        avg_total = sum(t["total_ms"] for t in query_timings) / len(query_timings)
        print("\nQuery Serving Phase (Average per query):")
        print(f"  - Hybrid Search (M2):     {avg_search:.2f} ms")
        print(f"  - Cross-Encoder Rerank (M3): {avg_rerank:.2f} ms")
        print(f"  - LLM Generation:         {avg_llm:.2f} ms")
        print(f"  - End-to-End Query Total: {avg_total:.2f} ms")
    print("=" * 60)

    failures = failure_analysis(results.get("per_question", []))
    os.makedirs("reports", exist_ok=True)
    save_report(results, failures, path="ragas_report.json")
    save_report(results, failures, path="reports/ragas_report.json")
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker, build_latencies = build_pipeline()
    evaluate_pipeline(search, reranker, build_latencies)
    print(f"\nTotal: {time.time() - start:.1f}s")
