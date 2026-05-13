from utils import model
from faiss_client import faq_index
from synonyms import synonyms_dict
from reranker import reranker


# =========================
# 1. Query Rewrite（增强版）
# =========================
def expand_query(query):

    query = query.lower().strip()

    expanded = [query]

    # ⭐ 语义增强词（关键升级）
    semantic_boost = []

    for key, syns in synonyms_dict.items():

        key = key.lower()

        syns = [s.lower() for s in syns]

        if key in query:
            expanded.extend(syns)

            # ⭐ 强化语义组合（重点优化）
            semantic_boost.append(key)
            semantic_boost.extend(syns)
            semantic_boost.append(" ".join([key] + syns))

        for s in syns:
            if s in query:
                expanded.append(key)
                semantic_boost.append(key)

    # ⭐ 如果是发货类问题，强制加语义锚点（非常重要）
    if "发" in query or "货" in query:
        semantic_boost.append("订单 发货时间 物流 什么时候发出")

    expanded.extend(semantic_boost)

    return list(set(expanded))


# =========================
# 2. Hybrid Search
# =========================
def hybrid_search(query_text, faq_list, top_k=5):

    query_text = query_text.lower().strip()

    print("\n" + "=" * 70)
    print(f"[用户问题] {query_text}")
    print("=" * 70)

    # =========================
    # FAQ MAP（修复安全版）
    # =========================
    faq_map = {i: item for i, item in enumerate(faq_list)}

    # =========================
    # exact match
    # =========================
    for idx, item in faq_map.items():
        if item["question"].lower().strip() == query_text:
            return [{
                "id": idx,
                "question": item["question"],
                "answer": item["answer"],
                "score": 1.0
            }]

    # =========================
    # expand queries
    # =========================
    queries = expand_query(query_text)

    all_candidates = {}

    print("\n================ VECTOR SEARCH ================")

    # =========================
    # vector recall
    # =========================
    for q in queries:

        query_vec = model.encode(
            q,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        D, I = faq_index.search(query_vec, top_k=8)

        print("\n[Query]", q)
        print("召回:", I)

        for score, idx in zip(D, I):

            if idx == -1:
                continue

            idx = int(idx)

            item = faq_map[idx]

            # =========================
            # ⭐ candidate fusion（关键升级）
            # =========================
            if idx not in all_candidates:
                all_candidates[idx] = {
                    "idx": idx,
                    "question": item["question"],
                    "answer": item["answer"],
                    "vector_score": float(score)
                }
            else:
                # 取最大值（避免噪声）
                all_candidates[idx]["vector_score"] = max(
                    all_candidates[idx]["vector_score"],
                    float(score)
                )

    candidates = list(all_candidates.values())

    if not candidates:
        return []

    # =========================
    # rerank
    # =========================
    print("\n================ RERANK ================")

    pairs = [(query_text, c["question"]) for c in candidates]
    scores = reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    # =========================
    # final sort（融合分数）
    # =========================
    candidates.sort(
        key=lambda x: (
            0.3 * x["vector_score"] +   # recall
            0.7 * x["rerank_score"]     # semantic
        ),
        reverse=True
    )

    print("\n================ FINAL RESULT ================")

    for c in candidates[:top_k]:
        print(
            f"ID={c['idx']} | "
            f"Vec={c['vector_score']:.4f} | "
            f"Rerank={c['rerank_score']:.4f} | "
            f"{c['question']}"
        )

    return [
        {
            "id": c["idx"],
            "question": c["question"],
            "answer": c["answer"],
            "score": float(c["rerank_score"])
        }
        for c in candidates[:top_k]
    ]