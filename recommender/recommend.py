import argparse

import pandas as pd

ap = argparse.ArgumentParser(description="Recommend an LLM serving configuration")
ap.add_argument("--users", type=int, required=True, help="concurrent users: 1, 2, 4 or 8")
ap.add_argument("--max-ttft-p95", type=float, default=float("inf"), help="ms")
ap.add_argument("--min-agg-tok-s", type=float, default=0.0)
ap.add_argument("--max-ppl", type=float, default=float("inf"), help="max perplexity")
ap.add_argument("--max-model-mib", type=float, default=float("inf"))
ap.add_argument("--hourly-cost", type=float, default=0.04, help="hardware $/hour (assumed)")
ap.add_argument("--top", type=int, default=5)
a = ap.parse_args()

sweep = pd.read_csv("results/sweep_results.csv")
qual = pd.read_csv("results/quant_results.csv")[["quant", "size_mib", "perplexity"]]
df = sweep.merge(qual, on="quant")
df["cost_per_1m_usd"] = a.hourly_cost / (df["agg_tok_s"] * 3600) * 1_000_000

df = df[df["users"] == a.users]
ok = df[
    (df["ttft_p95_ms"] <= a.max_ttft_p95)
    & (df["agg_tok_s"] >= a.min_agg_tok_s)
    & (df["perplexity"] <= a.max_ppl)
    & (df["size_mib"] <= a.max_model_mib)
].sort_values("cost_per_1m_usd")

cols = ["quant", "slots", "users", "ttft_p95_ms", "agg_tok_s",
        "per_user_tok_s", "perplexity", "size_mib", "cost_per_1m_usd"]
if ok.empty:
    print("No configuration meets all constraints. Closest by throughput:")
    print(df.sort_values("agg_tok_s", ascending=False)[cols].head(3).round(3).to_string(index=False))
else:
    print(f"{len(ok)} feasible configuration(s). Best first (lowest cost per 1M tokens):")
    print(ok[cols].head(a.top).round(3).to_string(index=False))
