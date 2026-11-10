import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="LLM Inference Optimizer", layout="wide")


@st.cache_data
def load():
    sweep = pd.read_csv("results/sweep_results.csv")
    qual = pd.read_csv("results/quant_results.csv")[["quant", "size_mib", "perplexity"]]
    return sweep.merge(qual, on="quant")


df = load()

st.title("LLM Inference Optimizer")
st.caption("Qwen2.5-1.5B on Apple M2 (llama.cpp, Metal). Cost uses an assumed hardware price.")

st.sidebar.header("Your constraints")
users = st.sidebar.selectbox("Concurrent users", [1, 2, 4, 8], index=2)
max_ttft = st.sidebar.number_input("Max p95 TTFT (ms)", value=1000.0, step=50.0)
min_tp = st.sidebar.number_input("Min aggregate tokens/s", value=0.0, step=10.0)
max_ppl = st.sidebar.number_input("Max perplexity (lower = better quality)", value=20.0, step=0.5)
hourly = st.sidebar.number_input("Hardware cost ($/hour, assumed)", value=0.04, format="%.3f")

df["cost_per_1m_usd"] = hourly / (df["agg_tok_s"] * 3600) * 1_000_000
cur = df[df["users"] == users]
ok = cur[
    (cur["ttft_p95_ms"] <= max_ttft)
    & (cur["agg_tok_s"] >= min_tp)
    & (cur["perplexity"] <= max_ppl)
].sort_values("cost_per_1m_usd")

cols = ["quant", "slots", "ttft_p95_ms", "agg_tok_s", "per_user_tok_s",
        "perplexity", "size_mib", "cost_per_1m_usd"]

st.subheader("Recommendation")
if ok.empty:
    st.warning("No configuration meets all constraints. Closest by throughput:")
    st.dataframe(cur.sort_values("agg_tok_s", ascending=False)[cols].head(3).round(3), hide_index=True)
else:
    b = ok.iloc[0]
    st.success(f"Use {b['quant']} with {int(b['slots'])} server slot(s). {len(ok)} configuration(s) meet your limits.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Throughput", f"{b['agg_tok_s']:.0f} tok/s")
    c2.metric("p95 TTFT", f"{b['ttft_p95_ms']:.0f} ms")
    c3.metric("Perplexity", f"{b['perplexity']:.2f}")
    c4.metric("Cost / 1M tokens", f"${b['cost_per_1m_usd']:.3f}")
    st.dataframe(ok[cols].round(3), hide_index=True)

st.subheader("Why: the measured data")
t1, t2, t3 = st.tabs(["Throughput vs users", "p95 TTFT vs users", "Quality vs speed"])
with t1:
    st.plotly_chart(px.line(df, x="users", y="agg_tok_s", color="quant", facet_col="slots",
                            markers=True, labels={"agg_tok_s": "aggregate tok/s"}),
                    use_container_width=True)
with t2:
    st.plotly_chart(px.line(df, x="users", y="ttft_p95_ms", color="quant", facet_col="slots",
                            markers=True, log_y=True, labels={"ttft_p95_ms": "p95 TTFT (ms, log)"}),
                    use_container_width=True)
with t3:
    st.plotly_chart(px.scatter(cur, x="agg_tok_s", y="perplexity", color="quant", symbol="slots",
                               hover_data=["ttft_p95_ms"],
                               title=f"{users} user(s): top-left is best (fast and accurate)"),
                    use_container_width=True)
st.caption("Each point is the median of 3 runs. Perplexity was measured on 20 WikiText-2 chunks, "
           "so small differences between Q4_K_M and Q8_0 are within error.")
