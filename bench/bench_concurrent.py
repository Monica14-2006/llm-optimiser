import asyncio
import csv
import json
import random
import statistics
import sys
import time

import aiohttp

URL = "http://localhost:8080/v1/chat/completions"
PROMPT = "Write a short paragraph explaining why the sky is blue."
MAX_TOKENS = 128
LEVELS = [1, 2, 4, 8]
ROUNDS = 3
LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"


def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


async def one_request(session):
    prompt = f"[{random.randint(0, 10**9)}] {PROMPT}"
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "cache_prompt": False,
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0
    async with session.post(URL, json=body) as resp:
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("delta", {}).get("content")
            if text:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_tokens += 1
    total = time.perf_counter() - t0
    tpot = (total - ttft) / (n_tokens - 1) if n_tokens > 1 else float("nan")
    return {"ttft": ttft, "tpot": tpot, "tokens": n_tokens, "total": total}


async def run_round(c):
    connector = aiohttp.TCPConnector(force_close=True, limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[one_request(session) for _ in range(c)])
        wall = time.perf_counter() - t0
    return results, wall


async def main():
    await run_round(1)  # warm-up, ignored
    rows = []
    for c in LEVELS:
        all_reqs = []
        agg = []
        for _ in range(ROUNDS):
            results, wall = await run_round(c)
            all_reqs.extend(results)
            agg.append(sum(r["tokens"] for r in results) / wall)
            await asyncio.sleep(2)
        ttfts = [r["ttft"] * 1000 for r in all_reqs]
        tpots = [r["tpot"] * 1000 for r in all_reqs]
        totals = [r["total"] for r in all_reqs]
        row = {
            "label": LABEL,
            "concurrency": c,
            "ttft_mean_ms": round(statistics.mean(ttfts), 1),
            "ttft_p95_ms": round(pct(ttfts, 95), 1),
            "tpot_mean_ms": round(statistics.mean(tpots), 2),
            "latency_mean_s": round(statistics.mean(totals), 2),
            "latency_p95_s": round(pct(totals, 95), 2),
            "per_user_tok_s": round(statistics.mean(1000 / t for t in tpots), 1),
            "agg_tok_s": round(statistics.mean(agg), 1),
        }
        rows.append(row)
        print(row)
        await asyncio.sleep(5)  # cool-down between levels
    path = f"results/concurrency_{LABEL}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("saved", path)


asyncio.run(main())
