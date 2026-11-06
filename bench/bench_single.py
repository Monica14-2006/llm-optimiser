import asyncio
import json
import random
import statistics
import time

import aiohttp

URL = "http://localhost:8080/v1/chat/completions"
PROMPT = "Write a short paragraph explaining why the sky is blue."
MAX_TOKENS = 128
N_WARMUP = 1
N_RUNS = 5


async def one_request(session):
    # Random prefix so the server cannot reuse a cached prompt
    prompt = f"[{random.randint(0, 10**9)}] {PROMPT}"
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "cache_prompt": False,
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
    return {
        "ttft_ms": ttft * 1000,
        "tpot_ms": tpot * 1000,
        "tokens": n_tokens,
        "total_s": total,
        "tok_per_s": n_tokens / total,
    }


async def main():
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as session:
        for _ in range(N_WARMUP):
            await one_request(session)
        results = []
        for i in range(N_RUNS):
            r = await one_request(session)
            results.append(r)
            print(
                f"run {i + 1}: ttft={r['ttft_ms']:.0f} ms  "
                f"tpot={r['tpot_ms']:.1f} ms  tokens={r['tokens']}  "
                f"total={r['total_s']:.2f} s  speed={r['tok_per_s']:.1f} tok/s"
            )
    print("\n--- averages over", N_RUNS, "runs ---")
    for key in ["ttft_ms", "tpot_ms", "tokens", "total_s", "tok_per_s"]:
        vals = [r[key] for r in results]
        print(f"{key:10s} mean={statistics.mean(vals):8.2f}  stdev={statistics.pstdev(vals):6.2f}")


asyncio.run(main())
