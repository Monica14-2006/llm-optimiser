import asyncio
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.request

import aiohttp

QUANTS = {
    "Q2_K": "models/qwen2.5-1.5b-instruct-q2_k.gguf",
    "Q4_K_M": "models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "Q8_0": "models/qwen2.5-1.5b-instruct-q8_0.gguf",
}
SLOTS = [1, 2, 4, 8]
USERS = [1, 2, 4, 8]
REPEATS = 3
CTX_PER_SLOT = 2048
MAX_TOKENS = 128
PORT = 8090
URL = f"http://localhost:{PORT}/v1/chat/completions"
PROMPT = "Write a short paragraph explaining why the sky is blue."
OUT = "results/sweep_results.csv"

QUICK = "--quick" in sys.argv
if QUICK:
    QUANTS = {"Q4_K_M": QUANTS["Q4_K_M"]}
    SLOTS = [1, 4]
    USERS = [1, 4]
    REPEATS = 1
    OUT = "results/sweep_quick.csv"


def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def start_server(model, slots):
    cmd = [
        "llama-server", "-m", model,
        "-c", str(CTX_PER_SLOT * slots), "-np", str(slots),
        "--port", str(PORT),
    ]
    log = open("results/server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)
    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError("server exited early, see results/server.log")
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            pass
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("server did not become ready in 120 s")


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


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
    n = 0
    async with session.post(URL, json=body) as resp:
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            choices = json.loads(data).get("choices") or []
            if not choices:
                continue
            if choices[0].get("delta", {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
    total = time.perf_counter() - t0
    tpot = (total - ttft) / (n - 1) if n > 1 else float("nan")
    return ttft, tpot, n, total


async def run_round(users):
    conn = aiohttp.TCPConnector(force_close=True, limit=0)
    async with aiohttp.ClientSession(connector=conn) as s:
        t0 = time.perf_counter()
        res = await asyncio.gather(*[one_request(s) for _ in range(users)])
        wall = time.perf_counter() - t0
    return res, wall


async def measure(users):
    await run_round(1)  # warm-up
    reps = []
    for _ in range(REPEATS):
        res, wall = await run_round(users)
        ttfts = [r[0] * 1000 for r in res]
        lats = [r[3] for r in res]
        reps.append({
            "ttft_mean_ms": statistics.mean(ttfts),
            "ttft_p95_ms": pct(ttfts, 95),
            "tpot_mean_ms": statistics.mean(r[1] * 1000 for r in res),
            "latency_p95_s": pct(lats, 95),
            "agg_tok_s": sum(r[2] for r in res) / wall,
        })
        await asyncio.sleep(3)
    out = {k: round(statistics.median(r[k] for r in reps), 2) for k in reps[0]}
    out["per_user_tok_s"] = round(1000 / out["tpot_mean_ms"], 1)
    out["agg_tok_s_min"] = round(min(r["agg_tok_s"] for r in reps), 1)
    out["agg_tok_s_max"] = round(max(r["agg_tok_s"] for r in reps), 1)
    return out


async def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    total = len(QUANTS) * len(SLOTS) * len(USERS)
    done = 0
    t_start = time.time()
    for qname, path in QUANTS.items():
        for slots in SLOTS:
            print(f"\n== starting server: {qname}, slots={slots}", flush=True)
            proc = start_server(path, slots)
            try:
                for users in USERS:
                    m = await measure(users)
                    row = {"quant": qname, "slots": slots, "users": users, **m}
                    rows.append(row)
                    done += 1
                    el = (time.time() - t_start) / 60
                    print(f"[{done}/{total}] {el:.1f} min  {row}", flush=True)
                    with open(OUT, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=rows[0].keys())
                        w.writeheader()
                        w.writerows(rows)
                    await asyncio.sleep(5)
            finally:
                stop_server(proc)
                time.sleep(20)  # cool-down between servers
    print("\nsaved", OUT)


asyncio.run(main())
