"""
在 CPU 上测量 Ollama 模型的「提示处理速度」与「生成速度」。

用法：
    1. 先确保 Ollama 在运行（任务栏有图标即可）
    2. 在命令行执行：  python bench_ollama.py
    3. 结果会打印在屏幕上，同时写入 bench_results.csv

它测的就是 llama-bench 里的 pp512 / tg128：
    pp_tps = 提示处理速度（每秒处理多少个输入 token）
    tg_tps = 生成速度（每秒生成多少个 token）

不用装任何第三方库，只用 Python 自带的模块。
"""

import csv
import json
import os
import statistics
import time
import urllib.error
import urllib.request

# ============ 需要你确认的配置 ============

# 要测哪些模型。名字必须是 `ollama list` 里能看到的名字。
MODELS = [
    "qwen3:1.7b",          # Q4_K_M 档（Ollama 默认）
    "qwen3:1.7b-q8_0",     # 高质量档
    "qwen3-lora:latest",   # 你已有的那个，一起对比
]

# 用几个线程。你的 CPU 有 18 个逻辑核，先用 16（留两个给系统）。
THREADS = 16

# 让模型生成多少个 token（对应 tg128 里的 128）
NUM_PREDICT = 128

# 每轮测几次，取中位数
RUNS = 3

API = "http://127.0.0.1:11434/api/generate"
OUT_CSV = "bench_results.csv"

# =========================================

# 造一段大约 500 个 token 的输入（对应 pp512 里的 512）
_SENTENCE = "请阅读下面这段文字并在最后回复“收到”两个字。这段话只是为了把输入长度撑到大约五百个 token，内容本身没有任何含义。"
PROMPT_BODY = _SENTENCE * 40


def call_ollama(model, nonce):
    """
    跑一次推理，返回本次的原始统计。

    nonce 会被放到提示词**最前面**。这很关键：Ollama 有前缀缓存，
    如果每次提示词完全一样，第二次之后输入处理会直接命中缓存，
    测出来的 pp 速度会虚高几十倍（看起来像几万 tok/s，CPU 根本做不到）。
    在开头加一个每次都不同的短标记，就能强制它重新算一遍。
    """
    payload = {
        "model": model,
        "prompt": f"[{nonce}]" + PROMPT_BODY,
        "stream": False,
        "keep_alive": "10m",          # 让模型留在内存里，避免每次都重新加载
        "options": {
            "num_predict": NUM_PREDICT,
            "num_thread": THREADS,
            "temperature": 0,
            "seed": 42,
        },
    }
    req = urllib.request.Request(
        API,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    wall_start = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    wall = time.time() - wall_start
    return data, wall


def to_tps(data, wall):
    """把 Ollama 返回的纳秒计时换算成 每秒 token 数。"""
    pp_count = data.get("prompt_eval_count") or 0
    pp_ns = data.get("prompt_eval_duration") or 0
    tg_count = data.get("eval_count") or 0
    tg_ns = data.get("eval_duration") or 0
    return {
        "pp_count": pp_count,
        "tg_count": tg_count,
        "pp_tps": round(pp_count / (pp_ns / 1e9), 2) if pp_ns else None,
        "tg_tps": round(tg_count / (tg_ns / 1e9), 2) if tg_ns else None,
        "load_s": round((data.get("load_duration") or 0) / 1e9, 2),
        "wall_s": round(wall, 2),
    }


def bench(model):
    print(f"\n===== {model} =====")
    print("预热（第一次调用包含模型加载时间，不计入结果）...")
    try:
        warm, _ = call_ollama(model, "warmup")
    except urllib.error.URLError as e:
        print(f"  x 连不上 Ollama：{e}")
        print("    请确认 Ollama 正在运行，且模型名字拼写正确（ollama list）。")
        return None
    print(f"  预热完成，加载耗时 {((warm.get('load_duration') or 0)/1e9):.1f} 秒")

    results = []
    for i in range(RUNS):
        data, wall = call_ollama(model, f"run{i+1}")
        r = to_tps(data, wall)
        results.append(r)
        print(
            f"  第 {i+1}/{RUNS} 次："
            f"输入 {r['pp_count']} tok @ {r['pp_tps']} tok/s  |  "
            f"生成 {r['tg_count']} tok @ {r['tg_tps']} tok/s"
        )

    pp = statistics.median([r["pp_tps"] for r in results if r["pp_tps"]])
    tg = statistics.median([r["tg_tps"] for r in results if r["tg_tps"]])
    load_s = results[0]["load_s"]

    print(f"  -> 中位数：pp {pp:.2f} tok/s，tg {tg:.2f} tok/s")
    if pp > 5000:
        print("  (!) pp 仍然偏高，可能还是命中了缓存，检查提示词是否每次不同")
    return {
        "model": model,
        "threads": THREADS,
        "pp_tokens": results[0]["pp_count"],
        "pp_tps": round(pp, 2),
        "tg_tokens": results[0]["tg_count"],
        "tg_tps": round(tg, 2),
        "load_s": load_s,
    }


def main():
    print("CPU 线程数设定：", THREADS)
    print("输入长度约：", len(PROMPT_BODY), "字符")
    rows = []
    for m in MODELS:
        r = bench(m)
        if r:
            rows.append(r)

    if not rows:
        print("\n没有任何结果，请先确认模型已下载：ollama list")
        return

    print("\n===== 汇总 =====")
    header = f"{'模型':<26}{'pp tok/s':>10}{'tg tok/s':>10}{'加载(s)':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:<26}{r['pp_tps']:>10}{r['tg_tps']:>10}{r['load_s']:>10}")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_CSV)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写入：{path}")


if __name__ == "__main__":
    main()
