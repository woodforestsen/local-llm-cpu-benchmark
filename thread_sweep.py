r"""
线程数扫描：找出你这台 CPU 的性价比拐点。

跑法：
    cd D:\桌面\issue1
    python thread_sweep.py

输出：
    thread_sweep.csv   — 原始数据

方法论说明（重要）：
    这里每次配置取 5 次中的**最大值**，而不是中位数。
    原因：CPU 上的干扰（杀毒软件、后台更新、其他进程）只会让结果**变慢**，
    不会让它变快。所以同一配置下最快的那个数，最接近机器的真实能力。
    这叫 best-of-N，是 CPU 性能测试的常规做法。
"""

import csv
import json
import sys
import urllib.error
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ============ 配置 ============

MODELS = [
    "qwen3:1.7b",
    "qwen3:1.7b-q8_0",
]

# 要扫哪些线程数。你的 CPU 有 18 个逻辑核，扫到 18 为止。
THREAD_LIST = [4, 8, 12, 16, 18]

NUM_PREDICT = 128
RUNS = 5          # 每个配置跑几次，取最快
API = "http://127.0.0.1:11434/api/generate"

# =========================================

_SENTENCE = "请阅读下面这段文字并在最后回复“收到”两个字。这段话只是为了把输入长度撑到大约五百个 token，内容本身没有任何含义。"
PROMPT_BODY = _SENTENCE * 40


def call_ollama(model, nonce, threads, keep_alive="10m"):
    payload = {
        "model": model,
        # nonce 放最前面，强制每次重新算输入，避免命中前缀缓存导致 pp 虚高
        "prompt": f"[{nonce}]" + PROMPT_BODY,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "num_predict": NUM_PREDICT,
            "num_thread": threads,
            "temperature": 0,
            "seed": 42,
        },
    }
    req = urllib.request.Request(
        API,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode("utf-8"))


def measure(model, threads):
    try:
        call_ollama(model, f"warm-{threads}", threads)   # 预热
    except urllib.error.URLError as e:
        print(f"    x 连不上或模型不存在：{e}")
        return None

    pp_list, tg_list = [], []
    for i in range(RUNS):
        d = call_ollama(model, f"t{threads}-r{i+1}", threads)
        pp_ns = d.get("prompt_eval_duration") or 0
        tg_ns = d.get("eval_duration") or 0
        if pp_ns:
            pp_list.append((d.get("prompt_eval_count") or 0) / (pp_ns / 1e9))
        if tg_ns:
            tg_list.append((d.get("eval_count") or 0) / (tg_ns / 1e9))
    return {
        "model": model,
        "threads": threads,
        "pp_tps": round(max(pp_list), 2) if pp_list else None,
        "tg_tps": round(max(tg_list), 2) if tg_list else None,
        "runs": RUNS,
    }


def main():
    rows = []
    for model in MODELS:
        print(f"\n===== {model} =====")
        for threads in THREAD_LIST:
            print(f"  线程 {threads:>2} ...", end=" ", flush=True)
            r = measure(model, threads)
            if r:
                rows.append(r)
                print(f"pp {r['pp_tps']:>7} tok/s   tg {r['tg_tps']:>6} tok/s")
            else:
                print("失败")
        # 测完一个模型就把它从内存卸载，避免下一个模型受内存压力影响
        try:
            call_ollama(model, "unload", 1, keep_alive="0s")
        except Exception:
            pass

    if not rows:
        print("\n没有任何结果。")
        return

    print("\n===== 线程数 vs 速度（best-of-%d）=====" % RUNS)
    header = f"{'模型':<22}{'线程':>6}{'pp tok/s':>12}{'tg tok/s':>12}{'相对最快':>10}"
    print(header)
    print("-" * len(header))

    best = {}
    for r in rows:
        if r["tg_tps"]:
            best[r["model"]] = max(best.get(r["model"], 0), r["tg_tps"])

    for r in rows:
        pct = f"{r['tg_tps'] / best[r['model']] * 100:.0f}%" if r["tg_tps"] and best.get(r["model"]) else "-"
        print(f"{r['model']:<22}{r['threads']:>6}{r['pp_tps']:>12}{r['tg_tps']:>12}{pct:>10}")

    with open("thread_sweep.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["model", "threads", "pp_tps", "tg_tps", "runs"])
        w.writeheader()
        w.writerows(rows)
    print("\n已写入 thread_sweep.csv")
    print("看「相对最快」那一列：从哪一档开始不再接近 100%，那就是你这台机器的性价比拐点。")


if __name__ == "__main__":
    main()
