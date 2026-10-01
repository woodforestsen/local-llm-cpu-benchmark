r"""
质量对比测试：拿同一批 20 道题去问每个模型，把回答存下来并给客观题自动判分。

用法：
    cd D:\桌面\issue1
    python quality_test.py

输出：
    answers\answers_<模型名>.md   每个模型的全部回答
    quality_results.csv           客观题得分汇总
"""

import csv
import json
import os
import sys
import urllib.error
import urllib.request

# Windows 控制台默认是 GBK，打印英文以外的符号会崩，这里强制成 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ============ 配置 ============

MODELS = [
    "qwen3:1.7b",
    "qwen3:1.7b-q8_0",
]

THREADS = 16
NUM_PREDICT = 512   # 留够额度；即使模型仍输出了思考内容，答案也不会被截断
API = "http://127.0.0.1:11434/api/generate"

# =========================================

# 题目格式：(编号, 分类, 题目, 判分关键词)
#   - 判分关键词不为 None 的算「客观题」，回答里出现任一个关键词就算 1 分
#   - 为 None 的算「主观题」，需要你自己看回答打分
QUESTIONS = [
    # --- 事实问答（客观）---
    ("A1", "事实", "中国的首都是哪座城市？只回答城市名。", ["北京"]),
    ("A2", "事实", "水的化学式是什么？只回答化学式。", ["H2O", "h2o"]),
    ("A3", "事实", "一年有多少个月？只回答数字。", ["12", "十二"]),
    ("A4", "事实", "光在真空中的速度大约是多少？用公里每秒表示，只回答数字。", ["30", "299792", "3×10", "3*10"]),
    ("A5", "事实", "地球上最大的洋是哪个？只回答名称。", ["太平洋"]),

    # --- 算术与推理（客观）---
    ("B1", "算术", "计算：37 × 24 = ? 只回答数字。", ["888"]),
    ("B2", "算术", "计算：(15 + 27) × 3 - 8 = ? 只回答数字。", ["118"]),
    ("B3", "算术", "一个班有 42 人，其中 3/7 是女生，女生有多少人？只回答数字。", ["18"]),
    ("B4", "推理", "小明比小红高，小红比小刚高。谁最矮？只回答名字。", ["小刚"]),
    ("B5", "算术", "一件商品原价 200 元，先涨价 10%，再打八折，最终价格是多少元？只回答数字。", ["176"]),

    # --- 中文写作与总结（主观）---
    ("C1", "写作", "用三句话向一个十岁小孩解释什么是「人工智能」。", None),
    ("C2", "写作", "把下面这段话压缩成一句话：「昨天下午三点，公司召开了第三季度总结会议，会上各部门负责人汇报了工作进展，最后总经理部署了第四季度的重点任务。」", None),
    ("C3", "写作", "写一句不超过 20 个字的广告语，推销一款可以自动记录睡眠的手环。", None),
    ("C4", "写作", "用中文解释「过拟合」是什么意思，举一个生活中的类比。", None),
    ("C5", "写作", "给同事写一条请假消息，说明你明天上午要去医院，下午回来上班。", None),

    # --- 指令遵循（主观，看格式是否听话）---
    ("D1", "指令", "用恰好 5 个字回答：今天天气怎么样？", None),
    ("D2", "指令", "把「人工智能正在改变世界」这句话倒过来念（从最后一个字到第一个字）。", None),
    ("D3", "指令", "列出三种水果，每种水果前面加一个数字编号，格式为「1. 苹果」，不要有任何其他文字。", None),
    ("D4", "指令", "回答时不要说任何解释，只输出 JSON：{\"status\": \"ok\"}", None),
    ("D5", "指令", "用英文回答：What is the capital of France? 只回答一个单词。", ["Paris", "paris"]),
]


def _post(payload):
    req = urllib.request.Request(
        API,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ask(model, prompt):
    base = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {"num_predict": NUM_PREDICT, "num_thread": THREADS, "temperature": 0, "seed": 42},
    }
    # Qwen3 这类「思考模型」默认先输出一大段推理过程，会把 num_predict 的额度吃光，
    # 正式回答还没开始就截断了 —— 表现就是答案栏一片空白。
    # 关掉思考模式，让它直接给答案。
    try:
        data = _post({**base, "think": False})
    except urllib.error.HTTPError:
        # 个别旧模型不认识 think 参数，退回普通调用
        data = _post(base)
    return (data.get("response") or "").strip()


def safe_name(model):
    return model.replace(":", "_").replace("/", "_")


def main():
    os.makedirs("answers", exist_ok=True)
    summary = []

    for model in MODELS:
        print(f"\n===== {model} =====")
        lines = [f"# {model} 的回答\n"]
        got = 0
        total = 0
        try:
            ask(model, "你好")  # 预热
        except urllib.error.URLError as e:
            print(f"  x 连不上 Ollama 或模型不存在：{e}")
            continue

        for qid, cat, q, keywords in QUESTIONS:
            try:
                ans = ask(model, q)
            except Exception as e:
                ans = f"(出错: {e})"
            is_obj = keywords is not None
            mark = ""
            if is_obj:
                total += 1
                hit = any(k.lower() in ans.lower() for k in keywords)
                got += 1 if hit else 0
                mark = "  [OK]" if hit else "  [NG]"
            lines.append(f"## {qid} [{cat}]\n\n**问：** {q}\n\n**答：** {ans}\n")
            print(f"  {qid} [{cat}]{mark}")

        path = os.path.join("answers", f"answers_{safe_name(model)}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  -> 客观题 {got}/{total}，回答已存到 {path}")
        summary.append({
            "model": model,
            "objective_score": f"{got}/{total}",
            "objective_pct": round(got / total * 100, 1) if total else 0,
        })

    if summary:
        print("\n===== 客观题汇总 =====")
        for r in summary:
            print(f"{r['model']:<26}{r['objective_score']:>8}   {r['objective_pct']}%")
        with open("quality_results.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["model", "objective_score", "objective_pct"])
            w.writeheader()
            w.writerows(summary)
        print("\n已写入 quality_results.csv")
        print("提醒：C 组和 D 组是主观题，必须自己抽检至少 5 题再下结论。")


if __name__ == "__main__":
    main()
