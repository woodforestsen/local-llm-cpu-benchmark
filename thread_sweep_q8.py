"""只重测 q8_0 的线程曲线（用于验证上次是不是热降频）。"""

import thread_sweep as t

t.MODELS = ["qwen3:1.7b-q8_0"]
t.RUNS = 5

t.main()
