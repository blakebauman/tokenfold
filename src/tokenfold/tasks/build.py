"""Build the eval set: sample every task and write data/eval/<task>.jsonl plus data/eval/all.jsonl.

  uv run tokenfold-build-eval                      # all tasks, 100 rows each
  uv run tokenfold-build-eval --n 200 --tasks ag_news,rte
Rows are validated through the same parser the server uses, so a malformed task fails here, not mid-eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..schema import parse_question
from . import BY_NAME, TASKS, Task
from .hf import HFRows


def build_task(task: Task, hf: HFRows, n: int, seed: int) -> list[dict]:
    raw, features = hf.sample(task.dataset, task.config, task.split, n, seed, task.page)
    names = features.get(task.label_field, {}).get("names")
    if task.expect_names is not None and tuple(names or ()) != task.expect_names:
        raise ValueError(f"{task.name}: label names changed upstream: {names}")
    question = task.question(names)
    parse_question(task.name, question)
    out = []
    for row in raw:
        conv = task.convert(row, names)
        if conv is None:
            continue
        state, label = conv
        out.append(
            {"task": task.name, "heldout": task.heldout, "state": state, "question": question, "label": label}
        )
        if len(out) == n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="rows per task (a task's own cap wins if smaller)")
    ap.add_argument("--tasks", help="comma-separated task names (default: all)")
    ap.add_argument("--out-dir", default="data/eval")
    ap.add_argument("--cache-dir", default="data/raw")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tasks = [BY_NAME[t] for t in a.tasks.split(",")] if a.tasks else TASKS
    hf = HFRows(a.cache_dir)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    everything = []
    for t in tasks:
        rows = build_task(t, hf, min(a.n, t.n or a.n), a.seed)
        (out_dir / f"{t.name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        everything += rows
        print(f"{t.name:18} {t.type:6} {'heldout' if t.heldout else '':8} {len(rows):4} rows")
    (out_dir / "all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in everything))
    print(f"{len(everything)} rows -> {out_dir / 'all.jsonl'}")


if __name__ == "__main__":
    main()
