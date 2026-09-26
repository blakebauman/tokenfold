"""Eval + calibration harness.

Dataset: JSONL, one row per (state, question, label), optionally tagged with `task` / `heldout`
(see tokenfold.tasks):
  {"task": "ag_news", "heldout": false, "state": <str|obj>,
   "question": {"type": "choice", "instructions": ..., "criteria": {...}}, "label": "world"}
  label is: an option key (choice) | a level index int (score) | true/false (noul)

  uv run tokenfold-evaluate --data data/eval/all.jsonl --backend llama --out calibration.json \
      --report data/reports/baseline.json

Each task is split into a calibration part and a test part. One temperature curve per question type
(T and an option-count slope, see calibration.py) is fit on the calibration parts of the non-heldout
tasks, then every task's test part is scored raw (T=1) and calibrated. Heldout tasks therefore show
whether the fitted temperatures transfer to unseen question shapes. Rows sharing the same state are
batched into one request (same as production), and each request's wall time is recorded as latency.

Scoring is the slow part (~20 min for the llama baseline), so `--save-logits` writes the raw logits and
`--from-logits` refits and re-reports from them without touching a model (no latency then)."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from .answers import choice_confidence, softmax
from .calibration import TYPES, Calibrator
from .engine import Engine
from .schema import Request, parse_question
from .server import BACKENDS, make_engine


def load_rows(path: str) -> list[dict]:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        r.setdefault("task", "default")
        r.setdefault("heldout", False)
        r["q"] = parse_question("row", r["question"])
        q = r["q"]
        lab = r["label"]
        if q.type == "choice":
            r["y"] = q.options.index(lab)
        elif q.type == "score":
            r["y"] = int(lab)
        else:
            r["y"] = 0 if lab in (True, "true", "yes", 1) else 1
    return rows


def collect_logits(engine: Engine, rows: list[dict], progress: bool = False) -> list[float]:
    """Attach raw backend logits to each row, batching rows by identical state.
    Returns per-request latency in milliseconds."""
    by_state: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_state[json.dumps(r["state"], sort_keys=True)].append(i)
    latencies = []
    t0 = time.perf_counter()
    for n, idxs in enumerate(by_state.values(), 1):
        req = Request(state=rows[idxs[0]]["state"], questions={f"q{i}": rows[i]["q"] for i in idxs})
        start = time.perf_counter()
        out = engine.raw_logits(req)
        latencies.append((time.perf_counter() - start) * 1000)
        for i in idxs:
            rows[i]["logits"] = out[f"q{i}"]
        if progress and (n % 50 == 0 or n == len(by_state)):
            elapsed = time.perf_counter() - t0
            eta = elapsed / n * (len(by_state) - n)
            print(
                f"  {n}/{len(by_state)} requests  {elapsed:.0f}s elapsed  ~{eta:.0f}s left", file=sys.stderr
            )
    return latencies


def row_key(r: dict) -> str:
    return hashlib.sha1(json.dumps([r["state"], r["question"]], sort_keys=True).encode()).hexdigest()[:16]


def save_logits(path: str, model: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "rows": [{"key": row_key(r), "logits": r["logits"]} for r in rows]}
    Path(path).write_text(json.dumps(payload))


def load_logits(path: str, rows: list[dict]) -> str:
    """Attach saved logits to rows, matched by (state, question) key, so a file may cover more rows than
    --data/--tasks selects (e.g. after merging scores for newly added tasks). Returns the model name."""
    saved = json.loads(Path(path).read_text())
    by_key = {s["key"]: s["logits"] for s in saved["rows"]}
    missing = sum(row_key(r) not in by_key for r in rows)
    if missing:
        raise SystemExit(f"{path} has no logits for {missing} of {len(rows)} rows; score them first")
    for r in rows:
        r["logits"] = by_key[row_key(r)]
    return saved["model"]


def ece(probs_list, ys, bins=10) -> float:
    conf = np.array([max(p) for p in probs_list])
    pred = np.array([int(np.argmax(p)) for p in probs_list])
    correct = (pred == np.array(ys)).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def metrics(rows: list[dict], cal: Calibrator | None = None) -> dict:
    """cal=None scores the raw logits (T=1)."""
    probs = [cal.apply(r["q"].type, r["logits"]) if cal else softmax(r["logits"]) for r in rows]
    ys = [r["y"] for r in rows]
    nll = -np.mean([np.log(max(p[y], 1e-12)) for p, y in zip(probs, ys, strict=True)])
    acc = np.mean([int(np.argmax(p)) == y for p, y in zip(probs, ys, strict=True)])
    conf = np.mean([choice_confidence(p) for p in probs])
    out = {
        "n": len(rows),
        "acc": round(float(acc), 4),
        "nll": round(float(nll), 4),
        "ece": round(ece(probs, ys), 4),
        "mean_confidence": round(float(conf), 4),
    }
    if rows and all(r["q"].type == "score" for r in rows):
        # the served score is the expected level, so judge that, not just the argmax
        exp = [float((np.arange(len(p)) * p).sum()) for p in probs]
        out["mae"] = round(float(np.mean([abs(e - y) for e, y in zip(exp, ys, strict=True)])), 4)
    return out


def latency_summary(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {
        "requests": len(a),
        "p50_ms": round(float(np.percentile(a, 50)), 1),
        "p95_ms": round(float(np.percentile(a, 95)), 1),
        "mean_ms": round(float(a.mean()), 1),
    }


def split_by_task(rows: list[dict], cal_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    fit, test = [], []
    for task, task_rows in by_task.items():
        task_rows = task_rows[:]
        # seeded per task, so adding or filtering tasks never reshuffles another task's split
        random.Random(f"{seed}/{task}").shuffle(task_rows)
        k = int(len(task_rows) * cal_frac)
        fit += task_rows[:k]
        test += task_rows[k:]
    return fit, test


def evaluate(rows: list[dict], cal_frac: float = 0.5, seed: int = 0) -> tuple[dict, Calibrator]:
    """rows must already carry logits. Returns (report, fitted calibrator)."""
    fit, test = split_by_task(rows, cal_frac, seed)
    fit = [r for r in fit if not r["heldout"]]
    cal = Calibrator()
    per_type = {}
    for t in TYPES:
        fit_t = [r for r in fit if r["q"].type == t]
        test_t = [r for r in test if r["q"].type == t]
        if len(fit_t) < 10 or not test_t:
            continue
        T = cal.fit(t, [r["logits"] for r in fit_t], [r["y"] for r in fit_t])
        per_type[t] = {
            "T": round(T, 4),
            "slope": round(cal.slope[t], 4),
            "fit_n": len(fit_t),
            "before": metrics(test_t),
        }
    for t in per_type:  # after every fit, so each type is scored with its own curve
        per_type[t]["after"] = metrics([r for r in test if r["q"].type == t], cal)
    per_task = {}
    for task in dict.fromkeys(r["task"] for r in test):
        tr = [r for r in test if r["task"] == task]
        t = tr[0]["q"].type
        per_task[task] = {
            "type": t,
            "heldout": tr[0]["heldout"],
            "n_options": len(tr[0]["logits"]),
            "T": round(cal.temperature(t, len(tr[0]["logits"])), 4),
            "before": metrics(tr),
            "after": metrics(tr, cal),
        }
    return {"per_type": per_type, "per_task": per_task}, cal


def print_table(report: dict) -> None:
    print(
        f"\n{'task':18} {'type':6} {'held':4} {'n':>4} {'opts':>4} {'T':>5}  {'acc':>5}"
        f"  {'ece raw':>7} {'ece cal':>7}  {'nll raw':>7} {'nll cal':>7}  {'mae':>5}"
    )
    for task, m in sorted(
        report["per_task"].items(), key=lambda kv: (kv[1]["type"], kv[1]["heldout"], kv[0])
    ):
        b, a = m["before"], m["after"]
        mae = f"{a['mae']:5.2f}" if "mae" in a else ""
        print(
            f"{task:18} {m['type']:6} {'yes' if m['heldout'] else '':4} {a['n']:4}"
            f" {m.get('n_options', 0):4} {m.get('T', 0):5.2f}  {a['acc']:5.2f}"
            f"  {b['ece']:7.3f} {a['ece']:7.3f}  {b['nll']:7.3f} {a['nll']:7.3f}  {mae}"
        )
    for t, m in report["per_type"].items():
        b, a = m["before"], m["after"]
        print(
            f"[{t}] T={m['T']} slope={m.get('slope', 0)}  acc {a['acc']:.3f}"
            f"  ece {b['ece']:.3f} -> {a['ece']:.3f}"
            f"  nll {b['nll']:.3f} -> {a['nll']:.3f}"
        )
    if "latency" in report:
        lat = report["latency"]
        print(f"latency: p50 {lat['p50_ms']} ms  p95 {lat['p95_ms']} ms  over {lat['requests']} requests")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="mock")
    ap.add_argument("--out", default="calibration.json", help="fitted temperatures")
    ap.add_argument("--report", help="write the full JSON report here")
    ap.add_argument("--tasks", help="comma-separated task names (default: all in --data)")
    ap.add_argument("--cal-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-logits", help="write raw logits here after scoring")
    ap.add_argument("--from-logits", help="skip scoring; reuse logits written by --save-logits")
    a = ap.parse_args()

    rows = load_rows(a.data)
    if a.tasks:
        keep = set(a.tasks.split(","))
        rows = [r for r in rows if r["task"] in keep]
    latency = {}
    if a.from_logits:
        model = load_logits(a.from_logits, rows)
        print(f"reusing {len(rows)} rows of {model} logits from {a.from_logits}", file=sys.stderr)
    else:
        engine = make_engine(a.backend, None)
        model = engine.backend.name
        print(f"scoring {len(rows)} rows with {model}", file=sys.stderr)
        latency = {"latency": latency_summary(collect_logits(engine, rows, progress=True))}
        if a.save_logits:
            save_logits(a.save_logits, model, rows)
            print(f"logits -> {a.save_logits}", file=sys.stderr)

    report, cal = evaluate(rows, a.cal_frac, a.seed)
    report = {"model": model, "data": a.data, "rows": len(rows), **report, **latency}
    cal.save(a.out)
    print_table(report)
    print(f"temperatures -> {a.out}")
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2))
        print(f"report -> {a.report}")


if __name__ == "__main__":
    main()
