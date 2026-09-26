"""Write a random labeled JSONL dataset for exercising the eval harness end to end.
Labels are random, so with the mock backend expect ~chance accuracy and a large fitted T.
  python examples/make_mock_dataset.py --n 300 --out mock.jsonl
  python -m tokenfold.evaluate --data mock.jsonl --backend mock --out calibration.json"""

import argparse
import json
import random

QUESTIONS = [
    (
        {
            "type": "choice",
            "instructions": "Which team should handle `ticket`?",
            "criteria": {
                "billing": "Payment issues",
                "technical": "Bugs or outages",
                "sales": "Pricing questions",
            },
        },
        lambda r: r.choice(["billing", "technical", "sales"]),
    ),
    (
        {
            "type": "score",
            "instructions": "How frustrated is the customer in `ticket`?",
            "criteria": ["Calm", "Frustrated but civil", "Very angry"],
        },
        lambda r: r.randrange(3),
    ),
    ({"type": "noul", "instructions": "Is `ticket` urgent?"}, lambda r: r.random() < 0.5),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of states (each gets every question)")
    ap.add_argument("--out", default="mock.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    with open(a.out, "w") as f:
        for i in range(a.n):
            state = {"ticket": f"mock ticket #{i}: {rng.getrandbits(32):08x}"}
            for q, label in QUESTIONS:
                f.write(json.dumps({"state": state, "question": q, "label": label(rng)}) + "\n")
    print(f"wrote {a.n * len(QUESTIONS)} rows -> {a.out}")


if __name__ == "__main__":
    main()
