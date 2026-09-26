"""Row sampling from the Hugging Face datasets-server JSON API. No `datasets`/pyarrow dependency, and
pages are cached on disk so a rebuild with the same seed is offline and byte-identical."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import requests

API = "https://datasets-server.huggingface.co/rows"
PAGE = 20  # small pages at many random offsets: some splits are sorted by label (clinc, amazon, ...)


class HFRows:
    def __init__(self, cache_dir: str | Path = "data/raw", timeout: float = 60.0, pause: float = 0.5):
        self.cache = Path(cache_dir)
        self.timeout = timeout
        self.pause = pause

    def _get(self, dataset: str, config: str, split: str, offset: int, length: int) -> dict:
        path = self.cache / dataset / config / split / f"{offset}-{length}.json"
        if path.exists():
            return json.loads(path.read_text())
        params = {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
        for attempt in range(8):
            time.sleep(self.pause)  # the public API rate-limits bursts; this is only paid on cache misses
            r = requests.get(API, params=params, timeout=self.timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                retry_after = r.headers.get("Retry-After", "")
                time.sleep(float(retry_after) if retry_after.isdigit() else min(60, 2 ** (attempt + 1)))
                continue
            r.raise_for_status()
            break
        else:
            r.raise_for_status()
        data = r.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return data

    def sample(
        self, dataset: str, config: str, split: str, n: int, seed: int, page: int = PAGE
    ) -> tuple[list[dict], dict]:
        """~n rows from random pages of the split, plus the features map {name: type}.
        Returns more than n so callers can drop unusable rows and still reach n."""
        head = self._get(dataset, config, split, 0, 1)
        total = head["num_rows_total"]
        features = {f["name"]: f["type"] for f in head["features"]}
        rng = random.Random(f"{dataset}/{config}/{split}/{seed}")
        n_pages = min(-(-int(n * 1.5) // page), -(-total // page))
        offsets = sorted(rng.sample(range(0, total, page), n_pages))
        rows = []
        for off in offsets:
            rows += [r["row"] for r in self._get(dataset, config, split, off, page)["rows"]]
        rng.shuffle(rows)
        return rows, features
