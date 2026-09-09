#!/usr/bin/env python3
"""Audit persisted data and paired intervals independently of training."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
r = json.loads((root / "result.json").read_text())
data = json.loads((root / "dataset.json").read_text())
rows = list(csv.DictReader((root / "masks.csv").open()))
assert r["metadata"]["source_sha256"] == hashlib.sha256((root / "run.py").read_bytes()).hexdigest()
assert len(rows) == 980
assert len(r["runs"]) == 20
assert r["counts"]["tuning_runs"] == 40
assert not any(v["at_grid_boundary"] for v in r["selected_lrs"].values())
split_ids = [set(s["indices"]) for s in data["splits"].values()]
assert len(set.union(*split_ids)) == 1797
assert sum(map(len, split_ids)) == 1797
for row in rows:
    assert len(row["mask"]) == int(row["model_depth"])
    assert row["mask"].count("1") == int(row["retained_depth"])
    assert 0 <= float(row["accuracy"]) <= 1
    assert float(row["ce"]) >= 0
    assert abs(float(row["ce"])-float(row["full_ce"])-float(row["excess_ce"])) < 1e-12
    assert (row["mask"][0] == "1") == (row["keeps_first"] == "True")
for seed in r["protocol"]["evaluation_seeds"]:
    for treatment in r["configs"]:
        group = [q for q in rows if int(q["seed"]) == seed and q["treatment"] == treatment]
        depth = r["configs"][treatment]["depth"]
        assert len(group) == 2**depth-1
        assert len({q["mask"] for q in group}) == len(group)
        full = next(q for q in group if q["mask"] == "1"*depth)
        assert float(full["excess_ce"]) == 0
        assert all(q["full_ce"] == full["ce"] for q in group)
for p in r["primary_comparisons"]:
    diffs = []
    for seed in r["protocol"]["evaluation_seeds"]:
        sparse = next(q for q in rows if int(q["seed"]) == seed and q["treatment"] == p["treatment"] and q["mask"] == "111100")
        dense = next(q for q in rows if int(q["seed"]) == seed and q["treatment"] == "dense6" and q["mask"] == "111100")
        diffs.append(float(sparse["excess_ce"])-float(dense["excess_ce"]))
    mean = statistics.mean(diffs)
    half = 2.7764451051977987 * statistics.stdev(diffs) / math.sqrt(5)
    assert abs(mean-p["difference"]["mean"]) < 1e-12
    assert abs(mean-half-p["difference"]["low"]) < 1e-12
    assert abs(mean+half-p["difference"]["high"]) < 1e-12
print("PASS: source hash, disjoint data split, tuning/evaluation counts, interior selected LRs, all980 exhaustive mask rows, full-depth anchors, and independent primary paired95% interval recalculation.")
