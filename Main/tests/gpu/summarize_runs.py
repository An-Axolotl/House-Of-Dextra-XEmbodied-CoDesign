"""Summarize one or more ghs_runner.py output directories (GPU runs): learning curve, best design, failed
designs, instability, and group consistency. Pass several dirs to compare arms (e.g. search vs random).

usage: python Main/tests/gpu/summarize_runs.py RUN_DIR [RUN_DIR ...]"""
import csv, glob, json, os, statistics, sys


def rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


curves = {}
for run in sys.argv[1:]:
    name = os.path.basename(os.path.normpath(run))
    stats = rows(os.path.join(run, "final_results", "iteration_group_stats.csv"))
    avg = [num(r["avg_score"]) for r in stats]
    curves[name] = avg
    ok = [a for a in avg if a is not None]
    print(f"\n== {name}: {len(stats)} iterations, {len(stats) - len(ok)} failed")
    for r in stats:
        print(f"   iter {int(r['iteration_idx']) + 1:3d} {r['group']:7s} avg={r['avg_score'] or '-':>8.8s} best={r['best_score_of_iter'] or '-':>8.8s}")
    if ok:
        k = min(3, len(ok))
        print(f"   mean of first {k} iterations {statistics.mean(ok[:k]):.4f} -> last {k} {statistics.mean(ok[-k:]):.4f}")
    best = os.path.join(run, "final_results", "best_design.json")
    if os.path.exists(best):
        b = json.load(open(best))
        print(f"   best design: score {b.get('reward'):.4f} codes {b.get('grammar_codes')} mode {b.get('placement_mode')}")
    n = sentinel = unstable = mismatch = 0
    for man in glob.glob(os.path.join(run, "iter_*", "batch_candidates", "manifest.csv")):
        folder_group = man.split(os.sep)[-3].split("_", 2)[-1]
        for r in rows(man):
            n += 1
            sentinel += (num(r["training_score"]) or 0) <= -999
            unstable += (num(r.get("unstable_episode_fraction")) or 0) > 0
            mismatch += r.get("effective_group") not in ("", folder_group)
    print(f"   designs {n}: failed (-1000) {sentinel}, unstable {unstable}, evaluated under wrong group {mismatch}")

if len(curves) > 1:
    print("\n== avg_score per iteration")
    names = list(curves)
    print("   iter " + " ".join(f"{n[:18]:>18s}" for n in names))
    for i in range(max(map(len, curves.values()))):
        cells = [(f"{curves[n][i]:.4f}" if i < len(curves[n]) and curves[n][i] is not None else "-") for n in names]
        print(f"   {i + 1:4d} " + " ".join(f"{c:>18s}" for c in cells))
