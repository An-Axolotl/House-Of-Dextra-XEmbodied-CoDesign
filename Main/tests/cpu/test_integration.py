"""Integration tests through CodesignPPOEvaluator.evaluate_designs_parallel and GraphHeuristicSearch.run,
using the fake build / convert / play stand-ins in sandbox/.
Run: python Main/tests/cpu/test_integration.py   (exit code 0 = all pass)"""
import os, sys, io, csv, random, contextlib, collections, inspect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import make, ghs_mod, ghs_runner, HandGroup, SB, report, finish
from fake_reward import hand_reward

def run_quiet(fn, logname):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): out = fn()
    open(os.path.join(SB, logname), "w").write(buf.getvalue()); return out, buf.getvalue()

# ---------- B1: same codes / different tips in one batch ----------
OUT = os.path.join(SB, "out_b1"); ghs, ev = make(OUT, groups=[HandGroup.SYM5], shuffle=False)
codes = ["230", "40", "310", "20", "550"]; tipsA = ["standard"] * 5; tipsB = ["wedged"] * 5
dA = ev.codes_to_designgraph(codes, tipsA, "symmetric", None); dA.group = "sym5"
dB = ev.codes_to_designgraph(codes, tipsB, "symmetric", None); dB.group = "sym5"
scores, log = run_quiet(lambda: ev.evaluate_designs_parallel([dA, dB]), "b1.log")
rA, rB = hand_reward(codes, tipsA, "sym5"), hand_reward(codes, tipsB, "sym5")
report("B1 same codes, different tips both evaluated with their own hand",
       not (abs(scores[0] - rA) < 1e-6 and abs(scores[1] - rB) < 1e-6),
       f"scores={[round(s,4) for s in scores]} true=[{rA:.4f}, {rB:.4f}]")
usds = sorted(os.path.basename(p) for p in __import__('glob').glob(os.path.join(OUT, "iter_0001_sym5", "batch_candidates", "*", "*.usd")))
print(f"    USD names: {usds}")

# ---------- B2: play emits no per-hand metrics ----------
OUT = os.path.join(SB, "out_b2"); ghs, ev = make(OUT, groups=[HandGroup.SYM5], shuffle=False)
d1 = ev.codes_to_designgraph(["230", "40", "310", "20", "550"], ["standard"]*5, "symmetric", None); d1.group = "sym5"
d2 = ev.codes_to_designgraph(["120", "70", "10", "80", "330"], ["thinner"]*5, "symmetric", None); d2.group = "sym5"
s_first, _ = run_quiet(lambda: ev.evaluate_designs_parallel([d1, d2]), "b2a.log")
ev.env["FAKE_PLAY_MODE"] = "no_hand_metrics"
s_second, _ = run_quiet(lambda: ev.evaluate_designs_parallel([d1, d2]), "b2b.log")
del ev.env["FAKE_PLAY_MODE"]
rows = list(csv.DictReader(open(ev.stats_path_csv)))
report("B2 missing per-hand metrics -> sentinels, not stale scores", s_second != [-1000.0, -1000.0],
       f"iteration1={[round(s,4) for s in s_first]} iteration2={s_second}; stats avg_score per row={[r['avg_score'] for r in rows]}")

# ---------- B3: play.py exits non-zero ----------
OUT = os.path.join(SB, "out_b3"); ghs, ev = make(OUT, groups=[HandGroup.SYM5], shuffle=False)
ev.env["FAKE_PLAY_MODE"] = "fail"
s3, log3 = run_quiet(lambda: ev.evaluate_designs_parallel([d1, d2]), "b3.log")
del ev.env["FAKE_PLAY_MODE"]
rows = list(csv.DictReader(open(ev.stats_path_csv)))
failed_line = [l.strip() for l in log3.splitlines() if "Parallel evaluation FAILED" in l or "Parallel evaluation ERROR" in l]
report("B3 non-zero play exit handled cleanly, one stats row", not (len(rows) == 1 and any("FAILED (return code 3)" in l for l in failed_line) and s3 == [-1000.0, -1000.0]),
       f"log={failed_line}; stats rows={len(rows)}; scores={s3}")

# ---------- B5: cycle boundary consistency + training-set accounting ----------
OUT = os.path.join(SB, "out_b5"); ghs, ev = make(OUT, groups=None, shuffle=True, candidates=4, iterations=7, batch_size=6, seed=3)
calls = []; orig_cg = ev._current_group
def logged_cg():
    g = orig_cg(); calls.append((ev._iter_idx, inspect.stack()[1].function, g.value)); return g
ev._current_group = logged_cg
ghs.train_value_network = lambda: None
best, _ = run_quiet(lambda: ghs.run(), "b5.log")
per_iter = collections.defaultdict(set)
for i, c, g in calls: per_iter[i].add(g)
report("B5 one group per iteration across all callers", any(len(v) > 1 for v in per_iter.values()), f"groups seen per iteration: {dict((i, sorted(v)) for i, v in per_iter.items())}")
order = [sorted(v)[0] for i, v in sorted(per_iter.items())]
print(f"    group order used: {order} (cycle 2 reshuffled: {order[:6] != order[6:7] + order[1:6]})")
mismatch = [d for d in ghs.seen_designs if hasattr(d, "_effective_group") and d.evaluated_group != d._effective_group]
report("B5b lookup/training group == evaluation group", len(mismatch) > 0, f"{len(mismatch)} mismatched entries")
n_eval = 7 * 4
complete = [d for d in ghs.seen_designs if not d._value_key.startswith("partial:")]
partial = [d for d in ghs.seen_designs if d._value_key.startswith("partial:")]
distinct = len({(d._value_key, d.evaluated_group) for d in complete})
report("B5c training set: each distinct hand once + its real partial states", not (len(complete) == distinct <= ghs.num_evaluated and len(partial) > 0),
       f"{ghs.num_evaluated} evaluations of {distinct} distinct hands -> {len(complete)} complete + {len(partial)} partial states (was 2x duplicates + truncated graphs)")
nonterm = sum(1 for d in partial if (d.to_tensor_data(d.evaluated_group)[0][1:, 1] == 0).any())
report("B5d partial training states have non-terminal fingers (match search-time inputs)", nonterm == 0, f"{nonterm}/{len(partial)} partial states contain a non-terminal finger")

finish("INTEGRATION TESTS")
