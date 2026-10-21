"""Value-network targets: every evaluated hand is trained toward its own best measured score, and every
partial state on its rollout toward the best score among its descendants (RoboGrammar V*).
Run: python Main/tests/cpu/test_targets.py   (exit code 0 = pass)"""
import os, sys, io, contextlib, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import make, ghs_mod, ghs_runner, HandGroup, SB, report, finish
OUT = os.path.join(SB, "out_targets")
ghs, ev = make(OUT, groups=[HandGroup.SYM5], shuffle=False, candidates=12, iterations=8, batch_size=12, seed=21)
ghs.train_value_network = lambda: None
measured = {}; traj_scores = []
_orig = ev.evaluate_designs_parallel
def wrapped(designs):
    s = _orig(designs)
    for d, sc in zip(designs, s):
        measured.setdefault(d._effective_key, []).append(sc)
        traj_scores.append((d, sc))
    return s
ev.evaluate_designs_parallel = wrapped
with contextlib.redirect_stdout(io.StringIO()): ghs.run()
wrong = [(k, max(v), ghs.lookup_table.get((k, "sym5"))) for k, v in measured.items() if abs(ghs.lookup_table.get((k, "sym5"), -9) - max(v)) > 1e-9]
n_part = sum(1 for (k, g) in ghs.lookup_table if k.startswith("partial:"))
print(f"{sum(len(v) for v in measured.values())} evaluations, {len(measured)} distinct hands; lookup: {len(measured)} hand keys + {n_part} partial-state keys")
report("hand targets equal their own best measured score", len(wrong) > 0, f"{len(wrong)} mismatches")
viol = 0; checked = 0
for d, sc in traj_scores:
    for s in getattr(d, "trajectory", []):
        checked += 1
        if ghs.lookup_table[(s.partial_state_key("sym5"), "sym5")] < sc - 1e-9: viol += 1
report("partial-state targets >= every descendant score", viol > 0 or checked == 0, f"{viol}/{checked} violations")
finish("TARGET TESTS")
