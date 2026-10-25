"""Unit tests for Main/graph_heuristic_search.py and the evaluator's design logic (no subprocesses).
Run: python Main/tests/cpu/test_unit.py   (exit code 0 = all pass)"""
import os, sys, random, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import make, ghs_mod, ghs_runner, HandGroup, SB, report, finish
import torch

OUT = os.path.join(SB, "out_unit")

ghs, ev = make(OUT, groups=None, shuffle=True, candidates=6, epsilon=0.4)

# ---- A1: _current_group() must be pure at a cycle boundary ----
ev._iter_idx = 6
vals = {ev._current_group().value for _ in range(40)}
report("A1 _current_group stable within an iteration at cycle boundary", len(vals) > 1, f"iter_idx=6: 40 calls returned {sorted(vals)}")
ev._iter_idx = 0

# ---- A2: partial-state keys must never alias a physical design key ----
random.seed(1); ghs.K = 4
cands = ghs.generate_candidate_designs()
ghs.lookup_table.clear(); ghs.seen_designs.clear(); ghs._seen_keys.clear()
for c in cands:
    c._effective_key = ghs.compute_effective_key_for_group(c, HandGroup.SYM5)
    ghs.update_lookup_table(c, 0.8, "sym5")
keys = [k for (k, g) in ghs.lookup_table]
eff = [k for k in keys if k.startswith("eff_")]; part = [k for k in keys if k.startswith("partial:")]
report("A2 partial states use their own key namespace", len(eff) != len(cands) or len(part) == 0 or any(k.startswith("eff_") for k in part),
       f"{len(cands)} hands -> {len(eff)} effective keys + {len(part)} partial-state keys, no overlap")
report("A2b every partial key is credited with max descendant reward", any(v != 0.8 for (k, g), v in ghs.lookup_table.items() if k.startswith("partial:")),
       "all partial values == 0.8 after one 0.8 evaluation")

# ---- A4: epsilon must change candidate generation ----
random.seed(123); ghs.epsilon = 0.0; k0 = [ghs.compute_effective_key_for_group(c, HandGroup.SYM3) for c in ghs.generate_candidate_designs()]
random.seed(123); ghs.epsilon = 1.0; k1 = [ghs.compute_effective_key_for_group(c, HandGroup.SYM3) for c in ghs.generate_candidate_designs()]
report("A4 --epsilon affects candidate generation", k0 == k1, f"candidates differ between epsilon=0.0 and 1.0: {k0 != k1}")
ghs.epsilon = 0.4

# ---- A5: rollout must survive a root state whose placeholder grammar_1 is non-zero ----
gen = ghs.generator; orig_init = gen.get_initial_design
def init_with_g1(group=None):
    d = orig_init(group)
    for n in d.nodes[1:]:
        if n.get("servo_count", 0) >= 2: n["grammar_1_count"] = 5
    return d
gen.get_initial_design = init_with_g1
try:
    d = ghs._generate_greedy_design(rng_seed=5, exploration=0.5, batch_path_tabu=set())
    report("A5 rollout robust to non-zero placeholder grammar_1", not d.is_complete(), f"complete design produced, {len(d.trajectory)} partial states recorded")
except AttributeError as e:
    report("A5 rollout robust to non-zero placeholder grammar_1", True, f"AttributeError: {e}")
gen.get_initial_design = orig_init

# ---- A6: anthro constraint must move the fingertip with its finger ----
d = ev.codes_to_designgraph(["40", "230", "20", "20", "20"], ["wedged", "thinner", "standard", "standard", "standard"], "symmetric", None)
codes, mode, ts, tps = ev.effective_codes_for(d, HandGroup.ANTH21)
report("A6 anthro swap keeps fingertips aligned", not (codes[0] == "230" and tps[0] == "thinner" and tps[1] == "wedged"), f"codes={codes} tips={tps}")

# ---- A7: ghost-finger tips must not change the effective key ----
d1 = ev.codes_to_designgraph(["230", "40", "310", "20", "550"], ["standard"]*5, "symmetric", None)
d2 = ev.codes_to_designgraph(["230", "40", "310", "20", "550"], ["standard"]*3 + ["wedged", "thinner"], "symmetric", None)
k1 = ghs.compute_effective_key_for_group(d1, HandGroup.SYM3); k2 = ghs.compute_effective_key_for_group(d2, HandGroup.SYM3)
report("A7 ghost-finger tips excluded from effective key", k1 != k2, f"keys equal: {k1 == k2}\n      {k1}")

# ---- A8: checkpoint round trip ----
ck = os.path.join(OUT, "ck"); ghs.best_design = cands[0]; ghs.best_reward = 0.8; ghs.save_checkpoint(1, save_dir=ck)
try:
    ghs.load_checkpoint(os.path.join(ck, "checkpoint_1.pt"))
    report("A8 --resume-checkpoint round trip", len(ghs.seen_designs) == 0, f"restored {len(ghs.seen_designs)} training states, {len(ghs._seen_keys)} keys, torch {torch.__version__}")
except Exception as e:
    report("A8 --resume-checkpoint round trip", True, f"{type(e).__name__}: {str(e)[:120]}")

# ---- A10: unassigned fields are masked out of the GNN input ----
root = gen.get_initial_design(group=HandGroup.SYM5)
nf, _ = root.to_tensor_data("sym5")
f1 = nf[1]
servo_oh, tip_oh, flags = f1[6:10], f1[16:19], f1[25:29]
report("A10 root state masks placeholder servo/g2/tip and exposes assignment flags",
       not (servo_oh.sum() == 0 and tip_oh.sum() == 0 and f1[5] == 0 and flags.sum() == 0),
       f"finger-1 root features: servo_oh={servo_oh.tolist()} g2n={f1[5].item()} tip_oh={tip_oh.tolist()} assigned_flags={flags.tolist()} (dim={nf.shape[1]})")
after = gen.apply_rule(root, "set_servo_count", {"finger_idx": 1, "servo_count": 3})
nf2, _ = after.to_tensor_data("sym5")
report("A10b assigning a field unmasks it", not (nf2[1][6:10].tolist() == [0, 0, 0, 1] and nf2[1][25] == 1), f"after servo=3: servo_oh={nf2[1][6:10].tolist()} flags={nf2[1][25:29].tolist()}")

finish("UNIT TESTS")
