"""One run of the improvement study (used by test_improvement.py).
argv: mode landscape seed iters K out.json [eval_noise_sd]
mode: search (shipped defaults: gumbel, temperature 1.0 -> 0.05, GNN trained) | random (uniform random designs,
same pipeline and budget) | multigroup (search over all six groups)."""
import os, sys, io, json, contextlib

mode, L, seed, iters, K, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
noise = sys.argv[7] if len(sys.argv) > 7 else "0"
os.environ["FAKE_REWARD_LANDSCAPE"] = L
os.environ["FAKE_PLAY_NOISE"] = noise
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import make, HandGroup, SB
from fake_reward import hand_reward, in_exception_family
import torch
torch.set_num_threads(2)

OUT = os.path.join(SB, "out_study", os.path.basename(out).replace(".json", ""))
groups = None if mode == "multigroup" else [HandGroup.SYM5]
ghs, ev = make(OUT, groups=groups, shuffle=(mode == "multigroup"), candidates=K, iterations=iters, batch_size=K, seed=seed)
if mode == "random":
    ghs.selection = "epsilon"; ghs.epsilon = 1.0; ghs.epsilon_final = 1.0; ghs.train_value_network = lambda: None

per_iter, best = [], {"true": -1.0, "info": None}
_orig = ev.evaluate_designs_parallel


def wrapped(designs):
    g = ev._current_group()
    scores = _orig(designs)
    trues = []
    for d, sc in zip(designs, scores):
        codes, _, _, tips = ev.effective_codes_for(d, g)
        t = hand_reward(codes, tips, g.value)            # noise-free truth, reporting only
        trues.append(t)
        if sc > -999 and t > best["true"]:
            best.update(true=t, info={"codes": codes, "tips": tips, "exception_family": in_exception_family(codes, tips)})
    per_iter.append({"iter": len(per_iter) + 1, "group": g.value, "n": len(designs),
                     "mean_true": sum(trues) / len(trues), "max_true": max(trues), "best_true_so_far": best["true"]})
    return scores


ev.evaluate_designs_parallel = wrapped
with contextlib.redirect_stdout(io.StringIO()):
    ghs.run()
json.dump({"mode": mode, "landscape": L, "seed": seed, "noise": noise, "per_iter": per_iter,
           "best_true": best["true"], "best": best["info"], "hands": ghs.num_evaluated}, open(out, "w"), indent=1)
print(f"{mode} L{L} seed{seed}: last mean={per_iter[-1]['mean_true']:.3f} best={best['true']:.3f}")
