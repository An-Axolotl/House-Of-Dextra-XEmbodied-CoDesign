"""Can the search improve designs? Search vs uniform random sampling at equal budget on three synthetic reward
landscapes with known optimum 0.90 (sandbox/fake_reward.py: monotone trend, coupled fingers, and an outlier
family across a valley), plus a noisy-evaluation run and a six-group run.

Run: python Main/tests/cpu/test_improvement.py [iters] [candidates] [parallel] [seeds]
     defaults 10 12 6 4 (~15-30 min); quick check: 6 8 8 2.
Exit code 0 = search beats random on final candidate quality with one-sided sign-test p < 0.05."""
import os, sys, json, subprocess, itertools, statistics, math, time

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.join(HERE, "sandbox")
sys.path.insert(0, SB)
from fake_reward import random_baseline, OPTIMUM

ITERS, K, PAR, SEEDS = (int(a) for a in (sys.argv[1:] + ["10", "12", "6", "4"][len(sys.argv) - 1:])[:4])
RES = os.path.join(SB, "study_results", f"i{ITERS}_k{K}")
os.makedirs(RES, exist_ok=True)
os.makedirs(os.path.join(SB, "Main"), exist_ok=True)
runner = os.path.join(HERE, "improvement_run.py")

jobs = []
for L, seed in itertools.product("012", range(SEEDS)):
    for mode in ("search", "random"):
        jobs.append((mode, L, seed, ITERS, K, os.path.join(RES, f"{mode}_L{L}_s{seed}.json"), "0"))
for seed in (0, 1):
    jobs.append(("search", "0", seed, ITERS, K, os.path.join(RES, f"noise_L0_s{seed}.json"), "0.05"))
    jobs.append(("random", "0", seed, ITERS, K, os.path.join(RES, f"noiserandom_L0_s{seed}.json"), "0.05"))
jobs.append(("multigroup", "0", 0, 18, 8, os.path.join(RES, "multigroup_L0_s0.json"), "0"))
jobs = [j for j in jobs if not os.path.exists(j[5])]
print(f"launching {len(jobs)} runs, {PAR} in parallel", flush=True)
t0, running, failed = time.time(), [], []
while jobs or running:
    while jobs and len(running) < PAR:
        j = jobs.pop(0)
        running.append((j, subprocess.Popen([sys.executable, runner] + [str(x) for x in j], stdout=open(j[5] + ".log", "w"),
                                            stderr=subprocess.STDOUT, cwd=os.path.join(SB, "Main"))))
    for j, p in running[:]:
        if p.poll() is not None:
            running.remove((j, p))
            if p.returncode:
                failed.append(j[5])
            print(f"  done {os.path.basename(j[5])} rc={p.returncode} ({time.time() - t0:.0f}s)", flush=True)
    time.sleep(2)
if failed:
    sys.exit(f"{len(failed)} runs failed, see the .log next to: {failed[:3]}")


def load(name): return json.load(open(os.path.join(RES, name)))
def first2(r): return statistics.mean(x["mean_true"] for x in r["per_iter"][:2])
def last2(r): return statistics.mean(x["mean_true"] for x in r["per_iter"][-2:])


print(f"\n=== IMPROVEMENT STUDY ({ITERS} iterations x {K} candidates, {SEEDS} seeds per landscape) ===")
pairs = []
for L in "012":
    base = random_baseline(L)
    S = [load(f"search_L{L}_s{s}.json") for s in range(SEEDS)]
    R = [load(f"random_L{L}_s{s}.json") for s in range(SEEDS)]
    print(f"\nLandscape {L}: uniform-random mean {base['mean']:.3f}, p99 {base['p99']:.3f}; optimum {OPTIMUM:.2f}")
    for name, runs in (("search", S), ("random", R)):
        b = [r["best_true"] for r in runs]
        print(f"  {name:6s}: candidate mean {statistics.mean(map(first2, runs)):.3f} -> {statistics.mean(map(last2, runs)):.3f}"
              f" | best found {statistics.mean(b):.3f} (regret {OPTIMUM - statistics.mean(b):.3f})")
    pairs += [(last2(s) - last2(r), s["best_true"] - r["best_true"]) for s, r in zip(S, R)]
    if L == "2":
        print(f"  outlier family found: search {sum(r['best']['exception_family'] for r in S)}/{SEEDS}, random {sum(r['best']['exception_family'] for r in R)}/{SEEDS}")

n = len(pairs)
wins_mean, wins_best = sum(d > 0 for d, _ in pairs), sum(d > 0 for _, d in pairs)
def p_one_sided(w): return sum(math.comb(n, k) for k in range(w, n + 1)) / 2 ** n
p_mean = p_one_sided(wins_mean)
print(f"\nSearch beats random on final candidate mean in {wins_mean}/{n} pairs (sign test p={p_mean:.4f}), "
      f"on best found in {wins_best}/{n} (p={p_one_sided(wins_best):.4f}); mean gains "
      f"{statistics.mean(d for d, _ in pairs):+.3f} / {statistics.mean(d for _, d in pairs):+.3f}")
NS = [load(f"noise_L0_s{s}.json") for s in (0, 1)]
NR = [load(f"noiserandom_L0_s{s}.json") for s in (0, 1)]
print(f"Eval noise sd 0.05: search final mean {statistics.mean(map(last2, NS)):.3f} vs random {statistics.mean(map(last2, NR)):.3f}")
by = {}
for x in load("multigroup_L0_s0.json")["per_iter"]:
    by.setdefault(x["group"], []).append(x["mean_true"])
print(f"Six-group run: groups improved from first to last visit: {sum(v[-1] > v[0] for v in by.values())}/{len(by)}")
ok = p_mean < 0.05
print(f"\n=== {'PASS' if ok else 'FAIL'}: search {'beats' if ok else 'does not beat'} random sampling ===")
sys.exit(0 if ok else 1)
