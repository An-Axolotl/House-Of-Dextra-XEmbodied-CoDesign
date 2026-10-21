"""Sanity-check real play.py evaluations (GPU). Reads GHS_HAND_METRICS_JSON / GHS_METRICS_JSON_FINAL lines.

usage: python Main/tests/gpu/check_metrics.py <metrics.jsonl | run_dir> [...]
       (a run_dir is searched for */batch_candidates/ghs_metrics.jsonl)

FAIL: evaluation truncated by --timeout, per-design total_episodes != env_count (one fixed-length round per
      env is expected; ~3x means the old episode triple-count), non-finite metrics, or no per-hand metrics.
WARN: unstable_episode_fraction > 0 (NaN state or object flung from the hand)."""
import glob, json, math, os, sys

PREFIXES = ("GHS_HAND_METRICS_JSON ", "GHS_METRICS_JSON_FINAL ")


def files_from(args):
    out = []
    for a in args:
        out += sorted(glob.glob(os.path.join(a, "**", "ghs_metrics.jsonl"), recursive=True)) if os.path.isdir(a) else [a]
    return out


def parse(path):
    hand, final = {}, {}
    for line in open(path, encoding="utf-8", errors="replace"):
        for prefix in PREFIXES:
            i = line.find(prefix)
            if i >= 0:
                payload = json.loads(line[i + len(prefix):])
                if prefix.startswith("GHS_HAND"):
                    hand = payload.get("hand_design_metrics", {})
                else:
                    final = payload
    return hand, final


def finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


fails = warns = 0
paths = files_from(sys.argv[1:])
if not paths:
    sys.exit("no metrics files found")
for path in paths:
    hand, final = parse(path)
    print(f"\n== {path}")
    if final:
        wall = final.get("eval_wall_time_s")
        print(f"   envs={final.get('num_envs')} episodes={final.get('episode_count')} steps={final.get('steps_completed')}"
              f" wall={wall if wall is None else round(wall, 1)}s truncated={final.get('evaluation_truncated')}")
        if final.get("evaluation_truncated"):
            print("   FAIL evaluation truncated by --timeout: raise --eval-timeout"); fails += 1
    if not hand:
        print("   FAIL no per-hand metrics"); fails += 1; continue
    print(f"   {'design':58s} {'envs':>5s} {'eps':>5s} {'norm_rew':>9s} {'omega':>7s} {'success':>8s} {'unstable':>8s}")
    for did, m in sorted(hand.items()):
        envs, eps = m.get("env_count"), m.get("total_episodes")
        unstable = m.get("unstable_episode_fraction")
        flags = []
        if envs != eps:
            flags.append(f"FAIL episodes {eps} != envs {envs}" + (" (looks like the old triple count)" if envs and eps == 3 * envs else ""))
        bad = [k for k in ("norm_reward", "average_reward", "omega_score") if k in m and not finite(m[k])]
        if bad:
            flags.append(f"FAIL non-finite {bad}")
        if unstable is None:
            flags.append("WARN no unstable_episode_fraction (old env code?)")
        elif unstable > 0:
            flags.append(f"WARN unstable {unstable:.0%}")
        fails += sum(f.startswith("FAIL") for f in flags); warns += sum(f.startswith("WARN") for f in flags)
        fmt = lambda v, n: f"{v:{n}.3f}" if finite(v) else f"{str(v):>{n}s}"
        print(f"   {did[:58]:58s} {str(envs):>5s} {str(eps):>5s} {fmt(m.get('norm_reward'), 9)} {fmt(m.get('omega_score'), 7)}"
              f" {fmt(m.get('success_rate'), 8)} {fmt(unstable, 8)}  {'; '.join(flags)}")
print(f"\n{len(paths)} file(s): {fails} FAIL, {warns} WARN")
sys.exit(1 if fails else 0)
