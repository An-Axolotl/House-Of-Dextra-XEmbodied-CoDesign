"""FAKE play.py used only by Main/tests/cpu: reads the hand JSON "USDs" written by fake_build.sh and
emits the same GHS_* metric lines as IsaacLab/scripts/rl_games/play.py, scored by fake_reward.py."""
import argparse, os, sys, json, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from fake_reward import hand_reward
p = argparse.ArgumentParser()
p.add_argument("--metrics_out", default="")
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--checkpoint", default="")
a, _ = p.parse_known_args()
mode = os.environ.get("FAKE_PLAY_MODE", "ok")
noise = float(os.environ.get("FAKE_PLAY_NOISE", "0"))
group = os.environ.get("CODESIGN_HAND_GROUP", "")
metrics_out = a.metrics_out or os.environ.get("GHS_METRICS_OUT", "")
def emit(line):
    print(line, flush=True)
    if metrics_out:
        with open(metrics_out, "a") as f: f.write(line + "\n")
usds = []
for i in range(100):
    v = os.environ.get(f"CODESIGN_HAND_USD_PATH_{i}")
    if v: usds.append(v)
    elif usds: break
print(f"[fake-play] group={group} checkpoint={a.checkpoint} n_usd={len(usds)} mode={mode}")
if mode == "fail":
    print("simulated play failure"); sys.exit(3)
rng = random.Random(os.environ.get("FAKE_PLAY_SEED", "0") + group + str(len(usds)))
summary = {}
for u in usds:
    did = os.path.basename(u).replace(".usd", "")
    with open(u) as f: spec = json.loads(f.read().strip())
    r = hand_reward(spec["codes"], spec["tips"], group)
    if noise: r = max(0.0, min(r + rng.gauss(0, noise), 0.999))
    summary[did] = {"env_count": max(1, a.num_envs // max(1, len(usds))), "total_episodes": max(1, a.num_envs // max(1, len(usds))), "unstable_episode_fraction": 0.0,
                    "average_reward": r * 30000.0, "norm_reward": r, "reward_cap": 30000.0,
                    "success_rate": 0.0, "omega_score": round(0.5 * r, 6), "rps_from_omega": r * 3, "turns_per_second": 0.0,
                    "raw_omega_z_rad_per_sec": 0.0, "raw_omega_z_deg_per_sec": 0.0, "raw_omega_z_rpm": 0.0,
                    "step_fraction_evaluated": 1.0, "contact_steps": 100, "evaluation_time_s": 60.0, "env_ids": [0]}
if mode != "no_hand_metrics":
    emit("GHS_HAND_METRICS_JSON " + json.dumps({"num_hand_designs": len(summary), "hand_design_metrics": summary}, separators=(",", ":")))
if mode == "crash_after_metrics":
    print("simulated crash after hand metrics"); sys.exit(1)
overall = sum(m["norm_reward"] for m in summary.values()) / max(1, len(summary))
final = {"episode_count": a.num_envs, "num_envs": a.num_envs, "evaluation_truncated": False, "steps_completed": 1800, "eval_wall_time_s": 0.1, "avg_metrics": {"norm_reward": overall}, "score": overall, "norm_reward": overall,
         "hand_design_metrics": summary if mode != "no_hand_metrics" else {}, "num_hand_designs": len(summary)}
emit("GHS_METRICS_JSON_FINAL " + json.dumps(final, separators=(",", ":")))
