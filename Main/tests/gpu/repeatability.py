"""Evaluation noise floor on the real simulator (GPU). Scores one design several times in the same batch next to
several different random designs, then compares the spread across copies (evaluation noise) with the spread
across designs (signal the search can use).

Run after sourcing .codesign_env.sh (same environment as quick_start_rotation.sh):
  python Main/tests/gpu/repeatability.py --group sym5 --copies 5 --others 5 --eval-envs 1024 [--repeats 2]"""
import argparse, os, random, shutil, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(MAIN)
sys.path.insert(0, MAIN)
import ghs_runner
from hand_groups import HandGroup
from graph_heuristic_search import FINGERTIP_CHOICES
from utils.isaac_python import resolve_isaac_python

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--group", default="sym5", choices=[g.value for g in HandGroup])
ap.add_argument("--copies", type=int, default=5, help="copies of the reference design in the batch")
ap.add_argument("--others", type=int, default=5, help="different random designs in the batch")
ap.add_argument("--ref-codes", default="230,40,310,20,550", help="reference design: five finger codes (see Generation/converters/README.md)")
ap.add_argument("--eval-envs", type=int, default=1024)
ap.add_argument("--eval-timeout", type=int, default=900)
ap.add_argument("--repeats", type=int, default=1, help="evaluate the same batch this many times (separate simulations)")
ap.add_argument("--output-dir", default="ghs_tests/repeatability")
ap.add_argument("--isaac-python", default=None)
ap.add_argument("--seed", type=int, default=0)
# test hooks (Main/tests/cpu/test_gpu_helpers.py runs this script against the fake tools)
ap.add_argument("--play-script", default=os.path.join(REPO, "IsaacLab", "scripts", "rl_games", "play.py"), help=argparse.SUPPRESS)
ap.add_argument("--build-script", default=os.path.join(REPO, "Generation", "converters", "batch_build_rand_hands2.sh"), help=argparse.SUPPRESS)
ap.add_argument("--workdir", default=MAIN, help=argparse.SUPPRESS)
args = ap.parse_args()

os.chdir(args.workdir)                           # the runner resolves ../IsaacLab and checkpoints relative to Main/
play, build = os.path.abspath(args.play_script), os.path.abspath(args.build_script)
env = ghs_runner.build_subprocess_env()
ghs_runner.add_isaaclab_pythonpath(env, play)
env.setdefault("CODESIGN_FIXED_DR_LEVEL", "5")
env.setdefault("CODESIGN_DISABLE_ADR_ADAPTATION", "1")
env["CODESIGN_GHS_EVALUATION"] = "1"
isaac_python = resolve_isaac_python(play_script=play, env=env, preferred=args.isaac_python)
env["CODESIGN_ISAAC_PYTHON"] = isaac_python
env["CODESIGN_PYTHON"] = env["CODESIGN_RUNNER_PYTHON"] = sys.executable
if shutil.which("xacrodoc"):
    env["CODESIGN_XACRODOC"] = shutil.which("xacrodoc")

group = HandGroup(args.group)
ev = ghs_runner.CodesignPPOEvaluator(
    build_script=build, play_script=play, output_dir=args.output_dir, isaac_python=isaac_python, eval_episodes=1,
    eval_envs=args.eval_envs, eval_timeout=args.eval_timeout, env=env, placement_mode="both", seed=args.seed,
    eval_groups=[group], shuffle_groups=False)

rng = random.Random(args.seed)


def random_design():
    codes, tips = [], []
    for _ in range(5):
        servos, g1, g2 = rng.choice([2, 3]), rng.randint(0, 9), rng.randint(0, 9)
        codes.append(f"{g1}{g2}0" if servos == 3 else f"{g2}0")
        tips.append(rng.choice(FINGERTIP_CHOICES))
    return codes, tips


ref = args.ref_codes.split(",")
batch = [(ref, ["standard"] * 5)] * args.copies + [random_design() for _ in range(args.others)]

copies_all, others_all = [], []
for rep in range(args.repeats):
    designs = [ev.codes_to_designgraph(c, t, "symmetric", None) for c, t in batch]
    scores = ev.evaluate_designs_parallel(designs)
    env_counts = [m.get("env_count", 0) for m in ev.last_hand_design_metrics.get(0, {}).values()]
    copies = [s for s in scores[:args.copies] if s > -999]
    others = [s for s in scores[args.copies:] if s > -999]
    copies_all += copies
    others_all += others
    print(f"\n[repeat {rep + 1}] copies {[round(s, 4) for s in scores[:args.copies]]}")
    print(f"[repeat {rep + 1}] others {[round(s, 4) for s in scores[args.copies:]]}")
    if env_counts:
        print(f"[repeat {rep + 1}] envs per design: min {min(env_counts)}, mean {statistics.mean(env_counts):.0f}")

print("\n=== REPEATABILITY ===")
if len(copies_all) < 2 or len(others_all) < 2:
    sys.exit("not enough successful evaluations (check the runner output above)")
sd_within, sd_between = statistics.stdev(copies_all), statistics.stdev(others_all)
ratio = sd_within / sd_between if sd_between > 0 else float("inf")
print(f"reference design ({len(copies_all)} evaluations): mean {statistics.mean(copies_all):.4f}, sd {sd_within:.4f}")
print(f"different designs ({len(others_all)} evaluations): mean {statistics.mean(others_all):.4f}, sd {sd_between:.4f}")
print(f"noise / signal sd ratio: {ratio:.2f}")
print("  < 0.3 good resolution | 0.3-0.7 noisy: use more envs per design | > 0.7 scores are mostly noise at this budget")
