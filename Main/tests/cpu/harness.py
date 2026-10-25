"""Shared setup for the CPU tests.

Runs the real Main/ search and runner code against the fake build / URDF->USD / play.py stand-ins in
sandbox/ (same CLIs, asset naming and GHS_* metric lines as the real tools, synthetic reward from
sandbox/fake_reward.py). No Isaac Sim or GPU needed. Outputs land in sandbox/ (gitignored)."""
import os, sys, shutil, random

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.join(HERE, "sandbox")
MAIN = os.path.dirname(os.path.dirname(HERE))            # the real Main/ under test
for p in (MAIN, SB):
    if p not in sys.path:
        sys.path.insert(0, p)
os.makedirs(os.path.join(SB, "Main"), exist_ok=True)
os.chdir(os.path.join(SB, "Main"))                        # the runner resolves ../IsaacLab relative to cwd

import torch
import graph_heuristic_search as ghs_mod
import ghs_runner
from hand_groups import HandGroup

BUILD = os.path.join(SB, "Generation", "converters", "fake_build.sh")
PLAY = os.path.join(SB, "IsaacLab", "scripts", "rl_games", "play.py")


def make(output_dir, *, groups=None, shuffle=True, candidates=6, iterations=3, batch_size=8, seed=0,
         epsilon=1.0, debug=False, selection="gumbel"):
    """Search + evaluator wired to the fake tools (same constructor arguments ghs_runner.main() uses)."""
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    random.seed(seed)
    torch.manual_seed(seed)
    ghs = ghs_runner.CodesignGraphHeuristicSearch(
        num_iterations=iterations, num_candidates=candidates, epsilon=epsilon, device="cpu",
        learning_rate=3e-4, opt_iterations=200, batch_size=batch_size, debug=debug, selection=selection)
    ghs.checkpoint_dir = os.path.join(output_dir, "ghs_checkpoint")
    ev = ghs_runner.CodesignPPOEvaluator(
        build_script=BUILD, play_script=PLAY, output_dir=output_dir, isaac_python=sys.executable,
        eval_episodes=1, eval_envs=64, eval_timeout=5, env=os.environ.copy(), placement_mode="both",
        debug=debug, wandb_run=None, seed=seed, eval_groups=groups, shuffle_groups=shuffle)
    ghs.evaluator = ev
    return ghs, ev


RESULTS = []


def report(name, failed, detail):
    RESULTS.append((name, bool(failed)))
    print(f"{'FAIL' if failed else 'PASS'} | {name} | {detail}")


def finish(title):
    n_fail = sum(f for _, f in RESULTS)
    print(f"\n=== {title}: {len(RESULTS) - n_fail}/{len(RESULTS)} passed ===")
    for name, failed in RESULTS:
        print(f"  {'FAIL' if failed else 'PASS'}  {name}")
    sys.exit(1 if n_fail else 0)
