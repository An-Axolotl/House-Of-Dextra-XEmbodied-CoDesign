"""End-to-end CLI test: runs Main/ghs_runner.py itself (argument parsing, main(), group cycling across a cycle
boundary, checkpoint + resume, best/worst export, --focus-metric, the random-baseline flags) against the
fake build / convert / play stand-ins in sandbox/.
Run: python Main/tests/cpu/test_cli.py   (~2-4 min, exit code 0 = all pass)"""
import os, sys, csv, json, glob, shutil, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import SB, MAIN, BUILD, PLAY, report, finish

RUNNER = os.path.join(MAIN, "ghs_runner.py")
OUT = os.path.join(SB, "out_cli")
shutil.rmtree(OUT, ignore_errors=True)
# resolve_isaac_python() only accepts an interpreter that can import isaaclab / isaaclab_tasks: empty stand-ins.
fake_site = os.path.join(OUT, "fake_site")
for mod in ("isaaclab", "isaaclab_tasks"):
    os.makedirs(os.path.join(fake_site, mod))
    open(os.path.join(fake_site, mod, "__init__.py"), "w").close()
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (fake_site, os.environ.get("PYTHONPATH", "")) if p),
           WANDB_MODE="disabled", PYTHONIOENCODING="utf-8")
COMMON = ["--play-script", PLAY, "--build-script", BUILD, "--isaac-python", sys.executable, "--device", "cpu",
          "--eval-envs", "64", "--eval-episodes", "1", "--placement-mode", "both"]


def run(name, *extra):
    out = os.path.join(OUT, name)
    p = subprocess.run([sys.executable, RUNNER, *COMMON, "--output-dir", out, *extra], cwd=os.path.join(SB, "Main"),
                       env=ENV, capture_output=True, text=True, timeout=1800)
    log = p.stdout + p.stderr
    open(out + ".log", "w").write(log)
    return p.returncode, out, log


def rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


# 1) seven iterations over all six groups: crosses a cycle boundary and writes checkpoint_5
rc, out, log = run("search", "--iterations", "7", "--candidates", "4", "--eval-groups", "all", "--shuffle-groups",
                   "--batch-size", "4", "--seed", "3")
report("runner exits cleanly", rc != 0 or "Traceback" in log, f"rc={rc}, see {out}.log")
stats = rows(os.path.join(out, "final_results", "iteration_group_stats.csv"))
groups = [r["group"] for r in stats]
report("one stats row per iteration, first cycle visits all six groups", len(stats) != 7 or len(set(groups[:6])) != 6, f"groups={groups}")
mism = []
for man in glob.glob(os.path.join(out, "iter_*", "batch_candidates", "manifest.csv")):
    folder_group = man.split(os.sep)[-3].split("_", 2)[-1]
    mism += [r["effective_group"] for r in rows(man) if r["effective_group"] != folder_group]
report("every design evaluated under its iteration's group", bool(mism), f"{len(mism)} mismatches")
best = os.path.join(out, "final_results", "best_design.json")
best_ok = os.path.exists(best) and json.load(open(best)).get("reward", 0) > 0
report("best design exported with a positive score", not best_ok, best)
need = ["final_results/best_design_hand.urdf", "final_results/evaluate_best_design.sh", "final_results/lookup_table.json",
        "final_results/rewards.csv", "ghs_checkpoint/checkpoint_5.pt"]
missing = [f for f in need if not os.path.exists(os.path.join(out, f))]
report("result artifacts and checkpoint written", bool(missing), f"missing={missing}")
partial_rows = [r for r in rows(os.path.join(out, "final_results", "rewards.csv")) if r["design_string"].startswith("partial:")]
report("rewards.csv lists physical hands only", bool(partial_rows), f"{len(partial_rows)} partial-state rows")
man0 = sorted(glob.glob(os.path.join(out, "iter_*", "batch_candidates", "manifest.csv")))[0]
report("manifest reports unstable_episode_fraction", "unstable_episode_fraction" not in (rows(man0) or [{}])[0], man0)
report("default selection is annealed Gumbel", "Successor selection: gumbel (exploration 1.0 -> 0.05)" not in log, "")

# 2) resume from the checkpoint
rc, out2, log = run("resume", "--iterations", "1", "--candidates", "4", "--eval-groups", "sym5", "--no-shuffle-groups",
                    "--resume-checkpoint", os.path.join(out, "ghs_checkpoint", "checkpoint_5.pt"))
report("--resume-checkpoint loads and runs", rc != 0 or "Resuming from GHS checkpoint" not in log, f"rc={rc}")

# 3) uniform-random baseline through the same pipeline
rc, out3, log = run("random", "--iterations", "2", "--candidates", "4", "--eval-groups", "sym5", "--no-shuffle-groups",
                    "--selection", "epsilon", "--epsilon", "1.0", "--epsilon-final", "1.0")
report("random baseline flags accepted", rc != 0 or "Successor selection: epsilon (exploration 1.0 -> 1.0)" not in log, f"rc={rc}")

# 4) --focus-metric rotation scores designs by omega_score (fake play: omega = 0.5 * reward, rps = 3 * reward)
rc, out4, log = run("rotation", "--iterations", "1", "--candidates", "4", "--eval-groups", "sym5", "--no-shuffle-groups",
                    "--focus-metric", "rotation")
man = glob.glob(os.path.join(out4, "iter_*", "batch_candidates", "manifest.csv"))
bad = [r for m in man for r in rows(m) if abs(float(r["training_score"]) - 0.5 * float(r["rps_from_omega"]) / 3) > 1e-4]
report("--focus-metric rotation scores by omega_score", rc != 0 or not man or bool(bad), f"rc={rc}, {len(bad)} mismatched rows")

finish("CLI TESTS")
