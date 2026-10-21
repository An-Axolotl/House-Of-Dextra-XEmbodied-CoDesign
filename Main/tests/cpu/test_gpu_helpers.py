"""Runs the GPU helper scripts (Main/tests/gpu/*) against the fake tools so they are known to work before GPU time
is spent: repeatability.py end to end, then check_metrics.py and summarize_runs.py on its output.
Run: python Main/tests/cpu/test_gpu_helpers.py   (exit code 0 = all pass)"""
import os, sys, shutil, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import SB, MAIN, BUILD, PLAY, report, finish

GPU = os.path.join(MAIN, "tests", "gpu")
OUT = os.path.join(SB, "out_gpu_helpers")
shutil.rmtree(OUT, ignore_errors=True)
fake_site = os.path.join(OUT, "fake_site")
for mod in ("isaaclab", "isaaclab_tasks"):
    os.makedirs(os.path.join(fake_site, mod))
    open(os.path.join(fake_site, mod, "__init__.py"), "w").close()
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (fake_site, os.environ.get("PYTHONPATH", "")) if p),
           FAKE_PLAY_NOISE="0.02", PYTHONIOENCODING="utf-8")


def run(script, *args):
    p = subprocess.run([sys.executable, os.path.join(GPU, script), *args], env=ENV, capture_output=True, text=True, timeout=900)
    open(os.path.join(OUT, script + ".log"), "w").write(p.stdout + p.stderr)
    return p.returncode, p.stdout + p.stderr


run_dir = os.path.join(OUT, "repeatability")
rc, log = run("repeatability.py", "--group", "sym5", "--copies", "3", "--others", "3", "--eval-envs", "64", "--repeats", "2",
              "--output-dir", run_dir, "--isaac-python", sys.executable, "--play-script", PLAY, "--build-script", BUILD,
              "--workdir", os.path.join(SB, "Main"))
report("repeatability.py runs and reports a noise/signal ratio", rc != 0 or "noise / signal sd ratio:" not in log,
       next((l for l in log.splitlines() if "ratio:" in l), f"rc={rc}"))
rc, log = run("check_metrics.py", run_dir)
report("check_metrics.py parses runner output and passes clean runs", rc != 0 or "0 FAIL" not in log,
       next((l for l in log.splitlines() if "file(s)" in l), f"rc={rc}"))
bad = os.path.join(OUT, "bad_metrics.jsonl")
open(bad, "w").write('GHS_HAND_METRICS_JSON {"hand_design_metrics": {"hand_x": {"env_count": 64, "total_episodes": 192, '
                     '"norm_reward": 0.1, "unstable_episode_fraction": 0.25}}}\n'
                     'GHS_METRICS_JSON_FINAL {"num_envs": 64, "episode_count": 64, "evaluation_truncated": true}\n')
rc, log = run("check_metrics.py", bad)
report("check_metrics.py flags truncation, triple counting and instability",
       rc == 0 or "truncated" not in log or "triple count" not in log or "WARN unstable" not in log, f"rc={rc}")
rc, log = run("summarize_runs.py", run_dir)
report("summarize_runs.py summarizes a run", rc != 0 or "designs 12" not in log, next((l for l in log.splitlines() if "designs " in l), f"rc={rc}"))
finish("GPU HELPER TESTS")
