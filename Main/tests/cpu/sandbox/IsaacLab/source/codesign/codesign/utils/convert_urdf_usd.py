"""FAKE URDF->USD converter used only by Main/tests/cpu (same CLI and <stem>/<stem>.usd layout)."""
import argparse, os, sys, shutil
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--in-dir", required=True); p.add_argument("--out-dir", required=True)
p.add_argument("--merge-joints", action="store_true"); p.add_argument("--headless", action="store_true")
a, _ = p.parse_known_args()
mode = os.environ.get("FAKE_CONVERT_MODE", "ok")
if mode == "fail":
    print("simulated converter failure"); sys.exit(1)
urdfs = sorted(Path(a.in_dir).rglob("*.urdf"))
ok = 0
for u in urdfs:
    d = Path(a.out_dir) / u.stem; d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(u, d / (u.stem + ".usd")); ok += 1
    print(f"{u} -> {d / (u.stem + '.usd')}")
if mode == "traceback":
    print("Traceback (most recent call last):\n  simulated warning traceback in stdout")
print(f"Conversion summary: {ok} succeeded, 0 failed")
