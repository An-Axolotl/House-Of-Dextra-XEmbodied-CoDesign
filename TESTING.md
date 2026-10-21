# Testing `fix/ghs-optimization`

This branch changes three things: the graph heuristic search (`Main/`), evaluation correctness in the
task and `play.py` (`IsaacLab/`), and asset naming for fingertip conditioning (`Generation/`).
The search and runner are verified on CPU by `Main/tests/cpu/`. The IsaacLab changes compile and are
reviewed but have not run in simulation yet; the GPU tests below cover that.

Scores from this branch are not comparable with older runs: each episode used to be counted three
times, and evaluation used to run at DR level 0 instead of the intended fixed level 5. The shipped
checkpoints are unaffected: with fingertip conditioning off (the default) they receive exactly the
inputs they were trained on.

## Setup (GPU machine)

```bash
git clone https://github.com/An-Axolotl/House-Of-Dextra-XEmbodied-CoDesign.git
cd House-Of-Dextra-XEmbodied-CoDesign && git checkout fix/ghs-optimization
./setup.sh                                   # see SETUP.md
source .codesign_env.sh
export REPO_ROOT=$PWD
export PYTHONPATH="$REPO_ROOT/IsaacLab/source/codesign:$REPO_ROOT/IsaacLab:$PYTHONPATH"
export CODESIGN_FIXED_DR_LEVEL=5 CODESIGN_DISABLE_ADR_ADAPTATION=1 CODESIGN_GHS_EVALUATION=1
mkdir -p Main/ghs_tests                      # all test outputs go here (gitignored)

nvidia-smi
"$CODESIGN_ISAAC_PYTHON" -c "import codesign; print(codesign.__file__)"   # must point inside this checkout
python Generation/converters/verify_generation_deps.py
ls IsaacLab/logs/rl_games/cd_hand_direct/*/nn/cd_hand_direct.pth          # group checkpoints present
```

## Minimum check before merging (about 30 to 45 min of GPU time)

Run after the setup above.

1. The CPU suite (section 1 below).
2. M1 below: the same sym3 hand evaluated on main and on this branch. It catches a crash in the
   environment changes, confirms the episode-count fix, and confirms an old policy still behaves the
   same (sym3 hands have ghost fingers, the case most sensitive to input changes).
3. G4: a two-iteration run through the real build, convert and evaluate pipeline.

If all three pass, merge. The rest of section 2 matters before relying on GHS results; G9 is the test
that shows the search improves designs in simulation.

### M1. Main versus branch on a sym3 hand

```bash
cd "$REPO_ROOT"; T=$REPO_ROOT/Main/ghs_tests/m1
bash Generation/converters/batch_build_rand_hands2.sh -o "$T" --mode manual -C "230,40,310,," \
  --placement-mode symmetric --palm-radius 0.06 --symmetric-start-deg 0 --symmetric-jitter-deg 0 \
  --slot-count 36 --min-sep-slots 4 --seed 10
(cd IsaacLab && "$CODESIGN_ISAAC_PYTHON" source/codesign/codesign/utils/convert_urdf_usd.py \
  --in-dir "$T" --out-dir "$T" --merge-joints --headless)
HAND=$(ls "$T"/*/*.usd | head -1)                       # ..._f4_f5_tsssxx.usd
git worktree add ../hod-main origin/main
for side in branch main; do
  root=$REPO_ROOT; [ "$side" = main ] && root=$(cd "$REPO_ROOT/../hod-main" && pwd)
  ( cd "$root/IsaacLab" && unset CODESIGN_FIXED_DR_LEVEL &&
    export PYTHONPATH="$root/IsaacLab/source/codesign:$root/IsaacLab:$PYTHONPATH" \
           CODESIGN_HAND_USD_PATH_0="$HAND" CODESIGN_HAND_GROUP=sym3 CODESIGN_JOINT_PRESET=symmetric &&
    "$CODESIGN_ISAAC_PYTHON" -c 'import codesign; print("codesign from", codesign.__file__)' &&
    "$CODESIGN_ISAAC_PYTHON" scripts/rl_games/play.py --task=Codesign-Reorientation-Direct-v0 --headless \
      --num_envs=256 --checkpoint=logs/rl_games/cd_hand_direct/group4_4096envs/nn/cd_hand_direct.pth \
      --max_episodes=1 --timeout=1800 --use_unexported --metrics_out "$REPO_ROOT/Main/ghs_tests/m1_$side.jsonl"
  ) > "$REPO_ROOT/Main/ghs_tests/m1_$side.log" 2>&1
done
grep "codesign from" Main/ghs_tests/m1_*.log                # each side must import its own checkout
python Main/tests/gpu/check_metrics.py Main/ghs_tests/m1_branch.jsonl
python Main/tests/gpu/check_metrics.py Main/ghs_tests/m1_main.jsonl   # flags episodes = 3x envs: expected on main
```

Pass:
- the branch passes `check_metrics.py`: episodes equal envs, not truncated, no unstable designs;
- main shows episodes equal to 3x envs (the old triple count);
- `omega` (rotation speed) is about the same on both, within roughly 15%: same policy, same inputs;
- `avg_rew` on the branch is about 3x main's: the counting fix.

## 1. CPU suite (any machine, a few minutes)

```bash
for t in unit integration targets cli gpu_helpers; do python Main/tests/cpu/test_$t.py > /tmp/t_$t.log 2>&1 && echo "PASS $t" || echo "FAIL $t (see /tmp/t_$t.log)"; done
python Main/tests/cpu/test_improvement.py 6 8 8 2      # search vs random, quick; no arguments = full study
```

Every test should pass. They run the real search and runner against fake build, convert and play
tools with a synthetic reward, so they check logic, not physics.

## 2. Full GPU test plan

| Test | Checks | Cost |
|---|---|---|
| G1 | Builder writes the fingertip token; URDF converts to USD | minutes |
| G2 | One hand evaluates; episodes counted once; new metric fields | minutes |
| G3 | Fingertip flag with an old checkpoint fails with a clear message | minutes |
| G4 | Runner end to end on a small run | under 1 h |
| G5 | Production-scale iteration: timing and timeouts | about 1 h |
| G6 | Evaluation noise versus differences between designs | under 1 h |
| G7 | Group cycling across all six groups, checkpoint resume | about 1 h |
| G8 | Simulation stability with ghost fingers | uses G4 to G7 output |
| G9 | Search beats random design sampling at equal budget | hours |
| G10 | Optional: fingertip-conditioned training starts | under 1 h |

Run them in order after the minimum check. Stop and report if one fails.

### G1. Build and naming

```bash
cd "$REPO_ROOT"; T=$REPO_ROOT/Main/ghs_tests/g1_build
bash Generation/converters/batch_build_rand_hands2.sh -o "$T" --mode manual -C "230,40,310,20,550" \
  --placement-mode symmetric --palm-radius 0.07 --symmetric-start-deg 0 --symmetric-jitter-deg 0 \
  --slot-count 36 --min-sep-slots 4 --seed 10 --fingertip-f1 wedged --fingertip-f2 thinner
(cd IsaacLab && "$CODESIGN_ISAAC_PYTHON" source/codesign/codesign/utils/convert_urdf_usd.py \
  --in-dir "$T" --out-dir "$T" --merge-joints --headless)
ls "$T"/*.urdf "$T"/*/*.usd
```

Pass: `hand_f1_230_f2_40_f3_310_f4_20_f5_550_twtsss.urdf` and the matching `.usd` exist.

### G2. Single-hand evaluation and episode accounting

```bash
HAND=$(ls "$T"/*/*.usd | head -1); cd "$REPO_ROOT/IsaacLab"
CODESIGN_HAND_USD_PATH_0="$HAND" CODESIGN_HAND_GROUP=sym5 CODESIGN_JOINT_PRESET=symmetric \
"$CODESIGN_ISAAC_PYTHON" scripts/rl_games/play.py --task=Codesign-Reorientation-Direct-v0 --headless \
  --num_envs=64 --checkpoint=logs/rl_games/cd_hand_direct/group3_4096envs/nn/cd_hand_direct.pth \
  --max_episodes=1 --timeout=1800 --use_unexported --metrics_out ../Main/ghs_tests/g2_branch.jsonl \
  2>&1 | tee ../Main/ghs_tests/g2_branch.log
cd "$REPO_ROOT"; python Main/tests/gpu/check_metrics.py Main/ghs_tests/g2_branch.jsonl
grep -E "\[morph\]|\[adr\]|\[ghs-eval\]" Main/ghs_tests/g2_branch.log
```

Pass: `check_metrics.py` exits 0 (64 envs, 64 episodes, not truncated, unstable 0). The log shows a
35-dim morphology vector, DR level fixed at 5 and an 1800-step episode. Note the reported `wall=`.

Optional comparison with main. Episodes should be 192 on main and 64 here, and the average reward
about 3× higher here. DR is unset in both runs so only the counting differs:

```bash
git worktree add ../hod-main origin/main
MAIN_PATH="$REPO_ROOT/../hod-main/IsaacLab/source/codesign:$REPO_ROOT/../hod-main/IsaacLab:$PYTHONPATH"
PYTHONPATH="$MAIN_PATH" "$CODESIGN_ISAAC_PYTHON" -c "import codesign; print(codesign.__file__)"   # must be in hod-main
# rerun the G2 play command twice with `env -u CODESIGN_FIXED_DR_LEVEL` in front:
#   from ../hod-main/IsaacLab with PYTHONPATH="$MAIN_PATH"  -> --metrics_out .../g2_main.jsonl
#   from $REPO_ROOT/IsaacLab                                -> --metrics_out .../g2_branch_dr0.jsonl
```

### G3. Fingertip flag guard

```bash
cd "$REPO_ROOT/IsaacLab"
CODESIGN_MORPH_FINGERTIPS=1 CODESIGN_HAND_USD_PATH_0="$HAND" CODESIGN_HAND_GROUP=sym5 CODESIGN_JOINT_PRESET=symmetric \
"$CODESIGN_ISAAC_PYTHON" scripts/rl_games/play.py --task=Codesign-Reorientation-Direct-v0 --headless --num_envs=16 \
  --checkpoint=logs/rl_games/cd_hand_direct/group3_4096envs/nn/cd_hand_direct.pth --max_episodes=1 --use_unexported 2>&1 | tail -20
```

Pass: the log shows a 55-dim morphology vector, then a RuntimeError saying the checkpoint does not
match the observation size with fingertip conditioning on.

### G4. Runner smoke run

```bash
cd "$REPO_ROOT/Main"
python ghs_runner.py --iterations 2 --candidates 6 --eval-envs 256 --eval-timeout 1800 --eval-groups sym3 \
  --no-shuffle-groups --batch-size 6 --seed 0 --output-dir ghs_tests/g4_smoke 2>&1 | tee ghs_tests/g4_smoke.log
python tests/gpu/check_metrics.py ghs_tests/g4_smoke
python tests/gpu/summarize_runs.py ghs_tests/g4_smoke
grep -cE "Traceback|Parallel evaluation (FAILED|ERROR)|Batch conversion FAILED|Build failed" ghs_tests/g4_smoke.log
```

Pass: `check_metrics.py` exits 0, both iterations have scores, no failed designs, the grep prints 0,
and `final_results/` contains `best_design.json` and `best_design_hand.urdf`. A camera warning from
the final best-design video is acceptable.

### G5. Production-scale iteration

```bash
python ghs_runner.py --iterations 1 --candidates 50 --eval-envs 2048 --eval-timeout 1800 --eval-groups sym5 \
  --no-shuffle-groups --seed 0 --output-dir ghs_tests/g5_scale 2>&1 | tee ghs_tests/g5_scale.log
grep -E "Successfully built|Batch conversion completed|Parallel evaluation completed" ghs_tests/g5_scale.log
python tests/gpu/check_metrics.py ghs_tests/g5_scale | grep "wall="
```

Pass: 50 of 50 built, conversion under 360 s (a hard limit in `ghs_runner.py`), not truncated.
Set `EVAL_TIMEOUT` in `quick_start_rotation.sh` to about 1.5× the reported `wall=`. The "Parallel
evaluation completed" time is the cost of one iteration; use it to budget G9.

### G6. Evaluation noise floor

```bash
python tests/gpu/repeatability.py --group sym5 --copies 5 --others 5 --eval-envs 400 --repeats 2
```

At 400 envs each design gets about 40, the same as G5. The script prints the ratio of evaluation noise
to the spread between designs. Below 0.3 is good. Above 0.7 means scores are mostly noise: give each
design more envs before G9.

### G7. Group cycling and resume

```bash
python ghs_runner.py --iterations 7 --candidates 4 --eval-envs 512 --eval-groups all --shuffle-groups \
  --batch-size 4 --seed 1 --output-dir ghs_tests/g7_groups 2>&1 | tee ghs_tests/g7_groups.log
python tests/gpu/check_metrics.py ghs_tests/g7_groups
python tests/gpu/summarize_runs.py ghs_tests/g7_groups
python ghs_runner.py --iterations 1 --candidates 4 --eval-envs 512 --eval-groups sym5 --no-shuffle-groups \
  --resume-checkpoint ghs_tests/g7_groups/ghs_checkpoint/checkpoint_5.pt --output-dir ghs_tests/g7_resume
```

Pass: the first six iterations visit all six groups once, "evaluated under wrong group" is 0, every
iteration passes `check_metrics.py`, and the resume run exits 0.

### G8. Stability with ghost fingers

```bash
python tests/gpu/check_metrics.py ghs_tests | grep -E "WARN unstable" || echo "no unstable designs"
```

Pass: no unstable designs. If some appear, mostly in sym3 or sym4 or with 2-servo fingers, test
heavier ghost links. In `Generation/converters/batch_build_rand_hands2.sh`, after the
`--ghost-range-eps` line, add `conv_flags+=("--tiny-mass" "1e-4" "--tiny-inertia" "1e-8")`. Rerun G6
with `--group sym3` and compare. Keep that edit out of commits until the results are in.

### G9. Search versus random at equal budget

```bash
for s in 1 2; do
  python ghs_runner.py --iterations 12 --candidates 24 --eval-envs 1024 --eval-timeout <from G5> \
    --eval-groups sym5 --no-shuffle-groups --seed $s --output-dir ghs_tests/g9_search_s$s
  python ghs_runner.py --iterations 12 --candidates 24 --eval-envs 1024 --eval-timeout <from G5> \
    --eval-groups sym5 --no-shuffle-groups --seed $s --output-dir ghs_tests/g9_random_s$s \
    --selection epsilon --epsilon 1.0 --epsilon-final 1.0
done
python tests/gpu/summarize_runs.py ghs_tests/g9_search_s1 ghs_tests/g9_random_s1 ghs_tests/g9_search_s2 ghs_tests/g9_random_s2
```

The random arm samples every grammar decision uniformly through the same pipeline. Pass: for both
seeds, the search's mean over its last three iterations rises above its first three and beats the
random arm, which stays flat. The search's best design should be at least as good as random's. Cost
is 48 iterations at the G5 per-iteration time, scaled by 24/50 candidates. Adjust `--eval-envs` if
G6 calls for it.

### G10. Fingertip-conditioned training smoke (optional)

```bash
cd "$REPO_ROOT"; A=$REPO_ROOT/Main/ghs_tests/g10_assets
bash Generation/converters/batch_build_rand_hands2.sh -o "$A" --mode all-random -n 40 --palm-radius 0.07 \
  --min-palm-fingers 5 --max-palm-fingers 5 --min-servos 2 --max-servos 3 --fingertip-random \
  --placement-mode symmetric --symmetric-start-deg 0 --symmetric-jitter-deg 3 --slot-count 36 --min-sep-slots 4 --seed 3
(cd IsaacLab && "$CODESIGN_ISAAC_PYTHON" source/codesign/codesign/utils/convert_urdf_usd.py --in-dir "$A" --out-dir "$A" --merge-joints --headless)
env -u CODESIGN_GHS_EVALUATION -u CODESIGN_FIXED_DR_LEVEL -u CODESIGN_DISABLE_ADR_ADAPTATION \
  CODESIGN_MORPH_FINGERTIPS=1 CODESIGN_HAND_GROUP=sym5 CODESIGN_JOINT_PRESET=symmetric \
  "$CODESIGN_ISAAC_PYTHON" IsaacLab/scripts/rl_games/train.py --task Codesign-Reorientation-Direct-v0 --headless \
  --num_envs 512 --max_iterations 20 --hand-assets-dir "$A"
```

Pass: the log shows `MORPH_DIM=55` and no "has no '_t<letters>' token" warning, and 20 iterations
complete. The evaluation variables are unset because training needs early termination and ADR.
Logs go to `./logs/rl_games/`.

## What to send back

- For the minimum check: the CPU results, both `check_metrics.py` outputs from M1, and the G4 summary.
- PASS or FAIL for each CPU test.
- The `check_metrics.py` output from G2, plus the G2 main comparison if you ran it.
- The G5 timing lines and `wall=`.
- The REPEATABILITY block from G6.
- Any `WARN unstable` lines from G8.
- The comparison table from `summarize_runs.py` in G9.
- For any failure, the log file named in that test.

Note: `--eval-episodes` counts episodes summed over all envs, so any value up to the env count gives
one 60 s round per env. That is the intended setting.
