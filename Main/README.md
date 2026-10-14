# Main Directory

This directory contains the Graph Heuristic Search (GHS) orchestration layer for CoDesign-InHand.
It is responsible for:

- sampling candidate hand morphologies,
- invoking the `Generation` pipeline to build assets,
- converting assets for simulation,
- evaluating designs in IsaacLab with pretrained checkpoints,
- logging/search bookkeeping and result packaging.

The main entrypoint for routine runs is:

- `./quick_start_rotation.sh`

## What `./quick_start_rotation.sh` Does

At runtime, the script:

1. Validates expected directory structure (`../IsaacLab`, `../Generation`).
2. Checks required scripts exist:
   - `../Generation/converters/batch_build_rand_hands2.sh`
   - `../IsaacLab/scripts/rl_games/play.py`
3. Verifies Python/Torch availability in the active environment.
4. Exports runtime environment variables for GHS evaluation behavior.
5. Launches `ghs_runner.py` with a configured set of search/evaluation flags.
6. Writes outputs under `Main/ghs_rotation_optimization/run_<timestamp>/`.

## Expected Layout From `Main/`

Run this script from inside `Main/`:

```bash
cd Main
./quick_start_rotation.sh
```

The script expects sibling directories:

- `../IsaacLab`
- `../Generation`

## Key Configuration Knobs

Open `quick_start_rotation.sh` and adjust the top-level config variables:

- `ITERATIONS`: number of GHS iterations.
- `CANDIDATES`: candidate designs evaluated per iteration.
- `EVAL_EPISODES`: episodes per design evaluation.
- `EVAL_ENVS`: number of parallel IsaacLab environments.
- `PLACEMENT_MODE`: hand placement strategy (`symmetric`, `anthro-top-heavy`, `both` via runner behavior).
- `EVAL_GROUPS`: evaluation groups (`all` or subset list).
- `SHUFFLE_GROUPS`: group-order reshuffling behavior for multi-group cycles.
- `SELECTION`: successor selection rule (`gumbel` = paper's Gumbel-perturbed argmax on standardized value predictions, `epsilon` = epsilon-greedy).
- `EPSILON`: initial exploration, annealed to 0.05 by the last iteration. Gumbel temperature on standardized value predictions (1.0 recommended; 0.4 is too greedy) or epsilon-greedy rate (0.4 recommended).
- `FOCUS_METRIC`: per-hand metric used as the design score (`reward` = normalized shaped reward, `rotation` = omega score, `success`).
- `EVAL_TIMEOUT`: wall-clock budget for one evaluation; must cover the fixed 60 s simulated episode across `EVAL_ENVS`.
- `MORPH_FINGERTIPS`: `1` makes the controller's morphology vector include fingertip type (requires policies trained with `CODESIGN_MORPH_FINGERTIPS=1`; see below).
- `BATCH_SIZE`: optimization batch size.
- `DEBUG`: optional debug mode.
- `OUTPUT_DIR`: run artifact path root.

## How the Search Works

Each iteration builds `CANDIDATES` hands by sequential grammar rollouts: finger by finger, the legal
values for servo count, grammar lengths and fingertip are scored by the design value network (a small
GNN over the palm + 5 finger slots, with unassigned fields masked), and the next value is chosen by
Gumbel-perturbed argmax on the standardized scores (`SELECTION=gumbel`, temperature annealed from
`EPSILON` to 0.05) or epsilon-greedy. All candidates are built, converted and evaluated in one
simulation with the pretrained policy of the current group. Every partial state on a rollout is
credited with the best reward of its descendants, and the value network is retrained on those targets
after each iteration. Groups rotate every iteration; the order is reshuffled once per cycle.

## Fingertip-aware policies (optional retraining)

The search optimizes fingertip type, but the shipped policies observe a 35-dim morphology vector
(finger presence, servo presence, link scales) that does not include it. To condition the controller
on fingertip type:

1. Rebuild assets with the current builder. Hand names now carry a `_t<letters>` token
   (`s`/`w`/`r`/`t` = standard/wedged/rounded/thinner per slot, `x` = ghost), which the simulator
   parses into a per-finger one-hot (morphology vector becomes 55-dim).
2. Train each group with `CODESIGN_MORPH_FINGERTIPS=1` exported (`scripts/rl_games/train.py`).
3. Point `_checkpoint_for_group` in `ghs_runner.py` at the new checkpoints and run with
   `MORPH_FINGERTIPS=1`. Mixing the flag and a checkpoint trained without it fails at load time
   with an explicit message.

## Environment Notes

This workflow uses two Python contexts:

- Runner Python:
  - executes `ghs_runner.py` and search logic.
- Isaac Python:
  - executes IsaacLab-facing scripts (e.g., `play.py`, URDF->USD conversion).

Isaac interpreter resolution is handled by `Main/utils/isaac_python.py` and can be overridden with:

- CLI: `--isaac-python <path>`
- env vars: `CODESIGN_ISAAC_PYTHON` or `ISAAC_SIM_PYTHON`

## Outputs

Each run creates a timestamped folder in:

- `Main/ghs_rotation_optimization/run_<timestamp>/`

Typical artifacts include:

- `iter_<n>_<group>/batch_candidates/`: built URDF/USD assets (`d<idx>_hand_..._t<tips>`), the play.py
  metrics file and `manifest.csv` (codes, scores, raw rotation metrics, `unstable_episode_fraction`),
- `final_results/`: `iteration_group_stats.csv/jsonl`, `cycle_group_stats.csv/jsonl`,
  `design_asset_map.csv`, `lookup_table.json`, `rewards.csv`, `best_design.json`, `worst_design.json`,
- `ghs_checkpoint/`: periodic search checkpoints every 5 iterations (resume with `--resume-checkpoint`),
- best/worst design URDF exports and copied meshes,
- optional evaluation videos (when enabled),
- generated helper script(s) for best-design re-evaluation.

## Related Files

- `ghs_runner.py`: primary orchestration and evaluation loop.
- `graph_heuristic_search.py`: core GHS search logic.
- `hand_groups.py`: evaluation group definitions.
- `quick_start_rotation.sh`: main run script.
