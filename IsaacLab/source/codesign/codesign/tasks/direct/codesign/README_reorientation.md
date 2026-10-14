# Reorientation Environment for Custom Hand

This directory contains the reorientation environment for the custom 20-joint hand with 5 fingers.

## Overview

The reorientation environment is based on the LEAP Hand reorientation environment and adapted for the custom hand with 20 joints (4 joints per finger, 5 fingers total). The environment tasks the hand with reorienting an object in-hand through continuous z-axis rotations.

## Files

- `reorientation_env.py`: Main environment implementation
- `reorientation_env_cfg.py`: Environment configuration
- `test_reorientation_env.py`: Test script to verify the environment works
- `README_reorientation.md`: This documentation file

## Environment Details

### Hand Configuration
- **Number of joints**: 20 (4 per finger)
- **Number of fingers**: 5
- **Actuated joints**: All 20 joints are actuated
- **Fingertips**: 5 fingertip bodies for contact sensing

### Joint Names
The 20 actuated joints are:
```
f1_finger1_palm_joint, f1_finger1_base_joint, f1_finger1_middle_joint, f1_finger1_end_joint
f2_finger1_palm_joint, f2_finger1_base_joint, f2_finger1_middle_joint, f2_finger1_end_joint
f3_finger1_palm_joint, f3_finger1_base_joint, f3_finger1_middle_joint, f3_finger1_end_joint
f4_finger1_palm_joint, f4_finger1_base_joint, f4_finger1_middle_joint, f4_finger1_end_joint
f5_finger1_palm_joint, f5_finger1_base_joint, f5_finger1_middle_joint, f5_finger1_end_joint
```

### Fingertip Names
The 5 fingertip bodies are:
```
f1_finger1_tip, f2_finger1_tip, f3_finger1_tip, f4_finger1_tip, f5_finger1_tip
```

## Environment Parameters

### Action Space
- **Size**: 20 (one for each joint)
- **Type**: Continuous, clamped to [-1, 1]
- **Action type**: Relative (can be changed to absolute in config)

### Observation Space
- **Size**: 168 by default = 3 history steps × 43 (20 normalized joint positions, 20 current targets,
  3 privileged object position) + 4 object-type one-hot + 35 morphology vector
- **Morphology vector**: per finger presence, per-link servo presence and grammar scale, parsed from
  the hand's USD file name. With `CODESIGN_MORPH_FINGERTIPS=1` a 4-way fingertip one-hot per finger is
  appended (35 -> 55, observation 188); policies must be trained and evaluated with the same setting.
- Joint positions are normalized by the joint range with the range clamped to >= 1e-3 rad, so ghost
  (removed-servo) joints with tiny ranges cannot produce huge or non-finite observations.

### Reward Function
The reward includes:
- **Distance reward**: Penalty for object distance from goal
- **Rotation reward**: Reward for orientation alignment
- **Action penalty**: Regularization on actions
- **Pose difference penalty**: Penalty for deviating from default pose
- **Torque penalty**: Regularization on joint torques
- **Success bonus**: Large bonus when goal is reached
- **Fall penalty**: Penalty when object falls out of hand

### Episode Termination
Episodes terminate when:
- Object falls out of hand (distance > fall_dist)
- Object is flipped (z-axis misalignment)
- Maximum episode length is reached

In GHS evaluation mode (`CODESIGN_GHS_EVALUATION=1`) early termination is disabled and every episode
runs for a fixed `fixed_eval_time_s` (60 s) so designs are compared on equal time. Per-design
accumulators are rolled up once per episode in `_reset_idx`; the initial reset does not count as an
episode. Envs whose state becomes non-finite (or whose object is flung > 2 m from the hand) are
flagged and reported as `unstable_episode_fraction` instead of poisoning the averages.

## Usage

### Registration
The environment is registered as `"Codesign-Reorientation-Direct-v0"` and can be used with IsaacLab's training framework.

### Testing
Run the test script to verify the environment works:
```bash
cd codesign/source/codesign/codesign/tasks/direct/codesign
python test_reorientation_env.py
```

### Training
The environment can be used with the existing RL Games PPO configuration in `agents/rl_games_ppo_cfg.yaml`.

## Configuration

Key configuration parameters in `reorientation_env_cfg.py`:

- `action_space`: 20 (number of joints)
- `observation_space`: 168 (see above; 188 with fingertip conditioning)
- `actuated_joint_names`: List of 20 joint names
- `fingertip_body_names`: List of 5 fingertip names
- `z_rotation_steps`: 16 (number of rotation steps for full 360°)
- `success_tolerance`: 0.2 (tolerance for success)
- `fall_dist`: 0.07 (distance threshold for fall penalty)

## Notes

- The environment uses the custom hand configuration from `cd_hand.py`
- ADR (Adaptive Domain Randomization) is enabled; `CODESIGN_FIXED_DR_LEVEL` pins the level for
  evaluation and `CODESIGN_DISABLE_ADR_ADAPTATION=1` stops it from changing mid-run
- The environment is designed for the DexCube object but can be modified for other objects
- Default joint positions are set to zeros but can be adjusted based on the hand's natural pose 