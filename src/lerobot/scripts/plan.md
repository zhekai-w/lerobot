# Plan: Train SmolVLA on UR5

## Context

Dataset: `All_Datasets/30hz`, LeRobot v2.1 format, 7-DOF UR5 + gripper, 2 RGB cameras (cam1=Azure Kinect, cam2=RealSense), 30 fps. Task: place cube on colored box. ~192 episodes / ~75K frames in `just_medium_encoded`; single-task variants (small/medium/large to red/green/orange) each ~64 episodes.

**Joint ordering in dataset is non-standard** — shoulder_pan is at index 5:
- Current:  `[shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, shoulder_pan, gripper]`
- Standard: `[shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, gripper]`

---

## Decision: Finetune from Base

Use `lerobot/smolvla_base`, not scratch. Reasons:
- Only ~75K frames — too few for scratch convergence
- Base has strong vision-language priors that transfer across embodiments
- SmolVLA pads actions to 32-dim and adapts per-embodiment regardless

Set `train_expert_only=false` since UR5 is new embodiment (not same as base training robots).

---

## Decision: Reorder Joints

**Yes, reorder before training.** Shoulder_pan has largest range of motion — putting it at index 0 is standard convention and avoids confusion when comparing against other UR5 datasets or policies.

Reorder: swap index 5 → 0, shift indices 0–4 → 1–5, for both `observation.state` and `action` columns in every parquet file.

---

## Step 1: Joint Reorder Preprocessing

Write `All_Datasets/reorder_joints.py`:
- Input: dataset dir (e.g. `30hz/just_medium_encoded`)
- For each parquet in `data/`: reorder state/action numpy arrays `[0,1,2,3,4,5,6]` → `[5,0,1,2,3,4,6]`
- Update `meta/info.json` joint name list to match new order
- Output: new dir e.g. `30hz_reordered/just_medium_encoded`

## Step 2: Verify

Load episode 0 from original and reordered. Assert:
```python
original["observation.state"][0, 5] ≈ reordered["observation.state"][0, 0]
```

## Step 3: Train

```bash
cd /home/zack/work/pkgs/lerobot

python -m lerobot.scripts.train \
  --policy.path=lerobot/smolvla_base \
  --dataset.root=/home/zack/work/All_Datasets/30hz_reordered/just_medium_encoded \
  --batch_size=32 \
  --steps=100000 \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=false \
  --output_dir=outputs/smolvla_ur5
```

Camera keys (`observation.images.cam1`, `observation.images.cam2`) auto-detected from dataset meta.
Depth cameras (`cam1_depth`, `cam2_depth`) skipped — SmolVLA doesn't handle depth natively.

## Step 4: Optional — Multi-task / Combined Dataset

To train on all size×color variants, concatenate encoded datasets via `MultiLeRobotDataset` or merge parquet files. All must be reordered first.

---

## Critical Files

| File | Purpose |
|------|---------|
| `pkgs/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py` | SmolVLA config (max_action_dim=32, freeze flags) |
| `pkgs/lerobot/src/lerobot/scripts/train.py` | Training entry point |
| `pkgs/lerobot/src/lerobot/datasets/lerobot_dataset.py` | Dataset loading logic |
| `All_Datasets/30hz/just_medium_encoded/meta/info.json` | Dataset metadata / joint names |

---

## Verification

1. Loss decreases within first 1K steps → embodiment loaded correctly
2. Check `outputs/smolvla_ur5/` for checkpoints at save_freq intervals
3. Run `visualize_dataset.py` on reordered dataset to sanity-check joint trajectories
