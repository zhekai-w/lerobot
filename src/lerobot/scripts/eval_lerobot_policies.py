#!/usr/bin/env python3
"""
Evaluate a LeRobot policy on a dataset without a simulation environment.
Runs inference on recorded observations and plots predicted vs ground-truth actions.

Works with any LeRobot policy (ACT, pi0, pi0fast, smolvla, diffusion, etc.)

Usage:
    python3 ./src/lerobot/scripts/eval_lerobot_policies.py \
        --policy_path /home/zack/work/models/smolvla_ur5_fruit/checkpoints/100000/pretrained_model \
        --dataset_root /home/zack/work/all_datasets/1_std_datasets/test_fruit_encoded \
        --episodes 0 1 2 \
        --save_plot_path eval_results/

    # HuggingFace hub model:
    python eval_policy.py \
        --policy_path lerobot/smolvla_base \
        --dataset_repo_id lerobot/pusht
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.constants import ACTION
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from filter_utils import savgol_chunk, rts_smoother_chunk, blend_chunk_boundary


def load_policy(policy_path: str, device: str):
    config = PreTrainedConfig.from_pretrained(policy_path)
    config.device = device
    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(policy_path, config=config)
    policy.eval()
    return policy


def make_batch(item: dict, device: str) -> dict:
    """Add batch dim and move to device for all tensor values. Pass strings through as-is."""
    batch = {
        k: v.unsqueeze(0).to(device, non_blocking=True)
        for k, v in item.items()
        if isinstance(v, torch.Tensor)
    }
    if "task" in item:
        batch["task"] = item["task"]
    return batch


def eval_episode(policy, dataset, local_ep_idx: int, device: str, max_steps: int,
                 chunk_filter_fn=None):
    """
    Run policy inference over one episode from the dataset.
    Returns predicted actions, raw (unfiltered) predicted actions, ground-truth actions
    (all numpy, shape T x action_dim), per-step inference times, and inference step indices.
    pred_raw is None when no chunk_filter_fn is provided.

    chunk_filter_fn: optional callable (H, D) -> (H, D) applied to each inferred chunk.
    """
    from collections import deque as _deque
    ep_from = dataset.episode_data_index["from"][local_ep_idx].item()
    ep_to = dataset.episode_data_index["to"][local_ep_idx].item()
    n_steps = min(ep_to - ep_from, max_steps)

    pred_actions = []
    pred_actions_raw = []   # unfiltered; populated only when chunk_filter_fn is set
    gt_actions = []
    inf_times = []
    inf_steps = []
    raw_queue = _deque()    # mirrors policy queue but holds unfiltered steps

    policy.reset()

    for i in range(n_steps):
        is_inference_step = len(policy._queues[ACTION]) == 0

        if is_inference_step:
            item = dataset[ep_from + i]
            batch = make_batch(item, device)
            batch = {k: v for k, v in batch.items() if k in policy.config.input_features or k == "task"}

            t0 = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(batch)
            elapsed = time.perf_counter() - t0

            if chunk_filter_fn is not None:
                queue = policy._queues[ACTION]
                action_np = action.squeeze(0).cpu().numpy()                          # (D,)
                rest = [q.cpu().numpy().reshape(action_np.shape) for q in queue]    # list of (D,)
                chunk_raw = np.stack([action_np] + rest)                             # (H, D)
                chunk_filtered = chunk_filter_fn(chunk_raw.copy())                  # (H, D)
                orig_q_shape = queue[0].shape if queue else action_np.shape
                queue.clear()
                for step in chunk_filtered[1:]:
                    queue.append(torch.from_numpy(step.reshape(orig_q_shape)).to(device))
                raw_queue.clear()
                for step in chunk_raw[1:]:
                    raw_queue.append(step)                                           # (D,)
                pred_actions_raw.append(chunk_raw[0])
                action = torch.from_numpy(chunk_filtered[0]).unsqueeze(0).to(device)

            inf_times.append(elapsed)
            inf_steps.append(i)
            print(f"    [inference] step {i:4d}  |  {elapsed*1000:.1f} ms")
        else:
            item = dataset.hf_dataset[ep_from + i]
            with torch.inference_mode():
                action = policy.select_action({})
            if chunk_filter_fn is not None:
                pred_actions_raw.append(raw_queue.popleft() if raw_queue else action.squeeze(0).cpu().numpy())

        pred_actions.append(action.squeeze(0).cpu().numpy())
        gt_actions.append(item["action"].numpy())

    pred_raw = np.stack(pred_actions_raw) if pred_actions_raw else None
    return (
        np.stack(pred_actions),   # (T, action_dim) — filtered
        pred_raw,                 # (T, action_dim) or None
        np.stack(gt_actions),     # (T, action_dim)
        np.array(inf_times),
        inf_steps,
    )


def mse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean((pred - gt) ** 2))


def plot_episode(
    pred: np.ndarray,
    gt: np.ndarray,
    episode_id: int,
    episode_mse: float,
    dim_names: list[str] | None = None,
    save_path: str | None = None,
    inf_steps: list[int] | None = None,
    pred_raw: np.ndarray | None = None,
):
    """Plot predicted vs GT action trajectories for each action dimension.

    pred_raw: unfiltered predictions (yellow). pred is the filtered output (dotted green).
    When pred_raw is None, pred is shown in yellow as the sole prediction.
    """
    import matplotlib
    if save_path:
        matplotlib.use("Agg")

    _, action_dim = pred.shape

    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(10, 4 * action_dim + 2))
    if action_dim == 1:
        axes = [axes]
    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)

    title = f"Episode {episode_id}  |  MSE: {episode_mse:.6f}"
    fig.suptitle(title, fontsize=14, fontweight="bold", color="#2E86AB", y=0.95)

    for dim, ax in enumerate(axes):
        label = dim_names[dim] if dim_names and dim < len(dim_names) else f"Action Dimension {dim}"
        ax.plot(gt[:, dim], label="gt action", linewidth=2)
        raw = pred_raw if pred_raw is not None else pred
        ax.plot(raw[:, dim], label="pred action", linewidth=2, alpha=0.5)
        if pred_raw is not None:
            ax.plot(pred[:, dim], label="chunk filtered action", linewidth=2, linestyle="--")
        if inf_steps:
            for j, s in enumerate(inf_steps):
                if j == 0:
                    ax.plot(s, gt[s, dim], "ro", label="inference point", markersize=6)
                else:
                    ax.plot(s, gt[s, dim], "ro", markersize=4)
        ax.set_title(label, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)

    if save_path:
        out = Path(save_path) / f"episode_{episode_id:04d}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    else:
        plt.show()

    plt.close(fig)


def plot_summary(all_mse: list[float], episode_ids: list[int], save_path: str | None = None):
    """Bar chart of MSE per episode."""
    fig, ax = plt.subplots(figsize=(max(6, len(all_mse) * 0.6 + 2), 4))
    ax.bar(range(len(all_mse)), all_mse, color="steelblue", alpha=0.8)
    ax.axhline(np.mean(all_mse), color="tomato", linestyle="--", label=f"mean={np.mean(all_mse):.5f}")
    ax.set_xticks(range(len(all_mse)))
    ax.set_xticklabels([str(e) for e in episode_ids], rotation=45, ha="right")
    ax.set_xlabel("Episode")
    ax.set_ylabel("MSE")
    ax.set_title("Action MSE per Episode")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        out = Path(save_path) / "summary_mse.png"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LeRobot policy on dataset — plots predicted vs GT action trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy_path", required=True,
                        help="Local checkpoint dir or HF repo id (e.g. lerobot/smolvla_base)")
    parser.add_argument("--dataset_repo_id", default=None,
                        help="LeRobot dataset repo id (e.g. ur5_combined). "
                             "If omitted and --dataset_root is set, the root folder name is used.")
    parser.add_argument("--dataset_root", default=None,
                        help="Local root directory for the dataset")
    parser.add_argument("--episodes", type=int, nargs="*", default=None,
                        help="Episode indices to evaluate. Default: all episodes.")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Max steps per episode")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device")
    parser.add_argument("--save_plot_path", default=None,
                        help="Directory to save PNG plots. If not set, shows interactively.")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip plotting entirely (timing/MSE only)")
    parser.add_argument("--action_dim_names", nargs="*", default=None,
                        help="Human-readable labels for each action dimension")
    parser.add_argument("--video_backend", default=None,
                        help="Video backend for decoding (e.g. 'pyav'). Default: torchcodec if available.")
    parser.add_argument("--n_action_steps", type=int, default=None,
                        help="Override policy n_action_steps (actions executed per inference call).")
    parser.add_argument("--chunk_filter", default="none", choices=["none", "savgol", "rts"],
                        help="Within-chunk smoothing applied to the full (H, D) action chunk.")
    parser.add_argument("--chunk_filter_window", type=int, default=7,
                        help="Savitzky-Golay window length (must be odd and < n_action_steps).")
    parser.add_argument("--chunk_filter_polyorder", type=int, default=3,
                        help="Savitzky-Golay polynomial order (must be < chunk_filter_window).")
    parser.add_argument("--chunk_filter_q", type=float, default=1e-3,
                        help="RTS process noise — larger = more responsive, less smooth.")
    parser.add_argument("--chunk_filter_r", type=float, default=1e-4,
                        help="RTS measurement noise — larger = more smoothing.")
    parser.add_argument("--boundary_blend", action="store_true",
                        help="Cosine-ramp first --boundary_blend_steps waypoints from previous chunk end.")
    parser.add_argument("--boundary_blend_steps", type=int, default=4,
                        help="Number of waypoints to blend at chunk boundaries.")
    parser.add_argument("--dt", type=float, default=0.05,
                        help="Time step between actions (seconds), used by RTS smoother and boundary blend.")
    args = parser.parse_args()

    print(f"Device: {args.device}")

    # load policy
    print(f"\nLoading policy: {args.policy_path}")
    policy = load_policy(args.policy_path, args.device)
    print(f"Policy type: {policy.config.type}")
    if args.n_action_steps is not None:
        from collections import deque
        from lerobot.constants import ACTION
        policy.config.n_action_steps = args.n_action_steps
        policy._queues[ACTION] = deque(maxlen=args.n_action_steps)
        print(f"Overriding n_action_steps: {args.n_action_steps}")

    # load dataset
    if args.dataset_repo_id is None:
        if args.dataset_root is None:
            parser.error("At least one of --dataset_repo_id or --dataset_root must be provided.")
        args.dataset_repo_id = Path(args.dataset_root).name
        print(f"\nNo --dataset_repo_id given; using folder name: {args.dataset_repo_id}")
    print(f"\nLoading dataset: {args.dataset_repo_id}")
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        episodes=args.episodes,
        video_backend=args.video_backend,
    )
    print(f"Episodes loaded: {dataset.num_episodes}  |  Frames: {dataset.num_frames}")
    print(f"Camera keys: {dataset.meta.camera_keys}")

    global_episode_ids = (
        args.episodes if args.episodes is not None
        else list(range(dataset.num_episodes))
    )

    n_action_steps = policy.config.n_action_steps
    if args.chunk_filter == "savgol":
        assert args.chunk_filter_window % 2 == 1, "--chunk_filter_window must be odd"
        assert args.chunk_filter_window < n_action_steps, (
            f"--chunk_filter_window ({args.chunk_filter_window}) must be < n_action_steps ({n_action_steps})"
        )
        assert args.chunk_filter_polyorder < args.chunk_filter_window, (
            "--chunk_filter_polyorder must be < --chunk_filter_window"
        )

    def _make_chunk_filter_fn():
        """Build a chunk filter closure with per-episode boundary state."""
        if args.chunk_filter == "none" and not args.boundary_blend:
            return None
        prev_state = {"end_pos": None, "end_vel": None}

        def chunk_filter_fn(chunk):
            if args.chunk_filter == "savgol":
                chunk = savgol_chunk(chunk, args.chunk_filter_window, args.chunk_filter_polyorder)
            elif args.chunk_filter == "rts":
                chunk = rts_smoother_chunk(chunk, dt=args.dt, q=args.chunk_filter_q, r=args.chunk_filter_r)
            if args.boundary_blend and prev_state["end_pos"] is not None:
                chunk = blend_chunk_boundary(
                    chunk, prev_state["end_pos"], prev_state["end_vel"],
                    args.dt, args.boundary_blend_steps,
                )
            prev_state["end_pos"] = chunk[-1].copy()
            prev_state["end_vel"] = (chunk[-1] - chunk[-2]) / args.dt
            return chunk

        return chunk_filter_fn

    # eval loop
    all_mse = []
    all_inf_times = []

    for local_idx, global_ep_id in enumerate(tqdm(global_episode_ids, desc="Episodes")):
        chunk_filter_fn = _make_chunk_filter_fn()
        pred, pred_raw, gt, inf_times, inf_steps = eval_episode(
            policy, dataset, local_idx, args.device, args.steps,
            chunk_filter_fn=chunk_filter_fn,
        )
        ep_mse = mse(pred, gt)
        all_mse.append(ep_mse)
        all_inf_times.extend(inf_times.tolist())

        print(
            f"  ep {global_ep_id}: steps={len(pred)}  MSE={ep_mse:.6f}  "
            f"inf_mean={inf_times.mean()*1000:.1f}ms"
        )

        if not args.no_plot:
            plot_episode(
                pred, gt,
                episode_id=global_ep_id,
                episode_mse=ep_mse,
                dim_names=args.action_dim_names,
                inf_steps=inf_steps,
                save_path=args.save_plot_path,
                pred_raw=pred_raw,
            )

    # summary 
    print("\n--- Summary ---")
    print(f"Episodes evaluated : {len(all_mse)}")
    print(f"Mean MSE           : {np.mean(all_mse):.6f}")
    print(f"Std MSE            : {np.std(all_mse):.6f}")
    print(f"Mean inference time (per chunk): {np.mean(all_inf_times)*1000:.1f} ms")
    print(f"Min / Max inf time             : {np.min(all_inf_times)*1000:.1f} / {np.max(all_inf_times)*1000:.1f} ms")

    if not args.no_plot and len(all_mse) > 1:
        plot_summary(all_mse, global_episode_ids, save_path=args.save_plot_path)


if __name__ == "__main__":
    main()
