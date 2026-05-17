#!/usr/bin/env python3
"""
Evaluate a LeRobot policy on a dataset without a simulation environment.
Runs inference on recorded observations and plots predicted vs ground-truth actions.

Works with any LeRobot policy (ACT, pi0, pi0fast, smolvla, diffusion, etc.)

Usage:
    python3 ./src/lerobot/scripts/eval_lerobot_policies.py \
        --policy_path /home/zack/work/Models/smolvla_ur5/checkpoints/100000/pretrained_model \
        --dataset_root /home/zack/work/all_datasets/30hz_reordered/combined_nodepth \
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


def eval_episode(policy, dataset, local_ep_idx: int, device: str, max_steps: int):
    """
    Run policy inference over one episode from the dataset.
    Returns predicted actions, ground-truth actions (both numpy, shape T x action_dim),
    and per-step inference times.
    """
    ep_from = dataset.episode_data_index["from"][local_ep_idx].item()
    ep_to = dataset.episode_data_index["to"][local_ep_idx].item()
    n_steps = min(ep_to - ep_from, max_steps)

    pred_actions = []
    gt_actions = []
    inf_times = []
    inf_steps = []

    policy.reset()

    for i in range(n_steps):
        is_inference_step = len(policy._queues[ACTION]) == 0

        if is_inference_step:
            item = dataset[ep_from + i]
            batch = make_batch(item, device)

            t0 = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(batch)
            elapsed = time.perf_counter() - t0

            inf_times.append(elapsed)
            inf_steps.append(i)
            print(f"    [inference] step {i:4d}  |  {elapsed*1000:.1f} ms")
        else:
            item = dataset.hf_dataset[ep_from + i]
            with torch.inference_mode():
                action = policy.select_action({})

        pred_actions.append(action.squeeze(0).cpu().numpy())
        gt_actions.append(item["action"].numpy())

    return (
        np.stack(pred_actions),   # (T, action_dim)
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
):
    """Plot predicted vs GT action trajectories for each action dimension."""
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
        ax.plot(pred[:, dim], label="pred action", linewidth=2, alpha=0.5)
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
    args = parser.parse_args()

    print(f"Device: {args.device}")

    # load policy
    print(f"\nLoading policy: {args.policy_path}")
    policy = load_policy(args.policy_path, args.device)
    print(f"Policy type: {policy.config.type}")

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

    # eval loop 
    all_mse = []
    all_inf_times = []

    for local_idx, global_ep_id in enumerate(tqdm(global_episode_ids, desc="Episodes")):
        pred, gt, inf_times, inf_steps = eval_episode(
            policy, dataset, local_idx, args.device, args.steps
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
