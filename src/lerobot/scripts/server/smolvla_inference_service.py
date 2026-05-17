# SPDX-License-Identifier: Apache-2.0
"""
SmolVLA Inference Service

Drop-in ZMQ server that serves a `SmolVLAPolicy` behind the same wire protocol
as `pkgs/Isaac-GR00T/scripts/inference_service.py`, so the existing
`pkgs/Isaac-GR00T/scripts/ur5_gr00t_simple_client.py` can hit it without any
changes.

The client sends:
    video.azure_kinect : (1, H, W, 3) uint8
    video.wfov         : (1, H, W, 3) uint8
    state.ur5_arm      : (1, 6) float64
    state.gripper      : (1, 1) float64
    annotation.human.task_description : list[str]

The server returns:
    action.ur5_arm : (action_horizon, 6) float64
    action.gripper : (action_horizon,)   float64

Server usage:
    python -m lerobot.scripts.server.smolvla_inference_service \
        --server --model-path /path/to/smolvla_ur5_ckpt --port 5555

Synthetic client sanity-check (no robot needed):
    python -m lerobot.scripts.server.smolvla_inference_service --client --port 5555
"""

import io
import time
from dataclasses import dataclass

import msgpack
import numpy as np
import torch
import tyro
import zmq

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


# ---------------------------------------------------------------------------
# Wire codec — msgpack + numpy. Bit-compatible with gr00t/eval/service.py
# ---------------------------------------------------------------------------


def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj):
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def pack(data: dict) -> bytes:
    return msgpack.packb(data, default=_encode)


def unpack(data: bytes) -> dict:
    return msgpack.unpackb(data, object_hook=_decode)


# ---------------------------------------------------------------------------
# SmolVLA ZMQ server
# ---------------------------------------------------------------------------


class SmolVLAZMQServer:
    def __init__(
        self,
        policy: SmolVLAPolicy,
        *,
        host: str,
        port: int,
        api_token: str | None,
        device: str,
        action_horizon: int,
        azure_kinect_key: str,
        wfov_key: str,
        state_key: str,
    ):
        self.policy = policy
        self.device = device
        self.api_token = api_token
        self.action_horizon = action_horizon
        self.azure_kinect_key = azure_kinect_key
        self.wfov_key = wfov_key
        self.state_key = state_key

        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"SmolVLA server listening on {addr}")
        while self.running:
            try:
                request = unpack(self.socket.recv())
                if self.api_token is not None and request.get("api_token") != self.api_token:
                    self.socket.send(pack({"error": "Unauthorized: Invalid API token"}))
                    continue

                endpoint = request.get("endpoint", "get_action")
                if endpoint == "ping":
                    self.socket.send(pack({"status": "ok", "message": "SmolVLA server running"}))
                elif endpoint == "kill":
                    self.running = False
                    self.socket.send(pack({"status": "ok"}))
                elif endpoint == "get_modality_config":
                    self.socket.send(pack(self._modality_config()))
                elif endpoint == "get_action":
                    action = self._get_action(request.get("data", {}))
                    self.socket.send(pack(action))
                else:
                    self.socket.send(pack({"error": f"Unknown endpoint: {endpoint}"}))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.socket.send(pack({"error": str(e)}))

    def _modality_config(self) -> dict:
        return {
            "video": {"modality_keys": [self.azure_kinect_key, self.wfov_key]},
            "state": {"modality_keys": [self.state_key]},
            "action": {"modality_keys": ["action.ur5_arm", "action.gripper"]},
        }

    def _get_action(self, obs: dict) -> dict:
        kinect = obs["video.azure_kinect"]
        wfov = obs["video.wfov"]
        arm = obs["state.ur5_arm"]
        gripper = obs["state.gripper"]
        task_list = obs["annotation.human.task_description"]
        task = task_list[0] if isinstance(task_list, (list, tuple)) else task_list

        batch = {
            self.azure_kinect_key: _image_to_tensor(kinect, self.device),
            self.wfov_key: _image_to_tensor(wfov, self.device),
            self.state_key: torch.from_numpy(
                np.concatenate([np.asarray(arm[0], dtype=np.float32),
                                np.asarray(gripper[0], dtype=np.float32)], axis=-1)
            ).unsqueeze(0).to(self.device),
            "task": task,
        }

        self.policy.reset()
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch)
        # chunk: (1, chunk_size, action_dim)
        actions = chunk[0, : self.action_horizon].detach().cpu().numpy().astype(np.float64)
        if actions.shape[-1] < 7:
            raise RuntimeError(
                f"SmolVLA returned action_dim={actions.shape[-1]}, expected >=7 "
                f"(6 arm + 1 gripper)."
            )
        return {
            "action.ur5_arm": np.ascontiguousarray(actions[:, :6]),
            "action.gripper": np.ascontiguousarray(actions[:, 6]),
        }


def _image_to_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    """(1, H, W, 3) uint8  ->  (1, 3, H, W) float32 in [0,1] on device."""
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    t = torch.from_numpy(img).to(device).permute(2, 0, 1).contiguous().float() / 255.0
    return t.unsqueeze(0)


# ---------------------------------------------------------------------------
# Synthetic client (sanity test only — production client is ur5_gr00t_simple_client.py)
# ---------------------------------------------------------------------------


def _smoke_test_client(host: str, port: int, api_token: str | None):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{host}:{port}")

    def call(endpoint, data=None):
        req: dict = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        if api_token:
            req["api_token"] = api_token
        sock.send(pack(req))
        return unpack(sock.recv())

    print("ping:", call("ping"))
    print("modality_config keys:", list(call("get_modality_config").keys()))

    obs = {
        "video.azure_kinect": np.random.randint(0, 256, (1, 360, 640, 3), dtype=np.uint8),
        "video.wfov": np.random.randint(0, 256, (1, 360, 640, 3), dtype=np.uint8),
        "state.ur5_arm": np.random.rand(1, 6).astype(np.float64),
        "state.gripper": np.random.rand(1, 1).astype(np.float64),
        "annotation.human.task_description": ["place the small cube on the red box."],
    }
    t0 = time.time()
    action = call("get_action", obs)
    print(f"get_action took {time.time() - t0:.3f}s")
    if "error" in action:
        print("ERROR:", action["error"])
        return
    for k, v in action.items():
        print(f"  {k}: shape={np.asarray(v).shape} dtype={np.asarray(v).dtype}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class ArgsConfig:
    """SmolVLA inference service (gr00t-protocol-compatible)."""

    model_path: str | None = None
    """Path to a SmolVLA checkpoint (HF hub id or local dir). Required for --server."""

    port: int = 5555
    """TCP port to bind/connect."""

    host: str = "0.0.0.0"
    """Host to bind (server) / connect (client)."""

    api_token: str | None = None
    """Optional API token for auth."""

    device: str = "cuda"
    """Torch device for SmolVLA inference."""

    action_horizon: int = 50
    """How many steps of the SmolVLA action chunk to return per get_action call."""

    azure_kinect_key: str = "observation.images.azure_kinect"
    """SmolVLA image key mapped from client's `video.azure_kinect`."""

    wfov_key: str = "observation.images.wfov"
    """SmolVLA image key mapped from client's `video.wfov`."""

    state_key: str = "observation.state"
    """SmolVLA state key."""

    server: bool = False
    """Run as ZMQ server."""

    client: bool = False
    """Run synthetic smoke-test client."""


def main(args: ArgsConfig):
    if args.server:
        assert args.model_path is not None, "Need --model-path for server mode"
        print(f"Loading SmolVLA from {args.model_path} on {args.device}...")
        policy = SmolVLAPolicy.from_pretrained(args.model_path)
        policy.to(args.device)
        policy.eval()
        print("Policy loaded.")

        server = SmolVLAZMQServer(
            policy,
            host=args.host,
            port=args.port,
            api_token=args.api_token,
            device=args.device,
            action_horizon=args.action_horizon,
            azure_kinect_key=args.azure_kinect_key,
            wfov_key=args.wfov_key,
            state_key=args.state_key,
        )
        server.run()
    elif args.client:
        host = "localhost" if args.host == "0.0.0.0" else args.host
        _smoke_test_client(host, args.port, args.api_token)
    else:
        raise ValueError("Pass --server or --client")


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
