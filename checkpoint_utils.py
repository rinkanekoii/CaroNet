"""Checkpoint and model-loading helpers shared by training, UI, and debug tools."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

MODEL_STATE_KEYS = ("model_state", "model_state_dict", "state_dict")
STATE_PREFIXES = ("module.", "_orig_mod.")


def unwrap_model(model):
    """Return the real nn.Module behind DataParallel/torch.compile wrappers."""
    changed = True
    while changed:
        changed = False
        for attr in ("module", "_orig_mod"):
            if hasattr(model, attr):
                model = getattr(model, attr)
                changed = True
    return model


def extract_model_state(checkpoint: Any):
    """Accept raw state_dict or checkpoint dict and return the model state_dict."""
    if not isinstance(checkpoint, Mapping):
        return checkpoint
    for key in MODEL_STATE_KEYS:
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def strip_state_prefixes(state_dict):
    """Remove wrapper prefixes such as `module.` and `_orig_mod.` from state keys."""
    if not isinstance(state_dict, Mapping):
        return state_dict
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in STATE_PREFIXES:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


LEGACY_KEY_MIGRATIONS = [
    (".context.0.weight", ".context.weight"),
    (".context.0.bias", ".context.bias"),
]

def normalize_legacy_state_keys(state_dict):
    """Map older checkpoint parameter names to the current model names."""
    if not isinstance(state_dict, Mapping):
        return state_dict
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in LEGACY_KEY_MIGRATIONS:
            new_key = new_key.replace(old, new)
        normalized[new_key] = value
    return normalized


def load_torch_file(path, device="cpu"):
    """torch.load wrapper that works across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        import warnings
        warnings.warn(
            f"Checkpoint tại '{path}' chứa non-tensor objects, "
            "load với weights_only=False — chỉ dùng với file đáng tin cậy.",
            stacklevel=2,
        )
        return torch.load(path, map_location=device, weights_only=False)


def save_checkpoint(
    path,
    model,
    optimizer=None,
    scaler=None,
    scheduler=None,
    iteration=None,
    **extra,
):
    """Save a portable checkpoint without DataParallel/torch.compile prefixes."""
    real_model = unwrap_model(model)
    data = {
        "model_state": real_model.state_dict(),
    }
    if optimizer is not None:
        data["optimizer_state"] = optimizer.state_dict()
    if scaler is not None:
        data["scaler_state"] = scaler.state_dict()
    if scheduler is not None:
        data["scheduler_state"] = scheduler.state_dict()
    if iteration is not None:
        data["iteration"] = iteration
    data.update(extra)
    
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_checkpoint(
    path, model, optimizer=None, scaler=None, scheduler=None, device="cpu", strict=True
):
    """Load checkpoint into model and optionally optimizer/scaler."""
    checkpoint = load_torch_file(path, device=device)
    model_state = normalize_legacy_state_keys(
        strip_state_prefixes(extract_model_state(checkpoint))
    )
    # NETWORK SURGERY: Expand input channels from 5 to 6 if needed
    net_state_dict = dict(unwrap_model(model).named_parameters())
    for key in ("stem.branch3.0.weight", "conv_input.0.weight", "stem.branch5.0.weight", "stem.branch7.0.weight"):
        if key in model_state and key in net_state_dict:
            state_w = model_state[key]
            mod_w = net_state_dict[key]
            if state_w.shape[1] == 5 and mod_w.shape[1] == 6:
                import logging
                logging.getLogger(__name__).info(f"Network Surgery on {key}: expanding input channels from 5 to 6")
                new_w = torch.zeros_like(mod_w)
                new_w[:, :5, :, :] = state_w
                model_state[key] = new_w

    incompatible = unwrap_model(model).load_state_dict(model_state, strict=strict)
    if not strict and (getattr(incompatible, "missing_keys", None) or getattr(incompatible, "unexpected_keys", None)):
        import logging
        logging.getLogger(__name__).warning(
            "Checkpoint loaded with strict=False. Missing keys: %s | Unexpected keys: %s",
            getattr(incompatible, "missing_keys", []),
            getattr(incompatible, "unexpected_keys", []),
        )

    if isinstance(checkpoint, Mapping):
        import logging
        logger = logging.getLogger(__name__)
        if optimizer is not None and "optimizer_state" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                # OPTIMIZER NETWORK SURGERY
                for name, param in unwrap_model(model).named_parameters():
                    if name in ("stem.branch3.0.weight", "conv_input.0.weight", "stem.branch5.0.weight", "stem.branch7.0.weight"):
                        if param in optimizer.state:
                            state = optimizer.state[param]
                            if "exp_avg" in state and state["exp_avg"].shape != param.shape:
                                if state["exp_avg"].shape[1] == 5 and param.shape[1] == 6:
                                    for state_key in ("exp_avg", "exp_avg_sq"):
                                        if state_key in state:
                                            old_t = state[state_key]
                                            new_t = torch.zeros_like(param, memory_format=torch.contiguous_format)
                                            new_t[:, :5, :, :] = old_t
                                            state[state_key] = new_t
                                    logger.info(f"Network Surgery on optimizer state for {name}: expanding from 5 to 6 channels")
                                else:
                                    del optimizer.state[param]
                                    logger.warning(f"Cleared optimizer state for {name} due to unhandled shape mismatch.")
            except Exception as exc:
                logger.warning("Failed to load optimizer state: %s", exc)
        if scaler is not None and "scaler_state" in checkpoint:
            try:
                scaler.load_state_dict(checkpoint["scaler_state"])
            except Exception as exc:
                logger.warning("Failed to load scaler state: %s", exc)
        if scheduler is not None and "scheduler_state" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            except Exception as exc:
                logger.warning("Failed to load scheduler state: %s", exc)
        return checkpoint.get("iteration")
    return None


def infer_use_coords_from_state_dict(state_dict, default=True):
    """Infer whether checkpoint expects coordinate planes from the stem input channels."""
    state_dict = strip_state_prefixes(state_dict)
    for key in ("stem.branch3.0.weight", "conv_input.0.weight"):
        if key in state_dict:
            in_channels = int(state_dict[key].shape[1])
            if in_channels == 3:
                return False
            if in_channels in (5, 6):
                return True
            import logging
            logging.getLogger(__name__).warning(f"Unexpected number of input channels: {in_channels}. Expected 3, 5, or 6.")
    return default


def infer_board_size_from_checkpoint(checkpoint, state_dict, default=15):
    """Infer board size from checkpoint metadata or old fully-connected policy heads."""
    if isinstance(checkpoint, Mapping):
        cfg = checkpoint.get("config") or {}
        for key in ("board_size", "size"):
            if key in cfg:
                return int(cfg[key])
        for key in ("board_size",):
            if key in checkpoint:
                return int(checkpoint[key])
    state_dict = strip_state_prefixes(state_dict)
    for key in ("policy_fc.weight",):
        if key in state_dict:
            moves = int(state_dict[key].shape[0])
            root = int(round(moves**0.5))
            if root * root == moves:
                return root
    return int(default)


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve model paths used from scripts launched in different folders."""
    path = Path(path)
    if path.exists():
        return path
    if base_dir is not None and not path.is_absolute():
        candidate = Path(base_dir) / path
        if candidate.exists():
            return candidate
    return path


def infer_model_channels(state_dict, default=320):
    """Infer ModelV8 channel count from checkpoint weights."""
    state_dict = strip_state_prefixes(state_dict)
    for key in (
        "final_norm.weight",
        "res_blocks.0.norm1.weight",
        "res_blocks.0.gn1.weight",
        "policy_local.0.weight",
    ):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    for key in ("stem.branch3.0.weight", "conv_input.0.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return default


def infer_res_blocks(state_dict, default=20):
    """Infer how many real residual blocks exist in a ModelV8 checkpoint."""
    state_dict = strip_state_prefixes(state_dict)
    indices = set()
    for key in state_dict.keys():
        parts = key.split(".")
        if len(parts) >= 3 and parts[0] == "res_blocks" and parts[1].isdigit():
            # ModelV8 PreNormResBlock has norm1/conv1; GlobalContextBlock does not.
            if parts[2] in {"norm1", "conv1", "norm2", "conv2"}:
                indices.add(int(parts[1]))
    return len(indices) if indices else default


def load_model_from_checkpoint(
    path,
    device="cpu",
    board_size=None,
    channels=None,
    num_res_blocks=None,
    dropout=0.1,
    strict=True,
    use_checkpoint=False,
    model_class=None,
):
    """Create model, infer basic architecture values, and load checkpoint weights."""
    resolved_path = resolve_path(path, Path(__file__).resolve().parent)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint. Đã tìm ở:\n"
            f"  {path}\n"
            f"  {Path(__file__).resolve().parent / path}"
        )
    checkpoint = load_torch_file(resolved_path, device=device)
    model_state = normalize_legacy_state_keys(
        strip_state_prefixes(extract_model_state(checkpoint))
    )
    if model_class is None:
        cfg = checkpoint.get("config", {}) if isinstance(checkpoint, Mapping) else {}
        model_class = cfg.get("model_class", cfg.get("model_version", "v8"))
    from models import get_model_class
    ModelClass = get_model_class(model_class)

    use_coords = infer_use_coords_from_state_dict(model_state, default=True)
    board_size = (
        infer_board_size_from_checkpoint(checkpoint, model_state)
        if board_size is None
        else int(board_size)
    )
    channels = infer_model_channels(model_state) if channels is None else int(channels)
    num_res_blocks = (
        infer_res_blocks(model_state) if num_res_blocks is None else int(num_res_blocks)
    )

    net = ModelClass(
        board_size=int(board_size),
        channels=channels,
        num_res_blocks=num_res_blocks,
        dropout=float(dropout),
        use_coords=use_coords,
        use_checkpoint=use_checkpoint,
    ).to(device)

    # NETWORK SURGERY: Expand input channels from 5 to 6 if needed
    for key in ("stem.branch3.0.weight", "conv_input.0.weight", "stem.branch5.0.weight", "stem.branch7.0.weight"):
        if key in model_state:
            state_w = model_state[key]
            try:
                mod_w = dict(net.named_parameters())[key]
                if state_w.shape[1] == 5 and mod_w.shape[1] == 6:
                    import logging
                    logging.getLogger(__name__).info(f"Network Surgery on {key}: expanding input channels from 5 to 6")
                    new_w = torch.zeros_like(mod_w)
                    new_w[:, :5, :, :] = state_w
                    model_state[key] = new_w
            except KeyError:
                pass

    net.load_state_dict(model_state, strict=strict)
    net.eval()
    return net


def export_to_onnx(model, path, board_size=15, use_coords=True):
    """Export PyTorch model to ONNX format."""
    import torch.onnx

    real_model = unwrap_model(model)
    actual_bs = getattr(real_model, "board_size", None)
    if actual_bs is not None and actual_bs != board_size:
        raise ValueError(
            f"board_size mismatch: model.board_size={actual_bs}, "
            f"nhưng export được gọi với board_size={board_size}"
        )

    real_model.eval()
    device = next(real_model.parameters()).device

    in_channels = 4 + (2 if use_coords else 0)
    dummy_input = torch.zeros(1, in_channels, board_size, board_size, device=device)

    print(f"Exporting model to ONNX: {path}")  # Suppress verbose torch.onnx warnings
    import logging

    logging.getLogger("torch.onnx").setLevel(logging.ERROR)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            real_model,
            dummy_input,
            path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["value", "policy"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "value": {0: "batch_size"},
                "policy": {0: "batch_size"},
            },
        )
    print("ONNX export successful.")
