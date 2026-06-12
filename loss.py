import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext

from augmentation import random_augment_batch

__all__ = ["train_network", "train_network_dataloader"]

EMPTY_PLANE_IDX = 2

def _get_autocast_context(use_amp, device_type):
    """Get autocast only where it is useful for this project."""
    if not use_amp or device_type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type)
    from torch.cuda.amp import autocast

    return autocast()


def _smooth_policy_over_legal_moves(pis_t, states_t, epsilon: float):
    """Apply label smoothing over empty cells only, not occupied cells."""
    if epsilon <= 0:
        return pis_t
    # state planes are [current player, opponent, empty, optional row coords, optional col coords]
    legal = states_t[:, EMPTY_PLANE_IDX].reshape(pis_t.shape).float()
    legal_count = legal.sum(dim=1, keepdim=True).clamp_min(1.0)
    legal_uniform = legal / legal_count
    smoothed = pis_t * (1.0 - epsilon) + legal_uniform * epsilon
    return smoothed  # Convex combination of valid dists is already normalized


def _optimizer_step(optimizer, scaler, grad_clip, use_amp, net):
    """Execute one optimizer step with optional AMP scaling and gradient clipping."""
    grad_norm = None
    if use_amp and scaler is not None:
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip).item()
        scaler.step(optimizer)
        scaler.update()
    else:
        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip).item()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm


def _compute_loss(
    net, states_t, pis_t, zs_t, policy_weight, value_weight, entropy_weight
):
    """Compute combined loss - shared between AMP and non-AMP paths."""
    v_pred, p_logits = net(states_t)

    # Policy CE intentionally uses the full softmax: illegal moves have target 0,
    # so putting probability mass there is still penalized. The entropy bonus below
    # is different: it should encourage exploration only among legal moves, not reward
    # the model for spreading probability over occupied cells. Humanity has invented
    # enough ways to reward bad behavior already.
    log_p = F.log_softmax(p_logits, dim=-1)
    safe_pis = pis_t.clamp(min=0.0)
    policy_loss = -(safe_pis * log_p).sum(dim=-1).mean()

    value_loss = F.mse_loss(v_pred.view(-1).float(), zs_t.float())

    if entropy_weight > 0:
        legal = states_t[:, EMPTY_PLANE_IDX].reshape(p_logits.shape).bool()
        neg_large = torch.finfo(p_logits.dtype).min
        legal_logits = p_logits.masked_fill(~legal, neg_large)
        legal_log_p = F.log_softmax(legal_logits, dim=-1)
        legal_probs = legal_log_p.exp().masked_fill(~legal, 0.0)
        policy_entropy = -(legal_probs * legal_log_p.masked_fill(~legal, 0.0)).sum(dim=-1).mean()
    else:
        policy_entropy = 0.0

    loss = (
        policy_weight * policy_loss
        + value_weight * value_loss
        - entropy_weight * policy_entropy
    )
    return loss, policy_loss, value_loss


def _tensor_to_float(v, n: int) -> float:
    return v.item() / n if isinstance(v, torch.Tensor) else float(v) / n


def _run_training_loop(
    net, data_iter, device, optimizer, scaler,
    use_amp, label_smoothing, grad_clip,
    policy_weight, value_weight, entropy_weight,
    accumulation_steps, is_dataloader=False
):
    """Inner loop dùng chung cho cả 2 training paths."""
    device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    autocast_ctx = _get_autocast_context(use_amp, device_type)
    
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0
    accum_count = 0
    
    optimizer.zero_grad(set_to_none=True)

    for batch in data_iter:
        states_t, pis_t, zs_t = batch
        
        if is_dataloader:
            use_pin_memory = device_type == "cuda"
            if use_pin_memory:
                states_t = states_t.contiguous(memory_format=torch.channels_last)
            states_t = states_t.to(device, non_blocking=use_pin_memory)
            pis_t = pis_t.to(device, non_blocking=use_pin_memory)
            zs_t = zs_t.to(device, non_blocking=use_pin_memory)
        else:
            if device_type == "cuda":
                states_t = states_t.contiguous(memory_format=torch.channels_last)
            states_t = states_t.to(device)
            pis_t = pis_t.to(device)
            zs_t = zs_t.to(device)

        pis_t = _smooth_policy_over_legal_moves(pis_t, states_t, label_smoothing)

        with autocast_ctx:
            loss, policy_loss, value_loss = _compute_loss(
                net,
                states_t,
                pis_t,
                zs_t,
                policy_weight,
                value_weight,
                entropy_weight,
            )
            if accumulation_steps > 1:
                loss = loss / accumulation_steps

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_count += 1
        total_loss += loss.detach() * (accumulation_steps if accumulation_steps > 1 else 1)
        total_policy_loss += policy_loss.detach()
        total_value_loss += value_loss.detach()
        num_batches += 1

        if accum_count >= accumulation_steps:
            _optimizer_step(optimizer, scaler, grad_clip, use_amp, net)
            accum_count = 0

    if accum_count > 0:
        _optimizer_step(optimizer, scaler, grad_clip, use_amp, net)

    n = max(1, num_batches)
    return (
        _tensor_to_float(total_loss, n),
        _tensor_to_float(total_policy_loss, n),
        _tensor_to_float(total_value_loss, n),
    )


def train_network_dataloader(
    net,
    buffer,
    device,
    optimizer,
    scaler=None,
    batch_size=256,
    epochs=3,
    use_amp=False,
    label_smoothing=0.0,
    grad_clip=1.0,
    policy_weight=1.0,
    value_weight=1.0,
    entropy_weight=0.0,
    num_workers=2,
    accumulation_steps=1,
):
    net.train()
    results = [0.0, 0.0, 0.0]
    
    device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    use_pin_memory = device_type == "cuda"
    effective_workers = num_workers if device_type == "cuda" else 0
    
    micro_batch = (
        max(1, batch_size // max(1, accumulation_steps))
        if accumulation_steps > 1
        else max(1, batch_size)
    )

    for _ in range(epochs):
        dataloader = buffer.get_dataloader(
            batch_size=micro_batch,
            num_workers=effective_workers,
            pin_memory=use_pin_memory,
            augment=True,
        )
        if dataloader is None:
            continue
            
        r = _run_training_loop(
            net, dataloader, device, optimizer, scaler,
            use_amp, label_smoothing, grad_clip,
            policy_weight, value_weight, entropy_weight,
            accumulation_steps, is_dataloader=True
        )
        results = [a + b for a, b in zip(results, r)]
        
    n = max(1, epochs)
    return tuple(x / n for x in results)


class _ManualSampleIter:
    def __init__(self, buffer, micro_batch, num_iters):
        self.buffer = buffer
        self.micro_batch = micro_batch
        self.num_iters = num_iters
        self.board_size = None
        if buffer.state_shape is not None:
            self.board_size = buffer.state_shape[-1]
        elif buffer.pis is not None:
            self.board_size = int(np.sqrt(buffer.pis.shape[1]))

    def __iter__(self):
        for _ in range(self.num_iters):
            states, pis, zs = self.buffer.sample(self.micro_batch)
            if states is None:
                continue
            if self.board_size is None:
                self.board_size = int(np.sqrt(pis.shape[1]))
            states, pis, zs = random_augment_batch(states, pis, zs, self.board_size)
            yield torch.from_numpy(states), torch.from_numpy(pis), torch.from_numpy(zs)


def train_network(
    net,
    buffer,
    device,
    optimizer,
    scaler=None,
    batch_size=256,
    epochs=3,
    use_amp=False,
    label_smoothing=0.0,
    grad_clip=1.0,
    policy_weight=1.0,
    value_weight=1.0,
    entropy_weight=0.0,
    accumulation_steps=1,
):
    net.train()
    results = [0.0, 0.0, 0.0]
    
    micro_batch = (
        max(1, batch_size // max(1, accumulation_steps))
        if accumulation_steps > 1
        else max(1, batch_size)
    )

    for _ in range(epochs):
        num_iters = max(1, len(buffer) // micro_batch)
        data_iter = _ManualSampleIter(buffer, micro_batch, num_iters)
        r = _run_training_loop(
            net, data_iter, device, optimizer, scaler,
            use_amp, label_smoothing, grad_clip,
            policy_weight, value_weight, entropy_weight,
            accumulation_steps, is_dataloader=False
        )
        results = [a + b for a, b in zip(results, r)]
        
    n = max(1, epochs)
    return tuple(x / n for x in results)
