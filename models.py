import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _compute_groups(channels: int, target: int = 8) -> int:
    g = max(1, min(target, channels // 8))
    while channels % g != 0:
        g -= 1
    return g

def _kaiming_init(module):
    import torch
    import torch.nn as nn
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0.0)

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Stochastic depth without .item(), so CUDA does not synchronize every block."""
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - float(drop_prob)
    if keep_prob <= 0.0:
        return torch.zeros_like(x)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob).div_(keep_prob)
    return x * mask

class ResidualBlock(nn.Module):
    def __init__(self, channels, groups=8, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.gn2(self.conv2(out))
        return F.relu(out + residual)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        y = x.mean(dim=(2, 3))
        y = torch.sigmoid(self.fc2(F.relu(self.fc1(y))))
        return x * y.unsqueeze(-1).unsqueeze(-1)

class ResidualBlockSE(nn.Module):
    def __init__(
        self, channels, groups=8, dropout=0.0, se_reduction=8, zero_init=False
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.se = SEBlock(channels, reduction=se_reduction)
        if zero_init and getattr(self.gn2, "weight", None) is not None:
            nn.init.constant_(self.gn2.weight, 0.0)

    def forward(self, x):
        residual = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)

class SpatialAttention(nn.Module):
    """Spatial attention for capturing long-range patterns on the board."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        attn = torch.sigmoid(self.conv(x))
        return x * attn

class BottleneckSE(nn.Module):
    """Efficient bottleneck block with SE attention - more capacity, same compute."""

    def __init__(
        self,
        channels,
        expansion=2,
        groups=8,
        dropout=0.0,
        se_reduction=8,
        zero_init=False,
    ):
        super().__init__()
        hidden = channels * expansion
        hidden_groups = max(1, min(32, hidden // 8))
        while hidden % hidden_groups != 0:
            hidden_groups -= 1

        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.gn1 = nn.GroupNorm(hidden_groups, hidden)
        self.conv2 = nn.Conv2d(
            hidden, hidden, 3, padding=1, groups=hidden, bias=False
        )
        self.gn2 = nn.GroupNorm(hidden_groups, hidden)
        self.conv3 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.gn3 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.se = SEBlock(channels, reduction=se_reduction)

        if zero_init and getattr(self.gn3, "weight", None) is not None:
            nn.init.constant_(self.gn3.weight, 0.0)

    def forward(self, x):
        residual = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = F.relu(self.gn2(self.conv2(out)))
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.gn3(self.conv3(out))
        out = self.se(out)
        return F.relu(out + residual)

class MultiScaleStem(nn.Module):
    """Multi-scale input processing to capture patterns at different scales."""

    def __init__(self, in_planes, channels):
        super().__init__()
        branch_ch = channels // 3
        rem = channels - branch_ch * 3

        groups_3 = max(1, min(8, branch_ch // 8))
        while branch_ch % groups_3 != 0:
            groups_3 -= 1
        groups_5 = max(1, min(8, branch_ch // 8))
        while branch_ch % groups_5 != 0:
            groups_5 -= 1
        out_ch = branch_ch + rem
        groups_7 = max(1, min(8, out_ch // 8))
        while out_ch % groups_7 != 0:
            groups_7 -= 1

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_planes, branch_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups_3, branch_ch),
            nn.ReLU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_planes, branch_ch, 5, padding=2, bias=False),
            nn.GroupNorm(groups_5, branch_ch),
            nn.ReLU(),
        )
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_planes, out_ch, 7, padding=3, bias=False),
            nn.GroupNorm(groups_7, out_ch),
            nn.ReLU(),
        )
        
        groups_fuse = _compute_groups(channels, target=32)
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 2 + out_ch, channels, 1, bias=False),
            nn.GroupNorm(groups_fuse, channels),
            nn.ReLU(),
        )

    def forward(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        return self.fuse(torch.cat([b3, b5, b7], dim=1))

class GlobalContextBlock(nn.Module):
    """Global context attention - lightweight alternative to self-attention."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.context = nn.Conv2d(channels, 1, 1)
        self.transform_in = nn.Conv2d(channels, hidden, 1, bias=False)
        self.norm = nn.LayerNorm(hidden)
        self.transform_out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        ctx = self.context(x).flatten(2)  # b, 1, hw
        ctx = torch.softmax(ctx, dim=-1)
        x_flat = x.flatten(2)  # b, c, hw
        global_ctx = torch.bmm(x_flat, ctx.transpose(1, 2))  # b, c, 1
        global_ctx = global_ctx.view(b, c, 1, 1)
        
        attn = self.transform_in(global_ctx)
        attn = self.norm(attn.view(b, -1)).view(b, -1, 1, 1)
        attn = self.transform_out(attn)
        return x * attn

class PreNormResBlock(nn.Module):
    """Pre-normalization residual block with stochastic depth."""

    def __init__(self, channels, groups=8, dropout=0.0, drop_path=0.0, dilation=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.drop_path_rate = drop_path
        self.se = SEBlock(channels, reduction=8)

        # Zero-init last conv for better training stability
        nn.init.constant_(self.conv2.weight, 0)

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x)))
        out = self.conv2(self.dropout(F.relu(self.norm2(out))))
        out = self.se(out)
        out = drop_path(out, self.drop_path_rate, self.training)
        return x + out

class TransformerBlock(nn.Module):
    """Lightweight transformer block for spatial attention with SDPA."""

    def __init__(self, channels, board_size, num_heads=4, mlp_ratio=2, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dropout_p = dropout
        
        self.pos_embed = nn.Parameter(torch.zeros(1, board_size * board_size, channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.norm1 = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.proj = nn.Linear(channels, channels)
        
        self.norm2 = nn.LayerNorm(channels)
        hidden = channels * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        
        # Add positional embedding
        x_flat = x_flat + self.pos_embed
        
        # Self-attention with SDPA
        normed = self.norm1(x_flat)
        q = self.q(normed)
        k = self.k(normed)
        v = self.v(normed)
        
        def split_heads(t):
            return t.reshape(b, -1, self.num_heads, c // self.num_heads).transpose(1, 2)
            
        attn_out = F.scaled_dot_product_attention(
            split_heads(q), split_heads(k), split_heads(v),
            dropout_p=self.dropout_p if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).reshape(b, -1, c)
        x_flat = x_flat + self.proj(attn_out)

        # MLP
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        # Reshape back
        return x_flat.permute(0, 2, 1).reshape(b, c, h, w)

class ModelV8(nn.Module):
    """
    Strongest architecture for 15x15 Gomoku (optimized for RTX 3070 8GB):
    - Multi-scale stem with 3x3, 5x5, 7x7 branches
    - Pre-norm residual blocks with stochastic depth
    - Global context attention blocks
    - Transformer block for global pattern recognition
    - Dual policy head with local + global features
    - Attention-weighted value head

    ~20M parameters, fits comfortably in 8GB VRAM for 17x17
    """

    def __init__(
        self,
        board_size=15,
        channels=300,
        num_res_blocks=20,
        dropout=0.1,
        use_coords=True,
        use_checkpoint=False,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.board_size = board_size
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.use_coords = use_coords
        self.use_checkpoint = use_checkpoint
        in_planes = 4 + (2 if use_coords else 0)

        groups = max(1, min(32, channels // 8))
        while channels % groups != 0:
            groups -= 1

        # Multi-scale stem
        self.stem = MultiScaleStem(in_planes, channels)

        # Stochastic depth schedule
        dpr = [
            drop_path_rate * i / max(1, num_res_blocks - 1)
            for i in range(num_res_blocks)
        ]

        # Main residual tower with pre-norm blocks
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(
                PreNormResBlock(
                    channels,
                    groups=groups,
                    dropout=dropout,
                    drop_path=dpr[i],
                    dilation=2 if 8 <= i < 12 else 1,  # Middle blocks use dilated convolutions
                )
            )
            # Add global context every 5 blocks
            if (i + 1) % 5 == 0:
                self.res_blocks.append(GlobalContextBlock(channels, reduction=4))

        # Transformer block for global attention (after conv tower).
        # MultiheadAttention requires channels % heads == 0.
        num_heads = min(8, channels)
        while channels % num_heads != 0:
            num_heads -= 1
        self.transformer = TransformerBlock(
            channels, board_size=board_size, num_heads=num_heads, mlp_ratio=2, dropout=dropout
        )

        # Final normalization
        self.final_norm = nn.GroupNorm(groups, channels)

        # Policy head: local (3x3) + global (1x1) combined
        policy_ch = 96
        policy_groups = max(1, min(8, policy_ch // 8))
        self.policy_local = nn.Sequential(
            nn.Conv2d(channels, policy_ch, 3, padding=1, bias=False),
            nn.GroupNorm(policy_groups, policy_ch),
            nn.ReLU(),
        )
        self.policy_global = nn.Sequential(
            nn.Conv2d(channels, policy_ch, 1, bias=False),
            nn.GroupNorm(policy_groups, policy_ch),
            nn.ReLU(),
        )
        self.policy_combine = nn.Sequential(
            nn.Conv2d(policy_ch * 2, policy_ch, 1, bias=False),
            nn.GroupNorm(policy_groups, policy_ch),
            nn.ReLU(),
            nn.Conv2d(policy_ch, 1, 1, bias=True),
        )

        # Value head with attention pooling
        value_ch = 96
        value_groups = max(1, min(8, value_ch // 8))
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, value_ch, 1, bias=False),
            nn.GroupNorm(value_groups, value_ch),
            nn.ReLU(),
        )
        self.value_attn = nn.Conv2d(value_ch, 1, 1, bias=True)
        self.value_fc = nn.Sequential(
            nn.Linear(value_ch, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

        # Initialize weights
        self._init_weights()
        # Preserve zero-init for stochastic depth stability
        for block in self.res_blocks:
            if isinstance(block, PreNormResBlock):
                nn.init.constant_(block.conv2.weight, 0)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        expected_ch = 4 + (2 if self.use_coords else 0)
        assert x.shape[1] == expected_ch, (
            f"Expected {expected_ch} input channels (use_coords={self.use_coords}), got {x.shape[1]}. "
            f"If use_coords=True, append normalized (row, col) coordinate maps after [player, opponent, empty]."
        )

        # Multi-scale stem
        x = self.stem(x)

        # Residual tower with global context
        for block in self.res_blocks:
            if (
                self.use_checkpoint
                and x.requires_grad
                and isinstance(block, PreNormResBlock)
            ):
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Global attention via transformer
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self.transformer, x, use_reentrant=False)
        else:
            x = self.transformer(x)

        # Final norm
        x = F.relu(self.final_norm(x))

        # Policy head
        p_local = self.policy_local(x)
        p_global = self.policy_global(x)
        policy = self.policy_combine(torch.cat([p_local, p_global], dim=1))
        policy = policy.flatten(1)

        # Value head with attention pooling
        v = self.value_conv(x)
        b, c, h, w = v.shape
        v_flat = v.flatten(2)
        attn = self.value_attn(v)
        attn = attn.flatten(2)
        attn = torch.softmax(attn, dim=-1)
        v_pooled = (v_flat * attn).sum(dim=-1)

        value = self.value_fc(v_pooled)
        value = torch.tanh(value).squeeze(-1)

        return value, policy

class _LazyModelRegistry(dict):
    """Registry that avoids importing legacy_models while legacy_models is importing us."""

    _legacy_names = {"v5", "v6", "v7", "v8_legacy"}

    def _load_legacy(self):
        if all(dict.__contains__(self, name) for name in self._legacy_names):
            return
        from legacy_models import ModelV5, ModelV6, ModelV7, ModelV8_Legacy

        super().update(
            {
                "v5": ModelV5,
                "v6": ModelV6,
                "v7": ModelV7,
                "v8_legacy": ModelV8_Legacy,
            }
        )

    def get(self, key, default=None):
        if key in self._legacy_names:
            self._load_legacy()
        return super().get(key, default)

    def __getitem__(self, key):
        if key in self._legacy_names:
            self._load_legacy()
        return super().__getitem__(key)

    def __contains__(self, key):
        return key == "v8" or key in self._legacy_names or super().__contains__(key)

    def keys(self):
        return ["v5", "v6", "v7", "v8", "v8_legacy"]

    def __iter__(self):
        return iter(self.keys())


_MODEL_REGISTRY = _LazyModelRegistry({"v8": ModelV8})


def get_model_class(version: str):
    ModelClass = _MODEL_REGISTRY.get(version)
    if ModelClass is None:
        raise ValueError(f"Unknown model version: {version}. Choose from {list(_MODEL_REGISTRY.keys())}")
    return ModelClass


def build_model(version: str, compile: bool = False, memory_format=None, **kwargs) -> nn.Module:
    model = get_model_class(version)(**kwargs)
    
    if memory_format is not None:
        model = model.to(memory_format=memory_format)
        
    if compile:
        model = torch.compile(model, mode="reduce-overhead")
        
    return model

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
