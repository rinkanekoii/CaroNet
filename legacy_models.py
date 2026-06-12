import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from models import (
    SEBlock, SpatialAttention, GlobalContextBlock,
    _compute_groups, _kaiming_init,
    ResidualBlock, ResidualBlockSE, BottleneckSE, MultiScaleStem,
    PreNormResBlock
)

class ImprovedPVNet(nn.Module):
    def __init__(
        self,
        board_size=17,
        channels=256,
        num_res_blocks=20,
        dropout=0.0,
        use_coords=False,
    ):
        super().__init__()
        self.board_size = board_size
        self.use_coords = use_coords
        in_planes = 3 + (2 if use_coords else 0)
        groups = _compute_groups(channels, target=32)

        self.conv_input = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(channels, groups=groups, dropout=dropout)
                for _ in range(num_res_blocks)
            ]
        )

        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False), nn.GroupNorm(4, 32), nn.ReLU()
        )
        self.policy_fc = nn.Linear(
            32 * board_size * board_size, board_size * board_size
        )

        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False), nn.GroupNorm(4, 16), nn.ReLU()
        )
        value_hidden = max(256, board_size * 16)
        self.value_fc1 = nn.Linear(16 * board_size * board_size, value_hidden)
        self.value_ln = nn.LayerNorm(value_hidden)
        self.value_dropout = nn.Dropout(0.3)
        self.value_fc2 = nn.Linear(value_hidden, 1)

        _kaiming_init(self)

    def forward(self, x):
        expected_ch = 3 + (2 if self.use_coords else 0)
        assert x.shape[1] == expected_ch, (
            f"Expected {expected_ch} input channels (use_coords={self.use_coords}), got {x.shape[1]}. "
            f"If use_coords=True, append normalized (row, col) coordinate maps after [player, opponent, empty]."
        )
        x = self.conv_input(x)
        for block in self.res_blocks:
            x = block(x)
        policy = self.policy_conv(x)
        policy = policy.flatten(1)
        policy = self.policy_fc(policy)
        value = self.value_conv(x)
        value = value.flatten(1)
        value = F.relu(self.value_ln(self.value_fc1(value)))
        value = self.value_dropout(value)
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)
        return value, policy



class ModelV5(nn.Module):
    """
    Wider receptive field + SE blocks, tuned defaults for 20x20 Caro.
    """

    def __init__(
        self,
        board_size=20,
        channels=192,
        num_res_blocks=20,
        dropout=0.1,
        use_coords=True,
        use_checkpoint=False,
    ):
        super().__init__()
        self.board_size = board_size
        self.use_coords = use_coords
        self.use_checkpoint = use_checkpoint
        in_planes = 3 + (2 if use_coords else 0)
        groups = _compute_groups(channels, target=32)

        self.conv_input = nn.Sequential(
            nn.Conv2d(in_planes, channels, 5, padding=2, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlockSE(
                    channels,
                    groups=groups,
                    dropout=dropout,
                    se_reduction=8,
                    zero_init=True,
                )
                for _ in range(num_res_blocks)
            ]
        )

        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 48, 1, bias=False), nn.GroupNorm(8, 48), nn.ReLU()
        )
        self.policy_fc = nn.Linear(
            48 * board_size * board_size, board_size * board_size
        )

        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False), nn.GroupNorm(8, 32), nn.ReLU()
        )
        value_hidden = max(512, board_size * 24)
        self.value_fc1 = nn.Linear(32 * board_size * board_size, value_hidden)
        self.value_gn = nn.GroupNorm(8, value_hidden)
        self.value_dropout = nn.Dropout(0.25)
        self.value_fc2 = nn.Linear(value_hidden, 1)

        _kaiming_init(self)

    def forward(self, x):
        x = self.conv_input(x)
        for block in self.res_blocks:
            if self.use_checkpoint and x.requires_grad:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        policy = self.policy_conv(x)
        policy = policy.flatten(1)
        policy = self.policy_fc(policy)
        value = self.value_conv(x)
        value = value.flatten(1)
        value = F.relu(self.value_gn(self.value_fc1(value)))
        value = self.value_dropout(value)
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)
        return value, policy



class ModelV6(nn.Module):
    """
    Deeper architecture with coordinate channels and SE Blocks.
    Optimized for complex board interactions.
    """
    def __init__(
        self,
        board_size=20,
        channels=192,
        num_res_blocks=20,
        dropout=0.1,
        use_coords=True,
        use_checkpoint=False,
    ):
        super().__init__()
        self.board_size = board_size
        self.use_coords = use_coords
        self.use_checkpoint = use_checkpoint
        in_planes = 3 + (2 if use_coords else 0)
        groups = _compute_groups(channels, target=32)

        self.conv_input = nn.Sequential(
            nn.Conv2d(in_planes, channels, 5, padding=2, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlockSE(
                    channels,
                    groups=groups,
                    dropout=dropout,
                    se_reduction=8,
                    zero_init=True,
                )
                for _ in range(num_res_blocks)
            ]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=True),
        )

        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
        )
        value_hidden = max(256, board_size * 16)
        self.value_fc1 = nn.Linear(32, value_hidden)
        self.value_ln = nn.LayerNorm(value_hidden)
        self.value_dropout = nn.Dropout(0.25)
        self.value_fc2 = nn.Linear(value_hidden, 1)

        _kaiming_init(self)

    def forward(self, x):
        x = self.conv_input(x)
        for block in self.res_blocks:
            if self.use_checkpoint and x.requires_grad:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        policy = self.policy_head(x)
        policy = policy.flatten(1)
        value = self.value_conv(x)
        value = value.mean(dim=(2, 3))
        value = F.relu(self.value_ln(self.value_fc1(value)))
        value = self.value_dropout(value)
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)
        return value, policy



class ModelV7(nn.Module):
    """
    Advanced architecture optimized for RTX 3070 (8GB VRAM):
    - Multi-scale stem: captures patterns at 3x3, 5x5, 7x7 scales
    - Bottleneck blocks with SE: more capacity with efficient compute
    - Spatial attention: long-range pattern recognition
    - Dual-head policy: local (3x3) + global (1x1) features combined
    - Attention-weighted value pooling: focuses on important board regions
    """

    def __init__(
        self,
        board_size=20,
        channels=256,
        num_res_blocks=16,
        dropout=0.1,
        use_coords=True,
        use_checkpoint=False,
    ):
        super().__init__()
        self.board_size = board_size
        self.use_coords = use_coords
        self.use_checkpoint = use_checkpoint
        in_planes = 3 + (2 if use_coords else 0)

        groups = max(1, min(32, channels // 8))
        while channels % groups != 0:
            groups -= 1

        # Multi-scale stem
        self.stem = MultiScaleStem(in_planes, channels)

        # Main residual tower with bottleneck blocks
        self.res_blocks = nn.ModuleList(
            [
                BottleneckSE(
                    channels,
                    expansion=2,
                    groups=groups,
                    dropout=dropout,
                    se_reduction=8,
                    zero_init=True,
                )
                for _ in range(num_res_blocks)
            ]
        )

        # Spatial attention after residual tower
        self.spatial_attn = SpatialAttention(channels)

        # Policy head: combines local (conv) and global (attention) features
        policy_ch = 64
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
        self.policy_combine = nn.Conv2d(policy_ch * 2, 1, 1, bias=True)

        # Value head with attention-weighted pooling
        value_ch = 64
        value_groups = max(1, min(8, value_ch // 8))
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, value_ch, 1, bias=False),
            nn.GroupNorm(value_groups, value_ch),
            nn.ReLU(),
        )
        self.value_attn = nn.Conv2d(value_ch, 1, 1, bias=True)
        value_hidden = 512
        self.value_fc1 = nn.Linear(value_ch, value_hidden)
        self.value_ln = nn.LayerNorm(value_hidden)
        self.value_dropout = nn.Dropout(0.25)
        self.value_fc2 = nn.Linear(value_hidden, 1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # Multi-scale stem
        x = self.stem(x)

        # Residual tower
        for block in self.res_blocks:
            if self.use_checkpoint and x.requires_grad:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Spatial attention
        x = self.spatial_attn(x)

        # Policy head: local + global features
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

        value = F.relu(self.value_ln(self.value_fc1(v_pooled)))
        value = self.value_dropout(value)
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)

        return value, policy




class MultiScaleStem_Legacy(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_ch = out_channels // 3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, 3, padding=1, bias=False),
            nn.GroupNorm(1, branch_ch),
            nn.ReLU()
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, 5, padding=2, bias=False),
            nn.GroupNorm(1, branch_ch),
            nn.ReLU()
        )
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels - 2*branch_ch, 7, padding=3, bias=False),
            nn.GroupNorm(1, out_channels - 2*branch_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)
        return torch.cat([x3, x5, x7], dim=1)

class GlobalContextBlock_Legacy(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.context = nn.Conv2d(channels, 1, 1)
        hidden = max(16, channels // reduction)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.LayerNorm([hidden, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        ctx = self.context(x).flatten(2)
        ctx = torch.softmax(ctx, dim=-1)
        x_flat = x.flatten(2)
        global_ctx = torch.bmm(x_flat, ctx.transpose(1, 2))
        global_ctx = global_ctx.view(b, c, 1, 1)
        attn = self.transform(global_ctx)
        return x * attn

class TransformerBlock_Legacy(nn.Module):
    def __init__(self, channels, num_heads=4, mlp_ratio=2, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * mlp_ratio, channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        out = self.norm1(x_flat)
        attn_out, _ = self.attn(out, out, out)
        x_flat = x_flat + attn_out
        out = self.norm2(x_flat)
        mlp_out = self.mlp(out)
        x_flat = x_flat + mlp_out
        return x_flat.transpose(1, 2).view(b, c, h, w)

class ModelV8_Legacy(nn.Module):
    def __init__(self, channels=256, board_size=15, num_res_blocks=20, dropout=0.1, use_coords=True, use_checkpoint=False):
        super().__init__()
        self.board_size = board_size
        self.use_coords = use_coords
        self.use_checkpoint = use_checkpoint
        
        in_planes = 3 + (2 if use_coords else 0)
        self.stem = MultiScaleStem_Legacy(in_planes, channels)
        
        blocks = []
        for i in range(num_res_blocks + 4):
            if i in [5, 11, 17, 23]:
                blocks.append(GlobalContextBlock_Legacy(channels, reduction=4))
            else:
                blocks.append(PreNormResBlock(channels, groups=max(1, min(32, channels // 8)), dropout=0.1))
        self.res_blocks = nn.ModuleList(blocks)
        
        self.transformer = TransformerBlock_Legacy(channels, num_heads=4, mlp_ratio=2, dropout=0.1)
        self.final_norm = nn.GroupNorm(max(1, min(32, channels // 8)), channels)
        
        policy_ch = 96
        policy_groups = max(1, min(8, policy_ch // 8))
        self.policy_local = nn.Sequential(
            nn.Conv2d(channels, policy_ch, 3, padding=1, bias=False),
            nn.GroupNorm(policy_groups, policy_ch),
            nn.ReLU()
        )
        self.policy_global = nn.Sequential(
            nn.Conv2d(channels, policy_ch, 1, bias=False),
            nn.GroupNorm(policy_groups, policy_ch),
            nn.ReLU()
        )
        self.policy_combine = nn.Sequential(
            nn.Conv2d(policy_ch * 2, policy_ch, 1, bias=False),
            nn.GroupNorm(policy_groups, policy_ch),
            nn.ReLU(),
            nn.Conv2d(policy_ch, 1, 1, bias=True)
        )
        
        value_ch = 96
        value_groups = max(1, min(8, value_ch // 8))
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, value_ch, 1, bias=False),
            nn.GroupNorm(value_groups, value_ch),
            nn.ReLU()
        )
        self.value_attn = nn.Conv2d(value_ch, 1, 1, bias=True)
        
        self.value_fc = nn.Sequential(
            nn.Linear(value_ch, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )
        
        _kaiming_init(self)
        for block in self.res_blocks:
            if isinstance(block, PreNormResBlock):
                nn.init.constant_(block.conv2.weight, 0)

    def forward(self, x):
        x = self.stem(x)
        for block in self.res_blocks:
            if self.use_checkpoint and x.requires_grad and isinstance(block, PreNormResBlock):
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.transformer(x)
        x = F.relu(self.final_norm(x))
        
        p_local = self.policy_local(x)
        p_global = self.policy_global(x)
        policy = self.policy_combine(torch.cat([p_local, p_global], dim=1))
        policy = policy.flatten(1)
        
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

