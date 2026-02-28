"""
Vesuvius Challenge Experiment Lab - 3D Segmentation Models
==========================================================
State-of-the-art 3D architectures for scroll surface detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ConvBlock3D(nn.Module):
    """Basic 3D convolution block with norm and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        norm_type: str = "batch",
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        # Normalization layer selection
        if norm_type == "batch":
            norm_layer = nn.BatchNorm3d(out_channels)
        elif norm_type == "instance":
            norm_layer = nn.InstanceNorm3d(out_channels, affine=True)
        elif norm_type == "group":
            norm_layer = nn.GroupNorm(min(32, out_channels), out_channels)
        else:
            norm_layer = nn.Identity()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            norm_layer,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            norm_layer,
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResConvBlock3D(nn.Module):
    """Residual 3D convolution block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch",
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        self.conv_block = ConvBlock3D(
            in_channels, out_channels,
            norm_type=norm_type, dropout_rate=dropout_rate
        )
        
        # Skip connection with optional projection
        self.skip = nn.Identity() if in_channels == out_channels else \
                    nn.Conv3d(in_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x) + self.skip(x)


class AttentionGate3D(nn.Module):
    """3D Attention Gate for focusing on relevant regions."""
    
    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: Optional[int] = None
    ):
        super().__init__()
        
        inter_channels = inter_channels or skip_channels // 2
        
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm3d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm3d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g: Gating signal from decoder (lower resolution)
        x: Skip connection from encoder (higher resolution)
        """
        # Upsample gate to match skip connection size
        g_up = F.interpolate(g, size=x.shape[2:], mode='trilinear', align_corners=True)
        
        g1 = self.W_g(g_up)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class Encoder3D(nn.Module):
    """3D U-Net Encoder with optional residual connections."""
    
    def __init__(
        self,
        in_channels: int,
        init_features: int,
        depth: int,
        use_residual: bool = False,
        norm_type: str = "batch",
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        features = init_features
        current_channels = in_channels
        
        for i in range(depth):
            block_class = ResConvBlock3D if use_residual else ConvBlock3D
            self.encoders.append(
                block_class(current_channels, features, norm_type=norm_type, dropout_rate=dropout_rate)
            )
            if i < depth - 1:
                # Use strided conv instead of MaxPool3d (MPS compatible + learnable)
                self.pools.append(
                    nn.Conv3d(features, features, kernel_size=2, stride=2, bias=False)
                )
            current_channels = features
            features *= 2
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skip_connections = []
        
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i < self.depth - 1:
                skip_connections.append(x)
                x = self.pools[i](x)
        
        return x, skip_connections


class Decoder3D(nn.Module):
    """3D U-Net Decoder with optional attention gates."""
    
    def __init__(
        self,
        init_features: int,
        depth: int,
        use_residual: bool = False,
        use_attention: bool = False,
        norm_type: str = "batch",
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        self.depth = depth
        self.use_attention = use_attention
        
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if use_attention else None
        
        # Calculate feature sizes (reversed from encoder)
        features = init_features * (2 ** (depth - 1))
        
        for i in range(depth - 1):
            out_features = features // 2
            
            self.upconvs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                    nn.Conv3d(features, out_features, kernel_size=3, padding=1)
                    )

            )
            
            if use_attention:
                self.attention_gates.append(
                    AttentionGate3D(out_features, out_features)
                )
            
            block_class = ResConvBlock3D if use_residual else ConvBlock3D
            # After concatenation: out_features (upconv) + out_features (skip) = out_features * 2
            self.decoders.append(
                block_class(features, out_features, norm_type=norm_type, dropout_rate=dropout_rate)
            )
            
            features = out_features
    
    def forward(
        self,
        x: torch.Tensor,
        skip_connections: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns final output and intermediate outputs for deep supervision."""
        
        intermediate_outputs = []
        
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            
            skip = skip_connections[-(i + 1)]
            
            # Apply attention gate if enabled
            if self.use_attention:
                skip = self.attention_gates[i](x, skip)
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
            
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)
            
            intermediate_outputs.append(x)
        
        return x, intermediate_outputs


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric segmentation.
    
    Features:
    - Flexible depth and feature count
    - Optional residual connections
    - Optional attention gates
    - Optional deep supervision
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_features: int = 32,
        depth: int = 4,
        use_residual: bool = False,
        use_attention: bool = False,
        use_deep_supervision: bool = False,
        norm_type: str = "batch",
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.use_deep_supervision = use_deep_supervision
        self.depth = depth
        
        self.encoder = Encoder3D(
            in_channels=in_channels,
            init_features=init_features,
            depth=depth,
            use_residual=use_residual,
            norm_type=norm_type,
            dropout_rate=dropout_rate
        )
        
        self.decoder = Decoder3D(
            init_features=init_features,
            depth=depth,
            use_residual=use_residual,
            use_attention=use_attention,
            norm_type=norm_type,
            dropout_rate=dropout_rate
        )
        
        # Final output layer
        self.final_conv = nn.Conv3d(init_features, out_channels, 1)
        
        # Deep supervision heads
        if use_deep_supervision:
            self.ds_heads = nn.ModuleList()
            features = init_features
            for i in range(depth - 2):  # Exclude last two levels
                features *= 2
            for i in range(depth - 1):
                self.ds_heads.append(nn.Conv3d(features, out_channels, 1))
                features //= 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        bottleneck, skip_connections = self.encoder(x)
        
        # Decode
        output, intermediate = self.decoder(bottleneck, skip_connections)
        
        # Final segmentation
        seg = self.final_conv(output)
        
        if self.use_deep_supervision and self.training:
            # Return list of outputs at different scales
            ds_outputs = [seg]
            for i, (inter, head) in enumerate(zip(intermediate[:-1], self.ds_heads)):
                ds_out = head(inter)
                # Upsample to full resolution
                ds_out = F.interpolate(ds_out, size=seg.shape[2:], mode='trilinear', align_corners=True)
                ds_outputs.append(ds_out)
            return ds_outputs
        
        return seg


class ResUNet3D(UNet3D):
    """Residual 3D U-Net variant."""
    
    def __init__(self, **kwargs):
        kwargs['use_residual'] = True
        super().__init__(**kwargs)


class AttentionUNet3D(UNet3D):
    """Attention 3D U-Net variant."""
    
    def __init__(self, **kwargs):
        kwargs['use_attention'] = True
        super().__init__(**kwargs)


def build_model(config) -> nn.Module:
    """Factory function to build model from config."""
    
    model_kwargs = {
        'in_channels': config.model.in_channels,
        'out_channels': config.model.out_channels,
        'init_features': config.model.init_features,
        'depth': config.model.depth,
        'use_residual': config.model.use_residual,
        'use_attention': config.model.use_attention,
        'use_deep_supervision': config.model.use_deep_supervision,
        'norm_type': config.model.norm_type,
        'dropout_rate': config.model.dropout_rate,
    }
    
    architecture = config.model.architecture.lower()
    
    if architecture == "unet3d":
        return UNet3D(**model_kwargs)
    elif architecture == "resunet3d":
        return ResUNet3D(**model_kwargs)
    elif architecture == "attention_unet3d":
        return AttentionUNet3D(**model_kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    x = torch.randn(2, 1, 64, 64, 64).to(device)
    
    for arch_name, arch_class in [
        ("UNet3D", UNet3D),
        ("ResUNet3D", ResUNet3D),
        ("AttentionUNet3D", AttentionUNet3D)
    ]:
        model = arch_class(in_channels=1, out_channels=1, init_features=16, depth=3).to(device)
        y = model(x)
        params = count_parameters(model)
        print(f"{arch_name}: input {x.shape} -> output {y.shape}, params: {params:,}")