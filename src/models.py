import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# CELL 4: MODEL ARCHITECTURE
# =============================================================================

def get_num_groups(channels, max_groups=32):
    for g in [max_groups, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class HybridConv3d(nn.Module):
    """3x3x1 + 1x1x3 for XY/Z balance."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = out_ch // 2
        self.conv_xy = nn.Conv3d(in_ch, mid_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.conv_z = nn.Conv3d(in_ch, out_ch - mid_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.norm = nn.GroupNorm(get_num_groups(out_ch), out_ch)
        self.act = nn.LeakyReLU(0.01, inplace=True)
    
    def forward(self, x):
        return self.act(self.norm(torch.cat([self.conv_xy(x), self.conv_z(x)], dim=1)))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_hybrid=False):
        super().__init__()
        if use_hybrid:
            self.conv = HybridConv3d(in_ch, out_ch)
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.GroupNorm(get_num_groups(out_ch), out_ch),
                nn.LeakyReLU(0.01, inplace=True),
            )
    
    def forward(self, x):
        return self.conv(x)


class MultiScaleResBlock(nn.Module):
    def __init__(self, channels, scales=2, use_hybrid=False):
        super().__init__()
        self.scales = scales
        self.width = channels // scales
        self.convs = nn.ModuleList([ConvBlock(self.width, self.width, use_hybrid=use_hybrid) for _ in range(scales - 1)])
        self.norm = nn.GroupNorm(get_num_groups(channels), channels)
    
    def forward(self, x):
        splits = torch.chunk(x, self.scales, dim=1)
        outputs = [splits[0]]
        for i, conv in enumerate(self.convs):
            out = conv(splits[i + 1] + outputs[-1]) if i > 0 else conv(splits[i + 1])
            outputs.append(out)
        return x + self.norm(torch.cat(outputs, dim=1))


class AttentionBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, max(channels // reduction, 8))
        self.fc2 = nn.Linear(max(channels // reduction, 8), channels)
        self.conv_spatial = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
    
    def forward(self, x):
        b, c = x.shape[:2]
        ca = torch.sigmoid(self.fc2(F.relu(self.fc1(self.gap(x).view(b, c))))).view(b, c, 1, 1, 1)
        x_ca = x * ca
        sa = torch.sigmoid(self.conv_spatial(torch.cat([x_ca.mean(1, keepdim=True), x_ca.max(1, keepdim=True)[0]], 1)))
        return x_ca * sa


class SurfaceRefinementBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.edge_conv = nn.Conv3d(in_ch, in_ch, 3, padding=1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv3d(in_ch * 2, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(out_ch), out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(out_ch), out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
    
    def forward(self, x):
        return self.refine(torch.cat([x, torch.abs(self.edge_conv(x))], dim=1))


class TopoPreservingUNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=None, n_blocks=None,
                 use_attention=True, use_hybrid_conv=True, use_surface_refinement=True, use_deep_supervision=True):
        super().__init__()
        features = features or [32, 64, 128, 256, 320, 320]
        n_blocks = n_blocks or [1, 2, 3, 4, 6, 6]
        self.features = features
        self.use_deep_supervision = use_deep_supervision
        
        self.encoders = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        for i, (feat, nb) in enumerate(zip(features, n_blocks)):
            in_channels = in_ch if i == 0 else features[i-1]
            layers = [ConvBlock(in_channels, feat, use_hybrid=use_hybrid_conv and i > 0)]
            layers.extend([MultiScaleResBlock(feat, scales=2, use_hybrid=use_hybrid_conv) for _ in range(nb)])
            self.encoders.append(nn.Sequential(*layers))
            self.attentions.append(AttentionBlock(feat) if use_attention and i >= 2 else nn.Identity())
            if i < len(features) - 1:
                self.pools.append(nn.Conv3d(feat, feat, 2, stride=2))
        
        self.ups = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        
        for i in range(len(features)-2, -1, -1):
            self.ups.append(nn.ConvTranspose3d(features[i+1], features[i], 2, stride=2))
            if use_surface_refinement and i == 0:
                self.dec_convs.append(SurfaceRefinementBlock(features[i]*2, features[i]))
            else:
                self.dec_convs.append(nn.Sequential(
                    ConvBlock(features[i]*2, features[i], use_hybrid=use_hybrid_conv),
                    MultiScaleResBlock(features[i], scales=2, use_hybrid=use_hybrid_conv),
                ))
        
        self.final = nn.Conv3d(features[0], out_ch, 1)
        if use_deep_supervision:
            self.ds_heads = nn.ModuleList([nn.Conv3d(features[-(i+2)], out_ch, 1) for i in range(min(3, len(features)-1))])
    
    def forward(self, x):
        enc_features = []
        for i, (enc, att) in enumerate(zip(self.encoders, self.attentions)):
            x = att(enc(x))
            enc_features.append(x)
            if i < len(self.pools): x = self.pools[i](x)
        
        enc_features = enc_features[::-1]
        x = enc_features[0]
        ds_outputs = []
        
        for i, (up, dec) in enumerate(zip(self.ups, self.dec_convs)):
            x = up(x)
            skip = enc_features[i+1]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
            if self.use_deep_supervision and self.training and i < len(self.ds_heads):
                ds_outputs.append(self.ds_heads[i](x))
        
        out = self.final(x)
        return {'output': out, 'deep': ds_outputs} if self.use_deep_supervision and self.training else out


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
