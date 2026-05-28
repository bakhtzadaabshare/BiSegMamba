# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch 
from einops import rearrange
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from BiSegMamba.network_architecture.neural_network import SegmentationNetwork
from mamba_ssm import Mamba
import torch.nn.functional as F



class DirectionalFusion3D(nn.Module):
    """
    Channel-wise directional fusion:
      - predicts [B, 3, C, 1, 1, 1] weights
      - softmax over the 3 directions for each channel
      - concatenate + 1x1x1 mix
      - optional SE refinement
      - residual-friendly gamma scaling
    """
    def __init__(self, channels, use_softmax_gate=True, use_se=False, reduction=8):
        super().__init__()
        self.channels = channels
        self.use_softmax_gate = use_softmax_gate
        self.use_se = use_se

        if use_softmax_gate:
            hidden = max(channels // 4, 8)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),                       # [B, 3C, 1, 1, 1]
                nn.Conv3d(channels * 3, hidden, 1, bias=True),
                nn.GELU(),
                nn.Conv3d(hidden, channels * 3, 1, bias=True) # [B, 3C, 1, 1, 1]
            )

        self.mix = nn.Sequential(
            nn.Conv3d(channels * 3, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
        )

        if use_se:
            self.se = SEBlock(channels, reduction=reduction)

        self.gamma = nn.Parameter(torch.ones((1, channels, 1, 1, 1)) * 1e-6)

    def forward(self, x1, x2, x3):
        """
        x1, x2, x3: [B, C, D, W, H]
        """
        B, C, D, W, H = x1.shape

        if self.use_softmax_gate:
            gate_in = torch.cat([x1, x2, x3], dim=1)                  # [B, 3C, D, W, H]
            logits = self.gate(gate_in).view(B, 3, C, 1, 1, 1)       # [B, 3, C, 1, 1, 1]
            weights = torch.softmax(logits, dim=1)                    # softmax across directions

            x = torch.stack([x1, x2, x3], dim=1)                      # [B, 3, C, D, W, H]
            x = x * weights
            x = x.reshape(B, 3 * C, D, W, H)
        else:
            x = torch.cat([x1, x2, x3], dim=1)

        y = self.mix(x)

        if self.use_se:
            y = self.se(y)

        return self.gamma * y


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )

        self.fusion = DirectionalFusion3D(
                channels=dim,
                use_softmax_gate=True,
                use_se=True,
                reduction=8
            )
    
    def mamba_forward(self, x):
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x = self.norm(x)
        x = self.mamba(x)
        x = x.transpose(-1, -2).reshape(B, C, *img_dims)

        return x 
    
    def forward(self, x):
        
        x_skip = x

        out_x_1 = self.mamba_forward(x)

        x_2 = rearrange(x, "b c d w h -> b c w d h")

        out_x_2 = self.mamba_forward(x_2)
        
        out_x_2 = rearrange(out_x_2, "b c w d h -> b c d w h")

        x_3 = rearrange(x, "b c d w h -> b c h w d")

        out_x_3 = self.mamba_forward(x_3)
        out_x_3 = rearrange(out_x_3, "b c h w d -> b c d w h")

        # out = out_x_1 + out_x_2 + out_x_3
        # learned fusion instead of naive summation
        out = self.fusion(out_x_1, out_x_2, out_x_3)

        out = out + x_skip

        return out


class BiMambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.fusion = DirectionalFusion3D(
            channels=dim,
            use_softmax_gate=True,
            use_se=True,
            reduction=8
        )

        # learnable forward/backward weighting
        # shape: [3 directions, 2 branches(fwd,bwd), C]
        # initialized to zeros -> softmax gives 0.5 / 0.5 at start
        self.fb_logits = nn.Parameter(torch.zeros(3, 2, dim))

    def _to_seq(self, x):
        # x: [B, C, D, W, H] -> [B, N, C]
        return x.flatten(2).transpose(1, 2).contiguous()

    def _from_seq(self, x, shape):
        # x: [B, N, C] -> [B, C, ...]
        B, N, C = x.shape
        return x.transpose(1, 2).contiguous().reshape(B, C, *shape)

    def _fuse_bidirectional(self, fwd, bwd, shape, dir_idx):
        """
        fwd, bwd: [B, N, C]
        shape: target spatial shape
        dir_idx: 0, 1, 2 for the 3 directions
        """
        fwd_3d = self._from_seq(fwd, shape)
        bwd_3d = self._from_seq(bwd.flip(1), shape)

        # [2, C] -> softmax across forward/backward
        w = torch.softmax(self.fb_logits[dir_idx], dim=0)
        wf = w[0].view(1, -1, 1, 1, 1)
        wb = w[1].view(1, -1, 1, 1, 1)

        return wf * fwd_3d + wb * bwd_3d

    def forward(self, x):
        # x: [B, C, D, W, H]
        B, C, D, W, H = x.shape
        x_skip = x

        # 3 directional views
        x1 = x
        x2 = rearrange(x, "b c d w h -> b c w d h")
        x3 = rearrange(x, "b c d w h -> b c h w d")

        # flatten to sequences
        s1 = self._to_seq(x1)
        s2 = self._to_seq(x2)
        s3 = self._to_seq(x3)

        # forward + backward sequences in one batched call
        all_seqs = torch.cat(
            [s1, s2, s3, s1.flip(1), s2.flip(1), s3.flip(1)],
            dim=0
        ).contiguous()

        all_seqs = self.norm(all_seqs)
        all_out = self.mamba(all_seqs)

        o1, o2, o3, o1b, o2b, o3b = all_out.chunk(6, dim=0)

        # learnable forward/backward fusion
        out1 = self._fuse_bidirectional(o1, o1b, (D, W, H), dir_idx=0)
        out2 = self._fuse_bidirectional(o2, o2b, (W, D, H), dir_idx=1)
        out3 = self._fuse_bidirectional(o3, o3b, (H, W, D), dir_idx=2)

        # permute back to common layout
        out2 = rearrange(out2, "b c w d h -> b c d w h")
        out3 = rearrange(out3, "b c h w d -> b c d w h")

        out = self.fusion(out1, out2, out3)
        out = out + x_skip
        return out
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):

        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_residual

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class LargeKernelConv(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.deep_path = nn.Sequential(
            ConvBlock(channels, channels, 5, padding=2),
            ConvBlock(channels, channels, 3, padding=1),
            ConvBlock(channels, channels, 3, padding=2, dilation=2),
        )

        self.shortcut_path = nn.Sequential(
            ConvBlock(channels, channels, 3, padding=1),
            ConvBlock(channels, channels, 1, padding=0)
        )

        self.se = SEBlock(channels)

    def forward(self, x):
        out = self.deep_path(x) + self.shortcut_path(x) + x
        return self.se(out)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, act="relu"):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.norm = nn.InstanceNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ProgressiveStemPatch3D(nn.Module):
    def __init__(self, in_ch, stem_ch, out_ch):
        super().__init__()

        # full-resolution shallow feature extraction
        self.conv1 = ConvGNAct(in_ch, stem_ch, k=5, p=2)
        self.conv2 = ConvGNAct(stem_ch, stem_ch, k=3, p=1)

        # gradual reduction
        self.down1 = nn.Sequential(
            nn.Conv3d(stem_ch, stem_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(stem_ch),
            nn.ReLU(inplace=True),
        )

        self.refine = nn.Sequential(
            nn.Conv3d(stem_ch, stem_ch, kernel_size=3, padding=1, groups=stem_ch, bias=False),
            nn.InstanceNorm3d(stem_ch),
            nn.GELU(),
        )

        self.down2 = nn.Sequential(
            nn.Conv3d(stem_ch, out_ch, kernel_size=3, stride=(1,2,2), padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        stem = self.conv2(self.conv1(x))   # full-res skip
        y = self.down1(stem)
        y = y + self.refine(y)
        y = self.down2(y)                  # encoder input
        return stem, y


class LightRefine(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.norm = nn.InstanceNorm3d(C)
        self.act = nn.GELU()
        self.dw = nn.Conv3d(C, C, 3, padding=1, groups=C, bias=False)
        self.pw = nn.Conv3d(C, C, 1, bias=False)

    def forward(self, x):
        return x + self.pw(self.dw(self.act(self.norm(x))))

class FinalUpAnisoV2(nn.Module):
    def __init__(self, in_ch, out_ch, stem_ch):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(2,2,2), stride=(2,2,2), bias=False),
            nn.InstanceNorm3d(out_ch), 
            nn.ReLU(inplace=True),
        )
        self.ref1 = LightRefine(out_ch)

        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(out_ch, out_ch, kernel_size=(1,2,2), stride=(1,2,2), bias=False),
            nn.InstanceNorm3d(out_ch), 
            nn.ReLU(inplace=True),
        )
        self.ref2 = LightRefine(out_ch)

        self.fuse = nn.Sequential(
            nn.Conv3d(out_ch + stem_ch, out_ch, 1, bias=False),
            nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True),
            ConvGNAct(out_ch, out_ch, k=3, p=1),
        )

    def forward(self, x, stem_skip):
        x = self.ref1(self.up1(x))
        x = self.ref2(self.up2(x))
        if x.shape[2:] != stem_skip.shape[2:]:
            print("please make sure to match the usample and downsample pitch_size")
            x = F.interpolate(x, size=stem_skip.shape[2:], mode="trilinear", align_corners=False)
        return self.fuse(torch.cat([x, stem_skip], dim=1))

class StrongSpatialMixer3D(nn.Module):
    def __init__(self, Cs, use_aniso=True):
        super().__init__()

        self.dw3 = nn.Conv3d(Cs, Cs, 3, padding=1, groups=Cs, bias=False)
        self.n3 = nn.InstanceNorm3d(Cs)

        self.dw5 = nn.Conv3d(Cs, Cs, 5, padding=2, groups=Cs, bias=False)
        self.n5 = nn.InstanceNorm3d(Cs)

        self.dwd = nn.Conv3d(Cs, Cs, 3, padding=2, dilation=2, groups=Cs, bias=False)
        self.nd = nn.InstanceNorm3d(Cs)

        self.use_aniso = use_aniso
        if use_aniso:
            self.dwa = nn.Conv3d(Cs, Cs, (1,5,5), padding=(0,2,2), groups=Cs, bias=False)
            self.na = nn.InstanceNorm3d(Cs)

        self.refine = nn.Conv3d(Cs, Cs, 3, padding=1, groups=Cs, bias=False)
        self.nr = nn.InstanceNorm3d(Cs)

        self.act = nn.GELU()
        self.gamma = nn.Parameter(torch.ones((Cs,1,1,1)) * 1e-6)
        self.se = SEBlock(Cs)

    def forward(self, x):
        y = self.act(self.n3(self.dw3(x)))
        y = y + self.act(self.n5(self.dw5(x)))
        y = y + self.act(self.nd(self.dwd(x)))
        if self.use_aniso:
            y = y + self.act(self.na(self.dwa(x)))

        y = self.act(self.nr(self.refine(y)))
        y = self.se(y)
        return x + self.gamma * y
        

class PatchMerge3D(nn.Module):
    def __init__(self, in_ch, out_ch, merge_size=(2, 2, 2), norm="in"):
        super().__init__()
        self.merge_size = merge_size
        merge_factor = merge_size[0] * merge_size[1] * merge_size[2]

        self.reduction = nn.Conv3d(in_ch * merge_factor, out_ch, kernel_size=1, bias=False)

        if norm == "in":
            self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        else:
            raise ValueError("for now use norm='in'")

        self.act = nn.GELU()

    def forward(self, x):
        B, C, D, H, W = x.shape
        md, mh, mw = self.merge_size

        pad_d = D % md
        pad_h = H % mh
        pad_w = W % mw
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
            B, C, D, H, W = x.shape

        x = rearrange(
            x,
            "b c (d pd) (h ph) (w pw) -> b (c pd ph pw) d h w",
            pd=md, ph=mh, pw=mw
        )

        x = self.act(self.norm(self.reduction(x)))
        return x

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        # stem = nn.Sequential(
        #       nn.Conv3d(dims[0], dims[0], kernel_size=3, stride=2, padding=1),
        #       )
        self.downsample_layers.append(nn.Identity())

        self.downsample_layers.append(
            PatchMerge3D(dims[0], dims[1], merge_size=(2, 2, 2), norm="in")
        )
        self.downsample_layers.append(
            PatchMerge3D(dims[1], dims[2], merge_size=(2, 2, 2), norm="in")
        )
        self.downsample_layers.append(
            PatchMerge3D(dims[2], dims[3], merge_size=(2, 2, 2), norm="in")
        )

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        cur = 0
        for i in range(4):
            if i == 0:
                gsc = nn.Sequential(
                    *[StrongSpatialMixer3D(dims[i], use_aniso=True) for j in range(depths[i])]
                )
                stage = None

            elif i == 1:
                gsc = nn.Sequential(
                    *[StrongSpatialMixer3D(dims[i], use_aniso=False) for j in range(depths[i])]
                )
                stage = None

            # elif i == 2:
            #     # gsc = nn.Sequential(
            #     #     *[LargeKernelConv(dims[i]) for j in range(depths[i])]
            #     # )

            #     gsc = nn.Identity()
            #     stage = nn.Sequential(
            #         *[BiMambaLayer(dim=dims[i]) for j in range(depths[i])]
            #     )
            else :
                gsc = nn.Identity()
                # gsc = nn.Sequential(
                #     *[LargeKernelConv(dims[i]) for j in range(depths[i])]
                # )
                stage = nn.Sequential(
                    *[BiMambaLayer(dim=dims[i]) for j in range(depths[i])]
                )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)

            if self.stages[i] is not None:
                x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, up_kernel=2, use_gate=True):
        super().__init__()

        if isinstance(up_kernel, int):
            up_kernel = (up_kernel, up_kernel, up_kernel)

        self.up = nn.Sequential(
            nn.ConvTranspose3d(
                in_ch,
                out_ch,
                kernel_size=up_kernel,
                stride=up_kernel,
                bias=False,
            ),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

        self.use_gate = use_gate

        if use_gate:
            inter_ch = max(out_ch // 2, 16)

            self.gate_g = nn.Sequential(
                nn.Conv3d(out_ch, inter_ch, kernel_size=1, bias=False),
                nn.InstanceNorm3d(inter_ch, affine=True),
            )

            self.gate_x = nn.Sequential(
                nn.Conv3d(skip_ch, inter_ch, kernel_size=1, bias=False),
                nn.InstanceNorm3d(inter_ch, affine=True),
            )

            self.psi = nn.Sequential(
                nn.Conv3d(inter_ch, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

            self.relu = nn.ReLU(inplace=True)

            # start as normal skip connection
            self.gate_gamma = nn.Parameter(torch.zeros(1))

        self.fuse = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

        self.refine = LightRefine(out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            print("please make sure to match the usample and downsample pitch_size, or add a proper align_corners argument if using interpolate")
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        if self.use_gate:
            g = self.gate_g(x)
            s = self.gate_x(skip)

            att = self.psi(self.relu(g + s))

            # residual gate: safer for small organs
            skip = skip * (1.0 + self.gate_gamma * att)

        y = self.fuse(torch.cat([x, skip], dim=1))
        y = self.refine(y)

        return y

class SegMamba(SegmentationNetwork):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
        do_ds: bool = False
    ) -> None:
        super().__init__()

        # added for compatibility with UNETR++ trainer, not used in this model
        self.conv_op = nn.Conv3d
        self.num_classes = out_chans
        self.do_ds = do_ds  # set True later if you add deep supervision
        
        # model actual parameters
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans, 
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                              )
        stem_ch = max(self.feat_size[0] // 2, 48)
        self.stem_patch = ProgressiveStemPatch3D(
            in_ch=self.in_chans,
            stem_ch=stem_ch,
            out_ch=self.feat_size[0]
        )
        
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # self.decoder5 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.hidden_size,
        #     out_channels=self.feat_size[3],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.decoder4 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[3],
        #     out_channels=self.feat_size[2],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.decoder3 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[2],
        #     out_channels=self.feat_size[1],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )

        self.decoder5 = UpBlock3D(
            in_ch=self.hidden_size,
            out_ch=self.feat_size[3],
            skip_ch=self.feat_size[3],
            up_kernel=2,
            use_gate=True,
        )

        self.decoder4 = UpBlock3D(
            in_ch=self.feat_size[3],
            out_ch=self.feat_size[2],
            skip_ch=self.feat_size[2],
            up_kernel=2,
            use_gate=True,
        )

        self.decoder3 = UpBlock3D(
            in_ch=self.feat_size[2],
            out_ch=self.feat_size[1],
            skip_ch=self.feat_size[1],
            up_kernel=2,
            use_gate=True,
        )
        self.final_up = FinalUpAnisoV2(
            in_ch=self.feat_size[1],   # 96, because dec1 comes from decoder3
            out_ch=self.feat_size[0],  # 48, because final head expects 48
            stem_ch=stem_ch            # 48 in your case
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0], out_channels=self.out_chans)
        self.ds_out1 = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],   # dec1
            out_channels=self.out_chans
        )

        self.ds_out2 = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],   # dec2
            out_channels=self.out_chans
        )

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        enc1, x0 = self.stem_patch(x_in)
        outs = self.vit(x0)
        
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.final_up(dec1, enc1)
        out_main = self.out(out)

        if self.do_ds:
            out_ds1 = self.ds_out1(dec1)
            out_ds2 = self.ds_out2(dec2)
            return (out_main, out_ds1, out_ds2)
                
        return out_main
    