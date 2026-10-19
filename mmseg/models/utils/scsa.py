# SCSA: Synergistic Channel-Spatial Attention
# Paper: https://arxiv.org/pdf/2407.05128
import torch
import torch.nn as nn
from einops import rearrange


class SCSA(nn.Module):
    """Synergistic Channel-Spatial Attention. Shape-preserving.

    Args:
        dim (int): Feature channels.
        head_num (int): Attention heads.
        window_size (int): Downsampling window for channel attention.
    """

    def __init__(
            self,
            dim: int,
            head_num: int = 4,
            window_size: int = 7,
            group_kernel_sizes=(3, 5, 7, 9),
            qkv_bias: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.window_size = window_size

        assert dim % 4 == 0
        self.group_chans = gc = dim // 4

        self.local_dwc = nn.Conv1d(gc, gc, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=gc)
        self.global_dwc_s = nn.Conv1d(gc, gc, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=gc)
        self.global_dwc_m = nn.Conv1d(gc, gc, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=gc)
        self.global_dwc_l = nn.Conv1d(gc, gc, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=gc)

        self.sa_gate = nn.Sigmoid() if gate_layer == 'sigmoid' else nn.Softmax(dim=2)
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(dim, dim, 1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(dim, dim, 1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(dim, dim, 1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Sigmoid() if gate_layer == 'sigmoid' else nn.Softmax(dim=1)

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        elif down_sample_mode == 'avg_pool':
            self.down_func = nn.AvgPool2d((window_size, window_size), stride=window_size)
        elif down_sample_mode == 'max_pool':
            self.down_func = nn.MaxPool2d((window_size, window_size), stride=window_size)
        elif down_sample_mode == 'recombination':
            self.down_func = self._space_to_chans
            self.conv_d = nn.Conv2d(dim * window_size ** 2, dim, 1, bias=False)

    def _space_to_chans(self, x):
        b, c, h, w = x.shape
        ws = self.window_size
        x = x.view(b, c, h // ws, ws, w // ws, ws)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(b, c * ws * ws, h // ws, w // ws)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h_, w_ = x.shape

        # spatial attention (multi-scale 1D conv along H and W)
        x_h = x.mean(dim=3)
        l_h, gs_h, gm_h, gl_h = torch.split(x_h, self.group_chans, dim=1)
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_h), self.global_dwc_s(gs_h),
            self.global_dwc_m(gm_h), self.global_dwc_l(gl_h)), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w = x.mean(dim=2)
        l_w, gs_w, gm_w, gl_w = torch.split(x_w, self.group_chans, dim=1)
        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_w), self.global_dwc_s(gs_w),
            self.global_dwc_m(gm_w), self.global_dwc_l(gl_w)), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        x = x * x_h_attn * x_w_attn

        # channel attention (self-attention with downsampling)
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_d, w_d = y.shape
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        q = rearrange(q, 'b (hn hd) h w -> b hn hd (h w)', hn=self.head_num, hd=self.head_dim)
        k = rearrange(k, 'b (hn hd) h w -> b hn hd (h w)', hn=self.head_num, hd=self.head_dim)
        v = rearrange(v, 'b (hn hd) h w -> b hn hd (h w)', hn=self.head_num, hd=self.head_dim)
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = attn @ v
        attn = rearrange(attn, 'b hn hd (h w) -> b (hn hd) h w', h=h_d, w=w_d)
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn * x
