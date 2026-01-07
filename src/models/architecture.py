import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=None):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv1d(cin, cout, k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm1d(cout)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, use_dropout: bool = False, p: float = 0.1):
        super().__init__()
        p_pad = k // 2
        self.conv = ConvBNReLU(cin, cout, k=k, s=1, p=p_pad)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p) if use_dropout else nn.Identity()
        nn.init.kaiming_normal_(self.conv.conv.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        skip = x
        x = self.pool(x)
        return x, skip
    
class DecoderBlock(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = ConvBNReLU(cin, cout, k)

    def forward(self, x, skip):
        x = self.up(x)
        # pad/crop to match skip length
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, (0, diff)) if diff > 0 else x[..., :skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    
# ---------- CNN-SWT ----------    
class CNNSWT(nn.Module):
    def __init__(self, channels: int, k: int = 8, dilations=(1, 2), use_bn: bool = False):
        """
        channels: feature channels (depthwise)
        k: kernel size low-pass h
        dilations: SWT levels (1 ~ level-1, 2 ~ zero-insert level-2)
        """
        super().__init__()
        assert k >= 2, "kernel size k >= 2"
        self.channels = channels
        self.k = k
        self.dilations = tuple(dilations)

        # 1 learnable parameter per channel (low-pass)
        # Init He/Kaiming for conv1d + ReLU depthwise
        h = torch.empty(channels, 1, k)
        nn.init.kaiming_normal_(h, nonlinearity="relu")
        self.h = nn.Parameter(h)

        # Alternating signs: [1, -1, 1, -1, ...] (buffer, not param)
        alt = torch.ones(k)
        alt[1::2] = -1.0
        self.register_buffer("alt_sign", alt)

        # Pointwise 1x1 to fuse branches -> channels
        in_ch = channels * 2 * len(self.dilations)  # (approx + detail) per level
        self.proj = nn.Conv1d(in_ch, channels, kernel_size=1, bias=True)
        self.act  = nn.ReLU(inplace=True)
        self.bn   = nn.BatchNorm1d(channels) if use_bn else nn.Identity()
    
    def _make_g_from_h(self) -> torch.Tensor:
        """
        Generate high-pass kernel g = flip(h) * [1,-1,1,-1,...] on kernel dim.
        weight-tying: g is not learned, derived from h.
        """
        # self.h: (C,1,K)
        g = torch.flip(self.h, dims=[-1])  # (C,1,K)
        g = g * self.alt_sign.view(1, 1, -1)  # alternating signs
        return g

    @staticmethod
    def _same_padding(L: int, k: int, d: int) -> int:
        # padding to keep length (stride=1)
        return d * (k - 1) // 2

    def forward(self, x):   # x: (B,C,L)
        B, C, L = x.shape
        assert C == self.channels, f"Expected {self.channels} channels, got {C}"

        h = self.h
        g = self._make_g_from_h()

        feats = []
        for d in self.dilations:
            pad = self._same_padding(L, self.k, d)
            # depthwise conv: groups = C, weight shape (C,1,K)
            approx = F.conv1d(x, h, bias=None, stride=1, padding=pad, dilation=d, groups=C)
            detail = F.conv1d(x, g, bias=None, stride=1, padding=pad, dilation=d, groups=C)
            feats.extend([approx, detail])

        z = torch.cat(feats, dim=1)   # (B, C*2*levels, L)
        z = self.proj(z)              # (B, C, L)
        z = self.bn(z)
        z = self.act(z)
        return z

# ---------- Transformer ----------
class Time2Vec(nn.Module):
    def __init__(self, k: int = 3):  # 3 sine + 1 linear
        super().__init__()
        self.k = k
        self.omega_lin = nn.Parameter(torch.randn(1))
        self.phi_lin   = nn.Parameter(torch.randn(1))
        self.omega = nn.Parameter(torch.randn(k))
        self.phi   = nn.Parameter(torch.randn(k))
    def forward(self, L, device):
        i = torch.arange(L, device=device, dtype=torch.float32)  # [0..L-1]
        linear = i * self.omega_lin + self.phi_lin               # (L,)
        sines  = torch.sin(i.view(-1,1) * self.omega + self.phi) # (L,k)
        return torch.cat([linear.view(-1,1), sines], dim=1)      # (L, k+1)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.t2v = Time2Vec(k=3) 
        self.in_proj = nn.Linear(d_model + 4, d_model)  # add 4 T2V channels
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x):  # x: (B, C, L)
        B, C, L = x.shape
        z = x.permute(0, 2, 1)                     # (B, L, C)
        t2v = self.t2v(L, x.device).unsqueeze(0).expand(B, -1, -1)  # (B, L, 4)
        z = torch.cat([z, t2v], dim=-1)            # (B, L, C+4)
        z = self.in_proj(z)                        # (B, L, C)
        z2, _ = self.attn(self.ln1(z), self.ln1(z), self.ln1(z), need_weights=False)
        z = z + z2
        z2 = self.ffn(self.ln2(z))
        z = z + z2
        return z.permute(0, 2, 1)

# ---------- Full Model ----------
class DenoiseCNN_SWT_Transformer(nn.Module):
    def __init__(self, in_ch=1, base_ch=16, nhead=4, ffn_mult=2, num_transformer_layers=2):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch*2, base_ch*4

        # Encoder k=3
        self.enc1 = EncoderBlock(in_ch, c1, k=3)
        self.enc2 = EncoderBlock(c1,  c2, k=3)
        self.enc3 = EncoderBlock(c2,  c3, k=3, use_dropout=True, p=0.5)
        self.pre_bottleneck = nn.Sequential(
            ConvBNReLU(c3, c3, k=3, s=1, p=1),
            nn.Dropout(0.5)
        )
        # Bottleneck
        bott_ch = c3
        self.swt = CNNSWT(bott_ch, k=7, dilations=(1,2), use_bn=False)

        self.transformer = nn.Sequential(*[
            TransformerBlock(d_model=bott_ch, nhead=nhead, dim_feedforward=bott_ch*ffn_mult)
            for _ in range(num_transformer_layers)
        ])
        # Decoder
        self.dec4 = DecoderBlock(bott_ch + c3, c3, k=3)
        self.dec3 = DecoderBlock(c3 + c2, c2, k=3)   # skip enc2
        self.dec2 = DecoderBlock(c2 + c1, c1, k=3)   # skip enc1
        self.dec1 = ConvBNReLU(c1, c1, k=3)
        self.out  = nn.Conv1d(c1, 1, kernel_size=1)

    def forward(self, x):
        # -------- Encoder --------
        x1, skip1 = self.enc1(x)    # (B,c1,L/2),  skip1: (B,c1,L)
        x2, skip2 = self.enc2(x1)   # (B,c2,L/4),  skip2: (B,c2,L/2)
        x3, skip3 = self.enc3(x2)   # (B,c3,L/8),  skip3: (B,c3,L/4)

        # -------- Pre-bottleneck + Bottleneck --------
        b = self.pre_bottleneck(x3)           # ConvReLU+BN+Dropout(0.5), (B,c3,L/8)
        b = self.swt(b)             # (B,c3,L/8)
        b = self.transformer(b)     # (B,c3,L/8)

        # -------- Decoder (3 times upsample symmetric) --------
        y = self.dec4(b,   skip3)   # (B,c3,L/4)
        y = self.dec3(y,   skip2)   # (B,c2,L/2)
        y = self.dec2(y,   skip1)   # (B,c1,L)
        y = self.dec1(y)            # (B,c1,L)

        noise_hat = self.out(y)     # (B,1,L)
        return x - noise_hat

class DenoiseCNN_NoSWT(nn.Module):
    """Variant WITHOUT SWT - only uses Transformer in bottleneck"""
    def __init__(self, in_ch=1, base_ch=16, nhead=4, ffn_mult=2, num_transformer_layers=2):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch*2, base_ch*4

        self.enc1 = EncoderBlock(in_ch, c1, k=3)
        self.enc2 = EncoderBlock(c1, c2, k=3)
        self.enc3 = EncoderBlock(c2, c3, k=3, use_dropout=True, p=0.5)
        self.pre_bottleneck = nn.Sequential(
            ConvBNReLU(c3, c3, k=3, s=1, p=1),
            nn.Dropout(0.5)
        )

        # Only Transformer (no SWT)
        self.transformer = nn.Sequential(*[
            TransformerBlock(d_model=c3, nhead=nhead, dim_feedforward=c3*ffn_mult)
            for _ in range(num_transformer_layers)
        ])

        self.dec4 = DecoderBlock(c3 + c3, c3, k=3)
        self.dec3 = DecoderBlock(c3 + c2, c2, k=3)
        self.dec2 = DecoderBlock(c2 + c1, c1, k=3)
        self.dec1 = ConvBNReLU(c1, c1, k=3)
        self.out  = nn.Conv1d(c1, 1, kernel_size=1)

    def forward(self, x):
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)

        b = self.pre_bottleneck(x3)
        b = self.transformer(b)  # Only Transformer

        y = self.dec4(b, skip3)
        y = self.dec3(y, skip2)
        y = self.dec2(y, skip1)
        y = self.dec1(y)
        noise_hat = self.out(y)
        return x - noise_hat

class DenoiseCNN_NoTransformer(nn.Module):
    """Variant WITHOUT Transformer - only uses SWT in bottleneck"""
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch*2, base_ch*4

        self.enc1 = EncoderBlock(in_ch, c1, k=3)
        self.enc2 = EncoderBlock(c1, c2, k=3)
        self.enc3 = EncoderBlock(c2, c3, k=3, use_dropout=True, p=0.5)
        self.pre_bottleneck = nn.Sequential(
            ConvBNReLU(c3, c3, k=3, s=1, p=1),
            nn.Dropout(0.5)
        )

        # Only SWT (no Transformer)
        self.swt = CNNSWT(c3, k=7, dilations=(1, 2), use_bn=False)

        self.dec4 = DecoderBlock(c3 + c3, c3, k=3)
        self.dec3 = DecoderBlock(c3 + c2, c2, k=3)
        self.dec2 = DecoderBlock(c2 + c1, c1, k=3)
        self.dec1 = ConvBNReLU(c1, c1, k=3)
        self.out  = nn.Conv1d(c1, 1, kernel_size=1)

    def forward(self, x):
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)

        b = self.pre_bottleneck(x3)
        b = self.swt(b)  # Only SWT

        y = self.dec4(b, skip3)
        y = self.dec3(y, skip2)
        y = self.dec2(y, skip1)
        y = self.dec1(y)
        noise_hat = self.out(y)
        return x - noise_hat

class DenoiseCNN(nn.Module):
    """Simple Autoencoder Baseline"""
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch*2, base_ch*4

        # Encoder k=3
        self.enc1 = EncoderBlock(in_ch, c1, k=3)
        self.enc2 = EncoderBlock(c1,  c2, k=3)
        self.enc3 = EncoderBlock(c2,  c3, k=3, use_dropout=True, p=0.5)
        self.pre_bottleneck = nn.Sequential(
            ConvBNReLU(c3, c3, k=3, s=1, p=1),
            nn.Dropout(0.5)
        )
        # Bottleneck (Empty / Identity just pass through)
        # Decoder
        self.dec4 = DecoderBlock(c3 + c3, c3, k=3) # Corrected concat dim in decoder
        self.dec3 = DecoderBlock(c3 + c2, c2, k=3)
        self.dec2 = DecoderBlock(c2 + c1, c1, k=3)
        self.dec1 = ConvBNReLU(c1, c1, k=3)
        self.out  = nn.Conv1d(c1, 1, kernel_size=1)

    def forward(self, x):
        # -------- Encoder --------
        x1, skip1 = self.enc1(x)    # (B,c1,L/2)
        x2, skip2 = self.enc2(x1)   # (B,c2,L/4)
        x3, skip3 = self.enc3(x2)   # (B,c3,L/8)

        # -------- Pre-bottleneck --------
        b = self.pre_bottleneck(x3)

        # -------- Decoder --------
        # Note: In the original baseline_A code, dec4 took (bott_ch + c3) = 2*c3
        # Here b is c3, skip3 is c3. So cat(b, skip3) is 2*c3.
        y = self.dec4(b,   skip3)
        y = self.dec3(y,   skip2)
        y = self.dec2(y,   skip1)
        y = self.dec1(y)

        noise_hat = self.out(y)
        return x - noise_hat
