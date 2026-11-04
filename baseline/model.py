"""Visuomotor student policy (arXiv 2603.11980, Fig. 2): observation sequence → velocity [v, ω].

    Each frame: (C, H, W) = target heatmaps (enemy, ally) + depth  [+2 coord channels with --coordconv]
      └─ encoder ──► feat (feat_dim,)   resnet18 (paper, 1024-d) | cnn | flat | spatial (soft-argmax)
    (T, feat_dim) [+ prev action (2,)] [+ VLA plan (4,)]
      └─ LSTM (paper: 1 layer, 128) ──► h_t for every step
      └─ head ──► per-step action  linear (paper: 2-layer MLP, L2 regression) | tanh | cls

The network works in a *normalized* action space: a_norm = a / act_scale, with a per-dimension
act_scale (robot units, robots.py: max |demonstrated command| per dim). `act()` returns robot
units ([v, ω]); `robots.Robot.to_command()` turns that into the robot API call.
For the cls head the bins are the robot's demonstrated levels (e.g. ω ∈ {-25, 0, +30} %), so the
predicted command is always one the operator actually used.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


V_MAX = 100.0   # fallback percent scale of the labels


# ── encoders ───────────────────────────────────────────────────────────────────

def add_coords(x):
    """Append normalized (x, y) coordinate channels in [-1, 1]. (B, C, H, W) → (B, C+2, H, W)"""
    B, _, H, W = x.shape
    ys = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    xs = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    return torch.cat([x, xs, ys], dim=1)


class SmallCNN(nn.Module):
    """Original baseline encoder (stride-2 ×3 → avg-pool 4×4)."""

    def __init__(self, in_ch=2, feat_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 16, feat_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class FlatCNN(nn.Module):
    """Same conv stack without pooling: keeps the full 8×8 spatial grid."""

    def __init__(self, in_ch=2, feat_dim=64, hw=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        s = hw // 8
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * s * s, feat_dim), nn.ReLU())

    def forward(self, x):
        return self.fc(self.conv(x))


class SpatialSoftmaxEncoder(nn.Module):
    """Conv → K keypoint maps → spatial soft-argmax (Levine et al. 2016).

    Features per frame:
        expected (x, y) of every keypoint map            2K
        max activation of every keypoint map (presence)   K
        max of every raw input heatmap channel            2   (is enemy / ally visible at all?)
    The presence terms disambiguate "no detection" (uniform softmax → coords ≈ 0)
    from "target exactly in the image centre".
    """

    def __init__(self, in_ch=2, feat_dim=64, n_kp=32, n_hm=2):
        super().__init__()
        self.n_hm = n_hm
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=1, padding=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),    # 32×32
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, n_kp, 1),
        )
        self.log_temp = nn.Parameter(torch.zeros(1))
        self.fc = nn.Sequential(nn.Linear(3 * n_kp + n_hm, feat_dim), nn.ReLU())

    def forward(self, x):
        presence_in = x[:, :self.n_hm].amax(dim=(2, 3))              # (B, 2)
        f = self.conv(x)                                              # (B, K, h, w)
        B, K, h, w = f.shape
        flat = f.view(B, K, h * w)
        attn = F.softmax(flat / self.log_temp.exp(), dim=-1)
        ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        ex = (attn * gx.reshape(1, 1, -1)).sum(-1)
        ey = (attn * gy.reshape(1, 1, -1)).sum(-1)
        presence_kp = flat.amax(dim=-1)
        return self.fc(torch.cat([ex, ey, presence_kp, presence_in], dim=1))


class ResNet18Encoder(nn.Module):
    """Paper's visual encoder: ResNet18 (ImageNet init, trained end-to-end) → feat_dim embedding.
    The first conv takes `in_ch` channels; its RGB filters are averaged and replicated."""

    def __init__(self, in_ch=3, feat_dim=1024, pretrained=True):
        super().__init__()
        import torchvision
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=w)
        old = net.conv1
        net.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            net.conv1.weight.copy_(old.weight.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * 3 / in_ch)
        net.fc = nn.Identity()
        self.net = net
        self.fc = nn.Sequential(nn.Linear(512, feat_dim), nn.ReLU())

    def forward(self, x):
        return self.fc(self.net(x))


ENCODERS = ('resnet18', 'cnn', 'flat', 'spatial')
HEADS = ('tanh', 'linear', 'cls')


# ── policy ─────────────────────────────────────────────────────────────────────

class VisuomotorPolicy(nn.Module):
    def __init__(self, seq_len=8, feat_dim=64, lstm_hidden=128, lstm_layers=2,
                 encoder='cnn', coordconv=False, head='tanh', n_bins=21,
                 prev_action=False, prev_action_dropout=0.5, hm_size=224, act_scale=V_MAX,
                 channels=('enemy', 'ally', 'depth'), vla_cond=False, vla_dropout=0.2,
                 head_hidden=64, levels=None, pretrained=True):
        super().__init__()
        s_ = torch.as_tensor(act_scale, dtype=torch.float32).flatten()
        self.register_buffer('act_scale', s_.expand(2).clone() if s_.numel() == 1 else s_)
        self.channels = tuple(channels)
        n_hm = sum(c != 'depth' for c in self.channels)
        hm_channels = len(self.channels)
        assert encoder in ENCODERS and head in HEADS
        self.coordconv, self.head_type, self.n_bins = coordconv, head, n_bins
        self.prev_action, self.prev_action_dropout = prev_action, prev_action_dropout
        in_ch = hm_channels + (2 if coordconv else 0)

        if encoder == 'resnet18':
            self.encoder = ResNet18Encoder(in_ch, feat_dim, pretrained)
        elif encoder == 'cnn':
            self.encoder = SmallCNN(in_ch, feat_dim)
        elif encoder == 'flat':
            self.encoder = FlatCNN(in_ch, feat_dim, hm_size)
        else:
            self.encoder = SpatialSoftmaxEncoder(in_ch, feat_dim, n_hm=n_hm)

        self.vla_cond, self.vla_dropout = vla_cond, vla_dropout
        lstm_in = feat_dim + (2 if prev_action else 0) + (4 if vla_cond else 0)
        self.lstm = nn.LSTM(lstm_in, lstm_hidden, num_layers=lstm_layers, batch_first=True,
                            dropout=0.1 if lstm_layers > 1 else 0.0)
        out_dim = 2 * n_bins if head == 'cls' else 2
        layers = [nn.Linear(lstm_hidden, head_hidden), nn.ReLU(), nn.Linear(head_hidden, out_dim)]
        if head == 'tanh':
            layers.append(nn.Tanh())
        self.head = nn.Sequential(*layers)
        # per-dim bin centres in normalized units; robot levels (e.g. ω {-25,0,30}/30) when given
        if levels is not None and all(len(lv) == n_bins for lv in levels):
            bc = torch.tensor(levels, dtype=torch.float32) / self.act_scale[:, None]
        else:
            bc = torch.linspace(-1, 1, n_bins).expand(2, n_bins).clone()
        self.register_buffer('bin_centers', bc)          # (2, n_bins)

    def forward(self, seq, prev_act=None, cond=None):
        """seq: (B, T, C, H, W);  prev_act: (B, T, 2) normalized previous actions;
        cond: (B, T, 4) VLA plan conditioning [plan_v, plan_w, age, has_plan] (see dataset.py).

        Returns raw head output for every step:
            tanh/linear → (B, T, 2)        cls → (B, T, 2, n_bins) logits
        """
        B, T, C, H, W = seq.shape
        return self.temporal(self.encode(seq.reshape(B * T, C, H, W)).view(B, T, -1), prev_act, cond)

    def encode(self, x):
        """per-frame encoder: (N, C, H, W) → (N, feat) — frames are independent, so deploy caches it."""
        if self.coordconv:
            x = add_coords(x)
        return self.encoder(x)

    def temporal(self, feats, prev_act=None, cond=None):
        """(B, T, feat) per-frame features → raw head output for every step."""
        B, T = feats.shape[:2]
        if self.prev_action:
            if prev_act is None:
                prev_act = feats.new_zeros(B, T, 2)
            if self.training and self.prev_action_dropout > 0:
                keep = (torch.rand(B, T, 1, device=feats.device) > self.prev_action_dropout)
                prev_act = prev_act * keep
            feats = torch.cat([feats, prev_act], dim=-1)
        if self.vla_cond:
            if cond is None:
                cond = feats.new_zeros(B, T, 4)   # no VLA available → has_plan = 0
            if self.training and self.vla_dropout > 0:
                # drop the VLA for the whole sequence: policy must also work without it
                keep = (torch.rand(B, 1, 1, device=feats.device) > self.vla_dropout)
                cond = cond * keep
            feats = torch.cat([feats, cond], dim=-1)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(feats)
        y = self.head(out)
        if self.head_type == 'cls':
            y = y.view(B, T, 2, self.n_bins)
        return y

    def predict(self, y, mode='argmax'):
        """Raw head output → normalized action in [-1, 1], shape (..., 2).
        cls head: 'argmax' (most likely command — what the robot executes) or 'mean' (expectation)."""
        if self.head_type == 'cls':
            if mode == 'argmax':
                idx = y.argmax(-1)                                       # (..., 2)
                return torch.stack([self.bin_centers[d][idx[..., d]] for d in range(2)], -1)
            return (F.softmax(y, dim=-1) * self.bin_centers).sum(-1)
        if self.head_type == 'tanh':
            return y
        return y          # paper: unbounded L2 regression (clipped to robot limits at deployment)

    def to_bins(self, a):
        """Normalized action (..., 2) → index of the nearest bin centre per dim (..., 2)."""
        return (a[..., None] - self.bin_centers).abs().argmin(-1)

    @torch.no_grad()
    def act(self, heatmap_seq_np, prev_actions_pct=None, vla_cond_np=None):
        """Inference helper.
        heatmap_seq_np  : (T, C, H, W) float32
        prev_actions_pct: (T, 2) previous actions in robot units (only with prev_action=True)
        vla_cond_np     : (T, 4) [plan_v, plan_w, age, has_plan] (only with vla_cond=True)
        Returns: (2,) float32 [v, ω] in robot units for the last frame (→ Robot.to_command).
        """
        self.eval()
        dev = self.bin_centers.device
        t = torch.from_numpy(heatmap_seq_np).unsqueeze(0).to(dev)
        p = None
        if prev_actions_pct is not None:
            p = torch.as_tensor(prev_actions_pct, dtype=torch.float32, device=dev).unsqueeze(0) / self.act_scale
        c = None
        if vla_cond_np is not None:
            c = torch.as_tensor(vla_cond_np, dtype=torch.float32, device=dev).unsqueeze(0)
        a = self.predict(self.forward(t, p, c))[0, -1]
        return (a * self.act_scale).cpu().numpy()

    @torch.no_grad()
    def act_feats(self, feats, prev_actions_pct=None, vla_cond_np=None):
        """Same as act() but from cached per-frame features (T, feat) torch tensor — O(1) encoder work per step."""
        self.eval()
        dev = self.bin_centers.device
        p = None
        if prev_actions_pct is not None:
            p = torch.as_tensor(prev_actions_pct, dtype=torch.float32, device=dev).unsqueeze(0) / self.act_scale
        c = None
        if vla_cond_np is not None:
            c = torch.as_tensor(vla_cond_np, dtype=torch.float32, device=dev).unsqueeze(0)
        a = self.predict(self.temporal(feats.unsqueeze(0), p, c))[0, -1]
        return (a * self.act_scale).cpu().numpy()


def _channels(c):
    return tuple(x.strip() for x in c.split(',')) if isinstance(c, str) else tuple(c)


def build_model(cfg):
    """Build from a dict/namespace of hyper-parameters (missing keys → baseline defaults)."""
    g = (lambda k, d: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d: getattr(cfg, k, d))
    return VisuomotorPolicy(
        seq_len=g('seq_len', 8), feat_dim=g('feat_dim', 64), lstm_hidden=g('lstm_hidden', 128),
        lstm_layers=g('lstm_layers', 2), encoder=g('encoder', 'cnn'), coordconv=g('coordconv', False),
        head=g('head', 'tanh'), n_bins=g('n_bins', 21), prev_action=g('prev_action', False),
        prev_action_dropout=g('prev_action_dropout', 0.5),
        act_scale=g('act_scale', V_MAX) or V_MAX, channels=_channels(g('channels', 'enemy,ally,depth')),
        head_hidden=g('head_hidden', 64), levels=g('levels', None), pretrained=g('pretrained', True),
        vla_cond=bool(g('vla_tag', '')), vla_dropout=g('vla_dropout', 0.2),
    )
