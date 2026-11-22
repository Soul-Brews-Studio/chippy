"""EmbeddingGemma 2 vision encoder for the ANE, v2: the same math rewritten in Apple's ANE layout.

v1 (convert_gemma2_vision.py) traces the model's own layers: [B, P, 768] tensors, nn.Linear, one
attention over all 12 heads at once ([12, P, P] scores, 152 MB in fp16 at P = 2,520). v2 follows
Apple's "Deploying Transformers on the Apple Neural Engine" (ml-ane-transformers):
  - channels-first [B, C, 1, P] tensors from end to end; every linear is a 1 x 1 Conv2d
  - attention split per head: 12 score maps of [B, P_keys, 1, P_queries] (12.7 MB each), softmax over
    the channel axis, einsum instead of transposes of the big tensors
  - optional query chunks (--qchunks N): each head's queries in N slices, smaller score maps again
  - optional key chunks (--kchunks N): online softmax over N key slices with a running max, as in
    flash attention; exact, and each score map is N times smaller
  - RMSNorm over the channel axis, divided by the channel max first (fp16-safe, as v1 --fp16-safe;
    in fp32 it is exactly RMSNorm)
Measured 2026-10-08 on m5 (M5 Max), 2,520 patches, v1 and v2 timed alternately on the same 6 images
(`ab`); soft tokens vs fp32 over 12 images (`parity`):
  variant                      ops on ANE   ms/image   vs v1   worst token   median
  v1 (model's own layers)         2,529       341-344    1.00      0.929      0.99997
  v2 per head (the default)      13,721       245-247    1.40      0.944      0.99997
  v2 --qchunks 4                 16,985         247      1.38       -         0.99999 vs v1
  v2 --fused                      2,683         245      1.39      0.882      0.99997
  v2 --kchunks 4                 24,861         283      1.22      0.957      0.99997
Op count is not the cost (fused: 5x fewer ops, same time). the ane-oracle lab's vision_profile.py splits
the 245 ms: attention score maps ~164 ms (312 GFLOP at 1.9 TFLOPS), MLP ~52 ms (11.0 TFLOPS), the
rest ~29 ms: the 1 x 1 convs run near full speed, and the [P, P] score maps bound what is left.
Query or key chunks (online softmax) do not shrink that on the ANE.
Inputs: v1's side_inputs, transposed on the CPU: patches and pos [B, 768, 1, P], cos/sin [B, 64, 1, P],
valid [B, 1, 1, P], poolT [B, P, T]. Output: soft [B, 512, T]. v1's packages stay as they are; v2's are
named eg2-vision-v2-*.

    python convert_gemma2_vision_v2.py check   [--n 6] [--qchunks 1]      # fp32 v2 vs get_image_features
    python convert_gemma2_vision_v2.py convert --patches 2520 [--qchunks 1]
    python convert_gemma2_vision_v2.py parity  --patches 2520 [--qchunks 1] [--n 12]
    python convert_gemma2_vision_v2.py ab      --patches 2520 [--qchunks 1]  # v1 vs v2 on the ANE, back to back
    python convert_gemma2_vision_v2.py probe   [--n 12]                    # largest |x| at each RMSNorm site
    python convert_gemma2_vision_v2.py export  [--out DIR]                 # the app's vision assets (v1 + v2 packages)
    (convert / parity / ab also take --fused: LayerFused, the per-head norms and rotary done on all heads
     at once; and --kchunks N)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from convert_gemma2 import NEG, patch_coremltools_cast, placement
from convert_gemma2_vision import OUT, load_full, parts, pixels, sample_images, side_inputs

INPUTS = ('patches', 'pos', 'cos', 'sin', 'valid', 'poolT')
EPS = 1e-6
PEAKS: dict | None = None   # probe: largest |input| per RMSNorm site


def conv(linear):
    """nn.Linear (or Gemma4ClippableLinear without clipping) -> 1 x 1 Conv2d with the same weights."""
    lin = getattr(linear, 'linear', linear)
    c = torch.nn.Conv2d(lin.in_features, lin.out_features, 1, bias=lin.bias is not None)
    c.weight.data = lin.weight.detach().float().clone()[:, :, None, None]
    if lin.bias is not None:
        c.bias.data = lin.bias.detach().float().clone()
    return c


def rms(x, weight=None, site='residual'):
    """RMSNorm over channels (dim 1) without squaring large values: x / rms(x) == y / rms(y) for
    y = x / max|x|. `site` only labels the probe's record."""
    if PEAKS is not None:
        PEAKS[site] = max(PEAKS.get(site, 0.0), float(x.abs().max()))
    s = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-2)
    y = x / s
    y = y * torch.pow((y * y).mean(1, keepdim=True) + EPS / (s * s), -0.5)
    return y if weight is None else y * weight


def channel_weight(norm):
    return norm.weight.detach().float().clone()[None, :, None, None]


class Layer(torch.nn.Module):
    """One Gemma 4 vision encoder layer on [B, 768, 1, P]."""

    def __init__(self, layer, heads: int, head_dim: int, qchunks: int, kchunks: int = 1):
        super().__init__()
        a, m = layer.self_attn, layer.mlp
        self.q, self.k, self.v, self.o = conv(a.q_proj), conv(a.k_proj), conv(a.v_proj), conv(a.o_proj)
        self.gate, self.up, self.down = conv(m.gate_proj), conv(m.up_proj), conv(m.down_proj)
        self.register_buffer('qn', channel_weight(a.q_norm)); self.register_buffer('kn', channel_weight(a.k_norm))
        for name in ('input_layernorm', 'post_attention_layernorm', 'pre_feedforward_layernorm', 'post_feedforward_layernorm'):
            self.register_buffer(name, channel_weight(getattr(layer, name)))
        self.heads, self.d, self.qchunks, self.kchunks = heads, head_dim, qchunks, kchunks

    def attend(self, q, kt, v, kmask):
        """softmax(q k) v for one head, scaling 1.0. With kchunks > 1: over key slices with a running max
        (online softmax, as in flash attention). Each slice's score map is kchunks times smaller and the
        slices combine exactly; a slice that is all padding gets weight exp(-1e4) = 0."""
        if self.kchunks == 1:
            w = (torch.einsum('bchq,bkhc->bkhq', q, kt) + kmask).softmax(dim=1)
            return torch.einsum('bkhq,bchk->bchq', w, v)
        P = kt.shape[1]; step = -(-P // self.kchunks)
        parts = []
        for s in range(0, P, step):
            sc = torch.einsum('bchq,bkhc->bkhq', q, kt[:, s:s + step]) + kmask[:, s:s + step]
            m = sc.amax(dim=1, keepdim=True)                                   # [B, 1, 1, Q]
            e = torch.exp(sc - m)
            parts.append((m, e.sum(dim=1, keepdim=True), torch.einsum('bkhq,bchk->bchq', e, v[..., s:s + step])))
        top = parts[0][0]
        for m, _, _ in parts[1:]:
            top = torch.maximum(top, m)
        o = l = None
        for m, lj, oj in parts:
            a = torch.exp(m - top)
            o = a * oj if o is None else o + a * oj
            l = a * lj if l is None else l + a * lj
        return o / l

    def rope(self, x, cos, sin):
        """2-D rotary, as apply_multidimensional_rope: two halves of the head (x, then y), rotate_half in each."""
        h, q = self.d // 2, self.d // 4
        r = torch.cat([-x[:, q:h], x[:, :q], -x[:, h + q:], x[:, h:h + q]], dim=1)
        return x * cos + r * sin

    def forward(self, h, cos, sin, kmask):
        x = rms(h, self.input_layernorm)
        q, k, v = self.q(x), self.k(x), self.v(x)
        heads = []
        for i in range(self.heads):
            sl = slice(i * self.d, (i + 1) * self.d)
            qi = self.rope(rms(q[:, sl], self.qn, 'q'), cos, sin)             # [B, 64, 1, P]
            ki = self.rope(rms(k[:, sl], self.kn, 'k'), cos, sin).transpose(1, 3)  # [B, P, 1, 64]
            vi = rms(v[:, sl], site='v')                                       # v_norm has no scale
            P = qi.shape[3]
            step = -(-P // self.qchunks)                    # slices, not torch.chunk: coremltools 9.0 fails on chunk(1)
            outs = [self.attend(qi[..., s:s + step], ki, vi, kmask) for s in range(0, P, step)]
            heads.append(outs[0] if len(outs) == 1 else torch.cat(outs, dim=3))
        h = h + rms(self.o(torch.cat(heads, dim=1)), self.post_attention_layernorm, 'attn_out')
        x = rms(h, self.pre_feedforward_layernorm)
        return h + rms(self.down(F.gelu(self.gate(x), approximate='tanh') * self.up(x)), self.post_feedforward_layernorm, 'mlp_out')


class BlockNorm(torch.nn.Module):
    """RMSNorm over channel blocks of a [B, C, 1, P] tensor, every block at once, without reductions or
    reshapes: the mean of x^2 per block is a 1 x 1 conv (C -> blocks, weights 1/size), its inverse root
    goes back to C channels by broadcast (one block) or a second conv (C ones). Each x^2 term is scaled
    before it is summed, so only |x| > 256 can overflow fp16: use where the probe peaks stay below that."""

    def __init__(self, channels: int, size: int, weight=None):
        super().__init__()
        n = channels // size
        down = torch.zeros(n, channels, 1, 1); up = torch.zeros(channels, n, 1, 1)
        for i in range(n):
            down[i, i * size:(i + 1) * size] = 1.0 / size
            up[i * size:(i + 1) * size, i] = 1.0
        self.register_buffer('down', down); self.register_buffer('up', up)
        self.single = n == 1
        w = None if weight is None else weight.detach().float().clone().repeat(n)[None, :, None, None]
        self.register_buffer('w', w)

    def forward(self, x):
        inv = torch.pow(F.conv2d(x * x, self.down) + EPS, -0.5)
        y = x * (inv if self.single else F.conv2d(inv, self.up))
        return y if self.w is None else y * self.w


def rotation(heads: int, d: int):
    """rotate_half of both halves of every head as one signed permutation matrix [C, C, 1, 1]."""
    C, h, q = heads * d, d // 2, d // 4
    R = torch.zeros(C, C)
    for b in range(0, C, d):
        for part in (0, h):
            for j in range(q):
                R[b + part + j, b + part + q + j] = -1.0
                R[b + part + q + j, b + part + j] = 1.0
    return R[:, :, None, None]


class LayerFused(Layer):
    """Layer with the per-head work done on all heads at once: q/k/v norms as BlockNorm (64-channel
    blocks), the 2-D rotary as x * cos + conv(x, R) * sin with cos/sin tiled to 768 channels, the
    attention output norm as a one-block BlockNorm. Attention itself stays split per head."""

    def __init__(self, layer, heads: int, head_dim: int, qchunks: int):
        super().__init__(layer, heads, head_dim, qchunks)
        a, C = layer.self_attn, heads * head_dim
        self.qnorm, self.knorm, self.vnorm = BlockNorm(C, head_dim, a.q_norm.weight), BlockNorm(C, head_dim, a.k_norm.weight), BlockNorm(C, head_dim)
        self.onorm = BlockNorm(C, C, layer.post_attention_layernorm.weight)
        self.register_buffer('R', rotation(heads, head_dim))

    def forward(self, h, cos, sin, kmask):
        x = rms(h, self.input_layernorm)
        q = self.qnorm(self.q(x)); q = q * cos + F.conv2d(q, self.R) * sin
        k = self.knorm(self.k(x)); k = k * cos + F.conv2d(k, self.R) * sin
        v = self.vnorm(self.v(x))
        kt = k.transpose(1, 3)                                                  # [B, P, 1, 768], once
        heads = []
        for i in range(self.heads):
            sl = slice(i * self.d, (i + 1) * self.d)
            w = (torch.einsum('bchq,bkhc->bkhq', q[:, sl], kt[..., sl]) + kmask).softmax(dim=1)
            heads.append(torch.einsum('bkhq,bchk->bchq', w, v[:, sl]))
        h = h + self.onorm(self.o(torch.cat(heads, dim=1)))
        x = rms(h, self.pre_feedforward_layernorm)
        return h + rms(self.down(F.gelu(self.gate(x), approximate='tanh') * self.up(x)), self.post_feedforward_layernorm, 'mlp_out')


class VisionCoreV2(torch.nn.Module):
    def __init__(self, vm, ev, qchunks: int = 1, fused: bool = False, kchunks: int = 1):
        super().__init__()
        cfg = vm.config
        self.inp = conv(vm.patch_embedder.input_proj)
        if fused:
            self.layers = torch.nn.ModuleList(LayerFused(l, cfg.num_attention_heads, cfg.head_dim, qchunks) for l in vm.encoder.layers)
        else:
            self.layers = torch.nn.ModuleList(Layer(l, cfg.num_attention_heads, cfg.head_dim, qchunks, kchunks) for l in vm.encoder.layers)
        self.proj = conv(ev.embedding_projection)
        self.tile = cfg.num_attention_heads if fused else 1

    def forward(self, patches, pos, cos, sin, valid, poolT):
        h = self.inp(2 * (patches - 0.5)) + pos                                # [B, 768, 1, P]
        kmask = ((1.0 - valid) * NEG).transpose(1, 3)                          # [B, P, 1, 1]
        if self.tile > 1:                                                      # fused: cos/sin for all heads
            cos, sin = cos.repeat(1, self.tile, 1, 1), sin.repeat(1, self.tile, 1, 1)
        for layer in self.layers:
            h = layer(h, cos, sin, kmask)
        pooled = torch.matmul((h * valid).squeeze(2), poolT).unsqueeze(2)     # [B, 768, 1, T]
        return self.proj(rms(pooled, site='pooled')).squeeze(2)                # [B, 512, T]


def v2_inputs(pv, side):
    """v1's [N, P, C] inputs -> v2's channels-first ones (fp32 torch)."""
    cf = lambda t: t.transpose(1, 2).unsqueeze(2).contiguous()
    return {'patches': cf(pv), 'pos': cf(side['pos']), 'cos': cf(side['cos']), 'sin': cf(side['sin']),
            'valid': side['valid'][:, None, None, :].contiguous(), 'poolT': side['pool'].transpose(1, 2).contiguous()}


def package(patches, qchunks, fused=False, kchunks=1):
    tags = f'-q{qchunks}' if qchunks > 1 else ''
    tags += f'-k{kchunks}' if kchunks > 1 else ''
    tags += '-fused' if fused else ''
    return OUT / f'eg2-vision-v2-{patches}x1{tags}.mlpackage'


def cmd_probe(args):
    """Largest |input| at each RMSNorm site over sample images, fp32: which sites could square x in fp16
    directly (|x| < 256). Measured: residual 1,014, q 72, k 77, v 148, attention out 65, MLP out 243, pooled 930;
    --fused relies on this for its q/k/v and attention-output norms."""
    global PEAKS
    st = load_full(); hf, vm, ev = parts(st)
    core = VisionCoreV2(vm, ev).eval()
    PEAKS = {}
    for path in sample_images(args.n):
        pv, pid = pixels(st, path)
        with torch.no_grad():
            side, _ = side_inputs(vm, pid)
            core(**v2_inputs(pv, side))
    print('largest |x| into RMSNorm, by site:', {k: round(v, 1) for k, v in PEAKS.items()}, '(fp16 squares overflow past 256)')


def cmd_check(args):
    st = load_full(); hf, vm, ev = parts(st)
    core = VisionCoreV2(vm, ev, args.qchunks, args.fused, args.kchunks).eval()
    for path in sample_images(args.n):
        pv, pid = pixels(st, path)
        with torch.no_grad():
            ref = hf.get_image_features(pv, pid, return_dict=True).pooler_output[0]
            side, tmask = side_inputs(vm, pid)
            out = core(**v2_inputs(pv, side))[0].T[tmask[0]]
        cos = F.cosine_similarity(out, ref, dim=-1)
        print(f'{path.name[:34]:34} {tuple(ref.shape)}: cosine min {cos.min():.6f}, max |diff| {(out - ref).abs().max():.2e}', flush=True)


def cmd_convert(args):
    import coremltools as ct
    patch_coremltools_cast()
    st = load_full(); _, vm, ev = parts(st)
    P = args.patches; T = P // vm.config.pooling_kernel_size ** 2
    shapes = {'patches': (1, 768, 1, P), 'pos': (1, 768, 1, P), 'cos': (1, 64, 1, P), 'sin': (1, 64, 1, P),
              'valid': (1, 1, 1, P), 'poolT': (1, P, T)}
    path = package(P, args.qchunks, args.fused, args.kchunks)
    t0 = time.time()
    core = VisionCoreV2(vm, ev, args.qchunks, args.fused, args.kchunks).eval()
    ex = tuple(torch.rand(shapes[n]) if n == 'patches' else torch.ones(shapes[n]) * (n == 'valid') for n in INPUTS)
    with torch.no_grad():
        traced = torch.jit.trace(core, ex, check_trace=False)
    ml = ct.convert(traced, convert_to='mlprogram', minimum_deployment_target=ct.target.macOS15,
                    inputs=[ct.TensorType(name=n, shape=shapes[n], dtype=np.float16) for n in INPUTS],
                    outputs=[ct.TensorType(name='soft', dtype=np.float16)],
                    compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE)
    ml.short_description = (f'EmbeddingGemma 2 vision encoder + embed_vision, ANE layout (v2, {args.qchunks} query chunks, '
                            f'fused: {args.fused}, key chunks: {args.kchunks}), '
                            f'{P} patches -> soft [1, 512, {T}]')
    ml.save(str(path))
    print(f'{path.name}: converted in {time.time() - t0:.0f} s, {placement(path)}', flush=True)


def feeds(st, vm, paths, P):
    """Real images -> (v1 feed, v2 feed, soft-token mask, fp32 reference soft tokens)."""
    from convert_gemma2_vision import INPUTS as V1
    hf = next(m for m in st.modules() if type(m).__name__ == 'EmbeddingGemma2Model')
    for path in paths:
        pv, pid = pixels(st, path)
        if pv.shape[1] != P:
            continue
        with torch.no_grad():
            ref = hf.get_image_features(pv, pid, return_dict=True).pooler_output[0]
            side, tmask = side_inputs(vm, pid)
        f1 = {'patches': pv, **side}
        f1 = {n: f1[n].numpy().astype(np.float16) for n in V1}
        f2 = {n: t.numpy().astype(np.float16) for n, t in v2_inputs(pv, side).items()}
        yield path, f1, f2, tmask[0].numpy(), ref


def cmd_parity(args):
    import coremltools as ct
    st = load_full(); _, vm, _ = parts(st)
    ml = ct.models.MLModel(str(package(args.patches, args.qchunks, args.fused, args.kchunks)), compute_units=ct.ComputeUnit.CPU_AND_NE)
    worst, medians, times = [], [], []
    for path, _, f2, tmask, ref in feeds(st, vm, sample_images(args.n), args.patches):
        t0 = time.perf_counter(); y = ml.predict(f2)['soft'][0]; times.append(time.perf_counter() - t0)
        out = torch.from_numpy(y.T.astype(np.float32))[tmask]
        cos = F.cosine_similarity(out, ref, dim=-1)
        worst.append(float(cos.min())); medians.append(float(cos.median()))
        print(f'{path.name[:34]:34} soft-token cosine vs fp32: min {cos.min():.5f} median {cos.median():.5f}'
              f'{"  NaN!" if not torch.isfinite(out).all() else ""}', flush=True)
    print(f'{package(args.patches, args.qchunks, args.fused, args.kchunks).name} on CPU_AND_NE: worst token {min(worst):.5f}, median of medians '
          f'{np.median(medians):.5f}; predict median {1000 * np.median(times[1:] or times):.0f} ms')


def cmd_ab(args):
    """v1 and v2 on the same real images, alternating, so load changes hit both alike."""
    import os
    import coremltools as ct
    st = load_full(); _, vm, _ = parts(st)
    v1 = ct.models.MLModel(str(OUT / f'eg2-vision-{args.patches}x1-safe.mlpackage'), compute_units=ct.ComputeUnit.CPU_AND_NE)
    v2 = ct.models.MLModel(str(package(args.patches, args.qchunks, args.fused, args.kchunks)), compute_units=ct.ComputeUnit.CPU_AND_NE)
    data = list(feeds(st, vm, sample_images(args.n), args.patches))
    v1.predict(data[0][1]); v2.predict(data[0][2])                            # warm
    t1, t2, agree = [], [], []
    for r in range(args.rounds):
        for i, (_, f1, f2, tmask, _) in enumerate(data):
            order = ((v1, f1, t1), (v2, f2, t2)) if (r + i) % 2 == 0 else ((v2, f2, t2), (v1, f1, t1))
            ys = {}
            for ml, f, ts in order:
                t0 = time.perf_counter(); y = ml.predict(f)['soft'][0]; ts.append(time.perf_counter() - t0)
                ys[id(ml)] = y
            a = torch.from_numpy(ys[id(v1)].astype(np.float32))[tmask]
            b = torch.from_numpy(ys[id(v2)].T.astype(np.float32))[tmask]
            agree.append(float(F.cosine_similarity(a, b, dim=-1).median()))
    m1, m2 = 1000 * np.median(t1), 1000 * np.median(t2)
    print(f'{len(data)} images x {args.rounds} rounds, load {os.getloadavg()[0]:.1f}: v1 {m1:.0f} ms, v2 {m2:.0f} ms per image '
          f'({m1 / m2:.2f}x); v1 vs v2 soft tokens median cosine {np.median(agree):.5f}')


def cmd_export(args):
    """Vision assets for the app (ANE Embed 0.2+): <Application Support>/ANEEmbed/embeddinggemma2-vision-w16/
    manifest.json, pos_table.f32 (the two 10,240 x 768 position tables, float32) and both packages:
    v1 (convert_gemma2_vision.py --fp16-safe) and v2 (this file), so the app can run either layout.
    The text side stays in embeddinggemma2-w16, untouched: an older app never reads this folder."""
    import json
    import shutil
    from pathlib import Path
    st = load_full(); hf, vm, _ = parts(st)
    root = Path(args.out).expanduser()
    (root / 'packages').mkdir(parents=True, exist_ok=True)
    table = vm.patch_embedder.position_embedding_table.detach().float().contiguous().numpy()
    table.astype('<f4').tofile(root / 'pos_table.f32')
    rope = type(vm.encoder.rotary_emb)(vm.config)
    # the sequence the processor builds around one image: [bos, boi, image x n, eoi, eos]
    ids = st.preprocess([{'text': '<|image|>', 'image': str(sample_images(2)[0])}])['input_ids'][0].tolist()
    cfg = hf.config
    assert ids[1] == cfg.boi_token_id and ids[2] == cfg.image_token_id and ids[-2] == cfg.eoi_token_id, ids[:3] + ids[-3:]
    packages = {}
    for layout, src in (('v1', OUT / f'eg2-vision-{args.patches}x1-safe.mlpackage'), ('v2', package(args.patches, 1))):
        dst = root / 'packages' / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        packages[layout] = f'packages/{src.name}'
    manifest = {
        'model_tag': 'embeddinggemma2:ane-w16', 'hf_model': 'google/embeddinggemma-2',
        'patch_size': vm.config.patch_size, 'pooling': vm.config.pooling_kernel_size, 'max_patches': args.patches,
        'width': vm.config.hidden_size, 'out_dim': hf.config.text_config.hidden_size, 'head_dim': vm.config.head_dim,
        'table_size': table.shape[1], 'inv_freq': [float(x) for x in rope.inv_freq],
        'token_ids': {'bos': ids[0], 'boi': ids[1], 'image': ids[2], 'eoi': ids[-2], 'eos': ids[-1]},
        'packages': packages, 'default_layout': 'v2',
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{root}: pos_table.f32 {table.nbytes / 1e6:.0f} MB, packages {sorted(packages)}, token ids {manifest["token_ids"]}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('export')
    e.add_argument('--patches', type=int, default=2520)
    e.add_argument('--out', default='~/Library/Application Support/ANEEmbed/embeddinggemma2-vision-w16')
    for name in ('check', 'convert', 'parity', 'ab', 'probe'):
        s = sub.add_parser(name)
        s.add_argument('--qchunks', type=int, default=1)
        s.add_argument('--kchunks', type=int, default=1, help='online softmax over this many key slices (flash-attention style)')
        s.add_argument('--fused', action='store_true', help='per-head norms and rotary on all heads at once (LayerFused)')
        s.add_argument('--patches', type=int, default=2520)
        s.add_argument('--n', type=int, default={'check': 6, 'parity': 12, 'ab': 6}.get(name, 6))
        if name == 'ab':
            s.add_argument('--rounds', type=int, default=3)
    args = p.parse_args()
    {'check': cmd_check, 'convert': cmd_convert, 'parity': cmd_parity, 'ab': cmd_ab, 'probe': cmd_probe,
     'export': cmd_export}[args.cmd](args)


if __name__ == '__main__':
    main()
