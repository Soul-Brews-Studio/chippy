"""EmbeddingGemma 2 vision encoder for the Apple Neural Engine: images and video frames to soft tokens.

Split:
  CPU   the model's own processor: resize, cut into 16 x 16 patches, pad to max_patches (2,520 at the
        default 280-token image budget, 1,260 per video frame at 140). From the patch positions alone:
        the 2-D position embeddings (two 10,240 x 768 tables), the axial rotary cos/sin, the valid-patch
        mask and the 3 x 3 average-pool matrix
  ANE   patches -> input_proj + positions -> 16 encoder layers (bidirectional, padding keys masked)
        -> pool -> embed_vision (RMSNorm, 768 -> 512) = soft tokens [B, T, 512], T = max_patches / 9
  CPU   keep the valid soft tokens and put them where the image tokens are in the text embeddings;
        the text core package from convert_gemma2.py does the rest, on the ANE as before
The pooler's x sqrt(768) is left out: embed_vision starts with an RMSNorm, which cancels any scale
(the model's own comment: the scaling is what overflows fp16). --fp16-safe is needed: hidden states
between layers reach 1,200 (check), and fp16 RMSNorm squares overflow past 256.

Measured 2026-10-08 (m5, M5 Max): all 2,529 ops on the Neural Engine; soft tokens vs fp32 median
cosine 0.99997; 340 ms per image at load 6 (424 ms at load 16). Attention through
scaled_dot_product_attention (attn_implementation='sdpa') converts to the same ops with
coremltools 9.0: 340 vs 340 ms, back to back.

    python convert_gemma2_vision.py check   [--n 8]              # fp32 wrapper vs the model's get_image_features
    python convert_gemma2_vision.py convert --patches 2520 --batch 1 [--fp16-safe]
    python convert_gemma2_vision.py parity  --patches 2520 --batch 1 [--fp16-safe] [--n 16]

Run with the same pinned deps as convert_gemma2.py (tools/ane-embed/eg2 wraps them).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from convert_gemma2 import MODEL_ID, NEG, REPO, patch_coremltools_cast, placement, scaled_rms

OUT = REPO / '.tmp/models/embeddinggemma2-vision-coreml'
INPUTS = ('patches', 'pos', 'cos', 'sin', 'valid', 'pool')


def load_full(device='cpu', dtype=torch.float32):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID, device=device, model_kwargs={'attn_implementation': 'eager', 'torch_dtype': dtype})


def package(args):
    return OUT / f'eg2-vision-{args.patches}x{args.batch}{"-safe" if args.fp16_safe else ""}.mlpackage'


def parts(st):
    """(EmbeddingGemma2Model, vision tower, embed_vision) inside the sentence-transformers wrapper."""
    for m in st.modules():
        if type(m).__name__ == 'EmbeddingGemma2Model':
            return m, m.vision_tower, m.embed_vision
    raise RuntimeError('EmbeddingGemma2Model not found')


def side_inputs(vm, pid):
    """Everything that depends only on the patch positions pid [N, P, 2] (-1 = padding), in fp32."""
    valid = (~(pid == -1).all(-1)).float()
    c = pid.clamp(min=0)
    table = vm.patch_embedder.position_embedding_table
    pos = (F.embedding(c[..., 0], table[0]) + F.embedding(c[..., 1], table[1])) * valid[..., None]
    cos, sin = vm.encoder.rotary_emb(pos, pid)
    k = vm.config.pooling_kernel_size
    T = pid.shape[1] // (k * k)
    width = c[..., 0].max(-1, keepdim=True)[0] + 1
    cell = torch.div(c, k, rounding_mode='floor')
    w = F.one_hot((cell[..., 0] + (width // k) * cell[..., 1]).long(), T).float() / (k * k)   # [N, P, T]
    side = {'pos': pos, 'cos': cos.float(), 'sin': sin.float(), 'valid': valid, 'pool': w.transpose(1, 2).contiguous()}
    return side, ~(w == 0).all(1)                    # soft-token mask [N, T], the same rule as the model's pooler


class VisionCore(torch.nn.Module):
    """patches [B, P, 768] in [0, 1], pos [B, P, 768], cos/sin [B, P, 64], valid [B, P], pool [B, T, P]
    -> soft tokens [B, T, 512] (all T; the caller keeps the valid ones)."""

    def __init__(self, vm, ev, patches: int):
        super().__init__()
        self.vm, self.ev = vm, ev
        self.register_buffer('pid', torch.zeros(1, patches, 2, dtype=torch.long))   # only its last dim (2-D) is read

    def forward(self, patches, pos, cos, sin, valid, pool):
        h = self.vm.patch_embedder.input_proj(2 * (patches - 0.5)) + pos
        keys = ((1.0 - valid) * NEG)[:, None, None, :]
        for layer in self.vm.encoder.layers:
            h = layer(h, position_embeddings=(cos, sin), attention_mask=keys, position_ids=self.pid)
        return self.ev(torch.matmul(pool, h * valid[:, :, None]))


def use_fp16_safe_norms():
    import transformers.models.gemma4.modeling_gemma4 as mod
    mod.Gemma4RMSNorm._norm = scaled_rms


def pixels(st, path):
    f = st.preprocess([{'text': '<|image|>', 'image': str(path)}])
    return f['pixel_values'].float(), f['image_position_ids']


def sample_images(n):
    # calibration/check images: put a few of your own .jpg/.png in .tmp/mm/img and .tmp/mm/shots
    icons = sorted((REPO / '.tmp/mm/img').glob('*.jpg'))
    shots = sorted((REPO / '.tmp/mm/shots').glob('*.png'))
    a, b = n // 2, n - n // 2
    return icons[::max(1, len(icons) // max(a, 1))][:a] + shots[::max(1, len(shots) // max(b, 1))][:b]


def cmd_check(args):
    st = load_full(); hf, vm, ev = parts(st)
    peak = [0.0]
    hooks = [l.register_forward_hook(lambda m, i, o: peak.__setitem__(0, max(peak[0], float(o.abs().max()))))
             for l in vm.encoder.layers]
    for path in sample_images(args.n):
        pv, pid = pixels(st, path)
        with torch.no_grad():
            ref = hf.get_image_features(pv, pid, return_dict=True).pooler_output[0]
            side, tmask = side_inputs(vm, pid)
            out = VisionCore(vm, ev, pv.shape[1]).eval()(pv, **side)[0][tmask[0]]
        cos = F.cosine_similarity(out, ref, dim=-1)
        print(f'{path.name[:34]:34} {tuple(pv.shape)} -> {tuple(ref.shape)} soft tokens; '
              f'cosine min {cos.min():.6f}, max |diff| {(out - ref).abs().max():.2e}', flush=True)
    for h in hooks:
        h.remove()
    print(f'largest |hidden state| between encoder layers: {peak[0]:.1f} (fp16 RMSNorm squares overflow past 256)')


def cmd_convert(args):
    import coremltools as ct
    patch_coremltools_cast()
    if args.fp16_safe:
        use_fp16_safe_norms()
    st = load_full(); _, vm, ev = parts(st)
    P, B = args.patches, args.batch
    T = P // vm.config.pooling_kernel_size ** 2
    shapes = {'patches': (B, P, 768), 'pos': (B, P, 768), 'cos': (B, P, 64), 'sin': (B, P, 64), 'valid': (B, P), 'pool': (B, T, P)}
    OUT.mkdir(parents=True, exist_ok=True)
    path = package(args)
    t0 = time.time()
    core = VisionCore(vm, ev, P).eval()
    ex = tuple(torch.rand(shapes[n]) if n == 'patches' else torch.ones(shapes[n]) * (n == 'valid') for n in INPUTS)
    with torch.no_grad():
        traced = torch.jit.trace(core, ex, check_trace=False)
    ml = ct.convert(traced, convert_to='mlprogram', minimum_deployment_target=ct.target.macOS15,
                    inputs=[ct.TensorType(name=n, shape=shapes[n], dtype=np.float16) for n in INPUTS],
                    outputs=[ct.TensorType(name='soft', dtype=np.float16)],
                    compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE)
    ml.short_description = f'EmbeddingGemma 2 vision encoder + embed_vision, {P} patches x {B} -> {T} soft tokens of 512'
    ml.save(str(path))
    print(f'{path.name}: converted in {time.time() - t0:.0f} s, {placement(path)}', flush=True)


def cmd_parity(args):
    import coremltools as ct
    st = load_full(); hf, vm, ev = parts(st)
    P, B = args.patches, args.batch
    path = package(args)
    ml = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    worst, medians, times = [], [], []
    for path_img in sample_images(args.n):
        pv, pid = pixels(st, path_img)
        if pv.shape[1] != P:
            print(f'{path_img.name}: {pv.shape[1]} patches, package takes {P}: skipped'); continue
        with torch.no_grad():
            ref = hf.get_image_features(pv, pid, return_dict=True).pooler_output[0]
            side, tmask = side_inputs(vm, pid)
        feed = {'patches': pv, **side}
        feed = {n: np.repeat(feed[n].numpy().astype(np.float16), B, axis=0) for n in INPUTS}
        t0 = time.perf_counter(); y = ml.predict(feed)['soft']; times.append(time.perf_counter() - t0)
        out = torch.from_numpy(y[0].astype(np.float32))[tmask[0]]
        cos = F.cosine_similarity(out, ref, dim=-1)
        worst.append(float(cos.min())); medians.append(float(cos.median()))
        print(f'{path_img.name[:34]:34} soft-token cosine vs fp32: min {cos.min():.5f} median {cos.median():.5f}'
              f'{"  NaN!" if not torch.isfinite(out).all() else ""}', flush=True)
    print(f'{path.name} on CPU_AND_NE: worst token {min(worst):.5f}, median of medians {np.median(medians):.5f}; '
          f'predict median {1000 * np.median(times[1:] or times):.0f} ms per call of {B}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check'); c.add_argument('--n', type=int, default=8)
    for name in ('convert', 'parity'):
        s = sub.add_parser(name)
        s.add_argument('--patches', type=int, default=2520)
        s.add_argument('--batch', type=int, default=1)
        s.add_argument('--fp16-safe', action='store_true', help='RMSNorm divides by the row max before squaring')
        if name == 'parity':
            s.add_argument('--n', type=int, default=16)
    args = p.parse_args()
    {'check': cmd_check, 'convert': cmd_convert, 'parity': cmd_parity}[args.cmd](args)


if __name__ == '__main__':
    main()
