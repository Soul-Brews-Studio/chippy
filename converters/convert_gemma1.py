"""EmbeddingGemma (v1, 300M) text core -> Core ML packages for the Apple Neural Engine, and the
assets Chippy loads. Same approach as convert_gemma2.py, for google/embeddinggemma-300m:

  embeds [B, L, 768] (token embeddings x sqrt(768), looked up on the CPU), mask [B, L]
    -> 24 Gemma 3 layers, bidirectional (sliding layers see |i - j| < sliding_window) -> final norm
    -> hidden [B, 768, 1, L] fp16 on the ANE
  then on the CPU in float32 (Chippy's projectHidden): mean over real tokens -> @ dense1 (768x3072)
    -> @ dense2 (3072x768) -> L2 normalise.
Numerics, measured on the 128-token bucket (64 sample texts, cosine to sentence-transformers fp32,
texts that fit): the ANE computes in fp16 and flushes subnormals, and this graph needs two things
together to match the CPU:
  - the residual stream at 1/RES (Gemma 3's residual peaks ~155,000, fp16 max 65,504): embeds / RES and
    1/RES folded into the two post-norm gains; the norms that read the residual square x / 128
  - int8 per-channel conv weights (W8A16)
  fp16 weights, any variant of the graph ............ median 0.9904-0.9911 (CPU: 0.99994)
  + extra scaling of q/k or post-norm squares ........ no better, or worse with int8 (0.988)
  int8 conv weights, plain graph (this file) ......... median 0.99929, min 0.99901
The token table, mean pooling and the two Dense layers stay on the CPU in float32 (Chippy's projectHidden).

One package per length bucket (L tokens x B texts = 2,048 token slots per call), all on the ANE.

    ./eg1 check                     # torch wrapper vs sentence-transformers (fp32, no Core ML)
    ./eg1 convert --buckets all     # all packages (several minutes per bucket)
    ./eg1 parity --bucket 128x16    # ANE vectors vs sentence-transformers fp32
    ./eg1 export --root DIR         # what Chippy loads (default: ANEEmbed/embeddinggemma-w16)

--model picks the checkpoint: google/embeddinggemma-300m (official, gated: accept its license on
Hugging Face) or a mirror such as unsloth/embeddinggemma-300m. Both give the same weights.
Written by ane (AI, Claude) for Chippy. The ANE layout and the residual/int8 numerics follow what our
earlier EmbeddingGemma ANE experiments found; the code here is new.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from convert_gemma2 import NEG, patch_coremltools_cast, placement, unit
from samples import sample_texts

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = 'google/embeddinggemma-300m'
DEFAULT_OUT = REPO / '.tmp/models/embeddinggemma-coreml'
BUCKETS = '32x64,64x32,128x16,256x8,512x4,640x3,768x2,1024x2,2048x1'
RES, C_VAR = 8.0, 1 / 128          # residual stream / 8; RMSNorm variance computed on x / 128


def load_st(model_id):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model_id, device='cpu', model_kwargs={'attn_implementation': 'eager', 'torch_dtype': torch.float32})
    st.max_seq_length = 2048
    return st


def parts(st):
    """(Gemma3TextModel, dense1 Linear 768->3072, dense2 Linear 3072->768) from the sentence-transformers stack."""
    tm = st[0].auto_model
    dense = [m.linear for m in st if type(m).__name__ == 'Dense']
    assert type(tm).__name__ == 'Gemma3TextModel' and len(dense) == 2, 'unexpected EmbeddingGemma layout'
    return tm, dense[0], dense[1]


def conv1x1(lin, gain=None):
    """Linear(out, in) -> Conv2d 1x1 on [B, C, 1, S]; an optional per-input-channel gain folded in."""
    w = lin.weight.data if gain is None else lin.weight.data * gain[None, :]
    c = torch.nn.Conv2d(w.shape[1], w.shape[0], 1, bias=False)
    c.weight.data = w[:, :, None, None].clone()
    return c


def chan_rms(x, eps, c):
    """RMS over channels (dim 1) of [B, C, 1, S], squaring x * c so fp16 never overflows."""
    if c == 1.0:
        return x * torch.rsqrt((x * x).mean(1, keepdim=True) + eps)
    xc = x * c
    return x * (torch.rsqrt((xc * xc).mean(1, keepdim=True) + eps * c * c) * c)


class AneLayer(torch.nn.Module):
    """One Gemma 3 layer in the ANE's native layout: activations [B, C, 1, S], projections as 1x1 convs,
    attention as an einsum pair per head, no reshapes between layers. Same maths as the HF layer, with
    the residual stream at 1/RES and the fp16-safe norms described above."""

    def __init__(self, hf, cos, sin):
        super().__init__()
        a, m = hf.self_attn, hf.mlp
        self.H, self.D, self.scale = a.config.num_attention_heads, a.head_dim, a.scaling
        self.eps = hf.input_layernorm.eps
        self.q, self.k, self.v = conv1x1(a.q_proj), conv1x1(a.k_proj), conv1x1(a.v_proj)
        self.o = conv1x1(a.o_proj)
        self.gate, self.up, self.down = conv1x1(m.gate_proj), conv1x1(m.up_proj), conv1x1(m.down_proj)
        shape = (1, -1, 1, 1)
        self.register_buffer('g_in', (1.0 + hf.input_layernorm.weight.data).view(shape))
        self.register_buffer('g_pre', (1.0 + hf.pre_feedforward_layernorm.weight.data).view(shape))
        self.register_buffer('g_attn', ((1.0 + hf.post_attention_layernorm.weight.data) / RES).view(shape))
        self.register_buffer('g_ff', ((1.0 + hf.post_feedforward_layernorm.weight.data) / RES).view(shape))
        self.register_buffer('g_q', (1.0 + a.q_norm.weight.data).view(shape))
        self.register_buffer('g_k', (1.0 + a.k_norm.weight.data).view(shape))
        self.qk_eps = a.q_norm.eps
        self.register_buffer('cos', cos); self.register_buffer('sin', sin)          # [1, D, 1, S]

    def rope(self, x):
        h = self.D // 2
        return x * self.cos + torch.cat((-x[:, h:], x[:, :h]), dim=1) * self.sin

    def forward(self, x, mask):
        h = chan_rms(x, self.eps, C_VAR) * self.g_in
        k = self.rope(chan_rms(self.k(h), self.qk_eps, 1.0) * self.g_k)          # one KV head, shared
        v = self.v(h)
        kt = k.transpose(1, 3)                                                    # [B, S, 1, D]
        q = self.q(h)
        outs = []
        for i in range(self.H):
            qi = self.rope(chan_rms(q[:, i * self.D:(i + 1) * self.D], self.qk_eps, 1.0) * self.g_q)
            w = (torch.einsum('bchq,bkhc->bkhq', qi, kt) * self.scale + mask).softmax(dim=1)
            outs.append(torch.einsum('bkhq,bchk->bchq', w, v))
        x = x + chan_rms(self.o(torch.cat(outs, dim=1)), self.eps, 1.0) * self.g_attn
        h = chan_rms(x, self.eps, C_VAR) * self.g_pre
        f = self.down(torch.nn.functional.gelu(self.gate(h), approximate='tanh') * self.up(h))
        return x + chan_rms(f, self.eps, 1.0) * self.g_ff


class AneEncoder(torch.nn.Module):
    """embeds [B, L, 768], mask [B, L] -> hidden [B, 768, 1, L], all in the ANE layout."""

    def __init__(self, tm, length: int):
        super().__init__()
        cfg = tm.config
        self.types = list(cfg.layer_types)[: cfg.num_hidden_layers]
        pos = torch.arange(length)
        band = ((pos[:, None] - pos[None, :]).abs() >= cfg.sliding_window).float() * NEG   # [S_k, S_q]
        self.register_buffer('band', band[None, :, None, :])
        self.register_buffer('off_diag', (1.0 - torch.eye(length))[None, :, None, :])
        layers = []
        with torch.no_grad():
            x = torch.zeros(1, length, cfg.hidden_size)
            rot = {t: tm.rotary_emb(x, pos.unsqueeze(0), t) for t in set(self.types)}
        for i, hf in enumerate(tm.layers[: len(self.types)]):
            cos, sin = rot[self.types[i]]
            layers.append(AneLayer(hf, cos.float().transpose(1, 2)[:, :, None, :], sin.float().transpose(1, 2)[:, :, None, :]))
        self.layers = torch.nn.ModuleList(layers)
        self.norm_eps = tm.norm.eps
        self.register_buffer('g_out', (1.0 + tm.norm.weight.data).view(1, -1, 1, 1))

    def forward(self, embeds, mask):
        x = embeds.transpose(1, 2).unsqueeze(2) * (1.0 / RES)                    # [B, C, 1, S] at 1/RES
        keys = (1.0 - mask)[:, :, None, None] * NEG * self.off_diag              # [B, S_k, 1, S_q]
        masks = {'full_attention': keys, 'sliding_attention': keys + self.band}
        for t, layer in zip(self.types, self.layers):
            x = layer(x, masks[t])
        return chan_rms(x, self.norm_eps, C_VAR) * self.g_out


def project(hidden, mask, dense1, dense2):
    """What Chippy does on the CPU: float32 mean over real tokens, @ dense1 @ dense2 (not normalised)."""
    h = hidden.astype(np.float32)[:, :, 0, :]                            # [B, 768, L]
    m = mask.astype(np.float32)[:, None, :]
    pooled = (h * m).sum(2) / np.maximum(m.sum(2), 1.0)
    return pooled @ dense1 @ dense2


def dense_arrays(d1, d2):
    """dense1 [768, 3072] and dense2 [3072, 768], row-major float32 (Linear weights transposed)."""
    return d1.weight.detach().float().numpy().T.copy(), d2.weight.detach().float().numpy().T.copy()


def tokenize(st, texts, prompt='document'):
    feats = st.tokenize([st.prompts[prompt] + t for t in texts])
    return feats['input_ids'], feats['attention_mask']


def reference(st, texts, prompt='document'):
    return st.encode([st.prompts[prompt] + t for t in texts], batch_size=16, convert_to_numpy=True, normalize_embeddings=True)


def stage(tm, ids, mask, L, B):
    e = np.zeros((B, L, tm.config.hidden_size), np.float32); m = np.zeros((B, L), np.float32)
    ids, mask = ids[:, :L], mask[:, :L]
    with torch.no_grad():
        e[:ids.shape[0], :ids.shape[1]] = tm.embed_tokens(ids).numpy()      # includes * sqrt(768)
    m[:ids.shape[0], :ids.shape[1]] = mask.numpy()
    return e, m


def cmd_check(args):
    st = load_st(args.model); tm, d1, d2 = parts(st)
    texts = sample_texts(args.n)
    ref = reference(st, texts)
    L, B = 512, 8
    enc = AneEncoder(tm.eval(), L).eval()
    D1, D2 = dense_arrays(d1, d2)
    outs = []
    for i in range(0, len(texts), B):
        ids, mask = tokenize(st, texts[i:i + B])
        e, m = stage(tm, ids, mask, L, B)
        with torch.no_grad():
            hid = enc(torch.from_numpy(e), torch.from_numpy(m)).numpy()
        outs.append(project(hid, m, D1, D2)[:ids.shape[0]])
    cos = (unit(np.concatenate(outs)) * unit(ref)).sum(1)
    print(f'torch wrapper (fp32, L={L}) vs sentence-transformers: {len(texts)} texts, cosine min {cos.min():.6f} median {np.median(cos):.6f}')


def cmd_convert(args):
    import coremltools as ct
    patch_coremltools_cast()
    st = load_st(args.model); tm, d1, d2 = parts(st)
    tm.eval()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    H = tm.config.hidden_size
    for spec in (BUCKETS if args.buckets == 'all' else args.buckets).split(','):
        L, B = map(int, spec.split('x'))
        path = out_dir / f'eg1-w16-{L}x{B}.mlpackage'
        if path.exists():
            print(f'{path.name}: exists, skipped'); continue
        t0 = time.time()
        enc = AneEncoder(tm, L).eval()
        with torch.no_grad():
            traced = torch.jit.trace(enc, (torch.zeros(B, L, H), torch.ones(B, L)), check_trace=False)
        ml = ct.convert(traced, convert_to='mlprogram', minimum_deployment_target=ct.target.macOS15,
                        inputs=[ct.TensorType(name='embeds', shape=(B, L, H), dtype=np.float16),
                                ct.TensorType(name='mask', shape=(B, L), dtype=np.float16)],
                        outputs=[ct.TensorType(name='hidden', dtype=np.float16)],
                        compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE)
        if not args.fp16_weights:
            # int8 per-channel conv weights (W8A16): see the numerics table at the top
            # Only convs: a global config would also quantize the RoPE cos/sin tables (positions, not weights).
            import coremltools.optimize as cto
            cfg = cto.coreml.OptimizationConfig(op_type_configs={'conv': cto.coreml.OpLinearQuantizerConfig(
                mode='linear_symmetric', dtype='int8', granularity='per_channel')})
            ml = cto.coreml.linear_quantize_weights(ml, cfg)
        ml.short_description = f'EmbeddingGemma text core, {L} tokens x {B} -> hidden [B, 768, 1, L] after the final norm'
        ml.save(str(path))
        print(f'{path.name}: converted in {time.time() - t0:.0f} s, {placement(path)}', flush=True)


def cmd_parity(args):
    import coremltools as ct
    st = load_st(args.model); tm, d1, d2 = parts(st)
    D1, D2 = dense_arrays(d1, d2)
    L, B = map(int, args.bucket.split('x'))
    path = Path(args.out) / f'eg1-w16-{L}x{B}.mlpackage'
    units = ct.ComputeUnit.CPU_ONLY if args.cpu else ct.ComputeUnit.CPU_AND_NE
    ml = ct.models.MLModel(str(path), compute_units=units)
    texts = sample_texts(args.n)
    ref = reference(st, texts)
    outs, times = [], []
    for i in range(0, len(texts), B):
        ids, mask = tokenize(st, texts[i:i + B])
        e, m = stage(tm, ids, mask, L, B)
        t0 = time.perf_counter()
        hid = ml.predict({'embeds': e.astype(np.float16), 'mask': m.astype(np.float16)})['hidden']
        outs.append(project(hid, m, D1, D2)[:ids.shape[0]])
        times.append(time.perf_counter() - t0)
    cos = (unit(np.concatenate(outs)) * unit(ref)).sum(1)
    t = np.median(times[1:] or times)
    print(f'{path.name} on {"CPU" if units == ct.ComputeUnit.CPU_ONLY else "CPU_AND_NE"}: {len(texts)} texts (longer than {L} tokens are truncated)')
    print(f'  cosine vs sentence-transformers fp32: min {cos.min():.5f}  median {np.median(cos):.5f}')
    fit = np.array([int(m.sum()) <= L for m in tokenize(st, texts)[1]])
    if fit.any():
        print(f'  texts that fit the bucket ({int(fit.sum())}): min {cos[fit].min():.5f}  median {np.median(cos[fit]):.5f}')
    print(f'  predict: median {1000 * t:.1f} ms per call of {B} -> {B / t:.0f} texts/s')


def cmd_export(args):
    """Chippy's assets: tokenizer/tokenizer.json, embed_scaled.f16, dense1.f32, dense2.f32, packages/
    links, manifest.json (hidden format: pooling and dense on the CPU). First checks that the Rust tokenizer Chippy uses gives sentence-transformers' ids."""
    import hashlib, shutil
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    st = load_st(args.model); tm, d1, d2 = parts(st)
    D1, D2 = dense_arrays(d1, d2)
    snap = Path(snapshot_download(args.model, allow_patterns=['tokenizer.json']))
    tok = Tokenizer.from_file(str(snap / 'tokenizer.json'))
    tok.no_padding(); tok.enable_truncation(2048)
    texts = sample_texts()
    ids, mask = tokenize(st, texts)
    bad = sum(enc.ids != ids[i, :int(mask[i].sum())].tolist()
              for i, enc in enumerate(tok.encode_batch([st.prompts['document'] + t for t in texts], add_special_tokens=True)))
    print(f'tokenizer check: {len(texts) - bad} of {len(texts)} texts give identical ids')
    if bad:
        raise SystemExit('tokenizer mismatch: not exporting')
    root = Path(args.root).expanduser()
    if (root / 'manifest.json').exists() and not args.force:
        raise SystemExit(f'{root} already has assets (a running Chippy may use them); pass --force to replace')
    (root / 'tokenizer').mkdir(parents=True, exist_ok=True)
    dst = root / 'tokenizer' / 'tokenizer.json'; dst.unlink(missing_ok=True)
    shutil.copy(snap / 'tokenizer.json', dst)
    with torch.no_grad():
        table = (tm.embed_tokens.weight.float() * float(tm.embed_tokens.embed_scale)).to(torch.float16).numpy()
    table.tofile(root / 'embed_scaled.f16')
    D1.tofile(root / 'dense1.f32'); D2.tofile(root / 'dense2.f32')
    (root / 'packages').mkdir(exist_ok=True)
    specs = [tuple(map(int, b.split('x'))) for b in BUCKETS.split(',')]
    packages, digests = {}, {}
    for L, B in specs:
        src = Path(args.out).resolve() / f'eg1-w16-{L}x{B}.mlpackage'
        if not src.exists():
            raise SystemExit(f'missing {src.name}: run convert --buckets all first')
        link = root / 'packages' / src.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src)
        packages[str(L)] = f'packages/{src.name}'
        weight = src / 'Data/com.apple.CoreML/weights/weight.bin'
        digests[str(L)] = hashlib.sha256(weight.read_bytes()).hexdigest() if weight.exists() else None
    sha = lambda b: hashlib.sha256(b).hexdigest()
    ident = sha(json.dumps({'model': args.model, 'packages': digests, 'table': sha(table.tobytes()),
                            'dense1': sha(D1.tobytes()), 'dense2': sha(D2.tobytes())}, sort_keys=True).encode())
    manifest = {
        'model_tag': 'embeddinggemma:ane-w8a16',
        'vector_space_identity': f'embeddinggemma:coreml-w16-{ident}',
        'hf_model': args.model, 'hf_revision': snap.name,
        'vocab': int(table.shape[0]), 'hidden': int(table.shape[1]),
        'dense1': list(D1.shape), 'dense2': list(D2.shape), 'embed_scale': float(tm.embed_tokens.embed_scale),
        'slots_per_call': 2048, 'max_tokens': 2048, 'buckets': [L for L, _ in specs],
        'packages': packages,
        'prompts': {'document': st.prompts['document'], 'query': st.prompts['query']},
        'converter': 'converters/convert_gemma1.py',
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'exported to {root}: table {table.shape}, {len(packages)} packages, identity {manifest["vector_space_identity"][:44]}...')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=DEFAULT_MODEL)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check'); c.add_argument('--n', type=int, default=64)
    c = sub.add_parser('convert'); c.add_argument('--buckets', default='128x16'); c.add_argument('--out', default=str(DEFAULT_OUT))
    c.add_argument('--fp16-weights', action='store_true', help='keep fp16 conv weights (less precise on the ANE)')
    c = sub.add_parser('parity'); c.add_argument('--bucket', default='128x16'); c.add_argument('--n', type=int, default=64)
    c.add_argument('--cpu', action='store_true', help='run on the CPU only (isolates ANE fp16 effects)')
    c.add_argument('--out', default=str(DEFAULT_OUT))
    c = sub.add_parser('export'); c.add_argument('--out', default=str(DEFAULT_OUT)); c.add_argument('--force', action='store_true')
    c.add_argument('--root', default='~/Library/Application Support/ANEEmbed/embeddinggemma-w16')
    args = p.parse_args()
    {'check': cmd_check, 'convert': cmd_convert, 'parity': cmd_parity, 'export': cmd_export}[args.cmd](args)


if __name__ == '__main__':
    main()
