"""EmbeddingGemma 2 (text core) for the Apple Neural Engine: fixed-shape Core ML packages.

Split, the same as the EmbeddingGemma 1 packages Chippy runs:
  CPU   tokenize (sentence-transformers' own tokenizer + prompt) and look up the scaled token
        table (262,144 x 512, embed_tokens * sqrt(512)): 268 MB in fp16, cheaper on the CPU.
  ANE   embeds [B, L, 512] + mask [B, L] -> the 24 encoder layers (bidirectional attention,
        sliding window 512 on 5 of every 6 layers, per-layer inputs), final norm, mean pool over
        the mask, embedding_projection -> pooled [B, 768].
  CPU   L2-normalise.
Pooling before the projection is exact: the projection is linear and has no bias (the model's
own comment says so), and sentence-transformers mean-pools every token, prompt included.

    python convert_gemma2.py check  [--n 64]                      # fp32 wrapper vs sentence-transformers
    python convert_gemma2.py convert --buckets 128x8,512x4 [--out DIR]
    python convert_gemma2.py parity --bucket 128x8 [--n 64]      # Core ML on CPU_AND_NE vs sentence-transformers

Run with: uv run --prerelease allow --with 'sentence-transformers[image]>=6.1'
          --with 'transformers>=5.18.0.dev0' --with pillow --with coremltools==9.0 python ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

MODEL_ID = 'google/embeddinggemma-2'
REPO = Path(__file__).resolve().parents[1]
from samples import sample_texts   # public sample set, or SAMPLE_TEXTS=your.json
DEFAULT_OUT = REPO / '.tmp/models/embeddinggemma2-coreml'
NEG = -1e4                                    # additive mask; safe in fp16


def load_st():
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(MODEL_ID, device='cpu', config_kwargs={'vision_config': None, 'audio_config': None},
                             model_kwargs={'attn_implementation': 'eager', 'torch_dtype': torch.float32})
    st.max_seq_length = 2048
    return st


def stable_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, softcap=None, **kwargs):
    """eager_attention_forward with the softmax written out and the row max subtracted first.
    transformers upcasts its softmax to fp32; Core ML computes it in fp16, and with scaling 1.0
    (v2 normalises q and k itself) exp() of large scores overflows to inf on the ANE -> NaN
    (a 1,029-token text in the 2,048 bucket: 723 NaN on CPU_AND_NE, fine in torch fp16)."""
    groups = module.num_key_value_groups
    k = key.repeat_interleave(groups, dim=1)
    v = value.repeat_interleave(groups, dim=1)
    w = torch.matmul(query, k.transpose(2, 3)) * (module.head_dim ** -0.5 if scaling is None else scaling)
    if attention_mask is not None:
        w = w + attention_mask
    w = w - w.amax(dim=-1, keepdim=True)
    w = torch.exp(w)
    w = w / w.sum(dim=-1, keepdim=True)
    out = torch.matmul(w, v).transpose(1, 2).contiguous()
    return out, w


def scaled_rms(self, x):
    """RMSNorm without squaring large values: x / rms(x) == y / rms(y) for y = x / max|x|.
    transformers squares in fp32 (hidden_states.float()); Core ML squares in fp16, where any
    |x| > 256 overflows 65,504. Dividing by the row max first keeps every square <= 1."""
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-2)
    y = x / s
    return y * torch.pow((y * y).mean(-1, keepdim=True) + self.eps / (s * s), -0.5)


def use_stable_attention():
    """fp16-safe versions of the two ops transformers upcasts to fp32: softmax and RMSNorm.
    Used ONLY for the CPU bucket (2,048): on the ANE they cost precision on short texts
    (bench median cosine 0.99840 -> 0.97399, R@1 0.444 -> 0.324), and the shorter buckets
    never needed them (no NaN, median 0.9984 without)."""
    import transformers.models.embedding_gemma2.modeling_embedding_gemma2 as mod
    mod.eager_attention_forward = stable_attention
    mod.EmbeddingGemma2RMSNorm._norm = scaled_rms


def text_model(st):
    """The EmbeddingGemma2TextModel inside the sentence-transformers wrapper."""
    for m in st.modules():
        if type(m).__name__ == 'EmbeddingGemma2TextModel':
            return m
    raise RuntimeError('EmbeddingGemma2TextModel not found')


class Encoder(torch.nn.Module):
    """embeds [B, L, 512] fp, mask [B, L] (1 = token) -> pooled, projected [B, 768] (not normalised)."""

    def __init__(self, tm, length: int, window: int, fp16_safe: bool = False):
        super().__init__()
        if fp16_safe:
            use_stable_attention()
        self.tm = tm
        cfg = tm.config
        self.types = list(cfg.layer_types)
        pos = torch.arange(length)
        band = ((pos[:, None] - pos[None, :]).abs() > window).float() * NEG
        self.register_buffer('band', band[None, None])
        self.register_buffer('off_diag', (1.0 - torch.eye(length))[None, None])
        # Rotary cos/sin computed here in fp32 and stored as constants: computed inside the fp16
        # package, angles up to ~2,047 rad round to steps of 0.5-2 rad and long texts break
        # (one 1,029-token text came out at cosine 0.559). cos/sin in [-1, 1] are exact enough in fp16.
        with torch.no_grad():
            x = torch.zeros(1, length, cfg.hidden_size)
            for t in sorted(set(self.types)):
                cos, sin = tm.rotary_emb(x, pos.unsqueeze(0), t)
                self.register_buffer(f'cos_{t}', cos.float()); self.register_buffer(f'sin_{t}', sin.float())

    def forward(self, embeds, mask):
        tm = self.tm
        # padding keys masked, but every position always sees ITSELF: a padding row whose whole
        # sliding window is padding (> 512 pads in a row, only in the 2,048 bucket) would otherwise
        # be fully masked -> softmax 0/0 = NaN on the ANE, which leaks into real tokens through
        # 0 x NaN in the next layer. Real tokens are unchanged: their own key is real anyway.
        keys = (1.0 - mask)[:, None, None, :] * NEG * self.off_diag      # [B, 1, L, L]
        masks = {'full_attention': keys, 'sliding_attention': keys + self.band}
        per_layer = tm.ple(embeds)
        pe = {t: (getattr(self, f'cos_{t}'), getattr(self, f'sin_{t}')) for t in set(self.types)}
        h = embeds
        for i, layer in enumerate(tm.layers):
            h = layer(h, per_layer[:, :, i, :], attention_mask=masks[self.types[i]], position_embeddings=pe[self.types[i]])
        h = tm.norm(h)
        # weight each token by 1/count BEFORE summing: summing 1,000+ hidden states first and
        # dividing after overflows fp16 (max 65,504) on long texts
        m = mask[:, :, None]
        pooled = (h * (m / m.sum(1, keepdim=True).clamp(min=1.0))).sum(1)
        return tm.embedding_projection(pooled)


def tokenize(st, texts, prompt_name='Document'):
    prompt = st.prompts[prompt_name]
    feats = st.tokenize([prompt + t for t in texts])
    return feats['input_ids'], feats['attention_mask']


def scaled_embeds(tm, ids):
    return tm.embed_tokens(ids)                                           # includes * sqrt(512)


def reference(st, texts, prompt_name='Document'):
    """sentence-transformers fp32 vectors, cached in .tmp/eg2-ref/ by a hash of the exact inputs:
    recomputing 1,024 texts on the CPU took ~6 of every bench's ~8 minutes."""
    import hashlib
    key = hashlib.sha256(json.dumps([prompt_name, st.max_seq_length, texts]).encode()).hexdigest()[:16]
    path = REPO / '.tmp/eg2-ref' / f'{key}.npy'
    if path.exists():
        return np.load(path)
    out = st.encode(texts, prompt_name=prompt_name, batch_size=16, convert_to_numpy=True, normalize_embeddings=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def real_texts(n):
    return sample_texts(n)


def cmd_check(args):
    st = load_st(); tm = text_model(st).eval()
    texts = real_texts(args.n)
    ref = reference(st, texts)
    ids, mask = tokenize(st, texts)
    L = ids.shape[1]
    print(f'{len(texts)} texts, padded length {L}, longest {int(mask.sum(1).max())} tokens')
    for window in (512, 513):
        enc = Encoder(tm, L, window).eval()
        with torch.no_grad():
            out = enc(scaled_embeds(tm, ids), mask.float()).numpy()
        cos = (unit(out) * unit(ref)).sum(1)
        print(f'window {window}: cosine vs sentence-transformers min {cos.min():.8f} mean {cos.mean():.8f}')


def patch_coremltools_cast():
    """coremltools 9.0 + torch 2.14: aten::Int on a folded (1,)-shaped constant reaches
    dtype(np.array([n])), which numpy >= 2 refuses ("only 0-dimensional arrays can be converted
    to Python scalars"). Take the single element first; everything else is unchanged."""
    from coremltools.converters.mil.frontend.torch import ops
    original = ops._cast

    def _cast(context, node, dtype, dtype_name):
        x = ops._get_inputs(context, node, expected=1)[0]
        if x.can_be_folded_to_const() and isinstance(x.val, np.ndarray) and x.val.size == 1:
            from coremltools.converters.mil import Builder as mb
            context.add(mb.const(val=dtype(x.val.reshape(-1)[0].item()), name=node.name), node.name)
            return
        original(context, node, dtype, dtype_name)

    ops._cast = _cast


def cmd_convert(args):
    import coremltools as ct
    patch_coremltools_cast()
    st = load_st(); tm = text_model(st).eval()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.buckets.split(','):
        L, B = map(int, spec.split('x'))
        path = out_dir / f'eg2-w16-{L}x{B}.mlpackage'
        if path.exists():
            print(f'{path.name}: exists, skipped'); continue
        t0 = time.time()
        enc = Encoder(tm, L, args.window, fp16_safe=L in CPU_BUCKETS).eval()
        ex = (torch.zeros(B, L, 512), torch.ones(B, L))
        with torch.no_grad():
            traced = torch.jit.trace(enc, ex, check_trace=False)
        ml = ct.convert(traced, convert_to='mlprogram', minimum_deployment_target=ct.target.macOS15,
                        inputs=[ct.TensorType(name='embeds', shape=(B, L, 512), dtype=np.float16),
                                ct.TensorType(name='mask', shape=(B, L), dtype=np.float16)],
                        outputs=[ct.TensorType(name='pooled', dtype=np.float16)],
                        compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE)
        ml.short_description = f'EmbeddingGemma 2 text core, {L} tokens x {B}, mean-pooled + projected (not normalised)'
        ml.save(str(path))
        print(f'{path.name}: converted in {time.time() - t0:.0f} s, {placement(path)}', flush=True)


def placement(path):
    import coremltools as ct
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(model.get_compiled_model_path()), compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts = {}
    for op in plan.model_structure.program.functions['main'].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        dev = type(usage.preferred_compute_device).__name__.replace('MLComputeDevice', '') if usage else 'const'
        counts[dev] = counts.get(dev, 0) + 1
    return counts


def cmd_parity(args):
    import coremltools as ct
    st = load_st(); tm = text_model(st).eval()
    L, B = map(int, args.bucket.split('x'))
    path = Path(args.out) / f'eg2-w16-{L}x{B}.mlpackage'
    ml = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    texts = real_texts(args.n)
    ref = reference(st, texts)
    outs, times = [], []
    for i in range(0, len(texts), B):
        part = texts[i:i + B]
        ids, mask = tokenize(st, part)
        ids, mask = ids[:, :L], mask[:, :L]                               # bucket = truncation length
        e = np.zeros((B, L, 512), np.float16); m = np.zeros((B, L), np.float16)
        with torch.no_grad():
            e[:len(part), :ids.shape[1]] = scaled_embeds(tm, ids).numpy()
        m[:len(part), :ids.shape[1]] = mask.numpy()
        t0 = time.perf_counter()
        y = ml.predict({'embeds': e, 'mask': m})['pooled'][:len(part)]
        times.append(time.perf_counter() - t0)
        outs.append(y)
    out = np.concatenate(outs)
    cos = (unit(out) * unit(ref)).sum(1)
    trunc = sum(1 for t in texts if len(st.tokenize([st.prompts['Document'] + t])['input_ids'][0]) > L)
    print(f'{path.name} on CPU_AND_NE: {len(texts)} texts ({trunc} longer than {L} tokens, truncated)')
    print(f'  cosine vs sentence-transformers fp32: min {cos.min():.5f}  p1 {np.percentile(cos, 1):.5f}  median {np.median(cos):.5f}')
    full = [c for c, t in zip(cos, texts) if len(st.tokenize([st.prompts['Document'] + t])['input_ids'][0]) <= L]
    if full:
        print(f'  texts that fit the bucket: min {min(full):.5f}  median {np.median(full):.5f}')
    print(f'  predict: median {1000 * np.median(times[1:] or times):.1f} ms per call of {B}  ->  {B / np.median(times[1:] or times):.0f} texts/s')


CPU_BUCKETS = {2048}
BUCKETS = '32x64,64x32,128x16,256x8,512x4,640x3,768x2,1024x2,2048x1'   # 2,048 token slots per call, as v1


def cmd_bench(args):
    """All buckets: each text goes to the smallest bucket that holds it; full batches per call."""
    import coremltools as ct
    st = load_st(); tm = text_model(st).eval()
    texts = sample_texts()
    long = [t for t in texts if len(t) >= 400]
    specs = [tuple(map(int, b.split('x'))) for b in BUCKETS.split(',')]
    # the 2,048 bucket runs on the CPU: on the ANE it loses precision at that length (a 1,029-token
    # text: cosine 0.907 on CPU_AND_NE vs 0.99995 on CPU_ONLY); texts over 1,024 tokens are rare
    units = lambda L: ct.ComputeUnit.CPU_ONLY if L in CPU_BUCKETS and not args.all_ane else ct.ComputeUnit.CPU_AND_NE
    models = {L: (B, ct.models.MLModel(str(Path(args.out) / f'eg2-w16-{L}x{B}.mlpackage'),
                                       compute_units=units(L))) for L, B in specs}

    def embed(items, prompt_name):
        ids_all, mask_all = tokenize(st, items, prompt_name)
        lens = mask_all.sum(1).tolist()
        out = np.zeros((len(items), 768), np.float32)
        groups = {}
        for i, n in enumerate(lens):
            L = next(L for L, _ in specs if L >= min(n, 2048))
            groups.setdefault(L, []).append(i)
        ane_s = 0.0
        for L, idx in groups.items():
            B, ml = models[L]
            for k in range(0, len(idx), B):
                part = idx[k:k + B]
                e = np.zeros((B, L, 512), np.float16); m = np.zeros((B, L), np.float16)
                for row, i in enumerate(part):
                    n = min(int(lens[i]), L)
                    with torch.no_grad():
                        e[row, :n] = scaled_embeds(tm, ids_all[i:i + 1, :n]).numpy()[0]
                    m[row, :n] = 1
                t0 = time.perf_counter()
                out[part] = ml.predict({'embeds': e, 'mask': m})['pooled'][:len(part)]
                ane_s += time.perf_counter() - t0
        return unit(out), ane_s, {L: len(v) for L, v in sorted(groups.items())}

    for _ in range(2):                                                   # warm every bucket once
        embed(texts[:8], 'Document')
    t0 = time.perf_counter()
    docs, ane_s, groups = embed(texts, 'Document')
    wall = time.perf_counter() - t0
    ref = unit(reference(st, texts))
    cos = (docs * ref).sum(1)
    print(f'{len(texts)} texts by bucket: {groups}')
    print(f'ANE end to end: {len(texts) / wall:.1f} texts/s (wall {wall:.1f} s, of which ANE predict {ane_s:.1f} s)')
    print(f'cosine vs sentence-transformers fp32: min {cos.min():.5f}  p1 {np.percentile(cos, 1):.5f}  median {np.median(cos):.5f}')
    ids_all, mask_all = tokenize(st, texts, 'Document')
    lens = mask_all.sum(1).tolist()
    for i in np.argsort(cos)[:5]:
        print(f'  worst #{i}: cosine {cos[i]:.5f}, {int(lens[i])} tokens, {len(texts[i])} chars, starts {texts[i][:60]!r}')
    q, _, _ = embed([t[:200] for t in long], 'SearchQuery')
    d, _, _ = embed([t[200:] for t in long], 'Document')
    r = (np.argsort(-(q @ d.T), axis=1) == np.arange(len(q))[:, None]).argmax(1) + 1
    print(f'self-retrieval on the ANE ({len(long)} pairs): R@1 {np.mean(r <= 1):.3f}  R@10 {np.mean(r <= 10):.3f}  MRR {np.mean(1 / r):.3f}')


def cmd_debug(args):
    """One text through one bucket, four ways; report where inf/NaN first appears in fp16."""
    import coremltools as ct
    st = load_st(); tm = text_model(st).eval()
    text = sample_texts()[args.index]
    L, B = map(int, args.bucket.split('x'))
    ref = unit(reference(st, [text]))[0]
    ids, mask = tokenize(st, [text]); n = int(mask.sum())
    e = torch.zeros(B, L, 512); m = torch.zeros(B, L)
    with torch.no_grad():
        e[0, :n] = scaled_embeds(tm, ids[:, :n])[0]
    m[0, :n] = 1
    print(f'text #{args.index}: {n} tokens, bucket {L}x{B}')
    enc = Encoder(tm, L, 512).eval()
    with torch.no_grad():
        y = enc(e, m)[0].numpy()
    print(f'  torch fp32: cosine {float(unit(y[None])[0] @ ref):.6f}')
    enc16 = Encoder(tm, L, 512).eval().half()
    first = []
    def hook(name):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if not first and not torch.isfinite(o).all():
                bad = (~torch.isfinite(o)).nonzero()
                first.append(f'{name}: non-finite at {bad[0].tolist()} ({len(bad)} values), real tokens 0..{n - 1}')
        return f
    for i, layer in enumerate(tm.layers):
        layer.register_forward_hook(hook(f'layer {i}'))
        layer.self_attn.register_forward_hook(hook(f'layer {i} attention'))
        layer.mlp.register_forward_hook(hook(f'layer {i} mlp'))
    with torch.no_grad():
        y16 = enc16(e.half(), m.half())[0].float().numpy()
    c16 = float(unit(y16[None])[0] @ ref) if np.isfinite(y16).all() else float('nan')
    print(f'  torch fp16 (CPU): cosine {c16:.6f}; first non-finite: {first[0] if first else "none"}')
    path = Path(args.out) / f'eg2-w16-{L}x{B}.mlpackage'
    feed = {'embeds': e.numpy().astype(np.float16), 'mask': m.numpy().astype(np.float16)}
    for cu in ('CPU_ONLY', 'CPU_AND_NE'):
        ml = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, cu))
        out = ml.predict(feed)['pooled'][0].astype(np.float64)
        c = float(unit(out[None])[0] @ ref) if np.isfinite(out).all() else float('nan')
        print(f'  Core ML {cu}: cosine {c:.6f}, non-finite values {int((~np.isfinite(out)).sum())}')


def cmd_export(args):
    """Write what Chippy loads: <root>/tokenizer/tokenizer.json, embed_scaled.f16, packages/ links,
    manifest.json (format "pooled"). Checks first that the Rust tokenizer Chippy uses (HF tokenizers,
    add_special_tokens, truncation at max_tokens) gives sentence-transformers' ids exactly."""
    import hashlib, shutil
    from tokenizers import Tokenizer
    st = load_st(); tm = text_model(st).eval()
    snap = next((Path.home() / '.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots').iterdir())
    tok = Tokenizer.from_file(str(snap / 'tokenizer.json'))
    tok.no_padding(); tok.enable_truncation(2048)
    texts = sample_texts()
    items = [st.prompts['Document'] + t for t in texts] + [st.prompts['SearchQuery'] + t[:200] for t in texts[:64]]
    ids, mask = tokenize(st, [t[len(st.prompts['Document']):] for t in items[:len(texts)]], 'Document')
    bad = 0
    for i, enc in enumerate(tok.encode_batch(items[:len(texts)], add_special_tokens=True)):
        n = int(mask[i].sum())
        if enc.ids != ids[i, :n].tolist():
            bad += 1
    print(f'tokenizer check: {len(texts) - bad} of {len(texts)} texts give identical ids (Rust tokenizers vs sentence-transformers)')
    if bad:
        raise SystemExit('tokenizer mismatch: not exporting')

    root = Path(args.root).expanduser(); (root / 'tokenizer').mkdir(parents=True, exist_ok=True)
    dst = root / 'tokenizer' / 'tokenizer.json'
    dst.unlink(missing_ok=True)                    # the HF cache copy is read-only: replace, don't overwrite
    shutil.copy(snap / 'tokenizer.json', dst)
    with torch.no_grad():
        table = (tm.embed_tokens.weight.float() * float(tm.embed_tokens.embed_scale)).to(torch.float16).numpy()
    table.tofile(root / 'embed_scaled.f16')
    (root / 'packages').mkdir(exist_ok=True)
    specs = [tuple(map(int, b.split('x'))) for b in BUCKETS.split(',')]
    packages, digests = {}, {}
    for L, B in specs:
        src = Path(args.out).resolve() / f'eg2-w16-{L}x{B}.mlpackage'
        link = root / 'packages' / src.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src)
        packages[str(L)] = f'packages/{src.name}'
        weight = src / 'Data/com.apple.CoreML/weights/weight.bin'
        digests[str(L)] = hashlib.sha256(weight.read_bytes()).hexdigest() if weight.exists() else None
    ident = hashlib.sha256(json.dumps({'hf_revision': snap.name, 'packages': digests,
                                       'table': hashlib.sha256(table.tobytes()).hexdigest()}, sort_keys=True).encode()).hexdigest()
    manifest = {
        'model_tag': 'embeddinggemma2:ane-w16',
        'vector_space_identity': f'embeddinggemma2:coreml-w16-{ident}',
        'hf_model': MODEL_ID, 'hf_revision': snap.name,
        'format': 'pooled', 'vocab': int(table.shape[0]), 'hidden': int(table.shape[1]), 'output_dim': 768,
        'slots_per_call': 2048, 'max_tokens': 2048, 'buckets': [L for L, _ in specs],
        'cpu_buckets': sorted(CPU_BUCKETS), 'packages': packages,
        'prompts': {'document': st.prompts['Document'], 'query': st.prompts['SearchQuery']},
        'converter': 'tools/ane-embed/convert_gemma2.py',
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'exported to {root}: table {table.shape}, {len(packages)} packages, identity {manifest["vector_space_identity"][:40]}…')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check'); c.add_argument('--n', type=int, default=64)
    c = sub.add_parser('convert'); c.add_argument('--buckets', default='128x8,512x4')
    c.add_argument('--window', type=int, default=512); c.add_argument('--out', default=str(DEFAULT_OUT))
    c = sub.add_parser('debug'); c.add_argument('--index', type=int, default=921); c.add_argument('--bucket', default='2048x1')
    c.add_argument('--out', default=str(DEFAULT_OUT))
    c = sub.add_parser('export'); c.add_argument('--out', default=str(DEFAULT_OUT))
    c.add_argument('--root', default='~/Library/Application Support/ANEEmbed/embeddinggemma2-w16')
    c = sub.add_parser('bench'); c.add_argument('--out', default=str(DEFAULT_OUT))
    c.add_argument('--all-ane', action='store_true', help='run the 2,048 bucket on the ANE too (to show why it is not)')
    c = sub.add_parser('parity'); c.add_argument('--bucket', default='128x8'); c.add_argument('--n', type=int, default=64)
    c.add_argument('--out', default=str(DEFAULT_OUT))
    args = p.parse_args()
    {'check': cmd_check, 'convert': cmd_convert, 'parity': cmd_parity, 'bench': cmd_bench, 'debug': cmd_debug, 'export': cmd_export}[args.cmd](args)


if __name__ == '__main__':
    main()
