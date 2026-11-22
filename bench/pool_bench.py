"""Solo against cluster on ANE Embed 0.3: does the volunteer pool make embedding faster?

Every round sends the same work twice, once with "pool": "solo" (this Mac's ANE only) and once with
"pool": "cluster" (split over local plus every live volunteer by weight). The order alternates, so a
change in background load hits both modes alike. Only the pool's primary model is spread
(embeddinggemma:ane-w8a16); other models always run local.

Per round: CONCURRENCY requests of BATCH real texts at once. Reported: texts/s per mode (median of
rounds), the speed-up, how many texts each worker took in the cluster rounds (from /api/workers), and
whether the cluster vectors match the solo ones (they come from other Macs' ANEs: cosine, not equality).

    python3 bench/pool_bench.py [--rounds 4] [--batch 240] [--concurrency 4] [--out f.json]
    python3 bench/pool_bench.py --long 6000 --batch 64 --rounds 3      # long texts, the slow case
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from samples import sample_texts

REPO = Path(__file__).resolve().parents[1]
PORT = 11445


def call(method, path, body=None):
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=900)
    c.request(method, path, json.dumps(body) if body is not None else None, {'Content-Type': 'application/json'})
    r = c.getresponse()
    data = json.loads(r.read() or b'{}')
    if r.status != 200:
        raise RuntimeError(f'{path} HTTP {r.status}: {data}')
    return data


def workers():
    return {w['id']: w for w in call('GET', '/api/workers')['workers']}


def run(mode, parts):
    t = time.perf_counter()
    with ThreadPoolExecutor(len(parts)) as ex:
        out = list(ex.map(lambda p: call('POST', '/api/embed', {'input': p, 'pool': mode})['embeddings'], parts))
    return time.perf_counter() - t, out


def cosmin(a, b):
    lo = 1.0
    for x, y in zip(a, b):
        lo = min(lo, sum(p * q for p, q in zip(x, y)))
    return lo


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--batch', type=int, default=240)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--long', type=int, default=0, metavar='CHARS',
                   help='long-text mode: glue real texts into texts of about CHARS characters (the slow case: '
                        '~6000 fills the 2048-token bucket)')
    p.add_argument('--out')
    a = p.parse_args()
    texts = sample_texts()
    if a.long:
        glued, cur = [], ''
        for t in texts * 8:
            cur = (cur + '\n\n' + t) if cur else t
            if len(cur) >= a.long:
                glued.append(cur[:a.long]); cur = ''
        texts = glued
    need = a.batch * a.concurrency
    texts = (texts * (need // len(texts) + 1))[:need]
    parts = [texts[i * a.batch:(i + 1) * a.batch] for i in range(a.concurrency)]
    health = call('GET', '/health')
    w0 = workers()
    report = {'app_version': health.get('app_version'), 'model': health.get('model'), 'load1_start': round(os.getloadavg()[0], 1),
              'texts_per_round': need, 'chars_per_text_median': statistics.median(len(t) for t in texts), 'batch': a.batch, 'concurrency': a.concurrency,
              'workers': {k: v.get('weight') for k, v in w0.items()}}
    print(f"workers {report['workers']}, load {report['load1_start']}", flush=True)
    run('solo', parts[:1]); run('cluster', parts[:1])                     # warm both paths
    secs = {'solo': [], 'cluster': []}
    before = workers()
    solo_vecs = cluster_vecs = None
    for r in range(a.rounds):
        for mode in (('solo', 'cluster') if r % 2 == 0 else ('cluster', 'solo')):
            s, out = run(mode, parts)
            secs[mode].append(s)
            print(f'round {r + 1} {mode:7} {need / s:7.1f} texts/s ({s:.2f} s)', flush=True)
            if r == 0:
                if mode == 'solo': solo_vecs = out
                else: cluster_vecs = out
    after = workers()
    taken = {k: after[k]['texts'] - before.get(k, {}).get('texts', 0) for k in after}
    solo_rate = need / statistics.median(secs['solo'])
    cluster_rate = need / statistics.median(secs['cluster'])
    lo = min(cosmin(x, y) for x, y in zip(solo_vecs, cluster_vecs))
    report.update(solo_texts_per_s=round(solo_rate, 1), cluster_texts_per_s=round(cluster_rate, 1),
                  speedup=round(cluster_rate / solo_rate, 2), texts_taken_all_rounds=taken,
                  cluster_vs_solo_cosine_min=round(lo, 5), load1_end=round(os.getloadavg()[0], 1))
    print(f'solo {solo_rate:.1f} texts/s, cluster {cluster_rate:.1f} texts/s: x{cluster_rate / solo_rate:.2f}; '
          f'texts taken {taken}; cluster vs solo min cosine {lo:.5f}', flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
