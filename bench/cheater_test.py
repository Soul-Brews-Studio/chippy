"""A volunteer that lies, against the work-stealing pool: its answers must never reach a caller.

The cheater claims chunks and returns random unit vectors (right shape, right length, wrong
meaning: only the spot check can catch them). Meanwhile cluster requests run. Checks:
  - every cluster answer matches the solo answer for the same texts (cosine >= 0.999)
  - the cheater is cut off: after 3 bad answers its claim is refused (403)

    python3 bench/cheater_test.py
"""
from __future__ import annotations

import http.client
import json
import random
import threading
from pathlib import Path

from samples import sample_texts

REPO = Path(__file__).resolve().parents[1]
PORT = 11445
TOKEN = json.load(open(Path.home() / 'Library/Application Support/ANEEmbed/join.json'))['token']


def call(path, body=None, method='POST'):
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=300)
    c.request(method, path, json.dumps(body) if body is not None else None, {'Content-Type': 'application/json'})
    r = c.getresponse()
    return r.status, json.loads(r.read() or b'{}')


def cheat(identity, log, stop):
    while not stop.is_set():
        s, job = call('/api/work/claim', {'token': TOKEN, 'worker': 'cheater', 'identity': identity, 'wait': 5})
        if s == 204:
            continue
        if s != 200:
            log.append(('claim', s, job.get('error')))
            return
        vecs = []
        for _ in job['input']:
            v = [random.gauss(0, 1) for _ in range(768)]
            n = sum(x * x for x in v) ** 0.5
            vecs.append([x / n for x in v])
        s2, reply = call('/api/work/result', {'token': TOKEN, 'worker': 'cheater', 'id': job['id'], 'embeddings': vecs})
        log.append(('result', s2, reply.get('error')))


def main():
    identity = call('/health', method='GET')[1]['vector_space_identity']
    texts = sample_texts(240)
    log, stop = [], threading.Event()
    t = threading.Thread(target=cheat, args=(identity, log, stop), daemon=True)
    t.start()
    worst = 1.0
    for _ in range(6):
        _, solo = call('/api/embed', {'input': texts, 'pool': 'solo'})
        _, clus = call('/api/embed', {'input': texts, 'pool': 'cluster'})
        for a, b in zip(solo['embeddings'], clus['embeddings']):
            worst = min(worst, sum(x * y for x, y in zip(a, b)))
        if any(e[0] == 'claim' for e in log):
            break
    stop.set(); t.join(10)
    w = {x['id']: x for x in call('/api/workers', method='GET')[1]['workers']}.get('pull:cheater', {})
    print('cheater log:', log)
    print('cheater worker:', {k: w.get(k) for k in ('texts', 'errors', 'weight', 'lastError')})
    ok_vectors = worst >= 0.999
    ok_cut = any(e[0] == 'claim' and e[1] == 403 for e in log)
    print(f'cluster vs solo min cosine {worst:.5f}: {"OK" if ok_vectors else "FAIL"}; cheater cut off: {"OK" if ok_cut else "FAIL"}')
    raise SystemExit(0 if ok_vectors and ok_cut else 1)


if __name__ == '__main__':
    main()
