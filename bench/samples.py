"""Sample texts for parity checks and benchmarks.

If SAMPLE_TEXTS (or ./sample-texts.json) points at a JSON file {"texts": [...]}, those are used —
use your own real text for honest numbers, since real text is longer and messier than this set.
Otherwise a built-in public set is used: short and long English and Thai prose, code, SQL, logs,
mixed scripts, expanded deterministically to the count asked for.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

BASE = [
    "The Apple Neural Engine runs fixed-shape Core ML models at a fraction of the GPU's power.",
    "Embeddings turn text into vectors so that similar meanings land close together.",
    "A volunteer Mac asks the pool for work, embeds a chunk on its own ANE, and posts the vectors back.",
    "Work stealing: every worker takes the next chunk from one shared queue whenever it is free.",
    "SELECT id, title FROM notes WHERE created_at > now() - interval '7 days' ORDER BY created_at DESC;",
    "def cosine(a, b):\n    return sum(x * y for x, y in zip(a, b))",
    "func embed(_ texts: [String]) async throws -> [[Float]] { try await engine.embed(texts) }",
    "2026-11-22 10:30:01 INFO pool: chunk 7 claimed by pull:mba (32 texts), 0.41 s",
    "error: Library not loaded: /usr/lib/swift/libswift_Concurrency.dylib",
    "curl -fsS http://127.0.0.1:11445/health | python3 -m json.tool",
    "เครื่อง Mac หลายเครื่องช่วยกันฝังเวกเตอร์ได้ ถ้าแบ่งงานเป็นชิ้นเล็กแล้วให้ใครว่างก็หยิบไป",
    "Neural Engine ทำงานกับโมเดลที่ shape ตายตัว จึงต้องแบ่งข้อความตามความยาวเป็น bucket",
    "วันนี้ฝนตกหนักตั้งแต่เช้า รถติดยาวไปถึงสี่แยก แต่ร้านกาแฟหน้าออฟฟิศยังเปิดตามปกติ",
    "ถ้า volunteer ตอบเวกเตอร์ผิด spot check จะจับได้ และหลังจากผิดสามครั้งจะไม่ได้งานอีก",
    "สูตรแกงเขียวหวาน: พริกแกง กะทิ ไก่ มะเขือเปราะ ใบโหระพา น้ำปลา น้ำตาลปี๊บ",
    "Mixed: deploy เสร็จแล้ว แต่ health check ตอบ 503 อยู่สองนาที ก่อนจะกลับมาเป็น 200",
    "Meeting notes: agreed to ship the jump box behind Caddy; only volunteer paths pass through.",
    "The quick brown fox jumps over the lazy dog while the build cache warms up.",
    "Long-context retrieval needs chunking, overlap and a good prompt prefix for documents.",
    "task: search result | query: how do volunteers join the pool",
    "title: none | text: BOINC clients pull work units from a scheduler and report results.",
    "Q3 revenue grew 12% quarter over quarter, driven by subscriptions in Southeast Asia.",
    "ขอบคุณสำหรับข้อมูลครับ เดี๋ยวผมลองรันใหม่บนเครื่องที่เงียบกว่านี้แล้วจะส่งตัวเลขให้",
    "A 2048-token bucket costs far more than a 64-token one, so padding waste matters.",
]
TOPICS = ["the scheduler", "a cold cache", "the tokenizer", "Thai word breaks", "the vision encoder",
          "an idle laptop", "the shared queue", "a timeout", "fp16 overflow", "the jump box",
          "ตารางเวลา", "คิวกลาง", "เครื่องที่ช้าที่สุด", "ไฟล์ log", "ภาพหน้าจอ"]


def _expand(n: int) -> list[str]:
    out, i = [], 0
    while len(out) < n:
        base = BASE[i % len(BASE)]
        k = i // len(BASE)
        if k == 0:
            out.append(base)
        else:
            t = TOPICS[(i * 7 + k) % len(TOPICS)]
            reps = 1 + (i * 13 + k) % 6           # lengths vary, like real text
            out.append(f"{base} (note {k}: about {t}) " + " ".join([base] * (reps - 1)))
        i += 1
    return out


def sample_texts(n: int | None = None) -> list[str]:
    path = os.environ.get("SAMPLE_TEXTS") or ("sample-texts.json" if Path("sample-texts.json").exists() else "")
    texts = json.load(open(path))["texts"] if path else _expand(1024)
    return texts if n is None else (texts * (n // max(1, len(texts)) + 1))[:n]
