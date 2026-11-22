# Chippy — text and image embeddings on the Apple Neural Engine, and a pool of Macs that share the work

Chippy is a native Swift embedding server for Apple silicon. It runs EmbeddingGemma on the **Apple Neural
Engine (ANE)** through Core ML, serves an Ollama-compatible `/api/embed`, sits in the menu bar, and can
spread work over other Macs that **volunteer** their ANE, BOINC style: they pull chunks from a shared
queue, so a faster Mac simply does more.

- `app/` — the Swift package: `ane-embed` CLI and server, `ANEEmbedBar` menu bar app (Chippy), the pool,
  volunteers, and a Rust tokenizer (HF `tokenizers` over a C ABI)
- `converters/` — PyTorch → Core ML converters: EmbeddingGemma (300M, `eg1`) and EmbeddingGemma 2 (`eg2`):
  text cores, and EmbeddingGemma 2's vision encoder in two layouts (`v1` the model's own layers, `v2`
  rewritten in Apple's ANE layout, about 1.4x faster)
- `bench/` — solo vs cluster benchmark, a cheating-volunteer test, and `test_all.sh`

Written by ane, an AI oracle (Claude, via Claude Code), working with Nat Weerawan. Every number below was
measured on our machines; yours will differ.

## What it does

| | |
|---|---|
| Text embeddings | EmbeddingGemma (300M, W8A16) or EmbeddingGemma 2 (768-d) on the ANE, length buckets 32…2048 tokens |
| Image embeddings | EmbeddingGemma 2 vision encoder on the ANE, same vector space as the text |
| API | `POST /api/embed {"model", "input": [...]}` or `{"images": [base64...]}`, Ollama-shaped answer |
| Pool | `"pool": "solo"` (this Mac only) or `"cluster"` (default: shared queue over every volunteer) |
| Volunteers | `ane-embed join anejoin://HOST:PORT/TOKEN` — pulls 32-text chunks, 2 at a time (prefetch) |
| Trust | every volunteer chunk is spot-checked (cosine ≥ 0.995 against this Mac); 3 bad answers and it is cut off |

Measured (M5 Max + M6 + M4 MacBook Air, Wi-Fi, a busy machine — read them as rough):
short texts solo 103 → cluster 139 texts/s (x1.35); long texts (~6,000 chars) 10.1 → 19.9 texts/s (x1.97);
cluster vectors identical to solo (min cosine 1.00000).

## Requirements

- Apple silicon Mac, **macOS 15 or later** (the Core ML models target macOS 15; an M1 on an older macOS
  fails with `libswift_Concurrency.dylib` not found)
- Xcode command line tools (Swift 6), Rust (`cargo`) for the tokenizer
- For converting models: [`uv`](https://github.com/astral-sh/uv), and access to `google/embeddinggemma-2`
  on Hugging Face (accept its license)

## Build

```sh
cd app
(cd tokenizer-ffi && cargo build --release)
swift build -c release            # .build/release/ane-embed and ANEEmbedBar
swift test
APP_NAME=Chippy APP_ID=com.example.chippy zsh make_app.sh   # Chippy.app, version = CalVer of the build day
```

## Make the model assets

EmbeddingGemma (300M) — the menu bar app's primary model:

```sh
cd converters
./eg1 check                        # the ANE-layout graph in fp32 vs sentence-transformers: must be 1.000000
./eg1 convert --buckets all        # 9 Core ML packages, int8 conv weights (W8A16), ~1 min each
./eg1 parity --bucket 128x16       # ANE vectors vs sentence-transformers fp32 (we get median 0.9993)
./eg1 export                       # -> ~/Library/Application Support/ANEEmbed/embeddinggemma-w16
```

`--model google/embeddinggemma-300m` is the default (gated: accept its license on Hugging Face);
`--model unsloth/embeddinggemma-300m` is an ungated mirror with the same weights. Measured on our M5 Max:
vectors from these packages match our long-running production packages at cosine 1.00000 (short and
~6,000-character texts), so an index built with them stays valid. Why the graph looks the way it does
(residual stream scaled for fp16, int8 weights — fp16 weights drift to cosine 0.991 on the ANE) is in
the docstring of `converters/convert_gemma1.py`.

EmbeddingGemma 2 (served beside it, and the only model with images):

```sh
./eg2 convert                      # text core -> Core ML packages
./eg2 parity                       # ANE vectors vs sentence-transformers fp32
./eg2 export                       # -> ~/Library/Application Support/ANEEmbed/embeddinggemma2-w16
./eg2 vision2 convert              # optional: images (vision encoder, ANE layout)
```

Parity and benchmarks use a built-in public sample set (`converters/samples.py`). For honest numbers point
`SAMPLE_TEXTS=your.json` (`{"texts": [...]}`) at your own real text: real text is longer and messier.
`eg1 export` refuses to overwrite assets a running Chippy may use unless you pass `--force`.

## Run

```sh
app/.build/release/ane-embed serve --port 11445   # EmbeddingGemma from ANEEmbed/embeddinggemma-w16
curl -s 127.0.0.1:11445/api/embed -d '{"input": ["title: none | text: hello", "title: none | text: สวัสดี"]}'
curl -s 127.0.0.1:11445/api/embed -d '{"input": ["hello", "world"], "pool": "solo"}'
```

Volunteers: the server prints a join line (`serve --join`, add `--lan` for the local network). On another
Mac: `curl -fsS 'http://HOST:11445/join.sh?t=TOKEN' | NAME=my-mac zsh` installs the binary and model in
`~/ane-volunteer` and starts pulling work. The join listener only answers the volunteer paths (403 for the
rest). **Keep the token secret** — a volunteer sees the texts it embeds.

macOS Local Network privacy blocks processes started by launchd or some terminal multiplexers from
reaching LAN addresses; start the volunteer from Terminal or an ssh session, or allow the app under
System Settings › Privacy & Security › Local Network.

## Test

```sh
python3 bench/pool_bench.py                                   # solo vs cluster, alternating order
python3 bench/pool_bench.py --long 6000 --batch 64 --rounds 3 # long texts
python3 bench/cheater_test.py                                 # a lying volunteer must be cut off
zsh bench/test_all.sh                                         # everything (sends embed load)
```

ANE measurements are only meaningful on a quiet machine; the benches print the load average.

## License

Apache-2.0. EmbeddingGemma weights are Google's and come under their own terms; they are not in this repo.
