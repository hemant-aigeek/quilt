# Quilt

**Stitch spare machines into one language model.**

Quilt runs a transformer whose layers are spread across a mesh of ordinary
devices — a laptop, a desktop with an old GPU, a mini-PC — so the group can
serve a model none of them could hold alone. Every member keeps a replica of
the conversation journal and a warm standby copy of a neighbour's layers, so
when a machine disappears (and on a home or office network, one always does)
the conversation carries on from wherever it was.

```
$ quilt join --id desk  --secret s3cret                       # first member
$ quilt join --id lap   --secret s3cret --join desk:7700      # second
$ quilt join --id mini  --secret s3cret --join desk:7700      # third

$ quilt chat --url http://desk:7780
you> explain quilts
quilt> ...
   [128 tokens · 2.1s · plan v1 · term 1]
```

Pull the plug on `desk` mid-conversation and the next prompt is answered by
`lap` or `mini`, with the same history, a couple of seconds later.

## Why

Pipeline-parallel inference across machines is well understood. What makes it
fragile outside a datacenter is churn: lids close, Wi-Fi drops, phones sleep.
Most designs make one machine special — the one you typed into — and lose
everything when it goes. Quilt's premise is that *no member is special*:

- **Leadership is a role, not a machine.** The member driving generation
  holds a short lease; when it lapses, survivors pick a successor
  deterministically from the state they already have — no voting round.
- **The journal is everywhere.** Every completed turn is replicated to every
  member before the next one starts. A new leader has the full history.
- **Standbys are warm.** Each layer range has a second member holding a copy
  of its weights. Losing a worker is a one-line edit to the plan, not a
  re-partition and reload.
- **The cut is measured, not declared.** Members time a real decoder block on
  their own hardware at start-up; the planner balances the pipeline on those
  numbers and minimises the slowest stage.

Read the full [design document](docs/DESIGN.md).

## Install

```sh
pip install -e .            # core: torch, numpy, msgpack, aiohttp
pip install -e ".[hf]"      # + transformers, for real checkpoints
pip install -e ".[dev]"     # + pytest, ruff
```

Python 3.10+. CPU works out of the box; pass `--device cuda` or `--device mps`
to use an accelerator.

## Try it on one machine

No model download is needed: the built-in `toy` model generates its weights
from a seed, so every member derives identical parameters from the config
alone. It produces nonsense text, but it exercises every part of the system.

```sh
export QUILT_SECRET=demo
quilt join --id a --listen 127.0.0.1:7700 --api 127.0.0.1:7780 --advertise 127.0.0.1 &
quilt join --id b --listen 127.0.0.1:7701 --api 127.0.0.1:7781 --advertise 127.0.0.1 --join 127.0.0.1:7700 &
quilt join --id c --listen 127.0.0.1:7702 --api 127.0.0.1:7782 --advertise 127.0.0.1 --join 127.0.0.1:7700 &

quilt chat --url http://127.0.0.1:7781 --prompt "hello" --max-tokens 32
open http://127.0.0.1:7782          # dashboard: members, plan, journal, events
kill %1                             # drop the leader; chat again
```

## Run a real model

Any Llama-family checkpoint from the Hugging Face Hub (Llama, Qwen2, Mistral,
TinyLlama…):

```sh
quilt join --id desk --secret s3cret --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype float16
quilt join --id lap  --secret s3cret --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype float16 --join desk:7700
```

Each member loads only the layers it owns (plus its standby copy). Use
`--memory-gib` to cap what a member lends and `quilt bench` to see what a
machine would contribute before joining.

> The Hugging Face backend tracks the `transformers` decoder-layer calling
> convention, which has shifted across releases; it targets 4.45+ and
> normalises the two common shapes. Treat it as experimental.

## Command line

| Command | Purpose |
|---------|---------|
| `quilt join` | Start a member. `--join host:port` to find an existing quilt; the first member joins nobody. |
| `quilt chat` | REPL or one-shot prompt against any member's API; follows leadership redirects. |
| `quilt plan profiles.json` | Preview the layer layout the planner would produce for a set of device profiles. |
| `quilt bench` | Measure this machine's memory and layer rate. |

## HTTP API

Any member serves it. A member that is not the leader answers chat requests
with a `307` to the leader's address, so clients follow leadership wherever
it moves.

| Route | |
|-------|-|
| `POST /v1/chat` | `{"session", "prompt", "max_tokens", "temperature", "top_k"}` → newline-delimited JSON: `{"token": "…"}` events, then `{"done": true, "text", "tokens", "seconds", "plan_version", "term"}`. |
| `GET /v1/mesh` | Members, profiles, current plan, leader and term, journal sequence, recent events. |
| `GET /v1/sessions/{id}` | The turns of one session from the journal. |
| `DELETE /v1/sessions/{id}` | Forget a session and drop its KV caches on every member. |
| `GET /` | Dashboard. |

## How it works, in one screen

```
 client ──► leader: embed prompt ─► patch 0 ─► patch 1 ─► … ─► last patch ─► logits
                     ▲                 (hidden states hop member to member)      │
                     └──────────────── sample a token, stream it, repeat ◄──────┘
                     └─ after the turn: append to journal, replicate to all members
```

- **Planning**: minimise the slowest patch subject to memory; bisect over the
  finite set of candidate time budgets. The leader hosts patch 0 so prompt
  tokens never leave the machine they were typed on.
- **Execution**: hidden states travel directly from patch to patch; only the
  final logits return to the leader. Each patch keeps a per-session KV cache
  and refuses a step whose position does not match — caches are rebuilt from
  the journal after any disruption, never trusted across a plan change.
- **Failover**: `LEAD` heartbeats carry the term; pings carry each member's
  journal sequence. A lapsed lease triggers deterministic succession by
  `(journal seq, layer rate, node id)`. The winner bumps the term, reconciles
  its journal with survivors, and hands orphaned patches to their standbys.

## Tests

```sh
pytest
```

Covers the planner, the wire codec, pipeline-vs-whole-model equivalence,
incremental decoding, and multi-member scenarios in-process: leader loss,
worker loss with standby takeover, late joiners, and rejected secrets.

## Scope

Quilt is for devices you trust on a network you control — your own machines,
a team's lab, a classroom. The shared secret gates membership; after that,
members see each other's activations and the journal in the clear. It is
sized for a room, not the internet, and it favours availability over strong
consistency under partition. See the design document for the reasoning.

## License

Apache-2.0.
