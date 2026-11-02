# Quilt — design

Quilt turns a handful of ordinary machines into one inference engine for a
language model that none of them could hold alone. This document explains the
shape of the system, the decisions behind it, and what it deliberately does
not do.

## 1. Problem

A modern open-weight model is tens of gigabytes. A laptop, a desktop with an
old GPU, and a mini-PC under the TV each have a few gigabytes to spare. Taken
together they could run the model — if the layers were spread across them and
activations could be passed from one to the next quickly enough.

Pipeline-parallel inference across machines is not new. What makes it painful
in practice on a home or office network is *churn*: laptops close, phones
sleep, Wi-Fi drops. A design that treats one machine as special — the one you
typed the prompt into, say — loses the whole conversation when that machine
goes away. Quilt is built around the assumption that any member, including the
one driving a conversation, can vanish at any moment.

## 2. Vocabulary

| Term | Meaning |
|------|---------|
| **Member** | One process on one device, lending memory and compute. |
| **Quilt** | The set of members that share a secret and a model. |
| **Patch** | A contiguous range of transformer layers pinned to one member. |
| **Plan** | The ordered list of patches that covers every layer, plus a version number. |
| **Leader** | The member currently driving generation and writing the journal. |
| **Journal** | The append-only, replicated log of conversation turns. |
| **Term** | A monotonically increasing leadership epoch. |

## 3. Architecture

```
          ┌──────────── member A (leader) ────────────┐
 client ──│ HTTP API ─► generate() ─► patch 0 (L0-3)  │
          │              ▲            │ hidden state  │
          │   journal ◄──┘            ▼               │
          └───────────────────────────┼───────────────┘
                                      │ FORWARD (hop 1)
          ┌──────────── member B ─────▼───────────────┐
          │ journal replica     patch 1 (L4-6)        │
          │ standby copy of patch 0                   │
          └───────────────────────────┼───────────────┘
                                      │ FORWARD (hop 2)
          ┌──────────── member C ─────▼───────────────┐
          │ journal replica     patch 2 (L7) + head   │──► LOGITS back to A
          │ standby copy of patch 1                   │
          └───────────────────────────────────────────┘
```

Every member runs the same code and the same three roles at once:

1. **Patch host** — holds the layers the plan assigns to it (and a warm copy
   of a patch it stands by for), keeps a per-session KV cache, and executes
   `FORWARD` requests.
2. **Journal replica** — accepts and stores every turn the leader commits.
3. **Leader candidate** — watches the leader's lease and takes over when it
   lapses, if it is the best-ranked survivor.

The member that happens to be leader additionally plans the layout, drives
the token loop, and serves chat requests. Leadership is a *role*, not a
*machine*.

### 3.1 Module map

| Module | Responsibility |
|--------|----------------|
| `quilt/protocol.py` | Length-prefixed msgpack frames; inline tensor codec; message types. |
| `quilt/transport.py` | Authenticated TCP links; bidirectional request/response multiplexing. |
| `quilt/membership.py` | Full-mesh peer links, gossip of addresses and profiles, liveness. |
| `quilt/profile.py` | Device profile: memory lent and a *measured* layer rate. |
| `quilt/planner.py` | Bottleneck-minimising partition of layers into patches; standbys; fast re-plan after a loss. |
| `quilt/journal.py` | Append-only turn log with global sequence numbers. |
| `quilt/node.py` | The member: patch hosting, pipeline hops, leadership, generation. |
| `quilt/model/` | Backends: a deterministic built-in model (`toy`) and Hugging Face Llama-family checkpoints (`hf`). |
| `quilt/api.py`, `quilt/dashboard.html` | HTTP API (streaming chat, status) and a one-page dashboard. |
| `quilt/cli.py` | `quilt join / chat / plan / bench`. |

## 4. Wire protocol

Frames are `uint32 length || msgpack map`. Each map has a `t` (type). A
request carries an integer `rid`; the reply echoes `rid` with `reply: true`,
so one persistent link carries many in-flight calls in both directions —
there is no separate client/server socket pair per direction.

Tensors are shipped as `{dtype, shape, data}` with the raw buffer in a msgpack
`bin`. No pickle; bf16 is converted to fp16 on the wire because numpy has no
bf16.

Joining is gated by a shared secret. Both sides send a nonce and prove the
secret with `HMAC-SHA256(secret, nonce || node_id)`. The secret is a
membership gate, not a confidentiality layer — see §9.

## 5. Membership

The quilt is meant for a handful of devices, so every member keeps a link to
every other member. New members learn the rest through gossip: each hello and
each periodic gossip message carries known addresses, and unknown ones are
dialled. Liveness is a one-second ping; a member whose link drops or whose
pongs stop for `dead_after` seconds is considered gone and a membership
change is fired.

Pings double as a cheap state exchange: each carries the sender's journal
sequence and current term, which is all the election needs (§8).

## 6. Planning

### 6.1 Profiles

A member advertises two numbers: bytes of memory it will lend and a **layer
rate** — how many decoder layers of this model's width it can run per second
for one token. The rate is *measured at start-up* by timing a synthetic block,
not declared from a spec sheet, so a throttled laptop and a desktop with the
same CPU get different numbers.

### 6.2 The cut

The pipeline's throughput is set by its slowest patch, so the planner
minimises that. Given a time budget `T`, a member with rate `r` may take at
most `⌊T·r⌋` layers, and never more than fit in its memory (accounting for
embedding and head weights on the first and last patch, and a KV reservation
per layer). Members are visited in a fixed order and each takes as many
layers as it may. The smallest feasible `T` is found by bisecting the finite
set of candidate budgets `{k / rᵢ}`.

The member driving generation is placed first when possible, so the prompt's
tokens are embedded locally and only hidden states cross the network.

### 6.3 Standbys

Every patch is also assigned a **standby**: a different member with enough
spare memory to keep a warm copy of the patch's weights. Standbys load their
copy when the plan is applied. If the owner disappears, the standby becomes
the owner by a one-line plan edit — no re-partitioning, no weight loading on
the critical path. If a patch has no live standby the planner falls back to a
full re-plan over the survivors.

## 7. Execution

Generation is a chain, not a hub-and-spoke:

1. The leader sends `FORWARD{session, plan_version, hop=0, start_pos, tokens}`
   to the owner of patch 0.
2. Patch 0 embeds the tokens, runs its layers, and forwards the hidden state
   to the owner of patch 1 — directly, without returning to the leader.
3. The last patch applies the final norm and head and replies with the logits
   of the last position; the reply unwinds through the same request chain.
4. The leader samples, streams the token to the client, and repeats.

Each patch keeps a KV cache per session, keyed by `session_id`, and checks
that `start_pos` equals the cache length it holds — a mismatch is an error,
not a silent corruption. The leader tracks how many tokens of each session the
pipeline has cached (`_fed`) and sends only the delta: the whole context on
the first step of a turn (or after any disruption), one token thereafter.

If any hop fails — a link drops, a plan version mismatches because a re-plan
raced the step — the leader clears its `_fed` record for the session, waits
for the plan to be acknowledged again, and replays the full context. KV
caches are rebuilt from the journal, never trusted across a plan change.

## 8. Journal and leadership

### 8.1 Journal

The leader is the only writer. After each turn it appends two entries (the
user's prompt and the model's reply, with their token ids) and broadcasts them
to all members. Each entry carries a global `seq`; a replica that receives a
gap asks for a snapshot. A member that joins late pulls everything from the
leader. The journal can optionally be persisted to disk so a member survives
its own restart with its history intact.

Sessions are views over the journal: a session's context is simply the
concatenation of its entries' tokens, which is what the leader feeds the
pipeline at the start of a turn.

### 8.2 Lease

The leader broadcasts `LEAD{term, seq, plan}` every `lead_interval` seconds.
A follower that hears nothing for `lead_timeout` seconds, or whose link to
the leader drops, runs an election.

### 8.3 Deterministic succession

There is no voting round. Each survivor ranks the members it can still reach
by `(journal seq, layer rate, node id)` and the maximum wins. Because every
member learns everyone's `seq` from pings, survivors compute the same answer
from the same inputs. The winner raises its term to `max(known terms) + 1`,
starts broadcasting `LEAD`, reconciles its journal against any survivor that
holds a later entry, and re-plans.

A member that computed a different winner marks that candidate as *suspect*
if it fails to announce itself within one lease, and excludes it next time —
so a stuck candidate cannot wedge the quilt. Two leaders that collide on the
same term (a healed partition, or a member that crowned itself before it found
the mesh) are resolved by the same ranking: everyone yields to the
better-ranked claimant. A stale leader that reappears hears a higher term and
steps down on its next `LEAD` exchange.

What the design trades away: with a partitioned network, each side may elect
a leader. Quilt favours availability — a conversation continues on whichever
side you can reach — and reconciles journals by `seq` when the partition
heals. It is not a consensus system and does not pretend to be one; for the
intended use (your own devices on your own network) that is the right trade.

### 8.4 What survives

| Event | Result |
|-------|--------|
| A worker (non-leader) dies | Its standby takes the patch; in-flight step is replayed; no turn is lost. |
| The leader dies mid-turn | A survivor takes over within ~2 s; the committed journal is intact; the interrupted turn must be re-sent by the client. |
| A member joins | The leader re-plans to use it; it receives the journal. |
| A member restarts with a persisted journal | It rejoins with its `seq` and is ranked accordingly. |

## 9. Security model

The shared secret controls who may join. Everything else — hidden states,
logits, the journal — is visible to every member. Quilt is for devices you
trust: your own machines, a team's lab, a classroom. Transport encryption is
a reasonable future addition (TLS with the secret as a PSK) but it would not
change the trust model, since any member sees the plaintext anyway.

## 10. The built-in model

`--model toy` is a small decoder-only transformer (RMSNorm, causal attention
with a KV cache, SiLU MLP, no positional embedding — the causal mask over
absolute positions is enough for the plumbing it exists to exercise) whose weights are generated from a seed.
Every member derives identical parameters from the config alone, so the
entire system — planning, chaining, failover, journal replay — runs and is
tested on any machine with no download. It generates nonsense; that is not
the point. The Hugging Face backend plugs real Llama-family checkpoints into
exactly the same `PatchRuntime` interface.

## 11. Non-goals

- Thousands of members. The full mesh and gossip are sized for a room, not
  the internet.
- Byzantine members. The secret gates entry; after that, members are trusted.
- Tensor parallelism. Patches are layer ranges; a single layer is never split.
- Strong consistency under partition (see §8.3).

## 12. Testing

The test suite exercises each layer and the whole:

- the planner balances by measured speed, respects memory, hands orphaned
  patches to standbys and re-plans when it cannot;
- a split pipeline produces the same logits as the unsplit model, and
  incremental decoding matches full recomputation;
- three in-process members plan, chain, and replicate; killing the leader
  mid-conversation yields a new leader with the same journal and a session
  that continues; killing a worker triggers the standby; a late joiner
  receives the journal and is planned in; a wrong secret is rejected.
