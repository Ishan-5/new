# SLO-Aware Cross-Request Pipelined Batching (XRPB)
### A concurrent scheduling algorithm for U-shaped edge-cloud collaborative LLM inference

---

## 1. Problem restatement

In U-shaped edge-cloud split inference, the first and last model layers run on the edge (keeping raw inputs/outputs private), while the compute-heavy middle layers run in the cloud. Every prefill and decode step requires an edge → cloud → edge round trip. Processing requests strictly sequentially creates "bubbles" — periods where edge or cloud compute sits idle waiting on the other side or on the network.

**Goal:** a concurrent scheduling strategy that (1) maximizes system throughput and (2) ensures every request meets its latency SLO.

---

## 2. System model

- Model layers `1..L`. Edge holds layer 1 (embedding) and layer `L` (final block + LM head). Cloud holds layers `2..L-1`.
- Per token: edge encode → network up → cloud middle-layer forward → network down → edge decode/sample.
- Relevant SLOs: **TTFT** (time to first token) and **TPOT/ITL** (inter-token latency during decode).
- **Bubble** = idle edge or cloud time caused by processing requests one at a time instead of overlapping them.

**Core insight:** a single decode-step forward pass for one request is tiny relative to GPU capacity (decode is memory-bandwidth-bound, not compute-bound). Network RTT is close to a fixed cost that can't be shrunk much. So the lever isn't making one round trip faster — it's making one round trip serve many requests at once (batching), and never leaving edge or cloud idle while waiting (pipelining).

---

## 3. Architecture

```
        ACTIVATIONS (up)
 +-------------------+   ------------->   +---------------------------+
 |   EDGE DEVICE      |                   |      CLOUD CLUSTER         |
 |  (per user, x N)   |                   |   (shared across users)    |
 |                    |   <-------------  |                             |
 |  Embed + layer 1   |      RESULTS      |  Middle layers (batched)    |
 |  Layer L + sampling|      (down)       |  SLO scheduler + paged KV   |
 +-------------------+                   +---------------------------+
```

Only activations cross the network; raw prompts/tokens never leave the edge device. The cloud side is a single shared service: because all connected edge devices use the same base model and split point, their per-token activations can be batched together in one middle-layer forward pass — this is the main throughput lever.

---

## 4. Algorithm components

1. **Iteration-level cloud batching** (continuous batching, cf. Orca / vLLM). The cloud runs an event-driven loop: each tick, gather all activations that have arrived from any edge device, run one batched forward pass through the middle layers, stream results back as they complete. Never wait for a "full" batch or for one specific request.

2. **Chunked prefill** (cf. Sarathi-Serve / SplitFuse). Long prompts are split into fixed-size token chunks so one large prefill can't monopolize a batch slot and starve other requests' decode steps. This is what prevents head-of-line blocking.

3. **SLO-aware batch admission**: within each cloud iteration, decode steps (cheap, latency-critical) are admitted first; remaining compute budget is filled with prefill chunks ordered by slack = `deadline - now - estimated_remaining_work` (earliest-deadline-first). This is a greedy heuristic, not a provably optimal packing — see Section 7.

4. **Asynchronous edge pipelining**: the edge never blocks on one session's cloud reply before starting the next ready session's compute. Requires holding up to `K` concurrent sessions' local KV-cache slices in edge memory, bounded by device capacity.

5. **Paged KV cache on cloud** (cf. PagedAttention/vLLM), so batch composition can change every iteration across many concurrent sessions without memory fragmentation.

6. **Admission control**: reject or queue new requests when a calibrated load model predicts the SLO can't be met (see B* below), instead of letting the system degrade for everyone under overload.

7. **(Optional / stretch) Edge-resident speculative decoding**: a small draft model running entirely on-edge proposes γ tokens; one round trip verifies all γ at once. This is the only lever here that reduces the *number* of round trips per token rather than just hiding their cost — but it depends on having a draft model compatible with the target model and a workable acceptance rate. Not required for the core design.

---

## 5. Scheduler pseudocode

```python
# ---- Cloud (single shared instance, event-driven) ----
loop every tick:
    ready = inbound_queue.pop_all()
    decode_reqs    = [r for r in ready if r.stage == DECODE]      # 1 token each
    prefill_chunks = [r for r in ready if r.stage == PREFILL]     # chunked

    batch  = decode_reqs[:]                          # decode admitted first
    budget = SLO_TPOT_floor - net_rtt_estimate        # remaining compute budget
    prefill_chunks.sort(key=slack)                    # EDF: soonest deadline first

    for chunk in prefill_chunks:
        if cost_model(batch + [chunk]) <= budget and len(batch) < B_max:
            batch.append(chunk)
        # else: deferred to next tick, not dropped

    if not batch:
        continue

    outputs = middle_layers.forward(batch, paged_kv_cache)
    for req, out in zip(batch, outputs):
        async_send(req.edge_addr, out)                # non-blocking
        paged_kv_cache.update(req, out)


# ---- Edge (per device, async) ----
on new_prompt(p):
    for chunk in split(p, chunk_size):
        act = first_layer.forward(chunk)
        enqueue_send(cloud, act)                       # non-blocking, don't wait

on cloud_response(req_id, activation):
    out   = last_layer.forward(activation)
    token = sample(out)
    emit(token)
    if not done:
        enqueue_send(cloud, first_layer.forward(token))  # next decode step
    # edge remains free to service other sessions between these events
```

---

## 6. Performance analysis

Let per-token cloud compute time for a batch of size `B` be approximately linear:

```
C(B) ≈ c0 + c1·B         (decode is memory-bandwidth-bound, batching is
                           nearly free until bandwidth/compute limits)
```

Let `r` = network RTT (up + down); edge compute is treated as negligible relative to `C(B)` and `r`.

- **System throughput** ≈ `B / C(B)` tokens/sec.
- **Per-request latency** ≈ `C(B) + r` (roughly independent of `B` for any one request riding in the batch).
- **SLO constraint**: `C(B) + r ≤ SLO_TPOT`, which gives the maximum safe concurrency:

```
B* = floor( (SLO_TPOT - r - c0) / c1 )
```

`B*` is the number to hand to admission control: the system should not admit more concurrent decode-active requests than `B*` without violating the TPOT SLO.

**Illustrative example only (not measured — replace with profiled numbers before citing):**
if `r = 20ms`, `SLO_TPOT = 50ms`, `C(B) = 5ms + 0.3ms·B`, then
`B* = floor((50 − 20 − 5) / 0.3) ≈ 83` concurrent requests.

TTFT/prefill follows the same shape, with chunk size as the tunable knob against the decode-SLO budget of co-scheduled requests in the same batch.

---

## 7. Limitations and risks (read before submitting as-is)

- **Every component is an existing technique**, not a new one: continuous batching (Orca/vLLM), chunked prefill (Sarathi-Serve), paged KV cache (vLLM), deadline scheduling (FastServe / real-time scheduling theory), prefill-decode disaggregation (DistServe/Splitwise). The contribution here is applying this combination to the edge-cloud U-split, not inventing new primitives. If judges know the LLM-serving literature, they will recognize the sources.
- **The RTT model is optimistic.** Treating `r` as a constant ignores that edge networks (especially wireless/mobile) have bursty, heavy-tailed RTT. A real system should budget against p95/p99 RTT, not the mean, or measured SLO attainment will be worse than the formula predicts.
- **EDF + greedy-fill is a heuristic, not optimal.** Cloud compute time is a nonlinear function of batch composition (total tokens, not just request count), so the true packing problem is a bin-packing/knapsack variant — NP-hard in general. A learned or ILP-based scheduler could outperform this heuristic on a benchmark.
- **Cross-request batching assumes a shared base model and split point.** If per-user adapters (e.g., LoRA) or different models are allowed, batching breaks and needs heterogeneous-batching support (e.g., S-LoRA-style), which is not covered here.
- **Speculative decoding is the least de-risked component.** It's the most theoretically direct fix for the round-trip problem, but depends on a compatible, well-calibrated draft model. Treat as optional; don't claim it as implemented unless it is.
- **Privacy is a property of the U-split itself** (given by the problem architecture), not a contribution of this scheduler. The scheduler's obligation is to not break it — in particular, cross-user batching should not introduce timing side channels that leak information across users. This is a risk worth naming, not one solved here.

---

## 8. What's missing for a complete submission

- Empirical results from a simulator or testbed: throughput, SLO attainment rate, p50/p95/p99 latency, and idle/bubble time fraction, before vs. after each component.
- A comparison against at least a naive FCFS/sequential baseline, and ideally one stronger baseline (e.g., continuous batching without SLO-awareness) to isolate what each component actually contributes.
- Sensitivity analysis across network conditions — this is the design's weakest assumption (Section 7).
- A working implementation. Algorithm design alone rarely wins systems contests; implementation quality and honest empirical results typically carry more weight than the write-up.

---

## 9. Bottom line

This design correctly identifies the mechanism behind the bubble problem (serialized round trips) and addresses it with defensible, literature-grounded techniques plus a concrete latency/throughput derivation (`B*`) rather than hand-waving. It is a synthesis of known methods applied to a new setting, not a novel algorithm — whether that satisfies the contest's judging criteria depends on the rubric. It does not by itself guarantee a win; that depends on implementation and measured results, which this document does not include.
