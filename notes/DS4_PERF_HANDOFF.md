# DS4 Flash 0731 — 2-Node DGX Spark Performance Handoff

**Date:** 2026-08-25 (~08:50 UTC, session interrupted mid-optimization)
**Goal:** match vLLM on the same hardware (~50 tok/s single stream, 300-500 with concurrency).
**Current measured:** 0.5–0.6 tok/s (0.6 → after batching fix, unverified with the latest async-H2D change).
**Output correctness:** VERIFIED — "Hi" → "Hello! How can I help you today" on 2-node TP with NVFP4 KV.

---

## Environment (verified)

- `spark1` = 192.168.0.211, `spark2` = 192.168.0.212 (ssh aliases, `~/.config/NVIDIA/Sync/config/ssh_config`, nvsync.key).
- Both GB10 (sm_121), 121.7GB unified, CUDA 13.0, aarch64, NCCL 2.31.2.
- RoCE fabric: `10.0.1.1` (spark1) / `10.0.1.2` (spark2); management net `192.168.0.x`.
- Model: `~/models/ds4-flash-0731` (symlink to HF cache; 155.4GB, 48 shards). Config: hidden=4096, 43 layers, 256 routed experts (128 local per rank), moe_intermediate=2048, q_lora_rank=1024, 64 heads.
- Zig: spark1 `~/zig/zig`, spark2 `~/zig-aarch64-linux-0.16.0/zig`. Repo rsynced to `~/agave` on both.
- **GPU clocks are capped to idle (208 MHz) by the `gb10-clockcap` docker container.** Fix (passwordless sudo, survives until reboot/container):
  `sudo -n nvidia-smi -lgc 2418` on BOTH nodes. Verify with `nvidia-smi --query-gpu=clocks.sm --format=csv,noheader` (should be ~2411+). Without this, every GPU op is ~10x slower.
- vLLM containers were `docker stop vllm-ds4-0731` on both nodes (they hold ~95GB GPU each; if restarted, agave's CUDA init fails).

## Launch procedure (rank 0 FIRST — rank 1's connect has NO retry)

```bash
# spark1 (rank 0):
cd ~/agave && NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=5 \
  NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1 NCCL_IB_AR_THRESHOLD=0 \
  NCCL_NET_GDR_LEVEL=3 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_IB_RETRY_CNT=7 \
  NCCL_IB_TIMEOUT=22 nohup ./zig-out/bin/agave ~/models/ds4-flash-0731 \
  --backend cuda --tp 2 --rank 0 --peers 10.0.1.2 --transport nccl \
  --kv-type nvfp4_ds_mla -n 8 "Hi" > /tmp/x-tp0.log 2>&1 &
# wait for "waiting for rank 1 on port 49454..." in the log (~4-5 min cold load), THEN:
# spark2 (rank 1): same command with --rank 1 --peers 10.0.1.1 > /tmp/x-tp1.log
```

The NCCL env vars are REQUIRED (without them the allReduce is slow / wrong NIC). The launch timing is critical: launch rank 1 only after rank 0's listener line appears.

## What was done (all uncommitted locally; git HEAD = 31fc61c)

### Load path (cold load ~155GB → ~4-5 min)
- `src/format/safetensors.zig`:
  - Shard mmaps get `MADV.RANDOM` (decode-time demand paging; SEQUENTIAL would over-read whole shards).
  - `fuseDs4Flash0731` sets SEQUENTIAL on all shards during the fuse (scattered fp8 repack reads at ~22MB/s otherwise), restores RANDOM on exit.
- `src/main.zig`: preload (touches all 155GB) kept; it's ~4 min cold, fast warm.
- The old `cuMemHostRegister` UMA pinning is DISABLED (`max_uma_regions = 0`): the pin faulted cold pages at ~19MB/s (23 min for 8 shards) and locked 26GB. `registerHostRegion` now only records shard ranges for hint management.

### Resident working set (the core fix — kills the demand-page thrash)
- `src/backend/cuda.zig` `residentWeight()`: CPU-touch the source (fast fault path) → `cuMemcpyHtoD` into 8GiB chunks (`resident_chunks[16]`) → `madvise(DONTNEED)` the source (clean pages; device copy is the source of truth). Address-ordered per shard for sequential disk reads. `resident_map` (host addr → device ptr) checked first by `getOrUpload`.
- `src/models/deepseek4.zig` `prefaultLocalExperts()` (called from main.zig after `setTpTransport`): device-copies ALL local expert weights + E8M0 scales (16512 ranges, ~66GB/rank), PLUS the residual mmap'd GPU-read tensors (router `ffn_gate_inp`, HC weights, `output.weight`). Sets `experts_resident = true` and calls `restoreMmapHints()`.
- `restoreMmapHints()`: RANDOM hint + DONTNEED on all tracked shard ranges — frees the ~50GB of dead non-expert mmap pages that were keeping the 121GB machine at the limit (causing multi-second cuMemAlloc stalls at every layer's combine sync).
- FFN's per-token `prefetchRange` (madvise WILLNEED) is SKIPPED when `experts_resident` (the host pages are gone; prefetching re-reads the working set from disk per token).
- HC gemv moved to GPU (`hcPreCpu` uses `doGemv` on CUDA instead of direct CPU mmap reads).

### GPU execution path
- **Dedicated CUDA stream** (`stream` field; kernels + async copies on it). Blocking copies on the legacy null stream cost ~2ms per call on GB10.
- **Deferred-sync batching actually wired**: the DS4 FFN/attention used `self.computeBackend().beginBatch()`, which is the CPU backend's NO-OP on CUDA — every gemv synced individually (~2ms each). Changed to `self.gemvBackend().beginBatch()/endBatch()` (4 call sites in deepseek4.zig: wo_a group, FFN phase 1, FFN phase 3). `syncGemvOutput` defers into a pending list; `endBatch` drains with async D2Hs + ONE stream sync. Drains measured 0-1ms.
- Hot-path H2D re-uploads (getInputBuf/getInPlaceBuf/findContaining stale-refresh + uploadToDevice) are now ASYNC on the stream (ordered before the consuming kernel). **This is the last change; NOT yet measured on the nodes.**
- **A 256MiB bump pool was tried and REVERTED** (its first-touch page allocation became a new tax; direct cuMemAlloc is fine).

### Instrumentation (TEMP, remove before final commit)
- `DS4PERF` per-token attn/ffn/hc ms in deepseek4.zig forward.
- `FFNPERF` per-phase (norm/route/phase1/silu/phase3/combine) for layers 0-2 in ffnLayer.
- `TPPERF` allReduce latency (transport.zig, every 40th call).
- `CUDA: N launches, avg Xµs` (cuda.zig launch timer) + `CUDA drain: Xms for N copies, act_cache M entries`.

## Measured progression (decode, 2-node TP, -n 8, warm)

| Stage | tok/s | ffn/token | attn/token | Notes |
|---|---|---|---|---|
| Before (demand-paged mmap) | ~0.01 | 8-30s | ~400ms | page-fault thrash |
| Resident + DONTNEED-all + residual | 0.2 | 5-7s | ~400ms | still stalled |
| + NCCL RoCE env + prefetch skip + HC GPU | 0.2-0.6 | 1.3-2.3s | ~400ms | allReduce 1ms; multi-second stalls gone |
| + batching wired (gemvBackend) + stream | 0.6 | ~1.5s | ~400ms | drains 0-1ms; launches 3µs; GPU 44% util |
| + async H2D re-uploads | **UNMEASURED** | ? | ? | the current build |

**The remaining bottleneck hypothesis** (all CPU-side blocking ops eliminated except...): the FFN per-layer is ~30-90ms even with fast launches/drains/kernels, alternating between phase1 and combine — the sum per layer is the constant. The async-H2D change targets the stale→re-upload path (down-gemv inputs etc.). If that doesn't fix it, the next suspects:
1. The act_cache's per-address growth (comp buffers: ~+8 entries/layer, each a cuMemAlloc) — bounded at ~11K entries; the allocations may still stall under pressure. Consider pre-registering the csa_comp circular buffers.
2. `findContaining` linear scans over the growing act_cache.
3. The kernel execution itself at the (now locked) clocks — verify with `nvidia-smi --query-gpu=utilization.gpu` during decode.

## Next steps (in order)

1. **Sync + rebuild both nodes** with the async-H2D build (currently only LOCAL; `zig build` passes):
   ```bash
   for n in spark1 spark2; do rsync -az -e ssh src/backend/cuda.zig $n:~/agave/src/backend/cuda.zig; done
   # then rebuild both (zig build) and relaunch via the procedure above
   ```
2. Verify the decode numbers (expect the FFN to drop toward ~100-300ms if the re-uploads were the tax).
3. If still slow: instrument `getInputBuf`/`getOutputBuf` misses (count + time cuMemAlloc + findContaining), and/or pre-register the csa_comp buffers.
4. **Batched expert gemv kernel** (Phase1 gate/up → 1 launch, Phase3 down → 1 launch): the launch count is 25/layer; batching to ~4 cuts the remaining per-op overhead ~6x. Kernel source goes in `src/backend/kernels/cuda/gemv_mxfp4_st_batched.zig`, add to `build.zig` kernel_files + `all.zig`, then `zig build ptx -Dcuda-sm=sm_121` + copy `zig-out/ptx/*.ptx` to `src/backend/kernels/cuda/` (all.ptx is the embedded aggregate — regenerate via the all.zig build-obj path or concatenate).
5. **GPU SDPA/kvDot kernel** for attention (currently CPU per-head kvDot; ~400ms now, grows with context).
6. Concurrency/numseq testing, then remove the TEMP instrumentation and commit.

## Gotchas

- `pkill -f "zig-out/bin/aga[v]e /home"` — the `[v]` avoids killing your own ssh command line.
- The gauntlet/concurrent bots edit this repo and auto-commit — re-read files before editing; `scripts/build-web.sh` is THEIR change, exclude it from any commit.
- The `zig build test` suite is broken by an abandoned sdpa fuzz (Zig 0.16 codegen bug) — don't chase it.
- Memory budget: 66GB resident chunks + ~26GB heap (repacked attention) + small act/KV ≈ 95-108GB used; ~12GB headroom. Any new big allocation can stall.
- On UMA the KV cache is zero-copy (host pointer IS the device pointer via `registerRamKv`) — no KV uploads in the hot path.
- The `gb10-clockcap` container may reset the GPU clocks on reboot — re-apply `sudo -n nvidia-smi -lgc 2418`.
