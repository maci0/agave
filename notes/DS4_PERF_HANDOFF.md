# DS4 Flash 0731 — 2-Node DGX Spark Performance Handoff

**Goal:** match vLLM on 2× DGX Spark (~50 tok/s single stream, 300-500 concurrent).
**Status:** batched expert gemv kernel + async heap uploads validated on the tiny-random model (**272 tok/s**); the full Flash 2-node decode measurement is BLOCKED by a spark2 ssh outage (needs a reboot).

## Environment (verified)

- `spark1` = 192.168.0.211, `spark2` = 192.168.0.212 (ssh aliases, nvsync.key).
- GB10 (sm_121), 121.7GB unified, CUDA 13.0, aarch64, NCCL 2.31.2. RoCE: `10.0.1.1`/`10.0.1.2`; management `192.168.0.x`.
- Model: `~/models/ds4-flash-0731` (155.4GB, 48 shards; hidden=4096, 43 layers, 256 experts, ff=2048).
- Zig: spark1 `~/zig/zig`, spark2 `~/zig-aarch64-linux-0.16.0/zig`. Repo rsynced to `~/agave`.
- **GPU clocks:** `sudo -n nvidia-smi -lgc 2418,2418` on both nodes (min=max keeps the memory/P0 state up). The `gb10-clockcap` container re-caps to `-lgc 0,2200` every 300s — `docker update --restart=no gb10-clockcap && docker stop gb10-clockcap`.
- **CRITICAL: `docker stop vllm-ds4-0731` on BOTH nodes before benchmarking** — the vLLM container re-holds ~95GB GPU memory and blocks agave with `cuMemAlloc` error 700 (OOM) even with 105GB system free.

## Launch procedure (rank 0 FIRST — rank 1's connect has NO retry)

```bash
# spark1 (rank 0): NCCL env + nohup ./zig-out/bin/agave ~/models/ds4-flash-0731 \
#   --backend cuda --tp 2 --rank 0 --peers 10.0.1.2 --transport nccl \
#   --kv-type nvfp4_ds_mla -n 16 "Hi" > /tmp/x-tp0.log 2>&1 &
# wait for "waiting for rank 1 on port 49454..." THEN spark2 (rank 1, --peers 10.0.1.1).
NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=5 \
NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1 NCCL_IB_AR_THRESHOLD=0 \
NCCL_NET_GDR_LEVEL=3 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_IB_RETRY_CNT=7 NCCL_IB_TIMEOUT=22
```

## Key changes (committed `7bb600e`)

1. **Batched expert gemv kernel** (`gemv_mxfp4_st_batched.zig` + PTX + `gemvMxfp4StBatched` in cuda.zig + ffnLayer wiring): one launch for all active experts' gate+up / down via device pointer tables. The sustained memory traffic keeps the GB10 memory clock ramped (per-expert 25µs bursts left it idle, 4.2MB reads costing 2-5ms each). Tiny model: FFN 26ms → 1ms/layer, 39 → 272 tok/s.
2. **Async H2D re-uploads for heap act buffers** (getInputBuf/getInPlaceBuf/findContaining): blocking null-stream copies ~2ms each. Async H2D FAILS (`CUDA_ERROR_INVALID_VALUE`) on mmap'd memory — weights (getOrUpload) stay blocking.
3. **Free large repacked host buffers after their device upload** (`fmt.freeRepackedTensor` in doGemv) — the ~26GB fp8→bf16 attention repacks are dead weight once uploaded (prevents the OOM).
4. Fuse: SEQUENTIAL repack reads, then RANDOM + DONTNEED the shard pages (SEQUENTIAL readahead otherwise leaves ~155GB RSS and OOMs the resident copy).
5. Out-of-range expert routing ids skipped (tiny-random emits ids up to 240 vs 128-expert tables) — the CPU path always tolerated this; CUDA now skips too.
6. Resident machinery (session 1, commit `dda7bd3`): device-copy local experts + scales + residual weights, DONTNEED the dead mmap pages, no UMA pinning, dedicated CUDA stream + deferred-sync batching (drains measured 0-1ms).

## Gotchas

- **spark2's sshd hangs under load** (banner timeout, ping OK) — recovery has required a reboot (the node came back "up 2 min"). Ask the user to reboot it if unreachable.
- The concurrent gauntlet bot edits/commits the repo (it deleted this file once) — re-read before editing; exclude `scripts/build-web.sh` and untracked bot files from commits.
- `pkill -f "zig-out/bin/aga[v]e /home"` (the `[v]` avoids killing your own ssh).
- `zig build test` is broken by an abandoned sdpa fuzz — don't chase it.

## Next steps

1. Reboot/verify spark2, then the standard 2-node launch → measure the Flash decode with the batched kernel (expect the FFN to drop from ~2s toward the bandwidth floor).
2. If the attention (~400ms) persists, the per-call syncs (q_a→rmsNorm→q_b chains) are the tax — batch or GPU-ize the attention gemvs.
3. GPU SDPA/kvDot kernel for the attention scores (CPU now — grows with context).
4. Remove the TEMP instrumentation (DS4PERF/FFNPERF/TPPERF/CUDA timers), then commit.
5. Concurrency/numseq testing (`--numseq`, server mode) toward the 300-500 concurrent target.
