#!/usr/bin/env bash
# Test TP=2 (expert-parallel) over NCCL between two GB10 (DGX Spark) nodes.
# Usage: ./scripts/test-tp-nccl.sh <rank> <peer_ip> <model_path> [n_tokens]
# Example:
#   Node A: ./scripts/test-tp-nccl.sh 0 10.0.1.2 ~/DeepSeek-V4-Flash-0731-MLX-Q4/
#   Node B: ./scripts/test-tp-nccl.sh 1 10.0.1.1 ~/DeepSeek-V4-Flash-0731-MLX-Q4/
#
# TP=2 shards routed experts round-robin across the pair (isLocalExpert) and
# all-reduces the hidden state after each FFN. Attention stays redundant on
# both ranks. Rank 0 owns stdout; both ranks must run the SAME model path.
#
# Start rank 1 FIRST (downstream), then rank 0.
#
# Requires: zig build (with CUDA enabled, default), libnccl.so.2 on both nodes.

set -euo pipefail

RANK=${1:?Usage: $0 <rank> <peer_ip> <model_path>}
PEER=${2:?Missing peer IP}
MODEL=${3:?Missing model path (GGUF file or SafeTensors dir)}
N_TOKENS=${4:-20}
BACKEND=${BACKEND:-cuda}
TRANSPORT=${TRANSPORT:-nccl}
KV_TYPE=${KV_TYPE:-q8_0}

# NCCL configuration for ConnectX RoCE RDMA (override via env for other NICs).
export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-5}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1,enP2p1s0f1np1}"
export NCCL_IB_AR_THRESHOLD="${NCCL_IB_AR_THRESHOLD:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-3}"
export NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING:-1}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

echo "=== TP=2 NCCL Test (DGX Spark) ==="
echo "Rank: $RANK / 2"
echo "Peer: $PEER"
echo "Model: $MODEL"
echo "Backend: $BACKEND  Transport: $TRANSPORT  KV: $KV_TYPE"
echo "Tokens: $N_TOKENS"
echo ""

# DS4 Flash 0731 on GB10: sm_121 PTX is JIT'd from the built kernels; keep the
# -Dcuda-sm used at build time compatible (sm_90 default JITs to sm_121 too).
./zig-out/bin/agave "$MODEL" \
    --backend "$BACKEND" \
    --tp 2 \
    --rank "$RANK" \
    --peers "$PEER" \
    --transport "$TRANSPORT" \
    --kv-type "$KV_TYPE" \
    -n "$N_TOKENS" \
    "What is quantum computing?"

echo ""
echo "=== TP=2 rank $RANK done ==="
