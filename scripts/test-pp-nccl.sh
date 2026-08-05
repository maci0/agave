#!/bin/bash
# Test PP=2 NCCL between two GB10 nodes.
# Usage: ./scripts/test-pp-nccl.sh <rank> <peer_ip> <model_path>
# Example:
#   Node A: ./scripts/test-pp-nccl.sh 0 10.0.1.2 ~/Qwen3.5-0.8B-Q8_0.gguf
#   Node B: ./scripts/test-pp-nccl.sh 1 10.0.1.1 ~/Qwen3.5-0.8B-Q8_0.gguf
#
# Start rank 1 FIRST (downstream), then rank 0.

set -euo pipefail

RANK=${1:?Usage: $0 <rank> <peer_ip> <model_path>}
PEER=${2:?Missing peer IP}
MODEL=${3:?Missing model path}
N_TOKENS=${4:-20}

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

echo "=== PP=2 NCCL Test ==="
echo "Rank: $RANK"
echo "Peer: $PEER"
echo "Model: $MODEL"
echo "Tokens: $N_TOKENS"
echo ""

./zig-out/bin/agave "$MODEL" \
    --pp 2 \
    --rank "$RANK" \
    --peers "$PEER" \
    --transport nccl \
    -n "$N_TOKENS" \
    "What is quantum computing?"
