#!/bin/bash
# Convenience script to test/run Mooncake with correct environment settings
# Usage: source scripts/test_mooncake_env.sh

export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
export MC_GID_INDEX=0
export MC_IB_PORT=1

# For RDMA mode, you also need unlimited memlock:
#   sudo bash -c 'ulimit -l unlimited; source scripts/test_mooncake_env.sh; python ...'

echo "Mooncake environment configured:"
echo "  LD_PRELOAD=$LD_PRELOAD"
echo "  MC_GID_INDEX=$MC_GID_INDEX"
echo "  MC_IB_PORT=$MC_IB_PORT"
