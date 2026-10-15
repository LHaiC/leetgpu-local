#!/usr/bin/env bash
# Build the sandboxed judge image used by --sandbox mode.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${JUDGE_IMAGE:-leetgpu-judge}"

docker build -f "$ROOT/docker/Dockerfile.judge" -t "$IMAGE" "$ROOT"
echo "Built $IMAGE"
echo "Start the server with: python scripts/local_web.py --sandbox"
