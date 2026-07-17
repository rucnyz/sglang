#!/bin/bash
# Render the long-horizon vs swarm "bubble" illustrative figure.
# Output: dev/figures/bubble_two_workloads.{png,pdf}
#
# Synthetic data — illustrates that a static HBM partition leaves a
# bubble whose direction flips with workload (long-horizon = KV-high,
# recurrent-low; swarm = recurrent-high, KV-low). Used in motivation
# section of the paper.

set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python -u dev/figures/bubble_two_workloads.py
