#!/usr/bin/env bash
# S0 variant 2: same single-process FSDP2 + cpu_offload run, but with the
# `adafactor` optimizer (registered in lerobot/optim/optimizers.py as part of the
# C track). Adafactor alone already fits the optimizer state, so this variant is
# the fallback when cpu_offload is slow or unsupported.
#
# Kept as a separate file only because `overnight.sh` and `make_report.py` refer to
# the two variants by name; the body is `smoke_s0.sh` with different defaults.
set -uo pipefail
cd "$(dirname "$0")" || exit 1

S0_TAG="${S0_TAG:-adafactor}" \
S0_OPTIMIZER="adafactor" \
S0_CPU_OFFLOAD="${S0_CPU_OFFLOAD:-true}" \
S0_BS="${S0_BS:-2}" \
S0_STEPS="${S0_STEPS:-20}" \
  exec bash ./smoke_s0.sh
