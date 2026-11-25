#!/usr/bin/env bash
# Render `_overnight.yaml` + `gates/decision.json` -> `_overnight.generated.yaml`.
#
# The template is never edited and the main configs
# (`pi05_flashvla_bridge.yaml` / `..._fractal.yaml`) are never touched: only the
# three values that S5 measured are injected, so the offline smoke and the real
# run keep using the same field names.
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

DECISION="$P3_GATES/decision.json"
TEMPLATE="$FLASHVLA_ROOT/train/configs/pi05/simpler/_overnight.yaml"

[ -f "$DECISION" ] || { echo "render_cfg: missing $DECISION (run sweep_bs.sh first)"; exit 1; }

"$FV_PY" - "$TEMPLATE" "$DECISION" "$OVERNIGHT_CFG" "$BASE_CKPT" <<'PY'
import json
import sys

import yaml

template, decision_path, out_path, base_ckpt = sys.argv[1:5]
cfg = yaml.safe_load(open(template))
dec = json.load(open(decision_path))

bs = int(dec.get("bs") or 1)
cfg["batch_size"] = bs
cfg["grad_accum_steps"] = int(dec.get("grad_accum_steps") or max(1, 32 // bs))
cfg["optimizer"]["type"] = dec.get("optimizer") or "adamw"
# `betas` only exists on the Adam configs; draccus refuses it for adafactor.
if cfg["optimizer"]["type"] not in {"adam", "adamw", "multi_adam", "xvla-adamw"}:
    cfg["optimizer"].pop("betas", None)
cfg["fsdp"]["cpu_offload"] = str(dec.get("cpu_offload", "true")).lower() == "true"
# On one rank there is nothing to shard, and FSDP2's fp32 master weights plus
# fp32 gradient buffers cost ~30 GB for nothing; S0/S5 decide whether to keep it.
cfg["fsdp"]["enable"] = str(dec.get("fsdp", "true")).lower() == "true"
cfg["policy"]["pretrained_path"] = base_ckpt

with open(out_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print(f"rendered {out_path}")
print(f"  batch_size       = {cfg['batch_size']}")
print(f"  grad_accum_steps = {cfg['grad_accum_steps']}  (equiv batch "
      f"{cfg['batch_size'] * cfg['grad_accum_steps']})")
print(f"  optimizer.type   = {cfg['optimizer']['type']}")
print(f"  fsdp.cpu_offload = {cfg['fsdp']['cpu_offload']}")
print(f"  pretrained_path  = {cfg['policy']['pretrained_path']}")
print(f"  steps            = {cfg['steps']}  save_freq={cfg['save_freq']}")
PY
