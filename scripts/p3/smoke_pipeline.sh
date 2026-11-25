#!/usr/bin/env bash
# P4 / B2 second half: S1-S4 of `3.4_todo.md` §4.
#
#   S1  data re-check (G2 regression guard, 2 min, expected to pass)
#   S2  weight load + 2 real optimizer steps on one card, no FSDP:
#       the log is grepped for missing/unexpected keys, and the loss must be finite
#   S3  the step-1/step-2 checkpoints must reload through
#       `PreTrainedPolicy.from_pretrained`, and `train_config.json` must keep the
#       `robotwin_multitask` block (otherwise a resumed run loses its data root)
#   S4  what image resolution reaches the vision tower (`S4_shapes.txt`)
#
# Nothing here blocks the night: every step is recorded, failures are reported in
# `gates/P4_pipeline.json` and in `P3_REPORT.md`, the script exits 0 except for S2.
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

cd "$FLASHVLA_ROOT" || exit 1
S3_OUT="${S3_OUT:-$P3_LOG/s3_ckpt}"
rm -rf "$S3_OUT"

echo "== S1 data re-check =="
"$FV_PY" scripts/p2/check_lerobot.py --root "$DATA_ROOT" \
  --configs train/configs/pi05/simpler/pi05_flashvla_fractal.yaml \
            train/configs/pi05/simpler/pi05_flashvla_bridge.yaml \
  --skip-samples > "$P3_LOG/P4_s1_check_lerobot.log" 2>&1
s1=$?
echo "   S1 rc=$s1 ($(grep -c . "$P3_LOG/P4_s1_check_lerobot.log") log lines)"

echo "== S2 two optimizer steps, single card, no FSDP =="
CUDA_VISIBLE_DEVICES="$GPU" "$FV_PY" -m flashvla.train \
  --config_path="$SMOKE_CFG" \
  --output_dir="$S3_OUT" --steps=2 --save_freq=1 --log_freq=1 \
  --batch_size=1 --grad_accum_steps=1 --num_workers=0 --save_checkpoint=true \
  > "$P3_LOG/P4_s2_s3.log" 2>&1
s2=$?
key_warnings=$(grep -iE "missing key|unexpected key|size mismatch" "$P3_LOG/P4_s2_s3.log" | head -5)
if [ -n "$key_warnings" ]; then
  echo "   S2 state-dict warnings:"; echo "$key_warnings"
else
  echo "   S2 no missing/unexpected state-dict keys"
fi
"$FV_PY" scripts/p3/parse_train_log.py "$P3_LOG/P4_s2_s3.log" \
  --bs 1 --accum 1 --tag s2_single_step --rc "$s2" > "$P3_LOG/P4_s2.json"
grep -h '"loss"\|"ok"' "$P3_LOG/P4_s2.json" | tr -d ' '

echo "== S3 checkpoint reload =="
CUDA_VISIBLE_DEVICES="" "$FV_PY" - "$S3_OUT" <<'PY' > "$P3_LOG/P4_s3_reload.log" 2>&1
import json, os, sys

root = sys.argv[1]
last = os.path.join(root, "checkpoints", "last", "pretrained_model")
print("checkpoint:", last, "exists:", os.path.isdir(last))

import flashvla.configs  # noqa: F401  -- registers the "pi05-flashvla" config subclass
from lerobot.configs.policies import PreTrainedConfig
from flashvla.policies.pi05.modeling_pi05_flashvla import PI05FlashVLAPolicy

# Reload on the CPU on purpose: GPU4 is shared tonight, and this check only has to
# prove the checkpoint is structurally loadable.
policy_cfg = PreTrainedConfig.from_pretrained(last)
policy_cfg.device = "cpu"
policy = PI05FlashVLAPolicy.from_pretrained(last, config=policy_cfg)
print("reload OK:", type(policy).__name__, "device:", policy_cfg.device)

cfg = json.load(open(os.path.join(last, "train_config.json")))
print("train_config.robotwin_multitask:", json.dumps(cfg.get("robotwin_multitask")))
print("train_config.steps:", cfg.get("steps"), "batch_size:", cfg.get("batch_size"))
PY
s3=$?
tail -4 "$P3_LOG/P4_s3_reload.log"

echo "== S4 image resolution =="
CUDA_VISIBLE_DEVICES="" "$FV_PY" scripts/p3/check_shapes.py --config "$SMOKE_CFG" \
  > "$P3_LOG/S4_shapes.txt" 2>&1
s4=$?
grep -E "^\[(data|dataset|preprocessed|policy|json)\]" "$P3_LOG/S4_shapes.txt" | head -20

"$FV_PY" - "$P3_GATES/P4_pipeline.json" "$s1" "$s2" "$s3" "$s4" <<'PY'
import json, sys

out, s1, s2, s3, s4 = sys.argv[1], *(int(v) for v in sys.argv[2:6])
report = {"S1_data": s1, "S2_two_steps": s2, "S3_reload": s3, "S4_shapes": s4}
json.dump(report, open(out, "w"), indent=2)
print(json.dumps(report))
PY

[ "$s2" -eq 0 ] || { echo "S2 FAILED - see $P3_LOG/P4_s2_s3.log"; exit 1; }
echo "== S1-S4 done =="
