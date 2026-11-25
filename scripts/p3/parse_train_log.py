#!/usr/bin/env python
"""Turn one FlashVLA training log into the numbers S0 / S5 need.

The trainer prints one line per `log_freq` steps via `MetricsTracker`:

    step:20 smpl:640 ep:2 epch:0.01 loss:1.234 grdn:0.512 lr:5.0e-05 updt_s:1.812 data_s:0.004

`updt_s` is the pure compute time of one optimizer step (all micro-batches) and
`data_s` the dataloading time of the last micro-batch, so

    samples/s = batch_size * grad_accum_steps / (updt_s + data_s)

Usage:
    python scripts/p3/parse_train_log.py <log> --bs 2 --accum 16 --tag s0_x [--rc 0]
"""

import argparse
import json
import math
import re

FIELD = {
    "step": r"step:\s*([0-9]+)",
    "loss": r"loss:\s*(-?[0-9.eE+]+)",
    "updt_s": r"updt_s:\s*([0-9.eE+]+)",
    "data_s": r"data_s:\s*([0-9.eE+]+)",
}


def grab(pattern: str, line: str, cast=float):
    match = re.search(pattern, line)
    return cast(match.group(1)) if match else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--rc", type=int, default=0)
    ap.add_argument("--optimizer", default="")
    ap.add_argument("--cpu-offload", default="")
    ap.add_argument("--fsdp", default="")
    args = ap.parse_args()

    with open(args.log, errors="ignore") as f:
        lines = [line for line in f if "updt_s:" in line]

    result = {
        "tag": args.tag,
        "log": args.log,
        "ok": False,
        "rc": args.rc,
        "batch_size": args.bs,
        "grad_accum_steps": args.accum,
        "optimizer": args.optimizer,
        "cpu_offload": args.cpu_offload,
        "fsdp": args.fsdp,
        "log_lines": len(lines),
    }

    if lines:
        last = lines[-1]
        updt = grab(FIELD["updt_s"], last)
        data = grab(FIELD["data_s"], last) or 0.0
        loss = grab(FIELD["loss"], last)
        step_time = (updt or 0.0) + data
        result.update(
            {
                "steps": grab(FIELD["step"], last, int),
                "loss": loss,
                "updt_s": updt,
                "data_s": data,
                "s_per_step": step_time,
                "samples_per_s": (args.bs * args.accum / step_time) if step_time > 0 else 0.0,
            }
        )
        result["ok"] = (
            args.rc == 0
            and loss is not None
            and math.isfinite(loss)
            and step_time > 0
        )
    elif args.rc != 0:
        tail = ""
        with open(args.log, errors="ignore") as f:
            tail = "".join(f.readlines()[-5:]).strip()
        result["error_tail"] = tail

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
