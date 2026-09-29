"""Phase 0, R1: read-only checks on the RGBPose-SlowFast reference code (no training).

Checks
  1. Does modelrgbposeslowfast.py run unmodified? (it has no `import torch`)
  2. Temporal / spatial shapes at every lateral connection, using the clip lengths that
     main.py actually passes to the DataLoader (NoOfFrms2slct_s=4, NoOfFrms2slct_f=20)
     and a sweep of other lengths. Flags any add that silently broadcasts over time.
  3. BatchNorm eval-mode behaviour: does a sample's eval-mode output depend on the other
     samples in its batch?
  4. Parameter counts of both variants.
  5. pyflakes static check of all R1 files (undefined names, unused imports).

Usage:
  python tools/phase0/r1_readonly_checks.py --repo third_party/rgbpose_slowfast \
      --out results/phase0/r1_checks.json
"""
import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

import torch


def load_model_module(repo: Path, inject_torch: bool):
    src = (repo / "modelrgbposeslowfast.py").read_text()
    mod = types.ModuleType("modelrgbposeslowfast")
    if inject_torch:
        # Minimal shim: the only change is making the name `torch` resolvable.
        mod.__dict__["torch"] = torch
    exec(compile(src, str(repo / "modelrgbposeslowfast.py"), "exec"), mod.__dict__)
    return mod


def run_unmodified(repo: Path):
    mod = load_model_module(repo, inject_torch=False)
    net = mod.RgbPoseSlowFastResNet34(num_classes=3).eval()
    x1 = torch.randn(1, 3, 4, 224, 224)
    x2 = torch.randn(1, 17, 20, 56, 56)
    try:
        with torch.no_grad():
            net(x1, x2)
        return {"runs_unmodified": True}
    except Exception as e:  # noqa: BLE001 - we want to record whatever happens
        return {"runs_unmodified": False, "error": f"{type(e).__name__}: {e}"}


def trace_shapes(net, t_slow, t_fast, rgb_hw=224, pose_hw=56, batch=2):
    rec = {}
    names = ["relu_fast", "layer1_slow", "layer2_slow", "layer3_slow", "layer4_slow",
             "layer2_fast", "layer3_fast", "layer4_fast"]
    names += [f"lateral{k}_{s}" for k in (1, 2, 3) for s in ("slow", "fast")]
    hooks = []
    for n in names:
        m = getattr(net, n)
        hooks.append(m.register_forward_hook(
            lambda _m, _i, o, n=n: rec.__setitem__(n, list(o.shape))))
    x1 = torch.randn(batch, 3, t_slow, rgb_hw, rgb_hw)
    x2 = torch.randn(batch, 17, t_fast, pose_hw, pose_hw)
    out = {"t_slow_in": t_slow, "t_fast_in": t_fast}
    try:
        with torch.no_grad():
            y = net(x1, x2)
        out["ok"] = True
        out["logits"] = list(y.shape)
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
    for h in hooks:
        h.remove()
    out["shapes"] = rec
    # Pair up the tensors that are summed in forward():
    #   x1 = x1 + lateral{k}_fast(x2)   (x1 = output of layer{k}_slow)
    #   x2 = lateral{k}_slow(x1) + x2   (x2 = output of relu_fast / layer{k}_fast)
    fast_src = {1: "relu_fast", 2: "layer2_fast", 3: "layer3_fast"}
    adds = []
    for k in (1, 2, 3):
        for target, lat in ((f"layer{k}_slow", f"lateral{k}_fast"),
                            (fast_src[k], f"lateral{k}_slow")):
            if target in rec and lat in rec:
                a, b = rec[target], rec[lat]
                adds.append({
                    "add": f"{target} + {lat}",
                    "target_shape": a, "lateral_shape": b,
                    "temporal_match": a[2] == b[2],
                    "silent_temporal_broadcast": a[2] != b[2] and 1 in (a[2], b[2]),
                })
    out["adds"] = adds
    return out


def bn_eval_batch_dependence(net, t_slow, t_fast):
    net.eval()
    torch.manual_seed(0)
    x1 = torch.randn(4, 3, t_slow, 224, 224)
    x2 = torch.randn(4, 17, t_fast, 56, 56)
    x1b = torch.randn(4, 3, t_slow, 224, 224) * 3 + 1  # different batch companions
    x2b = torch.randn(4, 17, t_fast, 56, 56) * 3 + 1
    with torch.no_grad():
        alone = net(x1[:1], x2[:1])[0]
        in_batch = net(x1, x2)[0]
        other_batch = net(torch.cat([x1[:1], x1b[1:]]), torch.cat([x2[:1], x2b[1:]]))[0]
    return {
        "max_abs_logit_diff_alone_vs_batch": float((alone - in_batch).abs().max()),
        "max_abs_logit_diff_batch_vs_other_companions": float((in_batch - other_batch).abs().max()),
    }


def bn_config(net):
    cfg = {}
    for n, m in net.named_modules():
        if isinstance(m, torch.nn.BatchNorm3d):
            key = f"track_running_stats={m.track_running_stats},momentum={m.momentum}"
            cfg[key] = cfg.get(key, 0) + 1
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("third_party/rgbpose_slowfast"))
    ap.add_argument("--out", type=Path, default=Path("results/phase0/r1_checks.json"))
    args = ap.parse_args()
    torch.manual_seed(0)

    commit = subprocess.run(["git", "-C", str(args.repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    report = {"repo_commit": commit, "torch": torch.__version__}
    report["unmodified_import"] = run_unmodified(args.repo)

    mod = load_model_module(args.repo, inject_torch=True)
    report["shim"] = "only `torch` injected into the module namespace; no other change"
    variants = {
        "ResNet34_style (used by main.py:163)":
            lambda: mod.RgbPoseSlowFastResNet34(num_classes=3),
        "ResNet50_style":
            lambda: mod.RgbPoseSlowFastResNet50(num_classes=3),
    }
    # (4, 20) are the values main.py:115-116 passes to the DataLoader.
    sweep = [(4, 20), (4, 32), (8, 32), (8, 40), (16, 64)]
    report["variants"] = {}
    for vname, ctor in variants.items():
        net = ctor().eval()
        v = {
            "params_total": sum(p.numel() for p in net.parameters()),
            "batchnorm3d_config_counts": bn_config(net),
            "shape_sweep": [trace_shapes(net, ts, tf) for ts, tf in sweep],
            "bn_eval_batch_dependence_T4_20": bn_eval_batch_dependence(net, 4, 20),
        }
        report["variants"][vname] = v
        print(f"[{vname}] params={v['params_total']:,}")
        for s in v["shape_sweep"]:
            flags = [a["add"] for a in s["adds"] if a["silent_temporal_broadcast"]]
            print(f"  T_slow={s['t_slow_in']:>2} T_fast={s['t_fast_in']:>2} ok={s['ok']}"
                  f" broadcast_adds={flags} {s.get('error', '')}")
        print("  BN eval dependence:", v["bn_eval_batch_dependence_T4_20"])

    try:
        pf = subprocess.run([sys.executable, "-m", "pyflakes",
                             *sorted(str(p) for p in args.repo.glob("*.py"))],
                            capture_output=True, text=True)
        report["pyflakes"] = [ln.replace(str(args.repo) + "/", "")
                              for ln in pf.stdout.splitlines()]
    except FileNotFoundError:
        report["pyflakes"] = "NOT RUN (pyflakes not installed)"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
