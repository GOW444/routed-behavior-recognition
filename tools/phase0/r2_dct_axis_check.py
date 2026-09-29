"""Phase 0, R2: resolve which axis the DCT and the high/low split index act on in the
official FreqMixFormer code (https://github.com/wenhanwu95/FreqMixFormer).

Method (read-only, no training):
  1. Confirm torch_dct.dct / idct act on the LAST tensor dimension (unit test).
  2. Run the repo's Spatial_Freq_MixFormer block on CPU on an NTU-shaped input
     (N, C, T, V) = (2, 80, 64, 25), with torch_dct.dct / idct wrapped so every call logs
     its input shape. The last dim of each logged shape is the axis being transformed.
     CPU shim: the block hard-codes CUDA (spa_freq_mixf.py:95 `.cuda(...)`, :97
     device='cuda'); those two strings are replaced before exec. Nothing else changes.
  3. Record the hard-coded operator constants (l, h, d) found in the source, with line.

Usage:
  python tools/phase0/r2_dct_axis_check.py --repo third_party/freqmixformer \
      --out results/phase0/r2_dct_axis.json
"""
import argparse
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch_dct


def dct_last_dim_unit_test():
    torch.manual_seed(0)
    # Signal constant along V (last dim) but varying along T: a DCT over the last dim
    # puts all energy in coefficient 0; a DCT over T would not.
    x = torch.randn(2, 4, 16, 1).repeat(1, 1, 1, 25)          # (N, C, T, V)
    X = torch_dct.dct(x)
    energy_nonzero_coeffs = float(X[..., 1:].abs().max())
    roundtrip = float((torch_dct.idct(X) - x).abs().max())
    return {
        "input_shape": list(x.shape),
        "max_abs_coeff_k>=1_along_last_dim": energy_nonzero_coeffs,
        "idct(dct(x)) max error": roundtrip,
        "conclusion": ("torch_dct.dct transforms the LAST dim"
                       if energy_nonzero_coeffs < 1e-4 else "UNEXPECTED"),
    }


def run_block_with_logging(repo: Path):
    sys.path.insert(0, str(repo))
    src_path = repo / "model" / "spa_freq_mixf.py"
    src = src_path.read_text()
    shim = {".cuda(x0.get_device())": "", "device='cuda'": "device=x0.device"}
    for a, b in shim.items():
        assert a in src, f"expected string not found: {a}"
        src = src.replace(a, b)

    calls = []
    real_dct, real_idct = torch_dct.dct, torch_dct.idct
    fake = types.SimpleNamespace(
        dct=lambda t, norm=None: (calls.append(("dct", list(t.shape))), real_dct(t, norm))[1],
        idct=lambda t, norm=None: (calls.append(("idct", list(t.shape))), real_idct(t, norm))[1],
    )
    mod = types.ModuleType("spa_freq_mixf_cpu")
    exec(compile(src, str(src_path), "exec"), mod.__dict__)
    mod.dct = fake  # the file does `import torch_dct as dct`

    from graph.ntu_rgb_d import Graph  # noqa: E402  (repo module)
    A = Graph(labeling_mode="spatial").A
    N, C, T, V = 2, 80, 64, 25
    block = mod.Spatial_Freq_MixFormer(C, C, A).eval()
    with torch.no_grad():
        y = block(torch.randn(N, C, T, V))
    uniq = sorted({(k, tuple(s)) for k, s in calls})
    return {
        "shim": shim,
        "input_NCTV": [N, C, T, V],
        "output_shape": list(y.shape),
        "n_calls": len(calls),
        "unique_calls(kind, input_shape)": [[k, list(s)] for k, s in uniq],
        "axis_transformed": {
            k: ("V (joints)" if s[-1] == V and len(s) == 4 and s[-2] == T else
                "last dim of a VxV attention map (joint index w)" if s[-1] == s[-2] == V else
                f"last dim, size {s[-1]}")
            for k, s in uniq
        },
    }


def hardcoded_constants(repo: Path):
    out = {}
    for i, line in enumerate((repo / "model" / "spa_freq_mixf.py").read_text().splitlines(), 1):
        m = re.match(r"\s*([lhd])\s*=\s*([0-9.]+)", line)
        if m:
            out[m.group(1)] = {"value": float(m.group(2)), "line": f"model/spa_freq_mixf.py:{i}"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("third_party/freqmixformer"))
    ap.add_argument("--out", type=Path, default=Path("results/phase0/r2_dct_axis.json"))
    args = ap.parse_args()
    commit = subprocess.run(["git", "-C", str(args.repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    report = {
        "repo_commit": commit,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "dct_unit_test": dct_last_dim_unit_test(),
        "block_trace": run_block_with_logging(args.repo),
        "hardcoded_operator_constants": hardcoded_constants(args.repo),
    }
    print(json.dumps(report, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
