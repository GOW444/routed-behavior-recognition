# routed-behavior-recognition

Routed two-branch (social vs RRB) extension of the CIGPose-VideoMAE autism-behaviour classifier, with a leakage-safe evaluation protocol.

**Current state: Phase 0 (reference verification) only.** Start with `PHASE0_SUMMARY.md`.

```
docs/ARCHITECTURE_V2_EVAL_PROMPT.md   the full plan (Phases 0-6)
PHASE0_SUMMARY.md            findings + decisions needed
DECISIONS.md                 small judgment calls and why
references/
  REFERENCE_REGISTRY.md      every external paper/repo: URL, version, commit, license, what was read
  spec_notes/                one note per reference (R1-R4, SOC, others)
  README.md                  which PDFs to drop in and their exact names
tools/phase0/
  r1_readonly_checks.py      RGBPose-SlowFast: temporal shapes at laterals, BN eval dependence, runnability
  r2_dct_axis_check.py       FreqMixFormer: which axis the DCT / split index acts on
  freq_resolution.py         Nyquist / bin spacing for T frames at a given fps and stride
results/phase0/              JSON outputs of the checks (measured)
scripts/fetch_references.sh  clones R1/R2 at the pinned commits into third_party/ (git-ignored)
```

## Reproduce the Phase 0 checks (CPU only, no data, about 1-2 min)

```bash
pip install -r requirements-phase0.txt
./scripts/fetch_references.sh
python tools/phase0/r1_readonly_checks.py   # -> results/phase0/r1_checks.json
python tools/phase0/r2_dct_axis_check.py    # -> results/phase0/r2_dct_axis.json
python tools/phase0/freq_resolution.py --T 16 --fps 30 --stride 4   # once our T/fps are known
```

The committed JSON results were produced with torch 2.14.0 (CPU), torch-dct 0.1.6 and numpy 2.4.6.
