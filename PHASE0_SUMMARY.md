# Phase 0: Reference verification, summary and decisions

Date 2026-09-29. Scope: Phase 0 only. No Phase A audit or training was run. **The CIGPose-VideoMAE code is not in this repository yet** (only a LICENSE file existed), so nothing about our own model is stated anywhere in this deliverable.

## What was done

| Item | Status | Where |
|---|---|---|
| Reference registry (R1-R4, SOC, 13 other methods) | done | `references/REFERENCE_REGISTRY.md` |
| Spec notes | R1, R2 full (code-based); R3, R4, SOC, others: skeletons/checklists | `references/spec_notes/` |
| R1 read-only checks (temporal lengths, BN, runnability, protocol) | **done, measured** | `tools/phase0/r1_readonly_checks.py` → `results/phase0/r1_checks.json` |
| R2 DCT axis / split index N from code | **resolved for V1 code, measured**; V2 unresolvable (no code) | `tools/phase0/r2_dct_axis_check.py` → `results/phase0/r2_dct_axis.json` |
| R3 implementable-from-paper list | provisional (from your lead description, not the paper) | `references/spec_notes/R3_finex.md` |
| Frequency-resolution calculator (for Phase A / E3) | tool only; our T/fps **NOT MEASURED** | `tools/phase0/freq_resolution.py` |

The main limit: this session's network allowed only GitHub and PyPI. **No paper PDF could be read.** Every paper is therefore BLOCKED-UNVERIFIED, and only code-level facts are marked [MEASURED].

## Key findings

### R2 FreqMixFormer `[MEASURED]`
1. **The official repo contains V1 code only.** There is no FreqMixFormerV2 code. `UniSTFormer/` is a third, different paper.
2. **In V1 code, the DCT runs over the joint axis (V=25), not time.** `spa_freq_mixf.py:117-120` applies `torch_dct.dct` (last dim) to `(N,C,T,V)`. Time and channels are then average-pooled away (`:41-42`). The split `d=13` (`:129`) scales columns of 25×25 cross-attention maps. There is **no temporal DCT anywhere**; `tem_mixf.py` imports `torch_dct` but never calls it.
   - `[INFERRED]` "High/low frequency" here is spatial frequency along the NTU joint *numbering*, not motion frequency. It does not model rhythmic RRB motion.
3. The code hard-codes `l=0.5, h=1.5, d=13` (not the 0.2/1.2 you quoted for V2). These are constants, not learned.
4. **Checkpoints are chosen on the test split.** `main.py:458-460, 505-508` keep the best-test epoch; there is no validation split in the configs. Confirms Rule 10: don't transplant these values.
5. V1 has 2.07M params at 60 classes. 25 joints and CUDA are hard-coded.

### R1 RGBPose-SlowFast `[MEASURED]`
1. **Silent temporal broadcast confirmed.** With the lengths `main.py` uses (RGB T=4, pose T=20), all three slow→pose laterals output **T=1**, which is broadcast onto 20 pose frames. So RGB→pose carries no temporal information, only a clip-level bias; pose→RGB (stride 5: 20→4) matches. Other lengths (4/32, 8/32, 16/64) crash.
2. **BatchNorm:** the ResNet50-style blocks use `track_running_stats=False`, so eval predictions depend on the batch. Measured logit changes: 0.015 (batch 1 vs 4) and 0.22 (different batch companions). The ResNet34-style variant, which `main.py` actually trains, uses normal BN and is batch-independent.
3. **The code does not run as released.** Unmodified, it raises `NameError: torch`, and pyflakes finds about 30 undefined names across files.
4. **The released protocol trains on the held-out fold.** `main.py:179` calls `train(...)` on the val loader. Folds are clip-level and unseeded, the checkpoint is chosen on the held-out fold, val uses train augmentation, and the 4 RGB frames each get a *different* augmentation.
5. **There is no LICENSE file**, so we must not copy the code. Parts appear derived from Apache-2.0 mmaction2 (`GeneratePoseTarget` docstring match) `[INFERRED]`. The "ResNet34" variant (43.8M params) is larger than the "ResNet50" one (37.0M).

### R3 / R4 / SOC
- R3 FineX: paper unread; no code found. Nothing is implemented. See the provisional component table.
- R4 ACII: **the PDF was not attached to this session.**
- SOC (2409.20270): a **WACV 2025 published version exists** (use it over arXiv v1). No official code found.

## Decisions I need from you

1. **PDFs** (blocking): see `references/README.md` for the six files and their exact names, or allow `arxiv.org` in the environment's network settings.
2. **MSTS**: which method do you mean? MS-TCT (Dai et al. CVPR 2022) is my guess `[SPECULATION]`. The other candidates are MS-STS and MSSTNet.
3. **E7 (R1 as external baseline):** with no license and non-runnable code, the only option is an *adapted* reimplementation under our protocol, with the lateral broadcast fixed or kept, and labelled as such. Do you want (a) adapted reimplementation, (b) BLOCKED until the authors add a license or respond, or (c) permission from your group to use the code, since R1 is from your group?
4. **R2 / E3 direction:** V1 code's "frequency" is over joints. For RRB, the motivated version is a **temporal** DCT, which would be our adaptation, not the paper's. Should E3 include (a) only a temporal-DCT adapted arm, (b) a joint-axis arm as in V1 code as well, or (c) wait for the V2 paper?
5. **Code for Phase A:** please push (or point me to) the existing CIGPose-VideoMAE code and the AAD/StimASD label files. Phase A cannot start without them.
6. **License note:** VideoMAE V1 weights (our current RGB backbone), V-JEPA, π-ViT, Sharingan and Hiera weights are CC-BY-NC-4.0. Just confirm the project is non-commercial research.

Small judgment calls I made on my own are in `DECISIONS.md`.
