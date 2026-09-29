# Reference registry (Phase 0)

Compiled 2026-09-29. Each entry records what was actually read and what was not.

**Access constraints in this session (these limit everything below):**
- The container's network policy allowed **github.com (git + raw) and PyPI only**. arxiv.org, ieeexplore, dl.acm.org, openaccess.thecvf.com, publications.idiap.ch, semanticscholar, par.nsf.gov and alphaxiv all returned *blocked* (proxy 403 / egress blocked). **So no paper text was read in this session.**
- Bibliographic metadata marked `[SEARCH-SNIPPET]` comes from web-search result snippets only. It's a lead (title, authors, venue), not a verified read of the paper.
- Code marked `[READ]` was cloned and read line by line at the pinned commit. `scripts/fetch_references.sh` reproduces the R1/R2 checkouts.
- The GitHub REST API was also blocked for repos outside this session's scope. As a result, the GitHub "license" metadata, releases, and repository visibility could **not** be queried. License statements below come from the files present in each repo tree at the pinned commit.

Status vocabulary: `VERIFIED-CODE` (code read and checks run), `BLOCKED-UNVERIFIED` (source not readable, so nothing about it is implemented), `UNRESOLVED-NAME` (not yet clear which method is meant).

---

## Summary table

| ID | Item | Paper text | Code | License (from repo tree) | Weights | Status |
|---|---|---|---|---|---|---|
| R1 | RGBPose-SlowFast (ICIP 2022) | NOT READ (IEEE blocked) | `[READ]` @03978e1 | **none**: no LICENSE file | none in tree or README | code VERIFIED-CODE; paper **BLOCKED-UNVERIFIED** |
| R2-V1 | FreqMixFormer (ACM MM 2024) | NOT READ (arXiv blocked) | `[READ]` @1d3c2ad (`model/`) | **none**: no LICENSE file | none in README | code VERIFIED-CODE; paper **BLOCKED-UNVERIFIED** |
| R2-V2 | FreqMixFormerV2 (FG 2025, arXiv 2412.20621) | NOT READ | **not found in the official repo** | n/a | n/a | **BLOCKED-UNVERIFIED** (paper and code) |
| R3 | FineX (arXiv 2608.13458) | NOT READ | none found | n/a | n/a | **BLOCKED-UNVERIFIED** |
| R4 | ACII 2026 triadic paper (`ACII2026_CR_99.pdf`) | **file not received** | n/a | private manuscript | n/a | **BLOCKED-UNVERIFIED** |
| SOC | Loose Social-Interaction Recognition (arXiv 2409.20270 / WACV 2025) | NOT READ | none found | n/a | n/a | **BLOCKED-UNVERIFIED** |
| π-ViT | Just Add π! (CVPR 2024, arXiv 2311.18840) | NOT READ | repo exists @b927973 (README only) | CC-BY-NC-4.0 | README links "Pretrained models" | repo located; paper BLOCKED |
| Sharingan | Multi-person gaze following (CVPR 2024) | NOT READ | repo exists @006b69d (README only) | CC-BY-NC-4.0 (`LICENSES/`) | README says checkpoints are provided | repo located; paper BLOCKED |
| MSTS | ? | n/a | n/a | n/a | n/a | **UNRESOLVED-NAME** (see below) |
| V-JEPA | facebookresearch/jepa @51c59d5 | NOT READ | README only | CC-BY-NC-4.0 | not checked | repo located |
| V-JEPA 2 | facebookresearch/vjepa2 @204698b (arXiv 2506.09985) | NOT READ | README only | MIT (repo LICENSE; weights terms not checked) | not checked | repo located |
| VideoMAE (V1) | MCG-NJU/VideoMAE @14ef8d8 (arXiv 2203.12602) | NOT READ | README only | **CC-BY-NC-4.0** | not checked | repo located. **Note: our current RGB stream uses VideoMAE, so its weights license applies to us** |
| VideoMAE V2 | OpenGVLab/VideoMAEv2 @29eab1e (arXiv 2303.16727) | NOT READ | README only | MIT (repo) | not checked | repo located |
| Video Swin | SwinTransformer/Video-Swin-Transformer @db018fb (arXiv 2106.13230) | NOT READ | README only | Apache-2.0 | not checked | repo located |
| MViTv2 | facebookresearch/mvit @dfbdfdf (arXiv 2112.01526) | NOT READ | README only | Apache-2.0 | README has pre-trained models section | repo located |
| Hiera | facebookresearch/hiera @b12b842 | NOT READ | README only | code Apache-2.0; `LICENSE.models` CC-BY-NC-4.0 | not checked | repo located |
| UMT | OpenGVLab/unmasked_teacher @4fb4049 (arXiv 2303.16058) | NOT READ | README only | MIT (repo) | not checked | repo located |
| InternVideo2 | OpenGVLab/InternVideo @3965eef, `InternVideo2/` (arXiv 2403.15377) | NOT READ | README only | Apache-2.0 (top-level) | not checked | repo located |
| VideoMamba | OpenGVLab/VideoMamba @37355c2 (arXiv 2403.06977) | NOT READ | README only | Apache-2.0 | README links HF model | repo located |
| (context) | open-mmlab/mmaction2 @a5a167d | n/a | `pose_transforms.py` read | Apache-2.0 | n/a | used to trace R1's pose-heatmap code provenance |

"repo located" means only that the official repository was identified from its own README (which names the paper) and its license file was read. **Nothing about the method itself is verified.** Each of these still needs a paper read and a spec note before any implementation (Global Rule 2).

---

## R1. RGBPose-SlowFast for stimming (ICIP 2022)

- **Paper:** "Detecting a child's stimming behaviours for autism spectrum disorder diagnosis using RGBPose-SlowFast network", IEEE ICIP 2022, pp. 3356-3360, DOI 10.1109/ICIP46576.2022.9897867 (DOI and pages as given by you; not independently checked).
  - Author list: your prompt says "Berlin S, Pandian, Rajagopalan, Jayagopi". A `[SEARCH-SNIPPET]` lists Deepak Pandian, Shyam Sundar Rajagopalan, Dinesh-Babu Jayagopi, without Berlin S. **Unresolved:** the snippet may be truncated. It needs checking from the PDF.
  - An ICIP 2022 talk page exists at rc.signalprocessingsociety.org (`spsicip22vid0548`) `[SEARCH-SNIPPET]`, not opened.
- **Paper text read:** none. **Status: BLOCKED-UNVERIFIED.** Supply `references/ICIP2022_RGBPose_SlowFast.pdf`.
- **Repo:** https://github.com/Jeba-create/Stimming-Behaviour-detection-using-RGB-POSE-SlowFast-Network, HEAD `03978e1756b5d3501f7390e87ac92c94e237e45d` (2023-11-05). Files: `Dataloaderrgbpose.py`, `Poseaugmentation.py`, `main.py`, `modelrgbposeslowfast.py`, `README.md`.
- **License:** no LICENSE/COPYING file in the tree. Default copyright applies, so **we may not copy this code into our repo**. We can read it and reimplement from our own spec. `main.py:1` carries `# Copyright (c) OpenMMLab`, and `Poseaugmentation.py:4-5` (`GeneratePoseTarget`, "Generate pseudo heatmaps based on joint coordinates and confidence.") matches mmaction2 `mmaction/datasets/transforms/pose_transforms.py:127-128`. `[INFERRED]` Parts are derived from Apache-2.0 OpenMMLab code, which would require attribution the repo does not carry.
- **Weights:** none in the tree, none linked in the README.
- **What I read:** all four `.py` files in full, plus the README (abstract only).
- **Findings:** see `spec_notes/R1_rgbpose_slowfast.md`. Measured output is in `results/phase0/r1_checks.json`.

## R2. FreqMixFormer (V1) and FreqMixFormerV2

- **V1 paper:** "Frequency Guidance Matters: Skeletal Action Recognition by Frequency-Aware Mixed Transformer", Wu, Zheng, Yang, Das, Chen, Lu (the README header order; the README bibtex swaps Chen and Das), ACM MM 2024. arXiv **2407.12322** (from the repo README badge and bibtex, `[READ]`). A `[SEARCH-SNIPPET]` shows arXiv versions v1 and v2 exist. The latest version and its date could not be checked.
- **V2 paper:** "FreqMixFormerV2: Lightweight Frequency-aware Mixed Transformer for Human Skeleton Action Recognition", Wu, Wang, Chen, Lu, arXiv **2412.20621** (v1 seen) `[SEARCH-SNIPPET]`. It was accepted at IEEE FG 2025 per the repo README ("accepted ... 2024/12/13") `[READ]`. Note that **V2 has a different author list from V1** (Pengfei Wang; no Zheng, Yang, Das).
- **Paper text read:** none (arXiv blocked). **Status: BLOCKED-UNVERIFIED** for both papers. Supply `references/FreqMixFormer_V1_2407.12322.pdf` and `references/FreqMixFormerV2_2412.20621.pdf`.
- **Repo:** https://github.com/wenhanwu95/FreqMixFormer, HEAD `1d3c2add885888f6eaa93135b9fb11dfdbc42be2` (2026-04-28).
- **Which files are which version** `[MEASURED]` from the tree plus configs:
  - `model/skefreqmixformer.py`, `model/spa_freq_mixf.py`, `model/tem_mixf.py` make up the only frequency model. Every config (`config/*/joint.yaml`, `motion.yaml`) uses `model: model.skefreqmixformer.Model`. The README title and bibtex are V1's. `[INFERRED]` This is **V1 code**.
  - `UniSTFormer/` belongs to a **third, different paper** (UniSTFormer, arXiv 2508.08944, per `UniSTFormer/README.md`). It contains no DCT call (grep).
  - **No V2 code exists in the repo.** There are no files named or containing HFAB/LFAB, and nothing beyond V1 is registered in the configs. The README's "Codes updated 2024/11/10" predates V2's acceptance (2024/12/13). **V2 code: BLOCKED (not released here).**
- **License:** no LICENSE file in the tree. **Do not copy the code.** Reimplement from the spec.
- **Weights:** none linked in the README.
- **Dependency:** `torch_dct` (PyPI `torch-dct` 0.1.6). Its source `_dct.py:57-67` states that `dct` is DCT-II "over the last dimension".
- **Findings:** see `spec_notes/R2_freqmixformer.md` (DCT axis resolved from code). Measured output is in `results/phase0/r2_dct_axis.json`.

## R3. FineX (arXiv 2608.13458)

- "Fine-Grained Action Recognition with Cross-Attentive Latent Sparse Experts", Imtiaz Ul Hassan, Tasweer Ahmad, Nik Bessis, Ardhendu Behera `[SEARCH-SNIPPET]`. An alphaxiv snippet says "Submitted 13 Aug 2026". You said v2 is dated 14 Aug 2026; I could not check the version list.
- **Paper text read:** none. **Code:** none found (search returned no repository). **Venue:** none seen.
- **Status: BLOCKED-UNVERIFIED.** Supply `references/FineX_2608.13458v2.pdf`, including the supplement if it has one.
- Spec note (provisional, classifies pieces only): `spec_notes/R3_finex.md`.

## R4. ACII 2026 triadic-interaction paper

- "A Comprehensive Multimodal Framework for Autism Detection via Behavior Recognition in Triadic Interaction Settings" (Roy Choudhury, Berlin, et al.).
- **The PDF was not attached to this session.** Only `ARCHITECTURE_V2_EVAL_PROMPT.md` was uploaded. **Status: BLOCKED-UNVERIFIED.** Put it at `references/ACII2026_CR_99.pdf`. That path is git-ignored so a private manuscript never gets pushed; see `references/README.md` for how to hand it to me.
- Spec note skeleton: `spec_notes/R4_acii2026_triadic.md`.

## SOC. Loose Social-Interaction Recognition (arXiv 2409.20270)

- "Loose Social-Interaction Recognition in Real-world Therapy Scenarios", Abid Ali, Rui Dai, Ashish Marisetty, Guillaume Astruc, Monique Thonnat, Jean-Marc Odobez, Susanne Thümmler, François Brémond `[SEARCH-SNIPPET]`.
- **A published version exists:** WACV 2025 (CVF open access page `Ali_Loose_Social-Interaction_Recognition_in_Real-World_Therapy_Scenarios_WACV_2025_paper`, plus an Idiap PDF `Ali_WACV_2025.pdf`) `[SEARCH-SNIPPET]`. Prefer the WACV version over the arXiv v1 once readable.
- **Code:** none found by search. **Weights:** unknown.
- **Status: BLOCKED-UNVERIFIED.** Supply `references/Ali_WACV2025_LooseInteraction.pdf`, plus the supplement if any.
- Spec note skeleton: `spec_notes/SOC_loose_interaction.md`.

## Other external methods from the original plan

A short spec note for each is in `spec_notes/EXTERNAL_methods.md`. Key points:

- **MSTS: UNRESOLVED-NAME.** Search finds several different methods with similar acronyms: **MS-TCT** (Multi-Scale Temporal ConvTransformer, Dai et al., CVPR 2022, arXiv 2112.03902, action *detection*), **MS-STS** (multi-scale spatio-temporal split attention, video instance *segmentation*), and **MSSTNet** (facial-expression recognition). `[SPECULATION]` MS-TCT is the most plausible match: it shares an author (Rui Dai) and group with the SOC paper. **Please confirm which one you mean.**
- **"JEPA-style pose model" (E2):** only *video* JEPA repos are located (V-JEPA, V-JEPA 2). No skeleton/pose JEPA has been identified yet. This is deferred to E2 as the prompt allows.
- **License flag:** VideoMAE (V1), V-JEPA, π-ViT, Sharingan and Hiera model weights are **CC-BY-NC-4.0 (non-commercial)**. That's fine for academic research, but record it if the work may ever be used commercially.
