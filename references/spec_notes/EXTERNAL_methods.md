# Spec notes: other external methods (original plan, E1/E2/optional enrichments)

One section per method. For every method, **paper text is NOT READ** (arXiv/CVF blocked in this session). Only the official repo was located and its README header and license file read, at the listed commit. Status for all of them: **repo located; method BLOCKED-UNVERIFIED**. A full spec note (what it does, I/O shapes, what we take, what we change) is written only after the paper is read. Candidate-table columns for E1 (params, compute at 16×224×224, token shape) are **NOT FILLED**, because they need a paper/code read.

## π-ViT: "Just Add π! Pose Induced Video Transformers for Understanding Activities of Daily Living"
- Name resolved: CVPR 2024, arXiv 2311.18840, Reilly & Das `[SEARCH-SNIPPET]`. The repo README confirms the title and venue.
- Repo `dominickrei/pi-vit` @ b927973, **CC-BY-NC-4.0**. The README links "Pretrained models".
- Lead to verify: 2D/3D skeleton-induction modules are auxiliary, training-only losses, and are dropped at inference `[SEARCH-SNIPPET]`. If true, they fit our setting (pose used as a training signal for the RGB stream). Whether its 3D-pose module needs 3D skeletons we don't have: **check**.

## Sharingan: multi-person gaze following
- CVPR 2024, Tafasca, Gupta, Odobez (Idiap/EPFL). The repo README confirms it.
- Repo `idiap/sharingan` @ 006b69d, **CC-BY-NC-4.0** (`LICENSES/`). The README says code, data artefacts and checkpoints are provided.
- ChildPlay is among its reported benchmarks `[SEARCH-SNIPPET]`. It must be spot-checked on our child videos before use (Phase 4.2).

## MSTS: UNRESOLVED-NAME, please confirm
Candidates found:
- **MS-TCT**: Multi-Scale Temporal ConvTransformer, Dai et al., CVPR 2022, arXiv 2112.03902. Dense action *detection*.
- **MS-STS**: multi-scale spatio-temporal split attention transformer. Video instance *segmentation*.
- **MSSTNet**: multi-scale spatio-temporal CNN-Transformer. Dynamic *facial expression* recognition.

`[SPECULATION]` MS-TCT is most likely, since it shares an author with the SOC paper. It is a temporal model over frame features, not a token-producing video backbone, so the "multi-scale tokens" idea in E1 might mean something else. **Not resolved. Nothing is implemented.**

## JEPA / V-JEPA / V-JEPA 2
- V-JEPA: `facebookresearch/jepa` @ 51c59d5, CC-BY-NC-4.0.
- V-JEPA 2: `facebookresearch/vjepa2` @ 204698b, arXiv 2506.09985, repo LICENSE MIT (weights terms not checked).
- **Pose/skeleton JEPA:** none located yet (deferred to E2 as the prompt specifies).

## VideoMAE (V1, our current RGB backbone) and VideoMAE V2
- VideoMAE: `MCG-NJU/VideoMAE` @ 14ef8d8, arXiv 2203.12602, **CC-BY-NC-4.0**.
- VideoMAE V2: `OpenGVLab/VideoMAEv2` @ 29eab1e, arXiv 2303.16727 (CVPR 2023), MIT.

## Video Swin
- `SwinTransformer/Video-Swin-Transformer` @ db018fb, arXiv 2106.13230, Apache-2.0 (mmaction2-based).

## MViTv2 / Hiera
- MViTv2: `facebookresearch/mvit` @ dfbdfdf, arXiv 2112.01526 (CVPR 2022), Apache-2.0.
- Hiera: `facebookresearch/hiera` @ b12b842, code Apache-2.0, **model weights CC-BY-NC-4.0** (`LICENSE.models`). Paper arXiv id not confirmed from the README header.

## UMT (Unmasked Teacher)
- `OpenGVLab/unmasked_teacher` @ 4fb4049, arXiv 2303.16058, MIT.

## InternVideo2
- `OpenGVLab/InternVideo` @ 3965eef, subfolder `InternVideo2/`, arXiv 2403.15377, top-level Apache-2.0. Per-model weight terms not checked.

## VideoMamba
- `OpenGVLab/VideoMamba` @ 37355c2, arXiv 2403.06977, Apache-2.0. The README links HF weights.
