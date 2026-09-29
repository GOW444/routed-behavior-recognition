# Spec note R4: ACII 2026 triadic-interaction paper. BLOCKED-UNVERIFIED

- `ACII2026_CR_99.pdf` was **not received** in this session. Only the prompt file was uploaded.
- It is a private camera-ready manuscript. Place it at `references/ACII2026_CR_99.pdf` (git-ignored, so it is never pushed) or upload it in the chat.
- Nothing about this paper is verified. The leads in the prompt (YOLOv7 person detection; Siamese/triplet role matcher with about 15 reference crops per role per video and 97% role accuracy; 19 behaviours in 3 domains; separate-domain mAP 56/55/55 vs joint 43; behaviour durations of about 4 s and about 18 s) are **your reading, not checked**.

## What to extract once the PDF is available (checklist)

1. Role identification: detector, matcher architecture, where the reference crops come from (manual per video?), and how role accuracy was measured.
2. Triadic handling: how person features are ordered and fused; what happens when a person is missing.
3. Domain taxonomy (Fig. 1): which behaviours sit in which domain (joint attention, imitation).
4. Separate vs joint modelling: exact table, metric, number of classes, data split. This is **weak indirect evidence for H1 only**.
5. Behaviour duration statistics: table/section. Compare later with our clip length (Phase A3).
6. Referenced leads (Ji et al. TCSVT 2025 lag-aware imitation; ChildPlay, ICCV 2023): record their citations exactly for later verification.
7. Dataset availability: "data I could ask the authors for" goes into DATA_REPORT.md.
