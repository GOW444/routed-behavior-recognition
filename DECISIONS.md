# DECISIONS.md

Small judgment calls, with reasons. Anything that changes the science, data access or budget is instead listed as a question in `PHASE0_SUMMARY.md`.

| # | Date | Decision | Reason |
|---|---|---|---|
| D1 | 2026-09-29 | External repos are **cloned at pinned commits into `third_party/` (git-ignored)** by `scripts/fetch_references.sh`, not vendored. | R1 and R2 have no license file, so copying their code into this repo is not permitted. Pinning commits makes the Phase 0 checks reproducible. |
| D2 | 2026-09-29 | Search-engine snippets were used **only for bibliographic metadata** (title, authors, venue, arXiv id). They are tagged `[SEARCH-SNIPPET]` and never used for method details. | arXiv and publisher sites were blocked. Snippets are not a read of the paper (Rule 2). |
| D3 | 2026-09-29 | The R1 checks run the model with **one shim** (`torch` injected into the module namespace). The unmodified import is tested separately and its failure recorded. | The only way to measure shapes/BN without changing the architecture; the shim is minimal and documented. |
| D4 | 2026-09-29 | The R2 checks run `Spatial_Freq_MixFormer` on CPU by replacing exactly two strings (`.cuda(x0.get_device())` → `` and `device='cuda'` → `device=x0.device`). `torch_dct.dct/idct` are wrapped to log input shapes. | No GPU is needed; neither replacement changes the maths. |
| D5 | 2026-09-29 | The R2 axis question is answered **for V1 code only**. V2 is marked unresolvable. | The official repo has no V2 code; the V2 paper is unread. |
| D6 | 2026-09-29 | `*.pdf` under `references/` is git-ignored. | R4 is a private camera-ready and must not be redistributed. The others are copyrighted. |
| D7 | 2026-09-29 | Frequency-resolution numbers for **our** model are not reported, only the calculator. | Our T/fps/stride must be read from our code (Phase A). Guessing would violate Rule 3. |
| D8 | 2026-09-29 | The R3 "implementable from paper" table is built from **your prompt's description**, clearly labelled provisional. | The prompt asks for the list in Phase 0; the paper is unreadable. The table is a checklist, not a claim. |
