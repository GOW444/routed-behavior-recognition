# references/

- `REFERENCE_REGISTRY.md`: one entry per external paper/repo: URLs, versions, commits, licenses, weights, what was and was not read.
- `spec_notes/`: one note per reference (R1-R4, SOC, other methods).

## PDFs I need from you (Phase 0 is blocked on these)

| File to place here | Unblocks |
|---|---|
| `ICIP2022_RGBPose_SlowFast.pdf` | R1 paper-vs-code comparison, author list, SSBD/"Autism dataset" details |
| `FreqMixFormer_V1_2407.12322.pdf` (latest arXiv version or the ACM MM camera-ready) | R2 V1 paper-vs-code check |
| `FreqMixFormerV2_2412.20621.pdf` | R2 V2 (HFAB/LFAB, operator, N/l/h search protocol); V2 code is not public |
| `FineX_2608.13458v2.pdf` (+ supplement) | R3 component table |
| `ACII2026_CR_99.pdf` | R4 (private; never pushed) |
| `Ali_WACV2025_LooseInteraction.pdf` (+ supplement) | Social branch faithful reimplementation (Phase 4.2) |

`*.pdf` is git-ignored so private or copyrighted PDFs are never pushed. To give them to a future
Claude session, either upload them in the chat, or (only if this GitHub repo is private) add them
with `git add -f references/<file>.pdf`.

**Alternative:** allow `arxiv.org` (and optionally `openaccess.thecvf.com`, `ieeexplore.ieee.org`)
in the cloud environment's network settings. Everything except R1 (IEEE paywall) and R4 (private)
could then be fetched directly.
