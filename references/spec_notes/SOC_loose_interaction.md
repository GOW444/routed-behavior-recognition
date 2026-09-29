# Spec note SOC: Loose Social-Interaction Recognition (arXiv 2409.20270 / WACV 2025). BLOCKED-UNVERIFIED

- Title and authors (Ali, Dai, Marisetty, Astruc, Thonnat, Odobez, Thümmler, Brémond) come from a `[SEARCH-SNIPPET]`.
- **A published WACV 2025 version exists** (CVF open access + Idiap). Use it as the primary reference once readable, and diff it against arXiv v1.
- Paper text: **not read**. Official code: **none found**. Weights: unknown.
- Supply `references/Ali_WACV2025_LooseInteraction.pdf`, plus the supplement if any.
- Search snippets confirm only the high-level framing: a dual-path architecture for a leader and an assistant, a CNN backbone, and a "Global-Layer-Attention" cross-attention fusion. Reported datasets are a Loose-Interaction dataset, Autism datasets, and NTU interactive classes. **None of the architectural details in your prompt (multi-scale 3D-CNN, last three blocks, Abstract Projection → 768-D, class token, 1-D positional encoding, GLA Q/K/V assignment, §3.6 temporal-synchronisation variant) are verified.**

## Checklist for the faithful reimplementation (Phase 4.2 Step 1), once the PDF is read

- [ ] Backbone: exact network, pretraining, input crop size, clip length, frame rate; shared weights confirmed?
- [ ] Which blocks feed the Abstract Projection, and the pooling and MLP widths.
- [ ] Token layout: class token, positional encoding, LayerNorm placement.
- [ ] GLA: exact Q (leader after self-attention?) and K/V (MLP of assistant?) definitions, heads, depth, residuals.
- [ ] How leader vs assistant is decided in their data, and what they say about role reversal.
- [ ] §3.5 and §3.6 details; ablation tables; seeds/variance.
- [ ] Their autism dataset: which one, classes, split protocol (group-level or not?).
- [ ] Whether code/weights are promised or linked in the WACV version.

Note: one author (Rui Dai) is also first author of MS-TCT, which is one candidate meaning of "MSTS" (see `EXTERNAL_methods.md`).
