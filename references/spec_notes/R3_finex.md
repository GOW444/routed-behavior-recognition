# Spec note R3: FineX (arXiv 2608.13458). BLOCKED-UNVERIFIED

- Paper not read (arXiv blocked). No official code found. **Nothing below is a claim about FineX.** The component list is copied from **your prompt's lead description** so we can plan. Every row must be re-checked against the PDF (`references/FineX_2608.13458v2.pdf`, plus the supplement).
- FineX is **not** a group router (your note). It must not be used as the social-vs-RRB router.

## Which pieces could be implemented from a paper alone vs need code (provisional)

"From paper" means a standard, fully specified building block that a clear equation or diagram plus hyperparameters would pin down. "Needs code" means details that papers routinely leave out and that change behaviour.

| Component (per your lead) | From paper alone? | What must be confirmed in the PDF / what is usually missing |
|---|---|---|
| Frozen backbones (R(2+1)D, PoseC3D, ST-GCN++), penultimate features cached | Yes. These are public backbones (PoseC3D/ST-GCN++ in pyskl/mmaction2, Apache-2.0) | exact checkpoints, pretraining data, feature layer, temporal pooling, clip sampling at caching time |
| Linear projection of each stream to D | Yes | bias, norm, dropout |
| Symmetric pairwise cross-attention (each stream queries the other two, residual + LayerNorm, stream identity kept) | Mostly | pre- vs post-LN, whether the two K/V streams are concatenated or attended separately, stream/positional embeddings, token count per stream (1 pooled vector vs a sequence) |
| Streamwise sparse MoE (shared expert bank, bias-free linear router, top-k, bottleneck-FFN experts) | Partly | gate normalisation (softmax before or after top-k), capacity factor / token dropping, noise in gating, expert hidden size, residual around the MoE |
| Switch-style load-balance loss, weight 0.1 | Yes if the equation is given | exact formula variant (Switch vs GShard), per-stream vs pooled |
| Label-smoothed CE, mean over streams | Yes | smoothing value |
| Training recipe (optimiser, schedule, epochs, augmentation on cached features) | Needs code or supplement | usually incomplete in the main text |
| Seeds / variance / significance | n/a | your note: not seen in the main text; check the supplement |

**Decision for Phase 0:** nothing from FineX is implemented. E5 arms that cite FineX will be labelled **"adapted, FineX-inspired"** and written from our own spec. They stay blocked until the PDF is read. Global Rule 9 (stacking leakage) applies to any cached-feature version regardless.
