# ROLE AND MISSION

You are a careful ML research engineer in my existing CIGPose-VideoMAE repository (multimodal video classifier for autism-related behaviors). I am changing the architecture into a ROUTED, TWO-BRANCH model and I want a rigorous, hallucination-resistant evaluation of it. Correctness and honesty matter more than speed or impressive numbers. A negative result reported honestly is a success; an irreproducible positive result is a failure.

This prompt was revised on 2026-09-29 to incorporate four reference papers (registry below). Everything I say about those papers in this prompt is a LEAD from a quick read, not a verified fact. You must re-verify every one of them yourself under the anti-hallucination rules.

# ARCHITECTURE V2 (what we are building)

```
                    Input video clip
                          |
                 [ROUTER: social vs RRB]      <- router paper: PENDING (see Router section)
                 /                        \
   RRB BRANCH (existing 3-stream model)     SOCIAL-INTERACTION BRANCH (new)
   RGB (VideoMAE) + Pose (CIGPose)          Leader/Assistant dual path,
   + Local ROI, pose-queries-RGB fusion,    inspired by arXiv 2409.20270
   quality gate, dataset-adaptive head      (Loose Social-Interaction Recognition)
   NO between-person (dyadic) tokens
                 \                        /
              [Combine: routed class posterior]
```

Design intent, which you must explain in the docs (see Phase A):
- The existing model was built to recognize repetitive restricted behaviors (RRB) such as flapping, spinning, head banging. Those are mostly single-person, motion-pattern problems. Social classes (e.g., joint attention, imitation in AAD) depend on how TWO people relate over time. Mixing both in one network forces the pose relation encoder to carry dyadic reasoning it was not designed for, so we separate them and let a router decide which specialist handles a clip.
- RRB branch = the current model, with ONE removal: delete the between-person (dyadic) relation tokens (currently 12 of the 228 pose tokens). Within-person relation tokens (96) stay. Keep the old behavior available via a config flag (`relation.between_person: true|false`) so the old baseline can still be reproduced. Update every downstream shape (pose tokens 228 -> 216, cross-attention K/V, pooling, ROI alignment) and report the changes.
- Social branch = new, inspired by arXiv 2409.20270 (details below), with role identification informed by the ACII 2026 paper (R4 below).
- Router = decides, per clip, social vs RRB (details below).
- Design hypotheses to TEST, not assume: (H1) separating social and RRB specialists beats one monolithic model; (H2) removing dyadic tokens from the RRB branch does not hurt RRB classes; (H3) explicit frequency modelling helps rhythmic RRB classes beyond the existing FFT spectral encoder; (H4) richer cross-stream fusion helps in this low-data regime and is not just extra parameters. Write each hypothesis and its predicted direction before running anything.

# REFERENCE PAPERS: WHAT EACH IS FOR (leads to verify, not facts)

Create `references/REFERENCE_REGISTRY.md` with, for each entry: official URL, version/date, official repo URL, license, weights availability, what you read (sections/files), and what you could NOT read. If a paper or repo cannot be accessed, mark it BLOCKED-UNVERIFIED and tell me exactly what file I should drop into `references/`.

## R1. RGBPose-SlowFast for stimming (in-domain prior work from my group)
- Paper: Berlin S, Pandian, Rajagopalan, Jayagopi, "Detecting a child's stimming behaviours for autism spectrum disorder diagnosis using RGBPose-SlowFast network", IEEE ICIP 2022, pp. 3356-3360, DOI 10.1109/ICIP46576.2022.9897867. The IEEE page is JavaScript-rendered and I could not read the paper text. Do NOT infer its method or numbers from the README or from citing papers. If you cannot get the PDF, mark the paper part BLOCKED-UNVERIFIED and ask me for it (I will put it at `references/ICIP2022_RGBPose_SlowFast.pdf`).
- Repo: https://github.com/Jeba-create/Stimming-Behaviour-detection-using-RGB-POSE-SlowFast-Network (files: Dataloaderrgbpose.py, Poseaugmentation.py, main.py, modelrgbposeslowfast.py, README.md; no license file was visible in the listing and no weights were listed; verify both before reusing any code).
- What the model code appears to be (from modelrgbposeslowfast.py, verify by reading it yourself): two 3D-ResNet pathways; an RGB pathway (3-channel input, wider) and a pose pathway whose input is a 17-channel keypoint-heatmap volume (narrower); bidirectional lateral connections (temporal-conv based) at three stages; pooled features of both pathways concatenated into one linear classifier with dropout; a ResNet-50-style and a smaller ResNet-34-style variant; default class count 8.
- Code observations to verify before trusting or porting anything: (a) the ResNet-50-style blocks build BatchNorm with `track_running_stats=False, momentum=0`, so eval-time predictions depend on batch statistics; (b) the lateral-connection temporal strides (8 for slow-to-pose, 5 for pose-to-slow) are not reciprocal, so the actual temporal lengths of the two inputs must be read from the DataLoader, and any tensor add across mismatched temporal length may be silently broadcasting. Report what actually happens.
- README says it was evaluated on SSBD (Rajagopalan, Dhall, Goecke, ICCVW 2013) and an "Autism dataset" of child motor actions. Determine whether StimASD is SSBD, a superset, or something else, from the official data sources, and record it in the data report.
- What it is for here: (1) a domain-relevant external baseline for the RRB classes (E7); (2) three design ideas to test as single factors: pose sampled at a higher temporal rate than RGB, bidirectional exchange between pose and RGB instead of one-way pose-queries-RGB, and exchange at several depths rather than only at the end; (3) a heatmap-based pose representation as an alternative to keypoint tokens (E2).

## R2. Frequency modelling: FreqMixFormerV2 (and V1)
- The link I gave, arXiv 2412.20621, is FreqMixFormerV2 (lightweight version), NOT the original FreqMixFormer (Wu, Zheng, Yang, Chen, Das, Lu, "Frequency guidance matters: skeletal action recognition by frequency-aware mixed transformer", ACM MM 2024). Both point to https://github.com/wenhanwu95/FreqMixFormer. Find the V1 paper and its exact version yourself and tell me which repo files correspond to which version. V2 is the primary candidate (about 1.2M vs 2.0M parameters, which suits our small data); V1 is a secondary comparison only if its paper and code verify.
- What V2 appears to do (verify in paper and code): DCT-based frequency representation of joint trajectories; input split into two channel groups; a mixed spatial attention (self-attention map plus a cross map that shares keys across the two groups); separate high-frequency and low-frequency attention blocks (HFAB, LFAB); a "high-low frequency operator" that scales DCT coefficients (high-frequency amplified by h, low-frequency attenuated by l, with a split index N); temporal attention afterwards; validated on NTU RGB+D 60/120 and NW-UCLA.
- Points that need verification because the paper text is ambiguous or risky: (a) the DCT is described over the frame axis, but the split index N is described over the joint count J (N searched in 1..25 for 25 joints); find in the code which axis the DCT and N actually act on; (b) the best (N=13, l=0.2, h=1.2) appear to have been searched on the NTU-60 X-Sub evaluation numbers, so they may have been chosen on test data; do NOT transplant them; (c) the reported ablations look like single runs without variance; (d) our input differs from NTU (65 canonicalized features per joint, our own joint set, short clips). Compute and report our frequency resolution and Nyquist limit (from clip length in frames and fps) before deciding whether a DCT high/low split is even meaningful for us.
- What it is for here: E3 (within-person relation encoder) and E4 (frequency/physics prior), always compared against our existing FFT spectral encoder.

## R3. FineX (arXiv 2608.13458, v2 14 Aug 2026; note the spelling FineX)
- Fine-Grained Action Recognition with Cross-Attentive Latent Sparse Experts (Hassan, Ahmad, Bessis, Behera). Recent preprint; I did not find an official code link or venue in the text I read; check.
- What it appears to do (verify): three FROZEN backbones (RGB R(2+1)D, pose-heatmap PoseC3D, skeletal-graph ST-GCN++) whose penultimate features are pre-cached and projected to a shared space; stage 2 trains only a fusion module: symmetric pairwise cross-attention (each stream queries the other two, stream identity preserved, residual + LayerNorm), then a streamwise sparse Mixture-of-Experts (one shared expert bank, bias-free linear router, top-k experts per stream feature, bottleneck-FFN experts) with a Switch-style load-balancing auxiliary loss, label-smoothed CE, outputs mean-pooled over streams. Reported defaults: D=512, 3 layers, 8 heads, 16 experts, top-8, load-balance weight 0.1, about 7.5M trainable fusion parameters, trained on datasets with roughly 16k-23k training clips (Gym99/Gym288/Diving48).
- What it is NOT: it is not a group-level router. Its MoE refines features inside each stream; the paper itself reports experts shared across classes. Do NOT use it as the social-vs-RRB router paper. The router paper is still pending.
- Cautions to carry into our design: (a) the balance loss pushes toward uniform expert usage; a supervised 2-way group router with imbalanced groups (few social clips) must not receive that loss unmodified; (b) 16 experts and about 7.5M fusion parameters are sized for 20k-clip data, ours is orders of magnitude smaller; scale down and report parameters; (c) I saw no seeds, variance, or significance tests in the main text; check the supplement; (d) the two-stage recipe (fine-tune backbones, cache features, train fusion) creates a stacking-leakage risk described in Global Rule 9.
- What it is for here: E5 (fusion, gating, MoE); the load-balance/collapse monitoring ideas in Phase 2b; the heatmap-vs-graph split of "pose" as a candidate in E2; symmetric cross-attention as a variant of our pose-queries-RGB fusion and as a parallel to the social branch's symmetric-role variant.

## R4. ACII 2026 triadic-interaction paper (attached: `ACII2026_CR_99.pdf`; put it at `references/ACII2026_CR_99.pdf`; treat as a private camera-ready manuscript, do not redistribute or quote beyond short excerpts)
- "A Comprehensive Multimodal Framework for Autism Detection via Behavior Recognition in Triadic Interaction Settings" (Roy Choudhury, Berlin, et al.). Read it fully yourself. Leads to verify:
  - Setting: child + caregiver + clinically trained examiner, 13 activities, about 30-minute sessions, front camera at 30 fps, toddlers 12-42 months; 19 behaviors in three domains (in its Fig. 1 taxonomy joint attention is listed under the facial-based domain and imitation under social-and-communication; our A3 assignment must follow what evidence a class needs, not that paper's or any dataset's domain label).
  - Person handling: YOLOv7 person detection, then a role-matching step that labels each detected person as child, caregiver, or examiner using a Siamese network with a triplet loss and roughly 15 reference crops per role taken per video, reported 97% role accuracy on their data; person features ordered (child, caregiver, examiner); person-person and person-object interaction via multi-head attention; skeletons of all people fed to a 3D ResNet as one array. This is coarse relative to the leader/assistant cross-attention of 2409.20270, so treat it as a source of ideas for ROLE IDENTIFICATION, TRIADIC HANDLING and DATA, not as an architecture to port.
  - Reported evidence bearing on our routing hypothesis (H1): behavior-detection mAP was higher when domains were modelled separately (56/55/55) than when all 19 behaviors were modelled jointly (43), which the authors attribute to co-occurring behaviors across domains. Treat as weak, indirect support only: different data, different metric (mAP, not macro-F1), and fewer classes per model is an easier problem. Do not cite it as evidence that routing works.
  - Reported temporal scale: short behaviors last about 4 s while joint attention, reciprocal social interaction and social engagement last about 18 s (their numbers). Our clips are short (VideoMAE grid of 8x7x7 tokens suggests 16 frames). Measure our clip duration in seconds and the typical duration of each social class; if social classes need many seconds of context, a single short clip may not contain the interaction, and the social branch needs longer temporal context (report this before building).
  - Leads from its reference list, NOT read, not to be implemented without verification: dyadic imitation with a lag-aware dual-stream model (Ji et al., IEEE TCSVT 2025) as a possible source for the temporal-synchronisation idea; ChildPlay (Tafasca, Gupta, Odobez, ICCV 2023) as a children's gaze benchmark for spot-checking gaze models on child faces.
  - The Play Behavior dataset itself is not assumed to be available to us. If it would help the social branch, list it under "data I could ask the authors for" in DATA_REPORT.md; do not assume access.
- What it is for here: A3 (triadic reality check), 4.2 (role identification, which pair is the "leader/assistant" pair when three people are present), Router (person count and role-detection confidence as cheap routing features), and the failure-analysis flags.

## Already in the original plan (unchanged, still to be verified): arXiv 2409.20270 (loose social-interaction recognition), Sharingan, pi-ViT, MSTS/multi-scale, JEPA/V-JEPA/V-JEPA 2, VideoMAE V2, Video Swin, MViTv2/Hiera, UMT, InternVideo2, VideoMamba. Same rules apply.

# GLOBAL ANTI-HALLUCINATION RULES (apply to every step)

1. READ BEFORE YOU CLAIM. Before describing any part of the current architecture, open the code and cite file:line. If code and diagram disagree, code wins; report the discrepancy.
2. NO INVENTED FACTS ABOUT PAPERS OR REPOS. For every external method (the social-interaction paper, the router paper, R1-R4 above, JEPA/V-JEPA, Sharingan, pi-ViT, MSTS/multi-scale, Mamba variants, etc.):
   a. find the official paper and official repo; record URLs and the paper version;
   b. read the relevant sections and the official code before implementing;
   c. write a short spec note: what the method does, input/output shapes, what you took, what you changed;
   d. if you cannot verify paper, code, or weights, mark the idea BLOCKED-UNVERIFIED and continue with the others; never implement from vague memory and label it with the paper's name;
   e. I may misremember method names ("pi-ViT", "JePA", "MSTS"); one already turned out to be a version mix-up (2412.20621 is FreqMixFormerV2, not V1). Resolve the correct method first and tell me;
   f. distinguish "faithful reimplementation" (follows the paper) from "adapted" (our changes). Label every variant one or the other.
3. NO FABRICATED NUMBERS. Every number in any report comes from a logged run (config, git hash, manifest hash, seed, fold, metrics file). Never estimate or "expect" results. Failed or unrun work is written NOT RUN. Numbers quoted from papers are labeled [PAPER-REPORTED] with table/section reference and are never mixed into our result tables.
4. NO SILENT FALLBACKS. If a dependency, checkpoint, video, or paper is missing, stop that item and log it.
5. ONE CHANGE AT A TIME relative to a stated reference model. Combos only after individual results, labeled as combos.
6. SANITY-CHECK EVERYTHING NEW: shape and gradient-flow unit tests; overfit ~16 clips to ~100% train accuracy; shuffled-label test must give ~chance; report parameter/FLOP parity or the difference (a gain from more parameters is not a gain from the idea).
7. Label report statements [MEASURED], [INFERRED], or [SPECULATION].
8. When uncertain, ask me. Do not guess silently.
9. NO STACKING LEAKAGE IN STAGED OR CACHED-FEATURE TRAINING. Whenever a later stage is trained on outputs of an earlier stage that was trained on the same training clips (fusion trained on cached features of fine-tuned backbones, router trained on frozen branches, staged-then-joint modes), the later stage sees train-set outputs that are much cleaner than test-time outputs and learns to over-trust them. Either use frozen backbones that never saw our labels, or produce the earlier stage's training-set outputs out-of-fold (inner cross-validation over TRAINING groups only), and say which you did. Verify empirically: report the train-vs-inner-validation gap of the later stage.
10. NO HYPERPARAMETER TRANSPLANTING. Values tuned in a paper (frequency split N, operator coefficients h and l, expert count, top-k, load-balance weight, learning rates) were tuned on other data, sometimes on their test split. Use them only as starting points and re-select them on inner validation splits from training groups only, or fix them in advance and label them "paper default, untuned".
11. LABELS FOLLOW EVIDENCE, NOT TAXONOMY. Assign a class to SOCIAL or RRB by what evidence it needs (a two-person temporal relation vs a single-person motion pattern), verified from label files and sample clips, not by another dataset's or paper's domain names.

# PHASE 0: REFERENCE VERIFICATION (before Phase A experiments; can run alongside Phase A)

- Produce `references/REFERENCE_REGISTRY.md` and one spec note per R1-R4 and per external method (Global Rule 2).
- For R2, resolve the DCT-axis and N ambiguity from the official code and state the answer with file:line. For R1, run the read-only checks on temporal lengths and BatchNorm behavior (no training needed) and report.
- For R3, list which pieces could be implemented from the paper alone and which need code that may not exist; do not implement FineX from the abstract.
- Summary + decisions for me: which R-items are BLOCKED and what file I must supply.

# PHASE A: ARCHITECTURE EXPLANATION AND AUDIT (before any experiment)

Produce two documents.

A1. ARCHITECTURE_AUDIT.md (facts from code): for each block, file:line, tensor shapes in/out, parameter count. Must include:
  - VideoMAE token path: state whether tokens are the full 8x7x7 grid (392x768) linearly projected to 256-D, or pooled anywhere (verify in code);
  - CIGPose extraction, canonicalization, unary encoder, relation encoder (within-person and, before removal, between-person);
  - pose sequence length T and frame rate used by each stream, whether pose and RGB are sampled at the same frames, and the resulting clip duration in seconds; what pose representation is used (keypoint features, not heatmaps) and how many people are tracked;
  - the existing FFT spectral encoder: input, window, number of frequency bins, frequency resolution and Nyquist limit given T and fps, and where it enters the network;
  - ROI encoder, 6OI adapter, quality gate;
  - cross-attention fusion (direction of queries and keys), pooling, learned early gating, fixed late fusion (w_pose).

A2. ARCHITECTURE_EXPLANATION.md (plain-language, for me, a student who wants fundamentals, not just a parts list). For EACH of the router, RRB branch (all three streams), and social branch explain: what problem the block solves, why it is designed that way, exact shapes, what information it can and cannot capture, and known failure modes. Include a diagram (mermaid or ASCII), a table of every tensor with shape, and a "what changed vs the old model and why" section. Add a table "where each candidate idea attaches" (E1-E7 and R1-R4: which block it replaces or augments, and which shapes change). Mark [MEASURED] (from code) vs [INFERRED].

A3. Verify class structure from the label files, not from my description: list every class of AAD and StimASD and assign each to SOCIAL or RRB (my reading of the diagram: AAD joint attention and imitation are social; the other AAD classes and all StimASD classes are RRB). Apply Global Rule 11. Report anything ambiguous. For each dataset and class report: number of clips and subjects/source videos, number of people visible per clip (0/1/2/3+), whether roles (child, caregiver, examiner) are identifiable, typical class duration in seconds versus our clip length (see R4), and whether a second person is present and detectable, since the social branch needs two people. State whether any clip is triadic and how the pair for the social branch would be chosen.

# PHASE 1: STIMASD DATA COMPLETION

- Find the official source and read its access terms and license; follow them. Do not bypass authentication or access restrictions. Also establish the relationship between StimASD and SSBD (same, superset, different) from official sources.
- Build data/stimasd_manifest.csv: id, source URL/reference, download status (OK/dead/private/restricted/corrupt), file hash, duration, fps, resolution, label, subject/child/source-video group ID, failure reason.
- Retry with backoff; use alternative mirrors only where terms allow. Never fabricate or substitute missing videos.
- Report exact counts (listed, obtained, missing by reason, per-class before/after, imbalance). Verify each video decodes, labels match the official annotations, and there are no duplicates (hash plus near-duplicate check).
- Freeze the dataset version (manifest hash); every later experiment states its manifest hash.

# PHASE 2: RIGOROUS EVALUATION PROTOCOL (build before any experiment)

- Splits and leakage: leave-one-group-out CV where the group is the SUBJECT/CHILD or source video, never the individual clip. If subject IDs are unavailable, group by source video and tell me it is the weaker guarantee. LOOCV (group level) for StimASD; official split for AAD if it exists, otherwise grouped stratified k-fold. Automated leakage test asserting zero group overlap in every fold; it must fail loudly and block the runner.
- No peeking: never choose epochs, thresholds, w_pose, router settings, frequency-split parameters, expert counts, or any hyperparameter using the held-out fold. Use a nested inner validation split taken from TRAINING groups only, or a fixed budget chosen before seeing results. Normalization statistics and class weights come from training folds only.
- Staged and cached-feature training obey Global Rule 9; the runner must record, for every staged experiment, whether earlier-stage outputs used for training were out-of-fold or from frozen never-fine-tuned backbones, and refuse to run staged modes otherwise.
- Metrics: primary = macro-F1 (declared in advance). Also accuracy, balanced accuracy, per-class P/R/F1, pooled confusion matrix, per-fold values. Also ECE/NLL. Always report the number of test clips and test groups per class; with tiny classes, say so next to every per-class number.
- Routed-model metrics (report all four so errors can be attributed): (1) end-to-end macro-F1; (2) router accuracy/F1 for social-vs-RRB; (3) branch accuracy GIVEN correct routing; (4) oracle-routing macro-F1 (ground-truth group used at test time; a diagnostic upper bound, never a headline result).
- Statistics: >=3 seeds (ideally 5); mean +- std; group-level bootstrap 95% CIs; paired tests against the reference (paired bootstrap, McNemar, or Wilcoxon over folds); Holm or Benjamini-Hochberg correction, or label results exploratory.
- Determinism and logging: every run writes config.yaml, git hash, manifest hash, seed, fold, per-sample predictions (id, y_true, y_pred, probabilities, router weights), metrics.json.
- One runner: `run_experiment --config X --protocol loocv --seeds 0..4`; it refuses to run if the leakage test fails.

# PHASE 2b: LOSS-CURVE MONITORING (mandatory in every training run)

- Log per epoch AND per N steps: total loss; each component separately (RRB-branch CE, social-branch CE, router CE, any auxiliary/regularization losses such as load-balance or frequency-consistency terms); train loss; INNER-validation loss and metric (from training groups only); learning rate; gradient norm (global and per module: router, RRB streams, social branch); gate values (g, quality gate, router weights, w_pose); parameter-update ratio.
- If a variant has experts: per-expert utilization, router entropy, and fraction of samples per expert, per stream and per class, over training. If a variant has learned frequency operators: the learned high/low coefficients over time. If a variant has lateral or cross-stream exchange: the norm of the exchanged signal relative to the stream's own signal.
- The held-out fold's loss may be logged for post-hoc diagnosis only, clearly flagged "NOT USED FOR ANY DECISION". It must never drive early stopping, checkpoint choice, or tuning.
- Auto-plot per run (PNG + the underlying CSV/JSON): loss curves per component; train-vs-inner-val gap; gradient norms; gate/router-weight and expert-utilization distributions over time. Under LOOCV, save every fold's curve and also produce an aggregate plot (median and IQR across folds/seeds) plus a small-multiples grid so outlier folds are visible.
- Automatic health checks that write flags to the run log: NaN/inf; loss not decreasing over the first N steps; divergence or spikes; train-val gap growth (overfitting); plateau; vanishing/exploding gradients; a branch getting ~zero gradient (a dead branch); gate or router collapse (always ~0 or ~1, or entropy near 0); expert collapse (few experts taking nearly all tokens) or dead experts; one branch dominating the total loss.
- Add a `TRAINING_HEALTH.md` summary per experiment listing flagged folds/seeds. A run with unresolved critical flags cannot be reported as a result.

# PHASE 3: DIAGNOSTICS AND ERROR-ANALYSIS TOOLING

- Grad-CAM-style explanations. RGB/VideoMAE: token-level Grad-CAM or attention rollout on the 8x7x7 grid, overlaid on frames; state the exact method and layer and sanity-check it (map must change under weight or label randomization). Pose: per-region and per-joint saliency (gradient x input or integrated gradients) over the 10 regions and over time. Frequency: which frequency bins or bands drive the prediction for each class (for FreqMix-style variants also the high- and low-frequency attention maps; for the existing FFT encoder its bin-wise contribution), compared against the known dominant frequency of the clip. ROI: per-ROI importance. Fusion: cross-attention maps, gate g, quality-gate values, w_pose. Social branch: GLA attention over the three feature levels and per-person contribution (leader vs assistant). Router: per-sample social/RRB weights and which input features drove them. Experts (if any): expert usage per class and per stream.
- Misclassification browser: a static HTML report (or Streamlit) listing all misclassified held-out clips, filterable by true class, predicted class, fold/seed, and FAILURE TYPE (router error vs branch error given correct routing). For each clip: video with saliency overlays, pose overlay, ROI crops, top-3 probabilities, router and gate weights, and data-quality flags (low pose confidence, missing ROI, second person not detected, wrong or ambiguous person-role assignment, very short clip, clip shorter than the class's typical duration, possible label noise). Show the pooled confusion matrix and most-confused pairs.
- Report failure patterns as [MEASURED] counts first, [INFERRED] explanations second.

# PHASE 4: BUILD THE NEW BRANCHES

## 4.1 RRB branch (existing model, dyadic tokens removed)
- Reference model M0 = old architecture reproduced under the new protocol. M0' = M0 with between-person tokens removed. Report both, mean +- std, so the effect of the removal is measured, not assumed. M0' is the RRB reference for all RRB experiments.

## 4.2 Social-interaction branch (inspired by arXiv 2409.20270)
Read the whole paper (all sections, including §3.5, §3.6, experiments, ablations) and its official code if any; check for newer versions or a published version. My reading so far, which you must VERIFY and not trust: two inputs (leader and assistant person crops); a shared-parameter multi-scale 3D-CNN backbone; features from the last three blocks; "Abstract Projection" (spatiotemporal average pooling then MLP with GELU per block, giving one 768-D vector per block, plus a learnable class token, 1-D positional encoding, and LayerNorm); "Global-Layer-Attention" (query = leader tokens after self-attention; key/value = MLP of assistant tokens; cross-attention); a classification head; and a variant (§3.6) that adds temporal synchronisation modelling for tight interactions.
- Step 1, faithful reimplementation: build it as described; write the spec note; overfit and shuffled-label sanity checks; report params.
- Step 2, adaptation to our data (labeled "adapted"): decide and document
  (a) how to identify the two roles (child vs adult, or leader vs assistant), including what happens when roles are ambiguous or reversed (the paper notes roles can reverse for a less socially active child); test both orderings and a symmetric variant (both cross-attention directions). Use R4 as a source of options, each a separate single-factor comparison: annotated roles if the data has them; a simple child-vs-adult heuristic (size, position, pose scale); an appearance-based role matcher in the style of R4 only if reference crops can be obtained without manual per-video labeling that would not exist at deployment. State how role errors propagate and log role-assignment accuracy on a hand-checked sample;
  (b) how person crops are obtained (reuse the existing pose/person detection; state failure handling when a second person is missing). If a clip contains three people (child, caregiver, examiner), enumerate the candidate pairs, test child-with-adult pairs separately and pooled, and report which choice you made and why;
  (c) backbone choice: paper-style 3D-CNN vs our VideoMAE tokens on per-person crops;
  (d) temporal context: compare the class durations from A3 with clip length; if the social classes are longer than the clip, test a longer-context variant as a single factor.
  Test one change at a time against the faithful version.
- Optional enrichments, each a single-factor experiment: pose-derived person tokens; Sharingan gaze/attention targets (verify paper, repo, weights, license; visually spot-check outputs on N of OUR videos, and if ChildPlay is accessible on N of its clips, before trusting them, since child video quality may break it) to give joint-attention evidence; the §3.6 temporal-synchronisation variant if it fits; a lag-aware imitation model only if the R4 lead (Ji et al.) verifies.
- Ablations to prepare: no GLA (concat/late-fusion instead), self-attention only, use of only the last block vs last three, shared vs unshared backbones, swapped/symmetric roles, leader-only and assistant-only inputs.
- Warning to respect: the social classes may be only ~2 classes in AAD, so this branch's data is tiny. Report class counts, use the strongest regularization allowed by the protocol, and report confidence intervals honestly.

## 4.3 Router
- The router paper has NOT been provided yet. Do NOT guess it, and do NOT substitute FineX (R3), which is not a group router. Build the router as a swappable module with a clean interface, and mark the paper-inspired router BLOCKED-PAPER-NOT-PROVIDED until I send the link.
- Implement now these placeholders, each clearly labeled a baseline/diagnostic:
  R0 oracle (ground-truth group; diagnostic upper bound, never a headline result);
  R1 learned soft gate over the two branches from cheap shared features (document exactly which features; NO label leakage; e.g., pooled global RGB token, pose-derived features such as number of persons detected, inter-person distance, and role-detection confidence);
  R2 hard top-1 routing with a straight-through or Gumbel estimator.
- Training modes to compare (one change at a time): (a) staged: train each branch with ground-truth group masks, then train the router with branches frozen (obey Global Rule 9: the router must not train on branch outputs that are over-fit on its own training clips); (b) joint end-to-end; (c) staged then a short joint fine-tune.
- Losses: masked CE per branch (a branch only receives clips of its own classes); router CE against the group label; optional entropy or load-balance terms. Caution from R3: a standard load-balance term pushes toward equal use of both branches, which contradicts an imbalanced true group prior; if you test it, weight it against the group prior and report its effect, or leave it out. The final class posterior is the router-weighted mixture of branch posteriors over the branch's own classes. Log and plot every component (Phase 2b).
- Router failure modes to monitor: collapse to one branch, high confidence on wrong routes, and the leakage risk that routing features encode the dataset identity. Note that StimASD may have no social classes at all, in which case routing is trivial there and the router must be evaluated mainly on AAD; say so explicitly in the reports.

# PHASE 5: EXPERIMENTS (in this priority order; stop an item if its sanity checks fail)

For each: write the hypothesis and predicted direction BEFORE running; keep everything else fixed; use the Phase 2 protocol; match parameters or report the difference; log training cost and TRAINING_HEALTH. If compute is limited, run the experiments that operate on cached or frozen features (E3, E4, E5) before the ones that need backbone fine-tuning (E1), and tell me you did so.

E1. VIDEO PATH (RRB branch): confirm how 392 tokens are produced; then test multi-scale / MSTS-style tokens (first identify what MSTS refers to), pi-ViT (verify the exact paper and whether its pose modules fit our inputs), VideoMAE V2, V-JEPA/V-JEPA 2, Video Swin, MViTv2/Hiera, UMT, InternVideo2, VideoMamba variants. Deliver a candidate table first (paper URL, code URL, weights Y/N and license, input format, token shape, params, compute at 16x224x224; mark unverifiable). I pick which to run; if I do not answer, run the 2-3 with best verified availability and modest compute. Also test frozen vs partially fine-tuned backbone and layer-wise LR decay.

E2. POSE BACKBONE AND POSE REPRESENTATION (RRB branch): (a) JEPA-style pose model instead of or beside CIGPose. First verify what exists (skeleton/pose JEPA papers vs video V-JEPA). If nothing suitable is pretrained, implement a minimal JEPA-style self-supervised pretraining on training-fold pose sequences only (masked latent prediction, EMA target encoder), labeled "our JEPA-style adaptation". (b) Pose temporal rate: pose is cheap, so test pose sampled at a higher rate than the RGB stream (the idea behind R1's fast pose pathway) as a single factor; first check that higher-rate pose exists or can be extracted, and report the change in frequency resolution. (c) Pose representation candidates table (no implementation until I choose): current keypoint tokens; keypoint-heatmap volumes as in R1 and R3 (PoseC3D-style); skeletal graph (ST-GCN++ style, R3). Verify each one's paper, code, weights, and license.

E3. WITHIN-PERSON RELATION ENCODER WITH FREQUENCY AWARENESS (RRB branch; between-person tokens are removed): FreqMixFormerV2 (R2, primary) and FreqMixFormer V1 (secondary, only if verified), repo https://github.com/wenhanwu95/FreqMixFormer. State exactly what is ported vs re-implemented and how our 65-feature-per-joint canonicalized input and short clips differ from NTU skeletons. Arms, each a single factor against the plain relation encoder at equal parameters: frequency operator switched on with identity coefficients (h = l = 1) as the control; DCT high/low attention blocks; the high/low operator with coefficients selected on inner validation only. Also a "no relation encoder" ablation. Report frequency resolution and Nyquist for our T and fps up front; if the split is not meaningful at our clip length, say so and mark the arm NOT RUN instead of forcing it.

E4. PHYSICS-INSPIRED PRIOR (RRB pose path): start with the simplest testable form (kinematic-consistency features: bone-length constancy, jerk/energy, angular velocity/acceleration; oscillator-fit features for repetitive motion: dominant frequency, amplitude, damping/regularity; or an auxiliary consistency loss). Validate on synthetic sequences with known frequency/amplitude; compare against the existing FFT spectral encoder AND the E3 frequency variants so gains are not duplicated frequency information. Combine E3 and E4 only after each is measured alone.

E5. FUSION, QUALITY GATE AND EXPERTS (RRB branch):
  - Baselines to keep: learned late fusion vs fixed w_pose; current pose-queries-RGB cross-attention.
  - Fusion variants (each a single factor): Flamingo-style tanh-gated cross-attention, bottleneck-token fusion, FiLM, product-of-experts, mixture-of-experts; modality dropout; pose<->ROI fusion variants.
  - From R3, over our three streams (RGB, pose, ROI): symmetric pairwise cross-attention where each stream queries the other two while keeping its identity, vs our one-way pose-queries-RGB. Then, only if that helps, the streamwise sparse expert layer, with a parameter-matched dense-layer control, a scaled-down expert count and top-k chosen on inner validation, the load-balance weight treated as a hyperparameter (Rule 10), and Phase 2b expert-utilization plots. Follow Global Rule 9 for any cached-feature version.
  - From R1: bidirectional exchange between the pose and RGB streams, and exchange at several depths of the two encoders vs only at the end, as separate single-factor tests.
  - Improved quality gate: per-token gating; extra inputs such as pose confidence, ROI sharpness/size, occlusion, model uncertainty; entropy/sparsity regularization; calibration. Test whether cross-attention alone already down-weights corrupted pose (R3 claims this; check it on our data with injected pose noise).
  - Plot gate distributions to check for collapse.

E6. SOCIAL BRANCH AND ROUTER: the Phase 4.2 steps and 4.3 comparisons; when I provide the router paper, implement it (spec note first) and compare against R1/R2 with the four routed-model metrics.

E7. EXTERNAL DOMAIN BASELINE: the RGBPose-SlowFast model (R1) on the RRB classes under exactly the Phase 2 protocol, as a reference row next to M0 and M0'. Run only if the repo's license permits, the code runs after the read-only checks in Phase 0, and its input pipeline (17-channel pose heatmaps, RGB) can be built from our data; otherwise mark BLOCKED. Report which pretraining, if any, it uses. Numbers copied from its paper are [PAPER-REPORTED] and never placed in our tables.

# PHASE 6: RIGOROUS ABLATIONS (prepare the harness now, run at the end)

- Config-driven ablation table generator covering: each stream on/off (RGB, pose, ROI, pairs, all); each RRB component removed (relation encoder, spectral encoder, ROI adapter, quality gate, learned gating, late fusion); dyadic tokens on/off (M0 vs M0'); frequency components (FFT encoder only, DCT operator identity, DCT high/low blocks, physics features); fusion variants (one-way, symmetric, bidirectional, multi-depth; dense vs sparse experts at matched parameters); social branch components (Phase 4.2 list, including role-assignment method and pair choice); router variants and training modes; each new E1-E7 idea. Add the routed-vs-monolithic comparison: a single model with no router, and a model whose branches are both fed everything.
- Each row: same protocol, seeds, and budget; parameter count; paired significance vs the reference; per-class deltas.
- Robustness: pose noise/dropout, missing ROIs, missing second person, wrong role assignment, frame-rate/length variation, router confusion (feed wrong-group clips).
- Controls a reviewer will demand: parameter-matched random-init control; shuffled pose-to-video mismatch; shuffled-label control.
- LaTeX/markdown tables generated from metrics files only; no number typed by hand.

# DELIVERABLES

1. ARCHITECTURE_AUDIT.md and ARCHITECTURE_EXPLANATION.md (Phase A)
2. references/REFERENCE_REGISTRY.md and one spec note per external method, including R1-R4 (Phase 0)
3. stimasd_manifest.csv and DATA_REPORT.md (Phase 1), including the StimASD-vs-SSBD answer and the person-count and clip-duration tables from A3
4. Evaluation harness with leakage tests, including the staged-training check (Phase 2)
5. Loss-curve logging, plots, health checks, TRAINING_HEALTH.md per experiment (Phase 2b)
6. Grad-CAM tools and misclassification browser (Phase 3)
7. results/ with per-run logs and RESULTS.md with auto-generated tables
8. FINDINGS.md: what worked, what did not, BLOCKED-UNVERIFIED / BLOCKED-PAPER-NOT-PROVIDED / NOT RUN items, threats to validity, ranked next steps, all labeled [MEASURED]/[INFERRED]/[SPECULATION]

# WORKING STYLE

- Work phase by phase. After each phase, give a short summary and list decisions I must make (dataset access issues, which E1 backbones and E2 pose representations to run, compute budget, router paper, role-assignment choices, which reference PDFs I must supply).
- Make reasonable judgment calls yourself on small ambiguities and record them in a `DECISIONS.md` with your reasoning; only stop to ask me for things that change the science, the data access, or the budget.
- If an instruction is ambiguous or seems wrong, say so instead of guessing.
- Never report improvement without protocol-compliant numbers.
