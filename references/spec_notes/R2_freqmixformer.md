# Spec note R2: FreqMixFormer (V1) / FreqMixFormerV2

- **Papers:** V1 arXiv 2407.12322 (ACM MM 2024) and V2 arXiv 2412.20621 (FG 2025). **Neither was read** (arXiv blocked), so both are BLOCKED-UNVERIFIED.
- **Code:** `wenhanwu95/FreqMixFormer` @ `1d3c2ad`. This is **V1 only**. V2 code is not in the repo (see the registry). No LICENSE file.
- **Checks script:** `tools/phase0/r2_dct_axis_check.py`. Output: `results/phase0/r2_dct_axis.json`.
- **Label:** any E3 arm built from this is **"adapted"**. A "faithful V2" arm is **impossible** until V2 code or at least the V2 paper is available. A "faithful V1" arm would need the V1 paper to confirm that the code matches it.

## 1. V1 code structure `[MEASURED]`

`model/skefreqmixformer.py:58-162` `Model` works as follows:
- A joint embedding `Linear(in_channels=3 → 80)` plus a learned positional embedding over 25 joints (`:73-74`). Before the embedding, a graph "k-hop" transform `I − A_outward^8` is applied (`:67, :123-134`).
- Ten `Ske_MixF` blocks (`:76-85`), with channels 80→160→320 and temporal stride 2 at l5 and l8. The configs use T=64 frames (`window_size: 64`).
- Multi-scale skip "trams" (`:103-114`), then a mean over time and persons, then FC (`:155-162`).

Each `Ske_MixF` block is `Spatial_Freq_MixFormer`, then `Temporal_MixFormer`, plus a residual (`:40-55`).

Parameter count `[MEASURED]`: **2,065,048** (60 classes) and **2,084,308** (120 classes). This is consistent with the "about 2.0M" lead for V1.

## 2. RESOLVED: which axis the DCT and the split index N act on (V1 code)

**Answer `[MEASURED]`: the DCT is over the JOINT axis (V = 25), not over time. The split index (`d = 13`) indexes joint-axis DCT coefficients, i.e. columns of a 25×25 attention map. The V1 model contains no temporal DCT at all.**

Evidence:
1. `torch_dct.dct` is DCT-II over the **last** dimension (library source `torch_dct/_dct.py:57-67`). A unit test confirms it: a signal constant along the last dim has max |coeff k≥1| = 1.9e-6.
2. `model/spa_freq_mixf.py:117-120` calls `dct.dct(m[:, i, c0:c1, :, :])`, where `m` is laid out `(N, C, T, V)` (`:94, :100-101`). The logged call input shape is `(2, 20, 64, 25)`, so the transformed axis is **V = 25 joints**.
3. The coefficients go into `Spa_Atten` (`:37-46`), which permutes to `(N, V, C, T)` and **average-pools over C and T** (`:41-42`). The result is one scalar per joint-DCT coefficient. That vector passes through `Linear(25, 25)` (`:35`) and a ReLU, and an outer product plus softmax gives a 25×25 map.
4. The high/low operator (`:127-135`): `m1_2d[:, :, :, :d] *= l; m1_2d[:, :, :, d:] *= h` with `l = 0.5, h = 1.5, d = 13` hard-coded (`:127, :128, :129`). It acts on the **last axis of the 25×25 cross-attention maps** (`m1_2d, m2_3d, m3_4d` only; the self maps `m1d..m4d` are not scaled). So `d` splits the 25 joint-DCT coefficient indices, not frames.
5. `dct.idct` (`:137-140`) is then applied to those **attention maps**, with logged input `(2, 20, 25, 25)`. That is an inverse DCT along the joint-coefficient axis of an attention matrix.
6. `model/tem_mixf.py:7` imports `torch_dct` but **never calls it** (grep). The temporal module is not frequency-based.

`[INFERRED]` consequences:
- In this code, "high/low frequency" means spatial frequency along the **joint numbering order** (the NTU index order, which is arbitrary with respect to body topology), after averaging out time. It is **not** the temporal frequency of motion that the paper text reportedly describes. Rhythmic RRB motion (flapping, rocking) is a temporal-frequency phenomenon, so **V1 as coded does not model it**. The existing FFT spectral encoder does, over time.
- The prompt's point (a), "DCT over frames vs N over joints", is therefore resolved *for V1 code*: both are over joints. Whether V2 changes this is **unknown** (no V2 code; paper unread).
- `l` and `h` are **fixed constants**, not learned. Phase 2b's "learned high/low coefficients" logging does not apply to V1 as coded.

## 3. Other code facts relevant to porting `[MEASURED]`

- **25 joints are hard-coded:** `nn.Linear(25, 25)` (`:35, :59`), and `A` is reshaped to `[3, 1, 25, 25]` (`:56-57`). `A_final` is 25×25 (`:97`). Our joint set would need these generalised.
- **CUDA is hard-coded:** `self.A_SE.cuda(x0.get_device())` (`:95`), `device='cuda'` (`:97`). The check script replaces only these two strings to run on CPU.
- **Input is 3 channels (xyz) per joint:** `in_channels=3`, `to_joint_embedding` (`skefreqmixformer.py:73`). Ours has 65 canonicalised features per joint, so the embedding layer changes.
- **Deprecated init calls:** `nn.init.normal`, `nn.init.constant` and `nn.init.kaiming_normal` emit warnings in torch 2.x and still run.

## 4. Point (b), hyperparameters possibly chosen on test data

- `[MEASURED]` The released training script **selects the checkpoint on the test split**. `main.py:458-460` tracks `best_acc` on `loader_name=['test']`, and `:505-508` reloads the weights of the best-test epoch for the final evaluation. The configs define only `split: train` and `split: test` (`config/nturgbd-cross-subject/joint.yaml:5, :19`). **There is no validation split in the code.**
- `[INFERRED]` Any hyperparameter search run with this code (N/d, l, h) was necessarily scored on test data. Whether the paper's (N=13, l=0.2, h=1.2) search for V2 was done this way can't be confirmed without the paper. Either way, **Rule 10 applies: do not transplant d/l/h.** Note also that the V1 code uses l=0.5, h=1.5, which differ from the V2 values you quoted.

## 5. Points (c) and (d)

- (c) Single runs / variance: **NOT CHECKED** (paper unread). The code has no seed loop or variance reporting (`main.py` trains one run).
- (d) Input differences (ours vs NTU): 65 features/joint vs 3; our own joint set vs 25 NTU joints; short clips vs T=64; 1-2 persons, with our RRB branch having no between-person tokens. The NTU model pools over M=2 persons at the end (`skefreqmixformer.py:159-160`).

## 6. Frequency resolution for a temporal adaptation

This only matters for an **adapted** arm where we deliberately apply the DCT over time (which V1 code does not do):
- Use `tools/phase0/freq_resolution.py --T <frames> --fps <video fps> --stride <sampling stride>`. For the DCT, coefficient k ↔ k·f_eff/(2T) Hz, with spacing f_eff/(2T) and Nyquist f_eff/2.
- **Our T, fps and stride are NOT MEASURED yet.** The CIGPose-VideoMAE code is not in this repo, so this is a Phase A item. Whether a high/low split is meaningful is decided only after those numbers are read from our code.

## 7. What we would take / change (E3, pending the papers)

- **Take (adapted):** the mixed self/cross attention over two (or four) channel groups; a frequency-domain attention branch; a fixed or learned band re-weighting operator with an identity (l = h = 1) control arm.
- **Change:** explicitly choose the transform axis (time is the motivated axis for RRB) and label it "adapted"; generalise from 25 joints; select d/l/h on inner validation only; parameter-match against the plain relation encoder.
- **Do not run E3 "V2" arms** until V2 paper and code are available.
