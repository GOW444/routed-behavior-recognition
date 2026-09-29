# Spec note R1: RGBPose-SlowFast (stimming, ICIP 2022)

- **Paper:** NOT READ (BLOCKED-UNVERIFIED; need `references/ICIP2022_RGBPose_SlowFast.pdf`). Everything below is from **code only**. Where the paper might say something different, the paper has not been checked.
- **Code:** `Jeba-create/Stimming-Behaviour-detection-using-RGB-POSE-SlowFast-Network` @ `03978e1`. Paths below are relative to that repo (`third_party/rgbpose_slowfast/` after running `scripts/fetch_references.sh`).
- **Checks script:** `tools/phase0/r1_readonly_checks.py`. Output: `results/phase0/r1_checks.json` (torch 2.14.0, CPU, random inputs, no training, no data).
- **Label:** anything we later build from this is **"adapted"**, never "faithful". The released code is not runnable as-is (§5), so there is no faithful executable reference.

## 1. What the model is `[MEASURED]` from code

Two 3D-ResNet pathways, "slow" = RGB and "fast" = pose heatmaps, with lateral connections in **both** directions after stages 1, 2 and 3. The two pooled features are concatenated and passed through dropout and one linear layer.

| | RGB ("slow") pathway | Pose ("fast") pathway |
|---|---|---|
| input | `(B, 3, T_s, 224, 224)` | `(B, 17, T_f, 56, 56)`: 17 COCO keypoint Gaussian heatmaps |
| stem | Conv3d k(1,7,7) s(1,2,2) + BN + ReLU + MaxPool s(1,2,2) (`modelrgbposeslowfast.py:47-50`) | Conv3d k(1,7,7) s1, **no spatial downsampling, no maxpool** (`:52-54`) |
| stage 1 | `layer1_slow` (`:64`) | **none** (stem feeds lateral 1 directly) |
| stages 2-4 | `layer{2,3,4}_slow` (`:65-67`) | `layer{2,3,4}_fast` (`:70-72`), narrower |
| head | AdaptiveAvgPool3d(1) on each, concat, Dropout(0.5), Linear (`:84-87`, `:126-137`) | |

- **No temporal downsampling anywhere** (all temporal strides are 1). Temporal kernels of 3 appear only in stages 3-4.
- Laterals: fast→slow uses `Conv3d(k=(5,1,1), stride=(5,1,1), pad=2)` (`:79-81`), and slow→fast uses `Conv3d(k=(5,1,1), stride=(8,1,1), pad=2)` (`:75-77`). They are merged by **addition** (`:103-104, :109-110, :119-120`), not concatenation.
- Two variants: `RgbPoseSlowFastResNet50` (bottleneck `block3d`, `:156`, default `num_classes=8`) and `RgbPoseSlowFastResNet34` (`block3dSmall`, `:300`, default 8). `main.py:163` trains the **ResNet34-style** variant with `num_classes=3`.
- Parameter counts at `num_classes=3` `[MEASURED]`: **ResNet34-style 43,762,179; ResNet50-style 37,002,019**. The "small" variant is actually *larger*, because `block3dSmall` uses full-width (1,3,3)/(3,3,3) kernels without a bottleneck.

## 2. Temporal lengths actually used `[MEASURED]`

- `main.py:115-116`: `NoOfFrms2slct_s=4` RGB frames and `NoOfFrms2slct_f=20` pose frames per clip.
- RGB frames: `Dataloaderrgbpose.py:165-170` samples 4 frames spread **uniformly over the whole video** (random offset inside each segment). The pose sampler (`Poseaugmentation.py:288-305`) independently samples 20 frames over the whole video. **The RGB and pose frames are sampled independently, so they are not the same time instants** `[MEASURED]`. The pose rate is about 5 times the RGB rate *on average*. Because the clip spans the whole video, the effective fps depends on each video's length.

## 3. The lateral-stride question: silent broadcasting CONFIRMED `[MEASURED]`

With (T_s, T_f) = (4, 20), batch 2, the tensors that are summed in `forward()` are:

| add (code line) | target shape | lateral shape | result |
|---|---|---|---|
| `x1 + lateral1_fast(x2)` (:103) | (2,64,**4**,56,56) | (2,64,**4**,56,56) | OK: 20 → 4 via stride 5 |
| `lateral1_slow(x1) + x2` (:104) | (2,16,**20**,56,56) | (2,16,**1**,56,56) | **silent broadcast over time** |
| `x1 + lateral2_fast(x2)` (:109) | (2,128,4,28,28) | (2,128,4,28,28) | OK |
| `lateral2_slow(x1) + x2` (:110) | (2,32,20,28,28) | (2,32,**1**,28,28) | **silent broadcast** |
| `x1 + lateral3_fast(x2)` (:119) | (2,256,4,14,14) | (2,256,4,14,14) | OK |
| `lateral3_slow(x1) + x2` (:120) | (2,64,20,14,14) | (2,64,**1**,14,14) | **silent broadcast** |

(ResNet34-style shown. The ResNet50-style variant behaves the same with 4× the channels.)

- **Why:** a stride-8 conv over T_s=4 (pad 2, k 5) outputs 1 frame. That single slow→fast summary is added identically to all 20 pose frames. So the "RGB → pose" direction carries **no temporal information** at any depth. It acts as a per-clip bias.
- Other lengths: (4,32), (8,32) and (16,64) **crash** with a size mismatch. (8,40) runs, but also broadcasts. `[MEASURED]` From the conv arithmetic, fast→slow outputs ⌊(T_f−1)/5⌋+1 frames, which must equal T_s. Slow→fast outputs ⌊(T_s−1)/8⌋+1 frames, which is 1 (a broadcast) whenever T_s ≤ 8. So every working configuration with T_s ≤ 8 broadcasts. The design is tied to about 4/20.
- Implication for our E5 "bidirectional exchange" test `[INFERRED]`: R1 as released is effectively **one-way temporal exchange (pose → RGB) plus a global RGB → pose bias**. It does not demonstrate true bidirectional exchange. If we test bidirectionality, we must build it with matched temporal resolution and state that it is "adapted".

## 4. BatchNorm behaviour `[MEASURED]`

- `block3d` (ResNet50-style) builds all three BNs with `momentum=0, track_running_stats=False` (`:9, :11, :13`): 87 BN layers. The stems and downsample BNs are standard (9 layers).
  - In PyTorch, `track_running_stats=False` means **batch statistics are used in eval mode too** (`momentum` is irrelevant).
  - Test, eval mode: the logits of sample 0 differ by **0.0149** (max abs) between batch size 1 and batch size 4, and by **0.219** when the other three batch members are swapped for different clips. **Predictions depend on what else is in the batch.** Any test metric from this variant depends on batch size and batch composition.
- `block3dSmall` (ResNet34-style, the variant `main.py` actually trains) uses standard BN (67 layers, running stats). Eval outputs are batch-independent (diff 7e-9, 0.0).
- `main.py:159-160` uses `batch_size=2`. BN with batch 2 is itself very noisy during training `[INFERRED]`.

## 5. The code does not run as released `[MEASURED]` (pyflakes + execution)

- `modelrgbposeslowfast.py` never imports `torch` but calls `torch.cat` (`:131`, `:277`). Unmodified forward raises `NameError: name 'torch' is not defined`.
- `Dataloaderrgbpose.py`: undefined names `torch` (:76, :77, :184, :200), `os` (:189), `UniformSampler` (:85), and the globals `left_kps/right_kps/areaa_range/scalee/NoOfFrms2slct_s` that exist only in `main.py` (:99-170). It imports `poseaugmentation` (:12), but the file is `Poseaugmentation.py`, which fails on case-sensitive filesystems.
- `Poseaugmentation.py`: undefined `cp`, `Flip` (:267-268), `randrange` (:297), `random` (:472-473).
- `main.py`: `DataLoader` undefined (:159-160).
- `[INFERRED]` The released code is a partial snapshot of a Colab notebook (`/content/...` paths at `main.py:126-137`). The paper's numbers cannot have come from these files as-is.

## 6. Evaluation protocol in the released code: serious problems `[MEASURED]`

These matter if anyone compares against R1's paper numbers. Those numbers stay `[PAPER-REPORTED]` and are never placed in our tables.

1. **The model trains on the held-out fold.** `main.py:179` computes the "validation" loss with `train(net_sf, optimizer, criterion, dataloader_val, device)`. That calls `net.train()`, `loss.backward()` and `optimizer.step()` on the val fold every epoch, so the reported `Val acc` is training accuracy on data it just fit.
2. **Clip-level folds, no grouping.** `KFold(n_splits=3, shuffle=True)` over clips (`:147-150`), with no subject or source-video groups, so leakage across folds is possible. No `random_state` is set: `torch.manual_seed(42)` at `:144` does not seed sklearn/numpy, so folds are not reproducible.
3. **Checkpoint chosen on the held-out fold** (`:184-189`, min "val" loss).
4. **Reported fold result = last-epoch "val" accuracy** (`:208`), i.e. item 1.
5. **Val uses train augmentation.** A single dataset object with `phase='train'` (`:138`) feeds both loaders (`:159-160`). Also, `Dataloaderrgbpose.py:44` `if mode=='train' or 'val'` is always true, and `Poseaugmentation.py:279` hard-codes `self.mode='train'`. So sampling is always random.
6. **Per-frame inconsistent RGB augmentation.** `Dataloaderrgbpose.py:57-68` applies a different transform to each frame index: frame 0 is flipped (p=0.9), frame 1 resize+center-crop, frame 2 center-crop 128, frame 3 rotation, frame 4 color-jitter. With 4 frames, this breaks temporal consistency within a clip.
7. `with_limb=True` is passed (`main.py:138`) but ignored: `GeneratePoseTarget()` is built with defaults (`Dataloaderrgbpose.py:160`), so heatmaps are keypoint-only (17 ch, σ=0.6, scaled by keypoint score).
8. `Poseaugmentation.py:307-308` caches `Frmstoselect` into the shared annotation dict on first use. Whether pose frames are then frozen or resampled depends on DataLoader worker lifetime (`num_workers=2`, non-persistent) `[INFERRED]`.

## 7. What to take / what to change (for E2, E5, E7)

- **Take as ideas (adapted, reimplemented; no code copied because there is no license):** (a) pose at a higher temporal rate than RGB; (b) exchange at several depths; (c) 17-channel keypoint heatmap volumes as a pose representation (PoseC3D-style, provenance mmaction2/pyskl, Apache-2.0, which we can use directly with attribution).
- **Change:** matched-resolution bidirectional laterals (no broadcast); standard BN with running stats; RGB and pose sampled on a shared timeline; group-level protocol.
- **E7 external baseline:** `[INFERRED]` feasible only as an **adapted reimplementation** under our Phase 2 protocol. Running the released code needs fixes to about 30 undefined names, and there is no license. **Your decision is needed** (see `PHASE0_SUMMARY.md`).
- **SSBD / StimASD question:** not answerable from the code. The README says it evaluated on SSBD plus an "Autism dataset", and `main.py` uses `ssd_*` file names with 3 classes. `[INFERRED]` 3 classes matches SSBD's three behaviours (arm flapping, head banging, spinning), but that is unverified. Deferred to Phase 1.
