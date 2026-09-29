"""Frequency resolution / Nyquist calculator for a clip of T sampled frames.

Needed before deciding whether a temporal DCT high/low split (FreqMixFormer-style) or
the existing FFT spectral encoder can resolve rhythmic RRB motion at our clip length.

  effective_fps = video_fps / frame_stride      (the rate at which the T frames are sampled)
  duration_s    = T / effective_fps
  Nyquist       = effective_fps / 2
  rFFT bins     = T // 2 + 1, spacing  df = effective_fps / T
  DCT-II coeff k corresponds to  k * effective_fps / (2 T)  Hz,  spacing = effective_fps / (2T)

Our actual T, video fps and stride are NOT yet measured (the CIGPose-VideoMAE code is not in
this repo yet; Phase A must read them from code). Pass them in once known:

  python tools/phase0/freq_resolution.py --T 16 --fps 30 --stride 4
"""
import argparse
import json


def resolution(T: int, fps: float, stride: int = 1) -> dict:
    eff = fps / stride
    return {
        "T": T, "video_fps": fps, "frame_stride": stride,
        "effective_fps": eff,
        "clip_duration_s": T / eff,
        "nyquist_hz": eff / 2,
        "rfft_bins": T // 2 + 1,
        "rfft_bin_spacing_hz": eff / T,
        "dct_coeffs": T,
        "dct_coeff_spacing_hz": eff / (2 * T),
        "lowest_nonzero_rfft_hz": eff / T,
        "min_cycles_visible_at_1hz": T / eff,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    print(json.dumps(resolution(a.T, a.fps, a.stride), indent=2))


if __name__ == "__main__":
    main()
