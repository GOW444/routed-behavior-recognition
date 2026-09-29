#!/usr/bin/env bash
# Clone the external reference repositories at the exact commits inspected in Phase 0.
# Code is cloned into third_party/ (git-ignored) and is NOT vendored into this repo:
# R1 and R2 ship no license file (see references/REFERENCE_REGISTRY.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"

clone_at() {  # name url commit
  local name=$1 url=$2 commit=$3
  if [ ! -d "$TP/$name/.git" ]; then
    git clone -q "$url" "$TP/$name"
  fi
  git -C "$TP/$name" fetch -q origin "$commit" 2>/dev/null || true
  git -C "$TP/$name" checkout -q "$commit"
  echo "$name @ $(git -C "$TP/$name" rev-parse HEAD)"
}

# R1 RGBPose-SlowFast (ICIP 2022) code
clone_at rgbpose_slowfast \
  https://github.com/Jeba-create/Stimming-Behaviour-detection-using-RGB-POSE-SlowFast-Network \
  03978e1756b5d3501f7390e87ac92c94e237e45d
# R2 FreqMixFormer (V1 code; V2 code not found in this repo, see spec note)
clone_at freqmixformer \
  https://github.com/wenhanwu95/FreqMixFormer \
  1d3c2add885888f6eaa93135b9fb11dfdbc42be2
