#!/usr/bin/env bash
# FAKE stand-in for Generation/converters/batch_build_rand_hands2.sh used only by Main/tests/cpu:
# same CLI and hand-name rule (incl. the _t<tips> token); writes a JSON "URDF".
set -euo pipefail
OUT=""; CODES=""; MODE=""; PLACEMENT=""; THUMB=""
T1=standard; T2=standard; T3=standard; T4=standard; T5=standard
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) OUT="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    -C) CODES="$2"; shift 2;;
    --placement-mode) PLACEMENT="$2"; shift 2;;
    --thumb-fixed-slot) THUMB="$2"; shift 2;;
    --fingertip-f1) T1="$2"; shift 2;;
    --fingertip-f2) T2="$2"; shift 2;;
    --fingertip-f3) T3="$2"; shift 2;;
    --fingertip-f4) T4="$2"; shift 2;;
    --fingertip-f5) T5="$2"; shift 2;;
    --palm-radius|--symmetric-start-deg|--symmetric-jitter-deg|--slot-count|--min-sep-slots|--seed|--anthro-top-fixed-slots|--thumb-fixed-servos) shift 2;;
    --thumb-bottom-deg|--top-band-deg) shift 3;;
    --save-xml) shift;;
    -h|--help) echo "fake build help"; exit 0;;
    *) echo "fake_build: unknown arg $1" >&2; exit 2;;
  esac
done
[[ -n "$OUT" && -n "$CODES" ]] || { echo "fake_build: need -o and -C" >&2; exit 1; }
if [[ -n "${FAKE_BUILD_FAIL_SUBSTR:-}" && "$CODES" == *"${FAKE_BUILD_FAIL_SUBSTR}"* ]]; then
  echo "fake_build: simulated failure for $CODES" >&2; exit 1
fi
IFS=',' read -r C1 C2 C3 C4 C5 <<< "$CODES"
name="hand"
i=1
for c in "$C1" "$C2" "$C3" "$C4" "$C5"; do
  if [[ -n "$c" ]]; then name="${name}_f${i}_${c}"; else name="${name}_f${i}"; fi
  i=$((i+1))
done
tok=""; i=1
for c in "$C1" "$C2" "$C3" "$C4" "$C5"; do
  t="T$i"; if [[ -n "$c" ]]; then tok="${tok}${!t:0:1}"; else tok="${tok}x"; fi; i=$((i+1))
done
name="${name}_t${tok}"
mkdir -p "$OUT/meshes" "$OUT/robot_meshes"
printf '{"hand_name":"%s","codes":["%s","%s","%s","%s","%s"],"tips":["%s","%s","%s","%s","%s"],"mode":"%s","thumb":"%s"}\n' \
  "$name" "$C1" "$C2" "$C3" "$C4" "$C5" "$T1" "$T2" "$T3" "$T4" "$T5" "$PLACEMENT" "$THUMB" > "$OUT/${name}.urdf"
echo "=== Building ${name}"
echo "→ $OUT/${name}.urdf"
