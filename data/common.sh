# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Helpers shared by the data scripts. Source it:  . "$(dirname "$0")/common.sh"
# It sources experiments/config.sh, so every variable there (WORK, RAW_DIR, IDX_DIR, BIN, PY, THREADS, ...) is available.

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$DATA_DIR/../experiments/config.sh"

DL_DIR="${DL_DIR:-$WORK/download}"      # $DL_DIR/<ds>/  downloaded archives (can be deleted after convert.sh)
DATASETS_ALL="t2i msmarco glove sift"   # default list; the 100M sets (deep, t2i100m) are run only when named
M_HNSW="${M_HNSW:-40}"                  # HNSW out-degree parameter M (lib/shardlib.py M)
EFC_HNSW="${EFC_HNSW:-200}"             # efConstruction (lib/shardlib.py EFC)

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

fsize() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }   # GNU stat, then BSD stat

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi
}

# fbin_rows FILE -> the row count in the header
fbin_rows() { "$PY" -c "import numpy as np, sys; print(int(np.fromfile(sys.argv[1], np.int32, 1)[0]))" "$1"; }

# Environment overrides N=... and NORM=... (e.g. for the synthetic set), captured once so that loops can reuse the names
N_OVERRIDE="${N:-}"
NORM_OVERRIDE="${NORM:-}"
# data_n DS -> N used for DS (the six datasets from config.sh; anything else, e.g. synth, from the header of its base.fbin)
data_n() {
  if [ -n "$N_OVERRIDE" ]; then echo "$N_OVERRIDE"; return; fi
  ds_n "$1" 2>/dev/null || fbin_rows "$RAW_DIR/$1/base.fbin"
}
# data_norm DS -> 1 (normalize, cosine order) or 0 (raw L2: SIFT, DEEP)
data_norm() { if [ -n "$NORM_OVERRIDE" ]; then echo "$NORM_OVERRIDE"; else ds_norm "$1"; fi; }

# remote_size URL -> Content-Length after redirects (empty if the server does not say)
remote_size() {
  curl -sIL --max-time 60 "$1" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}'
}

# fetch URL OUT [EXPECTED_BYTES] [MD5]
#   Downloads with resume (OUT.part), checks the size and the MD5 when given, then renames to OUT. Skips if OUT is complete.
fetch() {
  local url=$1 out=$2 want=${3:-} md5=${4:-}
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ]; then
    [ -z "$want" ] || [ "$(fsize "$out")" = "$want" ] || die "$out exists with size $(fsize "$out"), expected $want"
    log "have $(basename "$out")"; return 0
  fi
  if [ -n "$want" ] && [ "${CHECK_REMOTE_SIZE:-1}" = 1 ]; then
    local rs; rs=$(remote_size "$url")
    [ -z "$rs" ] || [ "$rs" = "$want" ] || die "$url: server reports $rs bytes, expected $want (file changed?). CHECK_REMOTE_SIZE=0 skips this check"
  fi
  local have=0; [ -f "$out.part" ] && have=$(fsize "$out.part")
  if [ -z "$want" ] || [ "$have" != "$want" ]; then
    log "download $url -> $out"
    curl -fL --retry 10 --retry-delay 10 -C - -o "$out.part" "$url"
  fi
  [ -z "$want" ] || [ "$(fsize "$out.part")" = "$want" ] || die "$out.part: size $(fsize "$out.part"), expected $want"
  if [ -n "$md5" ]; then
    log "md5 $(basename "$out")"
    local got; got=$(md5_of "$out.part")
    [ "$got" = "$md5" ] || die "$out.part: MD5 $got, expected $md5 (delete the .part file and retry)"
  fi
  mv "$out.part" "$out"
}

# fetch_range URL OUT NBYTES [PARTS]
#   Downloads bytes 0..NBYTES-1 of URL in PARTS parallel byte ranges, each resumable, then concatenates them into OUT.
fetch_range() {
  local url=$1 out=$2 nbytes=$3 parts=${4:-8}
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ]; then
    [ "$(fsize "$out")" = "$nbytes" ] || die "$out exists with size $(fsize "$out"), expected $nbytes"
    log "have $(basename "$out")"; return 0
  fi
  local ps=$(( (nbytes + parts - 1) / parts )) i pids=()
  log "download bytes 0-$((nbytes - 1)) of $url in $parts ranges -> $out"
  for i in $(seq 0 $((parts - 1))); do
    local start=$((i * ps)) end=$(( (i + 1) * ps )); [ "$end" -gt "$nbytes" ] && end=$nbytes
    local f="$out.part$i" want=$((end - start)) have=0
    [ "$want" -gt 0 ] || continue
    [ -f "$f" ] && have=$(fsize "$f")
    [ "$have" -le "$want" ] || die "$f is longer than its range ($have > $want); delete it"
    if [ "$have" -lt "$want" ]; then
      ( curl -fsS --retry 10 --retry-delay 10 -r "$((start + have))-$((end - 1))" "$url" >> "$f" ) &
      pids+=($!)
    fi
  done
  local p; for p in "${pids[@]+"${pids[@]}"}"; do wait "$p" || die "a range download failed; rerun to resume"; done
  : > "$out.part"
  for i in $(seq 0 $((parts - 1))); do
    local start=$((i * ps)) end=$(( (i + 1) * ps )); [ "$end" -gt "$nbytes" ] && end=$nbytes
    [ $((end - start)) -gt 0 ] || continue
    [ "$(fsize "$out.part$i")" = $((end - start)) ] || die "$out.part$i: size $(fsize "$out.part$i"), expected $((end - start)); rerun to resume"
    cat "$out.part$i" >> "$out.part"
  done
  [ "$(fsize "$out.part")" = "$nbytes" ] || die "$out.part: wrong size after concatenation"
  mv "$out.part" "$out"
  for i in $(seq 0 $((parts - 1))); do rm -f "$out.part$i"; done
}
