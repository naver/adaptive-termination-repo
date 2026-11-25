#!/usr/bin/env bash
# adaptive-termination-repo
# Copyright (c) 2026-present NAVER Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# rank-K termination calibration of one cell: the same procedure as adaptive termination (calibration queries #1000-1499, gamma = 0
# record, lib/pred_calib.py cell, per-shard gamma_i), with the rule comparing against the K-th distance (rank K).
# Result: key S{S}_E{E} in $RESULTS/cells/<dsn>/dabs/calib_cells.json. Skipped when the key exists.
#
# At S = 1 adaptive termination's rank k_i equals K, so rank-K termination and adaptive termination are the same configuration (same record,
# same calibration). If adaptive termination's entry exists it is copied instead of recomputed.
#
# Usage: calib_dabs.sh <ds> <S> <E>
set -uo pipefail
EXP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$EXP/config.sh"
[ $# -eq 3 ] || die "usage: calib_dabs.sh <ds> <S> <E>"
DS=$1; S=$2; E=$3
N=$(ds_n "$DS") || exit 1
DSN=${DS}_N${N}; KEY=S${S}_E${E}
CELLS=$RESULTS/cells/$DSN/dabs/calib_cells.json
RULE_CELLS=$RESULTS/cells/$DSN/calib_cells.json
mkdir -p "$(dirname "$CELLS")"
[ -s "$CELLS" ] && $PY -c "import json,sys;sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1])) else 1)" "$CELLS" "$KEY" && exit 0

if [ "$S" = 1 ] && [ "$(rank_of dabs 1)" = "$(rank_of rule 1)" ] && [ -s "$RULE_CELLS" ] \
   && $PY -c "import json,sys;sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1])) else 1)" "$RULE_CELLS" "$KEY"; then
  $PY - "$RULE_CELLS" "$CELLS" "$KEY" <<'EOF' || die "copy of the S=1 entry failed"
import json, sys
from pred_calib import update_table
src, out, key = sys.argv[1:4]
update_table(out, key, json.load(open(src))[key])
EOF
  echo "  dabs $DSN $KEY = adaptive termination at S=1 (copied)"
  exit 0
fi

RANK_MODE=dabs "$EXP/cell.sh" table "$DS" "$S" "$E" || exit 1
$PY -c "import json,sys;g=json.load(open(sys.argv[1]))[sys.argv[2]]['gamma_rank_shard'];print('  dabs %s %s gamma_i min %s max %s off %d' % (sys.argv[3],sys.argv[2],min(x or 0 for x in g),max(x or 0 for x in g),sum(x is None for x in g)))" "$CELLS" "$KEY" "$DSN"
