#!/usr/bin/env bash
# Build and run the test suite with maximum safe parallelism.
#
#   tools/run_tests.sh              # light tests in parallel, GLM tests serialised
#   tools/run_tests.sh --fast       # skip the tests that map the 100 GB checkpoint
#   tools/run_tests.sh --warm       # pre-warm the checkpoint's page cache first
#   tools/run_tests.sh --only tok   # substring filter on the test name
#
# Two classes, for one hardware reason: a GLM test maps ~100 GB of weights, so
# two of them cannot run together on a 128 GB machine (they evict each other's
# pages and end up slower than serial), and Metal is a single device. Everything
# else - the compiles and the CPU/small-model tests - runs at full machine width.
#
# Every test links ALL sources except main.cpp: the per-test build lines in the
# file headers go stale as symbols move between translation units (that is
# exactly how this script's first version silently failed), so the runner owns
# the link line instead.
set -o pipefail
cd "$(dirname "$0")/.."
MLX=$(brew --prefix mlx)
MODEL=${GLM5_MODEL:-/Volumes/External/Models/MikeRoz/GLM-5.3-Flash-Uncensored-2.51bpw-h6-exl3}
JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 8)
OUT=/tmp/exl3_tests; mkdir -p "$OUT"
SRC=$(ls src/*.cpp | grep -v '^src/main.cpp$' | tr '\n' ' ')
FAST=0; WARM=0; ONLY=""
while [ $# -gt 0 ]; do case "$1" in
  --fast) FAST=1;; --warm) WARM=1;; --only) ONLY="$2"; shift;;
  *) echo "unknown flag: $1"; exit 2;; esac; shift; done
if [ "$WARM" = 1 ]; then
  echo "== warming the page cache =="
  ls "$MODEL"/model-*.safetensors | xargs -P "$JOBS" -n1 cat > /dev/null
fi
# L = light (parallel)   G = maps the checkpoint (serial)
PLAN="
kda_recurrence L
kda_rewind L
hc L
moe L
mla_incremental L
indexer_pooling L
indexer_select L
indexer_topk L
chat_template L
pretokenizer L
glm5_load G
glm5_forward G
glm5_mtp G
glm5_exl3_decode G
glm5_rewind G
glm5_batch G
glm5_bench G
"
sel() { awk -v want="$1" -v o="$ONLY" '$2==want && (o=="" || $1 ~ o){print $1}' <<< "$PLAN"; }
LIGHT=$(sel L); GLM=$(sel G)
[ "$FAST" = 1 ] && GLM=""
rc=0
echo "== building (parallel, $JOBS jobs): $(echo $LIGHT $GLM | wc -w | tr -d ' ') tests =="
rm -f "$OUT"/b_*.sh
for n in $LIGHT $GLM; do
  f=tests/${n}_test.cpp; [ -f "$f" ] || f=tests/${n}.cpp
  if [ ! -f "$f" ]; then echo "  MISSING SOURCE for $n"; rc=1; continue; fi
  printf '%s\n' "clang++ -std=c++20 -O2 -Ivendor -Isrc -I'$MLX'/include -o '$OUT/$n' '$f' $SRC -L'$MLX'/lib -lmlx -Wl,-rpath,'$MLX'/lib >'$OUT/$n.build.log' 2>&1 && echo '  built $n' || { echo '  BUILD FAILED $n -> $OUT/$n.build.log'; exit 1; }" > "$OUT/b_$n.sh"
done
ls "$OUT"/b_*.sh 2>/dev/null | xargs -P "$JOBS" -n1 bash || rc=1
if [ -n "$LIGHT" ]; then
  echo "== light tests (parallel) =="
  for n in $LIGHT; do echo "$n"; done | xargs -P "$JOBS" -n1 -I{} bash -c \
    'o=$("'"$OUT"'/{}" 2>&1); s=$?; printf "%-22s %s\n" "{}" "$([ $s = 0 ] && echo PASS || echo FAIL\($s\))"; echo "$o" | tail -2 | sed "s/^/    /"; exit $s' || rc=1
fi
for n in $GLM; do
  echo "== GLM test (serial, 100 GB): $n =="
  # The checkpoint tests take the model directory as their first argument; not
  # passing it made every one of them print usage and exit 2 (so the runner
  # reported rc=1 while running nothing).
  "$OUT/$n" "$MODEL" 2>&1 | tail -8 | sed 's/^/    /' || rc=1
done
echo "== done (rc=$rc) =="
exit $rc
