#!/usr/bin/env bash
# Request-level batching, measured through the OpenAI-compatible HTTP server:
# N concurrent requests, aggregate completion tokens/second, and the per-request
# latency distribution.
#
#   tools/batch_server_bench.sh                 # --batch 0 then --batch 8, N = 1/2/4/8
#   MAXTOK=48 REPS=3 CONC_LIST="1 4 8" tools/batch_server_bench.sh
#   SAME_PROMPT=1 tools/batch_server_bench.sh   # all N requests share one prompt
#
# Why the plan looks like this:
#
#   * one server process per --batch setting, and the two settings alternated per
#     round: the box drifts ~30% between sessions, so only same-session (and
#     interleaved) comparisons mean anything, and the checkpoint's 105 GB allow
#     exactly one server at a time;
#   * every run is preceded by a warm-up wave (the first ~24 decode steps of a
#     fresh process are 2-3x slower: Metal's shader cache and page faults on the
#     mmap'd weights);
#   * --batch 0 is the single-stream baseline: the requests queue and the
#     aggregate of N requests is exactly one stream's rate;
#   * WARM=1 reads every shard first: the 105 GB checkpoint only fits in a
#     128 GB machine's page cache if nothing else is resident, and a decode step
#     whose experts are not resident runs at the SSD's 2.6 GB/s instead of the
#     GPU's ~150-300 GB/s (docs/perf.md);
#   * SAME_PROMPT=1 measures the other end of the range - identical requests
#     route to identical experts, so the MoE's weight reads are shared too. The
#     default (distinct prompts) is the honest server workload.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${GLM5_MODEL:-/Volumes/External/Models/MikeRoz/GLM-5.3-Flash-Uncensored-2.51bpw-h6-exl3}
BIN=${BIN:-build/bin/exl3-server}
HOST=127.0.0.1
PORT=${PORT:-8099}
MAXTOK=${MAXTOK:-64}
REPS=${REPS:-3}
CONC_LIST=${CONC_LIST:-"1 2 4 8"}
BATCH_LIST=${BATCH_LIST:-"0 8"}
SAME_PROMPT=${SAME_PROMPT:-0}
WARM=${WARM:-0}
DRAFT_MTP=${DRAFT_MTP:-0}   # >0: speculative decoding on the serial path (the MTP baseline)
OUT=${OUT:-/tmp/exl3_batch_server}
mkdir -p "$OUT"

[ -x "$BIN" ] || { echo "no $BIN; run: cmake --build build -j"; exit 2; }
command -v python3 >/dev/null || { echo "python3 needed for the JSON parsing"; exit 2; }

PROMPTS=(
  "In one sentence, what is a trellis? Answer briefly."
  "Name the three largest planets in the solar system, in order."
  "Write the first eight digits of pi, separated by spaces."
  "What colour is a ripe banana, and why?"
  "Translate 'good morning' into Japanese and Korean."
  "List four prime numbers above twenty."
  "Explain what a hash table is in one sentence."
  "What is the capital of Australia?"
)

start_server() {
  local batch=$1
  : > "$OUT/server.log"
  if [ "$WARM" = 1 ]; then
    echo "   warming the page cache ($(ls "$MODEL"/model-*.safetensors | wc -l | tr -d ' ') shards)..."
    ls "$MODEL"/model-*.safetensors | xargs -P 8 -n1 cat > /dev/null
  fi
  local extra=()
  [ "$DRAFT_MTP" -gt 0 ] && extra+=(--draft-mtp "$DRAFT_MTP")
  "$BIN" --model "$MODEL" --serve --host "$HOST" --port "$PORT" --batch "$batch" --quiet \
    "${extra[@]}" >>"$OUT/server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 600); do
    if curl -s -m 2 "http://$HOST:$PORT/health" >/dev/null 2>&1; then return 0; fi
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died; see $OUT/server.log"; return 1; }
    sleep 1
  done
  echo "server did not come up"; return 1
}

stop_server() {
  kill "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
}

# One wave: N concurrent requests, returns "<wave_s> <prompt_tokens> <completion_tokens> <lat_min> <lat_max>"
wave() {
  local conc=$1 tag=$2
  local t0 t1
  rm -f "$OUT"/w_*.json "$OUT"/w_*.time
  t0=$(python3 -c 'import time;print(time.time())')
  for i in $(seq 0 $((conc - 1))); do
    local idx=$((i % ${#PROMPTS[@]}))
    local prompt="${PROMPTS[$idx]}"
    [ "$SAME_PROMPT" = 1 ] && prompt="${PROMPTS[0]}"
    (
      curl -s -m 900 -o "$OUT/w_$i.json" -w '%{time_total}' \
        "http://$HOST:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$MAXTOK,\"temperature\":0}" \
        > "$OUT/w_$i.time" 2>/dev/null
    ) &
  done
  wait
  t1=$(python3 -c 'import time;print(time.time())')
  python3 - "$OUT" "$t0" "$t1" "$conc" "$tag" <<'PY'
import glob, json, sys, statistics
out, t0, t1, conc, tag = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
comp = prompt = 0
lat = []
for f in sorted(glob.glob(out + "/w_*.json")):
    try:
        j = json.load(open(f))
    except Exception:
        continue
    comp += j.get("usage", {}).get("completion_tokens", 0)
    prompt += j.get("usage", {}).get("prompt_tokens", 0)
for f in sorted(glob.glob(out + "/w_*.time")):
    try:
        lat.append(float(open(f).read().strip()))
    except Exception:
        pass
wall = t1 - t0
agg = comp / wall if wall > 0 else 0.0
per = agg / conc if conc else 0.0
lat_s = " ".join(f"{x:.2f}" for x in sorted(lat))
print(f"{tag}\tconc {conc}\twall {wall:7.2f}s\tprompt {prompt}\tcompletion {comp}\t"
      f"aggregate {agg:6.2f} tok/s\tper-request {per:5.2f} tok/s\t"
      f"latency min/med/max {min(lat) if lat else 0:.2f}/{statistics.median(lat) if lat else 0:.2f}/"
      f"{max(lat) if lat else 0:.2f}s\t[{lat_s}]", flush=True)
PY
}

echo "== request-level batching through the HTTP server =="
echo "model   $MODEL"
echo "max_tokens $MAXTOK, $REPS waves per cell, warm-up wave of $MAXTOK steps per server"
echo "same_prompt $SAME_PROMPT, batch settings: $BATCH_LIST, concurrency: $CONC_LIST, draft_mtp $DRAFT_MTP"
echo

for round in $(seq 1 2); do
  for batch in $BATCH_LIST; do
    echo "-- round $round, --batch $batch --"
    start_server "$batch" || exit 1
    wave 1 "warm-up"
    for conc in $CONC_LIST; do
      for r in $(seq 1 "$REPS"); do
        wave "$conc" "batch${batch}"
      done
    done
    stop_server
    echo
  done
done
echo "== done (raw responses in $OUT) =="
