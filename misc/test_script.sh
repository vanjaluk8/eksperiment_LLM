#!/bin/bash
set -e
# Skripta za automatsko testiranje performansi modela na Triton serveru

MODELS=("gemma3" "mistral" "meta-llama") # modeli koji se testiraju
BATCH_SIZES=(8) # batch veličine koje se testiraju
VERSIONS=("1") # verzije modela koje se testiraju
MEASUREMENT=12000 # measurement interval u ms

CONCURRENCY_RANGES=("1:64:16")  # concurrency ranges u formatu "min:max:step"

TRITON_HOST="localhost" # triton server host
TRITON_PORT="8000"  # HTTP API port

RESULTS_DIR=/workspace/files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
COMBINED_CSV="$RESULTS_DIR/combined_results_${TIMESTAMP}.csv"

mkdir -p "$RESULTS_DIR"

HEADER_WRITTEN=0

> "$COMBINED_CSV"

for MODEL in "${MODELS[@]}"; do
  for BATCH in "${BATCH_SIZES[@]}"; do
    for VERSION in "${VERSIONS[@]}"; do
      for CONCURRENCY_RANGE in "${CONCURRENCY_RANGES[@]}"; do

        OUTFILE="$RESULTS_DIR/${MODEL}_v${VERSION}_batch${BATCH}_concurrency${CONCURRENCY_RANGE//:/_}.csv"
        echo "Testing model $MODEL, version $VERSION, batch size $BATCH, concurrency range $CONCURRENCY_RANGE..."
        perf_analyzer \
          -m "$MODEL" \
          -x "$VERSION" \
          -u "$TRITON_HOST:$TRITON_PORT" \
          --concurrency-range "$CONCURRENCY_RANGE" \
          -b "$BATCH" \
          $SHAPE_ARGS \
          --measurement-interval "$MEASUREMENT" \
          --percentile 95 \
          --verbose-csv \
          --collect-metrics \
          -f "$OUTFILE"
        # header se dodaje samo jednom
        if [[ $HEADER_WRITTEN -eq 0 ]]; then
          awk -v m="$MODEL" -v v="$VERSION" -v b="$BATCH" -v c="$CONCURRENCY_RANGE" 'NR==1{print "model,version,batch_size,concurrency_range," $0}' "$OUTFILE" > "$COMBINED_CSV"
          HEADER_WRITTEN=1
        fi
        # apendanje linije sa modelom i batch size
        tail -n +2 "$OUTFILE" | awk -v m="$MODEL" -v v="$VERSION" -v b="$BATCH" -v c="$CONCURRENCY_RANGE" 'NF{print m "," v "," b "," c "," $0}' >> "$COMBINED_CSV"
        echo "  Results saved to $OUTFILE"
      done
    done
  done
done

echo "All tests completed. Combined CSV: $COMBINED_CSV"
