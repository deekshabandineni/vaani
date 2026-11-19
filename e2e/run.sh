#!/usr/bin/env bash
# Full-stack browser test. Needs Python deps (ml + ml-api), Node 20+, Chrome.
#   MODEL_DIR=ml/runs/synth/export ./e2e/run.sh
# MODEL_DIR must contain vaani_fp32.onnx, vaani_int8.onnx, model_meta.json from
# `python -m vaani_ml.export` (the default is the synthetic smoke-test model).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT/ml/runs/synth/export}"
if [ ! -f "$MODEL_DIR/vaani_int8.onnx" ]; then
  echo "building the synthetic smoke-test model (~4 min on CPU)..."
  (cd "$ROOT/ml" && python3 -m vaani_ml.train --synthetic 1536 --epochs 30 --batch-size 32 --d-model 96 \
     --layers 2 --lr 3e-3 --out runs/synth && python3 -m vaani_ml.export runs/synth/best.pt --out runs/synth/export)
fi
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
TMP="$(mktemp -d)"
pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; rm -rf "$TMP"; }
trap cleanup EXIT

python3 "$ROOT/e2e/make_audio.py" "$TMP/said.wav"

(cd "$ROOT/services/ml-api" && PYTHONPATH="$ROOT/ml" VAANI_MODEL_PATH="$MODEL_DIR/vaani_fp32.onnx" \
  VAANI_CALIBRATION="$MODEL_DIR/calibration.json" python3 -m uvicorn app.main:app --port 8000 >"$TMP/ml.log" 2>&1) & pids+=($!)
(cd "$ROOT/services/gateway" && DATABASE_URL=memory JWT_SECRET=e2e-secret-0123456789 ML_API_URL=http://localhost:8000 \
  node src/server.js >"$TMP/gw.log" 2>&1) & pids+=($!)

(cd "$ROOT/client" && VITE_API_URL=http://localhost:3000 npm run build --silent >/dev/null)
mkdir -p "$ROOT/client/dist/models" && cp "$MODEL_DIR"/vaani_int8.onnx "$MODEL_DIR"/model_meta.json "$ROOT/client/dist/models/"
[ -f "$MODEL_DIR/calibration.json" ] && cp "$MODEL_DIR/calibration.json" "$ROOT/client/dist/models/"
(cd "$ROOT/client" && npx vite preview --port 4173 --strictPort >"$TMP/web.log" 2>&1) & pids+=($!)

for url in http://localhost:8000/health http://localhost:3000/health http://localhost:4173; do
  for _ in $(seq 60); do curl -sf "$url" >/dev/null && break; sleep 0.5; done
done

cd "$ROOT/e2e" && FAKE_AUDIO="$TMP/said.wav" APP_URL=http://localhost:4173 node e2e.mjs || {
  echo "--- ml-api log"; tail -20 "$TMP/ml.log"; echo "--- gateway log"; tail -20 "$TMP/gw.log"; exit 1; }
