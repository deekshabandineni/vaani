Exported models live here (git-ignored). Produce them with:

    cd ml && python -m vaani_ml.export runs/base/best.pt --out ../models

Expected files: `vaani_fp32.onnx` (server), `vaani_int8.onnx` (browser), `model_meta.json`,
and `calibration.json` (from `python -m vaani_ml.evaluate ... --calibration-out ../models/calibration.json`).
Docker Compose mounts this folder into both the ML service and the web client.
For `npm run dev` in client/, copy the same files into `client/public/models/`.
