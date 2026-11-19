# Training the phoneme model

Everything here runs from `ml/`. A single GPU is enough (a Colab T4 or similar). The default
model has 5.3 M parameters.

## 0. Install
```bash
pip install torch            # pick the CUDA build that matches your GPU
pip install -e ".[train,dev]"
pytest -q                    # ~1 min; includes a tiny end-to-end train/export run
```

## 1. Get data
| Corpus | Why | Size | Link |
|---|---|---|---|
| LibriSpeech `train-clean-100` | native speech: learn what "correct" sounds like | 6.3 GB | openslr.org/12 |
| speechocean762 | non-native speech **with expert phone scores**: validation, calibration, evaluation | 0.5 GB | openslr.org/101 |
| L2-ARCTIC (optional) | non-native speakers incl. Hindi L1 (ASI, RRBI, SVBI, TNI) | 8 GB, request form | psi.engr.tamu.edu/l2-arctic-corpus |

Start with LibriSpeech-100 + speechocean762 train; add L2-ARCTIC later and compare.

## 2. Build manifests
```bash
python scripts/prepare_data.py librispeech /data/LibriSpeech/train-clean-100 data/libri.jsonl
python scripts/prepare_data.py speechocean /data/speechocean762 train data/so_train.jsonl
python scripts/prepare_data.py speechocean /data/speechocean762 test  data/so_test.jsonl
python scripts/prepare_data.py l2arctic /data/l2arctic data/l2a_hindi.jsonl --speakers ASI RRBI SVBI TNI
```
Utterances with words missing from CMUdict are skipped (the script prints how many it kept).

## 3. Train
```bash
python -m vaani_ml.train --train data/libri.jsonl data/so_train.jsonl \
    --val data/so_test.jsonl --epochs 40 --batch-size 32 --workers 4 --out runs/base
```
- Watch `val_per` (phone error rate) in the log / `runs/base/history.json`.
- **CTC plateau:** for the first epoch or two, the loss sits around 3–4 and PER stays at 1.0 while
  the model predicts only blanks. That's normal; it breaks out. If it never does, lower `--lr`.
- Do a short run first (`--epochs 5`) to estimate time per epoch before committing.
- Speechocean762 test is used as validation here. For the final numbers in your write-up, hold out
  part of speechocean762 train for validation instead, so the test set is untouched.

## 4. Export + quantize
```bash
python -m vaani_ml.export runs/base/best.pt --out ../models          # FP32 + INT8 ONNX + model_meta.json
python -m vaani_ml.export runs/base/best.pt --out ../models --tflite # + TFLite (pip install onnx2tf tensorflow)
```

## 5. Evaluate and calibrate
```bash
python -m vaani_ml.evaluate ../models/vaani_int8.onnx data/so_test.jsonl \
    --fit-on data/so_train.jsonl --calibration-out ../models/calibration.json
```
Outputs phone-level and utterance-level Pearson correlation with expert scores and mispronunciation
detection precision/recall/F1. Compare with the GOP baselines reported in the speechocean762 paper
(Zhang et al., Interspeech 2021).

## 6. Benchmark the edge model
```bash
python -m vaani_ml.benchmark ../models/vaani_fp32.onnx ../models/vaani_int8.onnx \
    --manifest data/so_test.jsonl --md ../docs/benchmark.md
```
Also measure in the browser: the app shows on-device scoring time under each result. Try a laptop
and your phone, and record both.

## 7. Experiments worth running (and writing up)
1. **Data:** LibriSpeech only vs. + speechocean762 vs. + L2-ARCTIC Hindi speakers → PCC and F1.
2. **Size:** `--d-model 144 --layers 4` vs. default → accuracy vs. size vs. latency.
3. **Quantization:** FP32 vs. INT8 → PCC drop (the benchmark reports score differences).
4. **Fusion weight:** vary `FUSION_AZURE_WEIGHT` on a set of your own recordings; look at the
   `agreement` field the API returns.
