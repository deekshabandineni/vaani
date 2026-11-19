"""Export a trained checkpoint for edge and server inference.

    python -m vaani_ml.export runs/base/best.pt --out ../client/public/models
produces
    vaani_fp32.onnx   full precision (server)
    vaani_int8.onnx   INT8 dynamic quantization (browser / mobile)
    model_meta.json   tokens + feature spec, read by the client

Optional TFLite: `pip install onnx2tf tensorflow` then pass --tflite. onnx2tf
converts the FP32 ONNX graph to a SavedModel and .tflite (FP32 + dynamic-range
INT8). The ONNX path is the primary one: it runs in the browser through
onnxruntime-web and on Android/iOS through ONNX Runtime Mobile.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from . import features
from .phonemes import IPA, TOKENS
from .train import load_checkpoint


def export_onnx(model, path: Path, opset: int = 17):
    model.eval()
    dummy = torch.randn(1, 300, features.N_MELS)
    kwargs = dict(input_names=["feats"], output_names=["logits"],
                  dynamic_axes={"feats": {1: "frames"}, "logits": {1: "out_frames"}},
                  opset_version=opset, do_constant_folding=True)
    try:
        torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(model, (dummy,), str(path), **kwargs)


def quantize_int8(src: Path, dst: Path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = dst.with_suffix(".pre.onnx")
    try:
        quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
        source = pre
    except Exception:  # pre-processing is an optimisation, not a requirement
        source = src
    quantize_dynamic(str(source), str(dst), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])
    pre.unlink(missing_ok=True)


def write_meta(out: Path, config: dict):
    meta = {
        "tokens": TOKENS, "ipa": IPA, "blank": 0, "subsampling": 2, "model_config": config,
        "features": {"sample_rate": features.SAMPLE_RATE, "n_fft": features.N_FFT,
                     "win_length": features.WIN_LENGTH, "hop_length": features.HOP_LENGTH,
                     "n_mels": features.N_MELS, "f_min": features.F_MIN, "f_max": features.F_MAX,
                     "preemph": features.PREEMPH, "log_eps": features.LOG_EPS},
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default="exported")
    ap.add_argument("--tflite", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(args.checkpoint)
    fp32, int8 = out / "vaani_fp32.onnx", out / "vaani_int8.onnx"
    export_onnx(model, fp32)
    quantize_int8(fp32, int8)
    write_meta(out, model.config)
    print(f"fp32: {fp32.stat().st_size / 1e6:.2f} MB   int8: {int8.stat().st_size / 1e6:.2f} MB")
    if args.tflite:
        subprocess.run(["onnx2tf", "-i", str(fp32), "-o", str(out / "tflite"), "-oiqt"], check=True)


if __name__ == "__main__":
    main()
