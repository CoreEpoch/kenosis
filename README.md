<div align="center">

# Kenosis

**Production-grade ONNX model quantization. Pure Rust. Zero Python.**

[![CI](https://github.com/CoreEpoch/kenosis/actions/workflows/ci.yml/badge.svg)](https://github.com/CoreEpoch/kenosis/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

</div>

---

Kenosis is a Rust CLI toolkit for quantizing, validating, inspecting, and comparing ONNX models. Its flagship feature is **static INT8 quantization** with ReLU-aware QDQ placement that achieves full QLinearConv fusion on stock ONNX Runtime — no custom operators required.

## Production Results

Kenosis quantizes the **PP-YOLOE+ object detection models** deployed in production edge AI pipelines, delivering production-validated performance gains:

| Model | Resolution | Cosine | Latency | Speedup | Size |
|-------|-----------|--------|---------|---------|------|
| PP-YOLOE+ Small | 320×320 | 0.998 | **23ms** vs 44ms | **1.89×** | 7.9 MB (3.9× smaller) |
| PP-YOLOE+ Small | 416×416 | 0.998 | **43ms** vs 77ms | **1.80×** | 7.9 MB (3.9× smaller) |
| PP-YOLOE+ Small | 640×640 | 0.999 | **111ms** vs 187ms | **1.68×** | 7.9 MB (3.8× smaller) |

> Real-world inference on the 320 model: **19–25ms** per frame. CPU utilization drops from ~85% to ~60% at 30fps, enabling 3–4 camera scaling on the same hardware.

### Classifier Benchmarks (Kenosis vs ORT Python Quantizer)

| Model | Cosine | Kenosis Latency | ORT Latency | Kenosis Advantage |
|-------|--------|----------------|-------------|-------------------|
| SqueezeNet 1.1 | 0.999 | **2.82ms** | 4.25ms | **51% faster** |
| ResNet50 v2 | 0.999 | **38.0ms** | 49.5ms | **24% faster** |

> Kenosis achieves **26/26 QLinearConv fusion** on SqueezeNet (vs ORT's 26/26), plus **8/8 QLinearConcat** and full pool fusion — with fewer residual DequantizeLinear nodes. The advantage comes from ReLU-aware QDQ placement that matches ORT's internal Conv+ReLU fusion patterns.

## Key Features

| Feature | Kenosis | ORT Python |
|---------|---------|------------|
| Static INT8 with ReLU-aware QDQ | ✅ | ❌ |
| Detection model mixed-precision | ✅ | ❌ |
| Non-vision tensor protection | ✅ | ❌ |
| Multi-input model calibration | ✅ | ❌ |
| INT32 bias quantization w/ DQL | ✅ | ✅ |
| Per-channel weight quantization | ✅ | ✅ |
| Built-in validation + benchmarking | ✅ | ❌ |
| PaddlePaddle Constant extraction | ✅ | ❌ |
| Zero Python dependency | ✅ | ❌ |
| Cross-platform single binary | ✅ | ❌ |

## Install

```bash
cargo install kenosis-cli
```

Or build from source:

```bash
git clone https://github.com/CoreEpoch/kenosis.git
cd kenosis
cargo build --release
```

## Usage

### Static INT8 Quantization (recommended)

The primary quantization mode. Produces QDQ-format models that run on stock ONNX Runtime with full INT8 acceleration.

```bash
# Standard vision model (SqueezeNet, ResNet, EfficientNet, etc.)
kenosis quantize model.onnx -o model_int8.onnx --static-int8

# Per-channel weights (better for models with high channel counts like ResNet)
kenosis quantize model.onnx -o model_int8.onnx --static-int8 --per-channel

# PaddlePaddle models (PP-YOLOE+, PP-LCNet, etc.)
kenosis quantize ppyoloe.onnx -o ppyoloe_int8.onnx --static-int8 --extract-constants

# Custom calibration sample count
kenosis quantize model.onnx -o model_int8.onnx --static-int8 --n-calib 40

# External calibration data (raw f32 binary files)
kenosis quantize model.onnx -o model_int8.onnx --static-int8 --calib-dir ./calib_data/
```

### Validate Quantized Models

Compare a quantized model against its FP32 baseline — measures cosine similarity, Top-1 agreement, and latency side-by-side.

```bash
# Basic validation (50 samples, 200 timed runs)
kenosis validate model.onnx model_int8.onnx

# Custom sample counts
kenosis validate model.onnx model_int8.onnx -n 500 --timed 500
```

Output:
```
  ════════════════════════════════════════════════════════
  📊  Kenosis Validation Report
  ════════════════════════════════════════════════════════
  ▸ Cosine similarity:  0.999128 (min 0.9986)
  ▸ Top-1 agreement:    83/100 (83%)
  ▸ Latency:            2.82ms vs 6.03ms (2.13× speedup)
  ▸ Size:               1.24 MB vs 4.73 MB (3.8× smaller)
  ▸ Verdict:             EXCELLENT — production ready
  ════════════════════════════════════════════════════════
```

### Inspect a Model

```bash
# Basic stats — ops, params, size, data types, largest tensors
kenosis inspect model.onnx
```

### Utility Commands

```bash
# Cast to FP16/BF16
kenosis cast model.onnx -o model_fp16.onnx --precision fp16

# Compare two models
kenosis diff model.onnx model_int8.onnx
```

## How Static INT8 Works

Kenosis's static INT8 pipeline applies seven coordinated optimizations:

1. **Self-calibration** — Automatically generates synthetic calibration inputs and runs them through the model via ONNX Runtime to collect per-tensor activation ranges. No external calibration data required. Multi-input models (detection models with scale_factor, etc.) are handled automatically.

2. **Weight quantization** — INT8 symmetric per-tensor or per-channel. All scale computations in f64 to match ORT's internal precision.

3. **INT32 bias quantization** — `scale = activation_scale × weight_scale`, zero_point = 0. Wrapped with DequantizeLinear for ORT kernel fusion.

4. **Zero-point nudged activation quantization** — UINT8 asymmetric with post-hoc range adjustment ensuring `float 0.0` maps exactly to the quantized zero. Prevents rounding asymmetry from compounding across layers.

5. **ReLU-aware QDQ placement** — ORT's Python quantizer places QDQ nodes on every Conv output independently. Kenosis detects Conv→ReLU pairs at graph level and places QDQ *after* the ReLU instead, giving ORT's runtime optimizer a cleaner `Conv → ReLU → QuantizeLinear` pattern that fuses into a single INT8 kernel. Combined with second-pass wrapping of Add, Concat, MaxPool, Sigmoid, Mul, and Clip outputs, this maximizes the number of operations ORT can execute in INT8.

6. **Non-vision tensor protection** — For multi-input models (detection, segmentation), tensors reachable from non-primary inputs (scale_factor, image_shape) are traced through the graph and excluded from quantization. This prevents metadata paths from being crushed by INT8 range limits.

7. **Model output protection** — Tensors that are direct model outputs are never QDQ-wrapped, preserving full FP32 precision in detection head scores and bounding box coordinates.

## Detection Model Support

Kenosis handles the specific challenges of quantizing object detection models:

- **Multi-input calibration** — Auto-generates appropriate default values for secondary inputs (scale_factor → 1.0, shape tensors → 0.0)
- **PaddlePaddle weight handling** — Extracts inline Constant nodes, deduplicates shared weights (deepcopy tensors), and upgrades opset attributes (Squeeze, Unsqueeze, BatchNorm, Dropout)
- **Mixed-precision detection head** — Backbone and neck are fully INT8; detection head outputs and metadata paths stay FP32
- **Scale factor preservation** — The bounding box rescaling path remains live and dynamic, not frozen to calibration values


## Architecture

```
kenosis/
├── crates/
│   └── kenosis-core/           # Library: quantization engine
│       └── src/
│           ├── model.rs        # OnnxModel load/save/traversal + Constant extraction
│           ├── static_int8.rs  # Static INT8 QDQ quantization pipeline
│           ├── inspect.rs      # Stats and analysis
│           ├── cast.rs         # FP16/BF16 casting
│           ├── diff.rs         # Model comparison
│           ├── proto.rs        # ONNX protobuf type definitions
│           └── error.rs        # Error types
├── apps/
│   └── kenosis-cli/            # Binary: CLI interface
│       └── src/commands/
│           ├── quantize.rs     # quantize command (static INT8)
│           ├── validate.rs     # validate command (accuracy + latency)
│           ├── inspect.rs      # inspect command
│           ├── cast.rs         # cast command
│           └── diff.rs         # diff command
```

## License

Apache-2.0 — see [LICENSE](LICENSE).

Built by [Core Epoch](https://coreepoch.dev).
