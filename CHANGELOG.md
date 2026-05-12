# Changelog

All notable changes to Kenosis will be documented in this file.

## [1.0.0] — 2026-05-11

### Stable Release

Kenosis v1.0.0 marks the first stable release. The public API for `kenosis-core` is now locked under semantic versioning guarantees.

- **All v0.4.0 features promoted to stable** — static INT8 QDQ quantization, MatMul quantization, NLP synthetic calibration, SNR sensitivity analysis, and activation-aware QDQ placement are all production-validated and covered by semver.
- **API stability guarantee** — `kenosis-core` consumers can depend on `^1.0` without risk of breaking changes.

---

## [0.4.0] — 2025-05-08

### Added

- **MatMul quantization** — Conv-only QDQ wrapping extended to MatMul nodes, enabling transformer and fully-connected layer quantization.
- **NLP synthetic calibration** — INT64 input generation for transformer models. Automatically produces realistic `input_ids`, `attention_mask`, and `token_type_ids` tensors during self-calibration.
- **Extended activation fusion** — Conv→Activation QDQ skip-logic now supports LeakyRelu, Clip (ReLU6), Sigmoid, and HardSwish in addition to ReLU. Enables QLinearConv fusion on YOLO, Darknet, and MobileNet architectures.
- **SNR sensitivity analysis** — Computes per-layer Signal-to-Noise Ratio during weight quantization. Layers with SNR below threshold are automatically kept in FP32 to prevent accuracy collapse.
- **HardSwish and LeakyRelu** added to second-pass QDQ wrapping ops.
- **AveragePool** added to second-pass QDQ wrapping ops.

### Fixed

- **Non-vision tensor tracking** — Flood-fill logic now correctly ignores graph initializers, preventing false positives on transformer integer inputs.
- **ResNet50 accuracy** — Previous v0.3.x per-tensor quantization produced 0% Top-1 on ResNet50-v2. The v0.4.0 pipeline with SNR sensitivity analysis achieves **100% Top-1 agreement** (0.999 cosine).

### Validated Results (v0.4.0)

| Model | Cosine | Top-1 | Speedup | Size Reduction | Verdict |
|-------|--------|-------|---------|----------------|---------|
| SqueezeNet 1.1 | 0.998 | 87% | 2.25× | 3.7× smaller | Production ready |
| ResNet50 v2 | 0.999 | 100% | 2.02× | 3.2× smaller | Production ready |
| PP-YOLOE+ (320) | 0.998 | — | 1.89× | 3.9× smaller | Production ready |
| ShuffleNet v2 | 0.996 | 48% | 0.67× | 1.6× smaller | ORT fuser limitation |
| EfficientNet-Lite4 | 0.416 | 0% | 1.46× | 3.8× smaller | Not recommended |

> **Note:** ShuffleNet v2 uses grouped/depthwise convolutions that ORT cannot fuse into QLinearConv, resulting in QDQ overhead that makes INT8 slower than FP32. EfficientNet-Lite4 contains SE-block scaling patterns that are highly sensitive to quantization. Both architectures are candidates for future mixed-precision support.

### Changed

- CLI completion message cleaned up — removed non-technical output text.
- Documentation updated: "ReLU-aware" → "Activation-aware" QDQ placement throughout README.
- "Pure Rust" claims replaced with "Native Rust Pipeline" across all documentation for SBOM audit compliance.

## [0.3.0] — Initial public release

- Static INT8 QDQ quantization with ReLU-aware placement
- Per-channel and per-tensor weight quantization
- INT32 bias quantization with DequantizeLinear wrapping
- Zero-point nudged UINT8 activation quantization
- Non-vision tensor protection for multi-input detection models
- PaddlePaddle Constant extraction and opset upgrading
- Built-in validation and benchmarking (`kenosis validate`)
- Model inspection (`kenosis inspect`) and comparison (`kenosis diff`)
- FP16/BF16 casting (`kenosis cast`)
