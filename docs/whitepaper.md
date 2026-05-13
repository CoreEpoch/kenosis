# Activation-Aware Quantization: Achieving Native Kernel Fusion in ONNX via Graph Reordering

**Author:** Core Epoch  
**Project:** Kenosis  
**Date:** May 2026

## Abstract

The deployment of deep neural networks on edge devices relies heavily on INT8 quantization to reduce memory footprint and inference latency. The industry-standard ONNX Runtime (ORT) provides Python-based quantization utilities that inject QuantizeLinear and DequantizeLinear (QDQ) nodes into the computation graph. However, the default node injection strategy isolates Convolution operations from subsequent Activation layers, inadvertently breaking hardware-level kernel fusion.

In this paper, we present **Kenosis**, a native Rust graph optimization engine that implements *Activation-Aware QDQ Placement*. By leveraging the commutative properties of non-linear activations under positive scalar multiplication, Kenosis safely reorders the computation graph to preserve Conv-Activation contiguity. Our benchmarks across five architectures demonstrate speedups of up to **2.46× over FP32 baselines** and up to **65% latency reduction against the ORT Python quantizer**, with cosine similarity scores of 0.983–0.999 against FP32 outputs — validated in production multi-camera edge deployments.

---

## 1. Introduction

The transition from FP32 (32-bit floating point) to INT8 (8-bit integer) computation is a critical step in preparing machine learning models for production edge environments. While the mathematical theory of quantization is well understood, the *implementation* of this math within a static computation graph often introduces severe performance bottlenecks.

In the ONNX ecosystem, static INT8 quantization is achieved by wrapping heavy mathematical operations (like `Conv` and `MatMul`) in QuantizeLinear (Q) and DequantizeLinear (DQ) nodes. This "QDQ pattern" signals the backend execution provider to map the operation to an accelerated 8-bit integer kernel.

We observed that standard Python-based quantization tools apply a naive, node-by-node injection strategy. This localized approach ignores the broader graph topology — specifically the relationship between Convolutional layers and their subsequent Non-Linear Activations (e.g., ReLU). This paper details how this naive placement causes "fusion breaking," leading to unnecessary memory thrashing, and how Kenosis solves this via topological graph awareness.

---

## 2. The Bottleneck: Broken Kernel Fusion

Modern CPU and GPU architectures achieve maximum throughput by minimizing trips to main memory. "Kernel Fusion" is the process by which a runtime engine collapses multiple sequential graph operations into a single, highly optimized hardware instruction.

In a standard FP32 vision model, a Convolution is almost always followed immediately by a ReLU activation:

```
Conv ➔ ReLU
```

When an ONNX execution provider sees this contiguous `Conv ➔ ReLU` pattern, it fuses them: performing the matrix multiplication and the negative-value zeroing in a single memory cycle.

When the standard ORT Python quantizer converts this to INT8, it evaluates the `Conv` node in isolation and wraps it in QDQ nodes:

```
Quantize ➔ Conv ➔ Dequantize ➔ ReLU
```

By injecting the `Dequantize` node between `Conv` and `ReLU`, the quantizer severs their contiguity. The runtime engine can no longer fuse them. The hardware is forced to: compute the INT8 Convolution, push the result to main memory, pull it back to Dequantize to FP32, push it back to memory, and finally pull it to apply ReLU. This memory thrashing completely neutralizes the computational speedup gained from INT8 math.

---

## 3. The Mathematical Guarantee of Reordering

To restore kernel fusion, the `Dequantize` node must be moved *after* the `ReLU` node. In computational graphs, altering the order of operations generally corrupts the output. Kenosis relies on a specific mathematical property of ReLU interacting with the Dequantization formula to guarantee that reordering is safe.

The Dequantize operation is defined as:

```
y = (x - zero_point) * scale
```

Kenosis designs Conv output bias quantization such that `zero_point` is exactly `0`. The Dequantization therefore simplifies to pure positive scalar multiplication:

```
y = x * scale   (where scale > 0)
```

The ReLU operation is defined as `y = max(0, x)`. Because the scale is strictly positive, the scalar multiplication is **commutative** with the `max(0, x)` operation:

- **Standard Path (Dequantize ➔ ReLU):** `max(0, x * scale)`
- **Kenosis Path (ReLU ➔ Dequantize):** `max(0, x) * scale`

Whether you multiply a negative integer by a positive scale and then clamp to zero, or clamp first and then multiply, the result is exactly `0.0`. This equivalence provides the formal proof required to safely rewrite the graph. The same commutativity holds for LeakyRelu, Clip, HardSwish, and Sigmoid activations, all of which Kenosis handles automatically.

The resulting Kenosis-optimized graph places `Dequantize` *after* the activation, restoring the contiguous `Conv ➔ ReLU` pattern that the execution provider maps to a single `QLinearConv` kernel:

```
Quantize ➔ Conv ➔ ReLU ➔ Dequantize
```

---

## 4. The Kenosis Pipeline

Kenosis is a native Rust graph optimization engine that applies eight coordinated optimizations statically, prior to deployment. Unlike standard tooling, Kenosis performs a topological traversal of the ONNX protobuf graph:

1. **Self-calibration:** Automatically generates synthetic calibration inputs and runs them through the model via ONNX Runtime to collect per-tensor activation ranges. No external calibration data required. Multi-input models and NLP inputs (token IDs, attention masks) are handled automatically.
2. **Weight quantization:** INT8 symmetric per-tensor or per-channel. All scale computations in f64 to match ORT's internal precision.
3. **INT32 bias quantization:** `scale = activation_scale × weight_scale`, zero_point = 0. Wrapped with DequantizeLinear for ORT kernel fusion.
4. **Zero-point nudged activation quantization:** UINT8 asymmetric with post-hoc range adjustment ensuring `float 0.0` maps exactly to the quantized zero. Prevents rounding asymmetry from compounding across layers.
5. **Activation-aware QDQ placement:** Detects `Conv/MatMul → Activation` pairs at graph level and places QDQ *after* the activation instead of between them. Combined with second-pass wrapping of Add, Concat, MaxPool, and AveragePool, this maximizes QLinear fusions.
6. **Non-vision tensor protection:** For multi-input models (detection, segmentation), tensors reachable from non-primary inputs (scale_factor, image_shape) are traced through the graph and excluded from quantization, preventing metadata paths from being crushed by INT8 range limits.
7. **Model output protection:** Tensors that are direct model outputs are never QDQ-wrapped, preserving full FP32 precision in detection head scores and bounding box coordinates.
8. **SNR sensitivity analysis:** Computes Signal-to-Noise Ratio for every layer's weight quantization. Automatically identifies mathematically fragile layers and protects them in FP32, recovering catastrophic Top-1 accuracy drops.

---

## 5. Benchmarks and Results

### Test Environment

| Component | Specification |
|-----------|--------------|
| CPU | Intel i5-13420H (8C/12T, 8 GB DDR5) |
| GPU | Disabled (CPU-only execution) |
| Runtime | ONNX Runtime 1.x (CPU EP), `ort` crate v2.0.0-rc.12 |
| Build | Release (`--release`), Rust 1.85+ |
| Isolated benchmarks | 200 timed iterations after 20 warmup runs, single-threaded ORT |
| Pipeline benchmarks | 30-second steady-state window after 20-second warmup |

Accuracy is reported as cosine similarity between INT8 and FP32 outputs — a direct measure of numerical fidelity that avoids ambiguity when comparing quantization methods across architectures.

---

### 5.1 Isolated Latency — PP-YOLOE+ Small (Kenosis INT8 vs FP32)

PP-YOLOE+ is the object detection architecture deployed in production multi-camera edge pipelines. Single-threaded isolated inference.

| Resolution | Cosine Similarity | INT8 Latency | FP32 Latency | Speedup | INT8 Size |
|------------|-------------------|--------------|--------------|---------|-----------
| 320×320 | 0.998 | **23ms** | 44ms | **1.89×** | 7.9 MB (3.9× smaller) |
| 416×416 | 0.998 | **43ms** | 77ms | **1.80×** | 7.9 MB (3.9× smaller) |
| 640×640 | 0.999 | **111ms** | 187ms | **1.68×** | 7.9 MB (3.8× smaller) |

---

### 5.2 Isolated Latency — Classifier Benchmarks (Kenosis INT8 vs FP32)

Standard vision classifiers quantized with per-tensor symmetric INT8 weights and self-calibrated activations.

| Architecture | Cosine Similarity | Kenosis INT8 | FP32 Baseline | Speedup | INT8 Size |
|--------------|-------------------|--------------|---------------|---------|-----------
| SqueezeNet 1.1 | 0.999 | **2.85ms** | 6.60ms | **2.32×** | 1.24 MB (3.8× smaller) |
| ResNet50 v2 | 0.995 | **27.8ms** | 68.4ms | **2.46×** | 30.6 MB (3.2× smaller) |
| MobileNetV2 | 0.990 | **4.61ms** | 6.53ms | **1.42×** | 7.10 MB (1.9× smaller) |
| EfficientNet-Lite4 | 0.983 | **14.2ms** | 26.8ms | **1.89×** | 16.5 MB (3.0× smaller) |

---

### 5.3 Direct Comparison — Kenosis vs ORT Python Quantizer (Isolated)

Head-to-head against the ORT Python quantizer, isolating the contribution of activation-aware QDQ placement. Both quantizers use per-tensor INT8 with synthetic calibration data. ORT quantizer uses `quantize_static` with QDQ format after `quant_pre_process`.

| Architecture | Kenosis Latency | ORT Latency | Kenosis Advantage |
|--------------|-----------------|-------------|-------------------|
| SqueezeNet 1.1 | **2.85ms** | 8.13ms | **65% faster** |
| ResNet50 v2 | **27.8ms** | 46.1ms | **40% faster** |
| MobileNetV2 | **4.61ms** | 6.29ms | **27% faster** |
| EfficientNet-Lite4 | **14.2ms** | 23.5ms | **40% faster** |

Notably, the ORT Python quantizer produces a SqueezeNet model that is **slower than the FP32 baseline** (8.13ms INT8 vs 6.60ms FP32) — a direct consequence of broken Conv-ReLU fusion caused by naive QDQ placement. Kenosis eliminates this regression entirely, delivering a 2.32× speedup over FP32.

---

### 5.4 Production Pipeline Comparison — Dual-Camera Deployment

Isolated latency benchmarks mask a critical dimension of quantization quality: CPU efficiency under multi-threaded production load. When ORT is given a thread pool, both FP32 and INT8 models execute as fast as the hardware allows — wall-clock latency appears similar because the thread pool saturates. What diverges significantly is *how hard the CPU works* to achieve that latency, and whether the pipeline can sustain frame rate at all.

The following results were captured from a dual-camera headless pipeline running PP-YOLOE+ Small 320×320, processing 640×480 @ 30fps input on two concurrent streams. CPU utilization was sampled at 1-second intervals via process-level polling over the steady-state window.

**Pipeline:** `cryphexd` — dual-camera (webcam-0 + webcam-1), output mode = none  
**Steady-state window:** 30 seconds (sampled after 20-second warmup)  
**ORT thread pool:** 4 threads/camera (auto-detected from 8 physical cores)

| Metric | FP32 Baseline | ORT Python INT8 | Kenosis INT8 |
|--------|--------------|-----------------|--------------|
| FPS (cam-0 / cam-1) | 29.5 / 30.1 | 22.5 / 22.7 | **29.5 / 30.1** |
| Avg Inference Latency | 21.6 ms | 43.8 ms | **20.7 ms** |
| CPU Utilization | ~53% | ~60% | **~41%** |
| Working Set Memory | 158.4 MB | 300.0 MB | **118.3 MB** |
| Model File Size | 30.4 MB | 30.7 MB | **7.9 MB** |
| Dropped Frames | 0 | 0 | **0** |

**Kenosis INT8 vs FP32 Baseline:**
- Throughput: identical — both sustain the full 30 fps capture rate with zero drops
- Latency: 4% faster (20.7ms vs 21.6ms) — thread pooling saturates CPU similarly, but individual INT8 passes are slightly faster
- CPU Utilization: **22% reduction** (~41% vs ~53%) — Kenosis achieves the same output while freeing significant compute headroom for additional workloads or camera streams
- Memory: **25% reduction** (118.3 MB vs 158.4 MB) — quantized weights and activations fit tighter in cache
- Model size: **3.8× smaller** (7.9 MB vs 30.4 MB)

**ORT Python Quantizer — Catastrophic Regression:**

The ORT Python quantizer does not merely underperform — it actively degrades the pipeline across every measured dimension:
- Throughput: **24% lower** (22.5 fps vs 29.5 fps) — cannot sustain the input capture rate
- Latency: **2× slower than FP32** (43.8ms vs 21.6ms) — broken QDQ fusion manifesting at pipeline scale
- CPU Utilization: **13% higher than FP32** (~60% vs ~53%) — inefficient execution graph burning more CPU to produce worse results
- Memory: **89% more than FP32** (300 MB vs 158.4 MB) — unoptimized QDQ graph bloat
- Model size: **larger than FP32** (30.7 MB vs 30.4 MB) — failed Constant extraction

The 22% CPU headroom reduction delivered by Kenosis INT8 has direct operational implications: on a fixed hardware platform, freed CPU capacity translates directly into additional concurrent camera streams without hardware upgrades.

---

## 6. Conclusion

The transition from Python-based, localized node injection to Rust-based, topologically aware graph rewriting represents a significant step forward in edge AI deployment. By aligning the static graph structure with the expectations of the underlying hardware execution providers, Kenosis achieves native kernel fusion without requiring custom C++ runtime extensions or modifications to the ONNX Runtime itself.

Activation-aware QDQ placement, combined with SNR-based layer protection and non-vision tensor exclusion, produces INT8 models with cosine similarity scores of 0.983–0.999 against their FP32 origins — while delivering the compute efficiency required to run high-density, production-grade computer vision pipelines on commodity edge hardware. In production pipeline testing, Kenosis INT8 matches FP32 throughput exactly while reducing CPU utilization by 22% and working memory by 25%, headroom that translates directly into camera density on fixed hardware.

---

*For source code, installation instructions, and deployment details, visit the [Kenosis repository on GitHub](https://github.com/CoreEpoch/kenosis).*
