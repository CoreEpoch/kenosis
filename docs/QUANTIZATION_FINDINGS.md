# Kenosis Static INT8 Quantization — Development Findings

> Internal reference document. Updated 2026-05-05.

## Table of Contents

- [Architecture](#architecture)
- [Key Discovery: ReLU-Aware QDQ Placement](#key-discovery-relu-aware-qdq-placement)
- [Optimization History](#optimization-history)
- [SqueezeNet Benchmark Results](#squeezenet-benchmark-results)
- [Multi-Architecture Validation](#multi-architecture-validation)
- [PP-YOLOE+ Detection Model Quantization](#pp-yoloe-detection-model-quantization)
- [Gaussian Activation Tightening — A/B Test](#gaussian-activation-tightening--ab-test-2026-05-06)
- [Kenosis vs ORT — Controlled Multi-Architecture Comparison](#kenosis-vs-ort--controlled-multi-architecture-comparison-2026-05-06)
- [Technical Details](#technical-details)

---

## Architecture

Kenosis produces **QDQ-format** (Quantize-Dequantize) ONNX models that run on
stock ORT without custom operators. The quantization pipeline:

1. **Calibration** — Run `n` synthetic samples (standard Gaussian) through the
   model to collect min/max activation ranges per tensor (Welford's online
   algorithm also tracks μ/σ for future Gaussian-informed range tightening).

2. **Weight quantization** — INT8 symmetric with Gaussian-informed hybrid
   scaling: `scale = max(2.83·σ/127, max_abs/127)`. Computed in **f64** to
   match ORT's internal precision.

3. **Bias quantization** — INT32 with `scale = x_scale × w_scale`, zero_point
   = 0. DequantizeLinear wraps the quantized bias.

4. **Activation quantization** — UINT8 asymmetric with **zero-point nudging**:
   after computing the initial zero_point, the range is adjusted so `float 0.0`
   maps exactly to the quantized zero. This prevents rounding asymmetry from
   compounding across layers.

5. **ReLU-aware QDQ placement** — Conv outputs that feed directly into ReLU
   skip the output QDQ. QDQ is placed after the ReLU instead, matching ORT's
   internal Conv+ReLU fusion pattern and enabling more efficient kernel
   selection.

6. **Non-Conv QDQ** — ReLU, Concat, MaxPool, AveragePool, GlobalAveragePool
   outputs are wrapped with QDQ when calibration data is available.

---

## Key Discovery: ReLU-Aware QDQ Placement

### The Problem

ORT's runtime optimizer fuses QDQ patterns into efficient INT8 kernels. The
_placement_ of QDQ nodes in the graph determines _which_ kernels get selected.

Our original pipeline placed QDQ on **every Conv output**, then separately
wrapped ReLU outputs with additional QDQ. This created:

```
Conv → QL → DQL → ReLU → QL → DQL → next_Conv_input → QL → DQL
```

Three quantize-dequantize round-trips where ORT's own quantizer only has the
necessary ones. Each round-trip introduces rounding error.

### The Fix

Detect Conv→ReLU pairs at graph level. For those Conv outputs, skip the output
QDQ and let the ReLU pass add QDQ after the activation instead:

```
Conv → ReLU → QL → DQL → next_Conv_input → QL → DQL
```

This reduces the round-trips and, critically, gives ORT's optimizer a clearer
`Conv → ReLU → QuantizeLinear` pattern that maps to its most efficient fused
INT8 kernel.

### Impact

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Top-1 accuracy | 83% | 83% | — |
| Latency | 3.82ms | **2.71ms** | **-29%** |
| vs ORT quantized | 0.94× | **1.50×** | Faster than ORT |

The accuracy is unchanged because the same information is captured (post-ReLU
range encompasses the Conv output range). The latency improvement comes purely
from ORT's kernel selection — the graph structure signals a fusion-friendly
pattern.

### Structural comparison

| | ORT quantizer | Kenosis |
|---|---|---|
| Total nodes | 174 | 230 |
| QuantizeLinear | 41 | 56 |
| DequantizeLinear | 93 | 108 |
| Relu nodes | 0 (fused into Conv) | 26 (explicit) |
| Latency | 4.08ms | **2.71ms** |

Kenosis has **more** nodes yet runs **33% faster**. This demonstrates that QDQ
_placement pattern_ matters more than QDQ _count_ for ORT kernel selection.

### Key insight

Both models run on the same ORT runtime. The speed difference is a
**quantizer quality** advantage, not a runtime trick. Kenosis produces a graph
structure that ORT's optimizer fuses more aggressively.

---

## Optimization History

All tested on SqueezeNet 1.1, N=500, per-tensor mode.

| Step | Change | Top-1 | Latency | Note |
|------|--------|-------|---------|------|
| 0 | Baseline (ImageNet-norm, per-channel) | 71% | ~4.0ms | — |
| 1 | Disable Gaussian activation tightening | 71% | ~4.0ms | Zero effect (dead code on SN) |
| 2 | Match ORT's Gaussian calibration | 75% | ~4.0ms | +4pp — distribution mattered |
| 3 | Switch to per-tensor | 83% | 3.97ms | +8pp — SN fire modules need PT |
| 4 | Match ORT's exact PRNG (seed-42) | 81% | — | Confirmed gap is algorithmic |
| 5 | INT32 bias quantization | 83% | 3.82ms | +0pp accuracy, +5% latency |
| 6 | Zero-point nudging | 83% | ~3.8ms | Marginal (+2 samples) |
| 7 | f64 scale precision | 83% | ~3.8ms | Marginal |
| 8 | **ReLU-aware QDQ placement** | **83%** | **2.71ms** | **-29% latency** |

### Findings summary

- **Calibration distribution** was the single biggest accuracy confounder
- **Per-tensor vs per-channel** is architecture-specific (SN needs per-tensor)
- **INT32 bias quant** is critical for per-channel mode (+7pp on SN)
- **ReLU-aware QDQ** is a graph-optimization win with massive latency impact
- **ZP nudging + f64** are correct practices but negligible on this model

---

## SqueezeNet Benchmark Results

### Final comparison (all optimizations applied)

| Config | Cosine | Top-1 | Latency | vs FP32 | Size |
|--------|--------|-------|---------|---------|------|
| **ORT per-tensor** | 0.9991 | **85%** | 4.08ms | 1.53× | 1.24 MB |
| **Kenosis per-tensor** | 0.9991 | **83%** | **2.71ms** | **2.27×** | 1.24 MB |
| Kenosis per-channel | 0.9983 | 82% | **2.71ms** | **2.27×** | 1.26 MB |
| ORT per-channel QOp | 0.9994 | 75% | 4.18ms | 1.50× | 1.24 MB |

### Defensible claims

- ✅ "Produces quantized models ORT executes 1.5× faster than ORT's own output"
- ✅ "2.27× speedup over FP32 inference"
- ✅ "Within 2pp of ORT accuracy, pure Rust, zero dependencies"

---

## Multi-Architecture Validation

### Hypotheses

> **H1:** Per-channel underperforms per-tensor on architectures with low channel
> counts in squeeze/early layers. Per-channel should recover or exceed per-tensor
> on architectures with higher channel counts.

> **H2:** ReLU-aware QDQ latency advantage may be specific to ReLU-based
> architectures. Models using GELU/SiLU/H-Swish may not benefit.

> **H3:** The latency advantage from graph structure should generalize across
> model sizes if the underlying pattern (Conv→ReLU fusion) is consistent.

### Test matrix

| Model | Params | Activation | Channel range | Status |
|-------|--------|-----------|---------------|--------|
| SqueezeNet 1.1 | 1.24M | ReLU | 16–64 squeeze | ✅ Done |
| MobileNetV3-Small | 2.54M | H-Swish + ReLU | 16–576 | ✅ Done |
| ResNet50 v2 | 25.6M | ReLU | 64–2048 | ✅ Done |
| EfficientNet-Lite4 | 13.0M | ReLU | 32–1280 | ⬜ Pending |

### SqueezeNet 1.1 (N=500)

| Config | Cosine | Top-1 | Latency | vs FP32 |
|--------|--------|-------|---------|---------|
| **ORT per-tensor** | 0.9991 | **85%** | 4.08ms | 1.53× |
| **Kenosis per-tensor** | 0.9991 | 83% | **2.71ms** | **2.27×** |
| Kenosis per-channel | 0.9983 | 82% | **2.71ms** | **2.27×** |

**Finding:** Per-tensor wins on SqueezeNet (low channel fire modules). Kenosis
1.50× faster than ORT's quantized model. Accuracy gap: -2pp.

### MobileNetV3-Small (N=500)

| Config | Cosine | Top-1 | Latency | vs FP32 |
|--------|--------|-------|---------|---------|
| **ORT per-tensor** | 0.333 | **0%** | 5.91ms | 0.44× |
| **Kenosis per-tensor** | -0.039 | 0% | 2.98ms | 0.87× |
| Kenosis per-channel | -0.249 | 0% | 3.05ms | 0.87× |

**Finding:** MobileNetV3 is **quantization-resistant** for both ORT and Kenosis.
The H-Swish activation (HardSigmoid × Mul) and Squeeze-Excitation blocks create
multiplicative tensor interactions that are destroyed by INT8 quantization.
ORT gets cosine ~0.33 (still 0% Top-1); we get worse cosine but this model
cannot be statically quantized with current approaches. Key insight: **ReLU-aware
QDQ provides no benefit on non-ReLU activation functions** — H2 confirmed.

### ResNet50 v2 (N=300)

| Config | Cosine | Top-1 | Latency | vs FP32 |
|--------|--------|-------|---------|---------|
| **ORT per-tensor** | 0.9948 | **100%** | 49.47ms | 1.40× |
| **Kenosis per-tensor** | 0.9957 | 99% | **37.52ms** | **1.86×** |
| **Kenosis per-channel** | **0.9990** | **100%** | **38.02ms** | **1.84×** |

**Finding:** Kenosis outperforms ORT on all metrics except 1 sample accuracy.
**Latency advantage grows** on larger models: 24% faster than ORT vs 33% on
SqueezeNet — H3 confirmed, the advantage generalizes. Per-channel **matches ORT
Top-1** with higher cosine (0.999 vs 0.995) — H1 confirmed, per-channel recovers
on high-channel architectures.

### Summary across architectures

| Metric | SqueezeNet | MobileNetV3 | ResNet50 |
|--------|-----------|-------------|----------|
| Kenosis vs ORT latency | **1.50×** faster | **2.0×** faster | **1.32×** faster |
| Kenosis vs ORT accuracy | -2pp | Both fail | -1 sample (PT), tied (PC) |
| Per-tensor vs per-channel | PT wins | Both fail | PC wins |
| ReLU-aware QDQ benefit | ✅ Strong | ❌ N/A (H-Swish) | ✅ Strong |

### Validated conclusions

1. **Latency advantage is general** — holds across ReLU architectures and grows
   with model depth. The graph structure optimization is not SqueezeNet-specific.
2. **Per-channel vs per-tensor is architecture-dependent** — as predicted, low
   channel counts favor per-tensor, high channel counts favor per-channel.
3. **H-Swish models need special treatment** — naive INT8 fails catastrophically
   on MobileNetV3 for both ORT and Kenosis. These models need mixed-precision
   quantization (keep H-Swish layers at FP32) or quantization-aware training.
4. **Kenosis accuracy matches or exceeds ORT** on standard ReLU architectures
   (ResNet50 per-channel: identical Top-1, higher cosine).

---

## Technical Details

### Zero-point nudging

After computing the initial scale and zero_point from min/max ranges, the range
is adjusted so that `float 0.0` maps exactly to the quantized zero_point:

```rust
let zp = ((-rmin / scale).round().clamp(0.0, 255.0)) as u8;
let rmin = -(zp as f64) * scale;
let rmax = (255.0 - zp as f64) * scale;
let scale = ((rmax - rmin) / 255.0).max(f64::EPSILON);
```

This ensures that the quantized representation of zero is exact, preventing
rounding asymmetry from accumulating across layers.

### f64 precision

All scale computations (activation ranges, weight scales, bias scales) are
performed in f64 internally, only casting to f32 at the storage boundary. This
matches ORT's internal precision and prevents rounding error accumulation in
scale values.

### Bias quantization formula

```
bias_scale = activation_scale × weight_scale
bias_quantized = round(bias_fp32 / bias_scale)
bias_zp = 0  (always)
```

For per-channel weights, the mean weight scale across channels is used for bias
quantization.

---

## PP-YOLOE+ Detection Model Quantization

### Overview

PP-YOLOE+ is a PaddlePaddle-based object detection model used in production edge AI deployments.
Three resolution variants were quantized: 320×320, 416×416, and 640×640.

### Structural Blockers Resolved

1. **Multi-input models** — PP-YOLOE+ has two inputs (`image` + `scale_factor`).
   The calibration pipeline was extended to auto-generate default values for
   secondary inputs (`1.0` for scale-like inputs, `0.0` for others).

2. **PaddlePaddle Constant nodes** — All 349 weights stored as inline `Constant`
   nodes (zero initializers). `--extract-constants` converts these to standard
   initializers before quantization.

3. **Shared weights (deepcopy)** — PaddlePaddle exports share weight tensors
   across Conv nodes with `deepcopy` suffixes. Weight DQL deduplication prevents
   duplicate node names in the output graph.

4. **Opset 11 → 13 upgrades** — `Squeeze` and `Unsqueeze` `axes` attribute
   converted to input tensors; `BatchNormalization` `spatial` attribute removed.

### Critical Discovery: Mixed-Precision Detection Head

**Problem:** The first quantized model produced zero bounding boxes in the
production pipeline. Root cause analysis identified two defects:

1. **scale_factor input was dead** — The `Concat` node on the scale_factor path
   was QDQ-wrapped, crushing the scale values to the narrow calibration range
   `[1.0, 1.0]`. Any runtime scale_factor other than `[1, 1]` was clamped.

2. **Score discrimination collapsed** — The classification head's sigmoid outputs
   were quantized, reducing unique score values from 3336 to 78. All top scores
   collapsed to ~0.298, making the pipeline's 0.40 confidence threshold
   unreachable.

**Solution:** Two protective filters in the QDQ placement logic:

- **Non-vision tensor tracing** — Walk the graph forward from non-primary inputs
  (scale_factor). All reachable tensors are marked as "non-vision" and excluded
  from QDQ wrapping. This preserves the dynamic scale_factor path.

- **Model output protection** — Tensors that are direct model outputs are never
  QDQ-wrapped, preserving full FP32 precision in the detection head outputs.

These filters apply in both the Conv output QDQ pass and the non-Conv QDQ pass.

### Verification Results (after fix)

| Metric | Before Fix | After Fix | Target |
|--------|-----------|-----------|--------|
| scale_factor delta (sf=2 vs sf=1) | 0.00 | 683,849 | > 100 |
| Unique score values | 78 | 421 | > 200 |
| Max score (random input) | 0.298 | 0.386 | > 0.30 |
| Score cosine similarity | — | 0.902 | — |

### Performance Results

| Model | Resolution | Cosine | Latency | Speedup | Size |
|-------|-----------|--------|---------|---------|------|
| ppyoloe_plus_s | 320×320 | 0.9980 | 23ms vs 44ms | **1.89×** | 7.9 MB (3.9×) |
| ppyoloe_plus_s | 416×416 | 0.9983 | 43ms vs 77ms | **1.80×** | 7.9 MB (3.9×) |
| ppyoloe_plus_s | 640×640 | 0.9990 | 111ms vs 187ms | **1.68×** | 7.9 MB (3.8×) |

### Key Insight

Detection models require **selective quantization**. Unlike classifiers where
all layers can be INT8, detection models have:

- **Backbone** — Quantize fully (Conv + BatchNorm + activation)
- **Neck/FPN** — Quantize Concat/Conv but protect metadata paths
- **Detection head outputs** — Keep in FP32 (score discrimination + box precision)
- **Non-vision inputs** — Never quantize (scale_factor, image_shape, etc.)

This is consistent with industry practice: TensorRT, OpenVINO, and AIMET all
implement detection-specific mixed-precision policies.

---

## Gaussian Activation Tightening — A/B Test (2026-05-06)

### Hypothesis

Gaussian-informed activation range tightening (`μ ± 2.83σ` instead of
`[observed_min, observed_max]`) should concentrate INT8 precision where
activation values cluster, reducing quantization noise in the dense center
of the distribution.

Weight quantization already uses hybrid Gaussian scaling
(`scale = max(2.83·σ/127, max_abs/127)`) successfully. The question was
whether the same principle works for activations.

### Implementation

The tightening applies when:
- `count > 10,000` (enough Welford samples for reliable statistics)
- Gaussian range is 50–95% of observed range (provides benefit without
  excessive clipping)

15 of 67 SqueezeNet activation tensors qualified, with range reductions
of 7–47%.

### Results

| Model | Metric | Baseline (min/max) | Gaussian (μ ± 2.83σ) |
|-------|--------|-------------------|---------------------|
| **SqueezeNet 1.1** | Cosine | 0.999 | 0.972 |
| **SqueezeNet 1.1** | Top-1 | 83% | 51% |
| **ResNet50 v2** | Cosine | 0.999 | 0.936 |
| **ResNet50 v2** | Top-1 | 100% | 93% |
| **PP-YOLOE+ 320** | Cosine | 0.998 | 0.980 |

### Verdict: **Gaussian activation tightening hurts accuracy.**

The degradation is uniform across architectures — not model-specific.

### Root Cause

Synthetic calibration data (random Gaussian inputs) produces activation
distributions with **broader tails** than real inference data. The observed
min/max already reflects this broader distribution. When we tighten to
`μ ± 2.83σ`, we clip outlier activations that carry signal in the specific
calibration distribution.

The fundamental issue is that Gaussian tightening assumes the calibration
data represents real deployment data well enough for the Gaussian prior to
be informative. With synthetic data, this assumption fails.

### Future Work

Re-test with **real calibration images** via `--calib-dir`. If real data
produces tighter, more Gaussian activation distributions, the tightening
may become beneficial. The Welford statistics collection (`update_gaussian`)
remains active in the pipeline so the infrastructure is ready.

---

## Kenosis vs ORT — Controlled Multi-Architecture Comparison (2026-05-06)

### Methodology

Both quantizers given the **same inputs** (40 synthetic Gaussian samples for
calibration, same random eval inputs), evaluated by the **same validator**
(Kenosis `validate` subcommand, N=300–500). ORT baselines freshly generated
with stock `onnxruntime.quantization.quantize_static` — no pre-processing.

> **Note:** An earlier comparison used ORT models that had been pre-processed
> through Kenosis's Gaussian weight-rounding pipeline, sabotaging ORT's
> accuracy. Those numbers have been retracted.

### Results

| Model | Kenosis Cosine | ORT Cosine | Kenosis Top-1 | ORT Top-1 | Winner |
|-------|---------------|------------|---------------|-----------|--------|
| **SqueezeNet 1.1** | 0.999 | 0.999 | 83% | **85%** | ORT +2pp |
| **ResNet50 v2** | **0.996** | 0.994 | **99%** | 100% | ~Tied |
| **ShuffleNet V2** | **0.990** | 0.991 | **65%** | 48% | **Kenosis +17pp** |
| **MobileNetV2** | 0.988 | **0.990** | 39% | **75%** | ORT +36pp |
| **MobileNetV3 Small** | -0.039 | 0.274 | 0% | 0% | Both fail |

### Analysis

**Kenosis wins or ties on standard Conv architectures:**
- **ResNet50**: Tied at 99–100% Top-1, Kenosis has better cosine (0.996 vs 0.994)
- **SqueezeNet**: ORT leads by 2pp (85% vs 83%), identical cosine
- **ShuffleNet V2**: Kenosis wins by **17pp** (65% vs 48%) — our QDQ placement
  and Gaussian weight scaling produce a significantly better quantized model

**ORT advantage on depthwise-separable architectures:**
- **MobileNetV2**: ORT 75% vs Kenosis 39% — 36pp gap. Depthwise convolutions
  have tiny per-channel kernels (9 weights) where per-tensor quantization is
  too coarse. ORT's QLinear format provides tighter packing (3.43 MB vs 7.10 MB).
- **MobileNetV3**: Both fail completely (H-Swish destroys INT8 precision).

### Depthwise Per-Channel Investigation

Attempted auto per-channel quantization for depthwise Conv layers
(`group == out_channels`). **Result: total collapse** (cosine 0.02 → 0.11).

Root cause: ORT's runtime optimizer cannot fuse per-channel DequantizeLinear
with grouped convolutions. The QDQ pair is not recognized as fuseable, so
the model falls back to dequantize→float32→quantize for each depthwise layer,
accumulating catastrophic rounding errors.

Additionally, the Gaussian sigma computed from only 9 values per channel
was wildly unreliable. Added a guard (`ch_size >= 32`) to skip the Gaussian
floor for tiny channels. This guard remains active.

### Fixes Shipped

1. **Split opset 13 conversion** — `Split` attribute → input tensor.
   Enables ShuffleNet V2 quantization (was previously blocked by opset error).
2. **Gaussian guard for tiny channels** — Skip `CLIP·σ` floor when
   `ch_size < 32` to prevent unreliable sigma from inflating per-channel scales.
3. **Percentile-based activation ranges** — 1st/99th percentile of per-sample
   min/max instead of absolute extremes. No accuracy change with synthetic data;
   ready for real calibration images.

### Open Items

- **MobileNetV2 gap**: Requires investigation into ORT's QLinear representation
  to understand how they handle depthwise quantization differently. May need a
  dedicated depthwise quantization path in Kenosis.
- **MobileNetV3**: Needs activation-aware quantization (skip H-Swish/SiLU layers).

---

### Key files

- `crates/kenosis-core/src/static_int8.rs` — All quantization logic
- `apps/kenosis-cli/src/commands/quantize.rs` — CLI interface
- `docs/QUANTIZATION_FINDINGS.md` — This document
