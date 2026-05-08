// Copyright (c) 2026 Core Epoch. Licensed under Apache-2.0.
//! Static INT8 quantization with ReLU-aware QDQ placement.
//!
//! Uses the **QDQ (Quantize-Dequantize)** format:
//! - Wraps each Conv with QuantizeLinear/DequantizeLinear nodes
//! - ReLU-aware placement enables full QLinearConv fusion in ORT
//! - **Min/max** symmetric weight scales with f64 precision
//! - **Calibration-derived** activation scales from running the model
//!
//! Requires the `calibrate` feature (pulls in the `ort` crate).

use std::collections::{HashMap, HashSet};

use crate::model::OnnxModel;
use crate::proto::{self, data_type, NodeProto, TensorProto};

/// Statistics returned after static INT8 quantization.
#[derive(Debug, Default)]
pub struct StaticInt8Stats {
    /// Number of Conv nodes wrapped with QDQ.
    pub conv_replaced: usize,
    /// Number of MatMul nodes wrapped with QDQ.
    pub matmul_replaced: usize,
    /// Total weight tensors processed.
    pub total_weights: usize,
    /// Number of activation tensors with calibration ranges.
    pub activation_tensors_calibrated: usize,
    /// Number of layers skipped by sensitivity analysis (too accuracy-sensitive).
    pub sensitivity_layers_skipped: usize,
}

/// Activation range observed during calibration.
///
/// Collects min/max bounds and per-sample ranges for percentile-based
/// activation scale computation (more robust than absolute min/max).
#[derive(Debug, Clone)]
struct ActivationRange {
    min: f32,
    max: f32,
    /// Per-sample minimum values for percentile computation
    sample_mins: Vec<f32>,
    /// Per-sample maximum values for percentile computation
    sample_maxs: Vec<f32>,
}

impl ActivationRange {
    fn new() -> Self {
        Self {
            min: f32::MAX,
            max: f32::MIN,
            sample_mins: Vec::new(),
            sample_maxs: Vec::new(),
        }
    }

    fn update(&mut self, min: f32, max: f32) {
        if min < self.min {
            self.min = min;
        }
        if max > self.max {
            self.max = max;
        }
        self.sample_mins.push(min);
        self.sample_maxs.push(max);
    }

    /// Compute UINT8 scale and zero_point using percentile-based ranges.
    ///
    /// Instead of using absolute min/max (sensitive to single outlier samples),
    /// takes a robust percentile of per-sample min/max values. This filters
    /// out rare extreme calibration samples that waste INT8 precision on
    /// empty range regions.
    fn to_uint8_params(&self) -> (f32, u8) {
        // Use f64 internally to match ORT's precision (avoids compounding
        // rounding errors across 26+ layers).

        // Percentile-based range: use 99.99th percentile of per-sample
        // min/max instead of absolute extremes. For N samples, this clips
        // at most the single most extreme sample.
        let (rmin, rmax) = if self.sample_mins.len() >= 3 {
            let mut mins = self.sample_mins.clone();
            let mut maxs = self.sample_maxs.clone();
            mins.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            maxs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Use 1st percentile for min (clip lowest outlier mins)
            // and 99th percentile for max (clip highest outlier maxs)
            let lo_idx = (mins.len() as f64 * 0.01).floor() as usize;
            let hi_idx = ((maxs.len() as f64 * 0.99).ceil() as usize).min(maxs.len() - 1);
            let pmin = (mins[lo_idx].min(0.0)) as f64;
            let pmax = (maxs[hi_idx].max(0.0)) as f64;
            (pmin, pmax)
        } else {
            ((self.min.min(0.0)) as f64, (self.max.max(0.0)) as f64)
        };

        // Compute initial scale and zero_point
        let scale = ((rmax - rmin) / 255.0).max(f64::EPSILON);
        let zp = ((-rmin / scale).round().clamp(0.0, 255.0)) as u8;

        // ── Zero-point nudging ──────────────────────────────────────
        // Adjust the range so that float 0.0 maps EXACTLY to the quantized
        // zero_point. Without this, rounding asymmetry compounds across layers.
        // This matches ORT's post-calibration range adjustment.
        let rmin = -(zp as f64) * scale;
        let rmax = (255.0 - zp as f64) * scale;
        // Recompute scale from nudged range
        let scale = ((rmax - rmin) / 255.0).max(f64::EPSILON);

        (scale as f32, zp)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Calibration data generation
// ─────────────────────────────────────────────────────────────────────────────

struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state & 0x00FF_FFFF) as f32 / 16777216.0
    }

    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(f32::EPSILON);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Generate calibration data matching ORT's Python quantizer.
///
/// ORT's standard usage feeds `np.random.randn(*input_shape).astype(np.float32)`
/// into calibration — raw standard Gaussian noise with no normalization.
/// We replicate that here using Box-Muller transform for N(0,1) samples.
///
/// The seed is deterministic per sample_idx to ensure reproducibility,
/// but intentionally different from ORT's numpy seed — the distribution
/// shape (standard Gaussian) is what matters for activation range estimation,
/// not the exact values.
fn generate_calib_data(shape: &[usize], sample_idx: usize) -> Vec<f32> {
    let total: usize = shape.iter().product();
    let mut rng = Xorshift64::new((sample_idx as u64 + 1) * 0xDEAD_BEEF);

    // Standard Gaussian noise — matches ORT's np.random.randn
    (0..total).map(|_| rng.next_normal()).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Calibration
// ─────────────────────────────────────────────────────────────────────────────

/// Augment the model to output every intermediate activation tensor,
/// then run `n_calib` calibration inputs and return per-tensor min/max ranges.
///
/// If `calib_dir` is provided, loads raw f32 binary files from that directory
/// instead of generating synthetic data. Files must be named `*.bin` and each
/// contain exactly `product(shape) * 4` bytes of little-endian f32 values.
#[cfg(feature = "calibrate")]
fn calibrate_activations(
    model: &OnnxModel,
    n_calib: usize,
    calib_dir: Option<&std::path::Path>,
) -> crate::Result<HashMap<String, ActivationRange>> {
    use ort::session::{builder::GraphOptimizationLevel, Session};
    use ort::value::Tensor;

    let mut aug = model.proto.clone();
    let graph = aug
        .graph
        .as_mut()
        .ok_or_else(|| crate::KenosisError::InvalidModel("no graph in model".into()))?;

    let existing_outputs: HashSet<String> = graph.output.iter().map(|o| o.name.clone()).collect();
    let init_names: HashSet<String> = graph.initializer.iter().map(|t| t.name.clone()).collect();

    for node in &graph.node {
        for out in &node.output {
            if out.is_empty() || existing_outputs.contains(out) || init_names.contains(out) {
                continue;
            }
            graph.output.push(proto::ValueInfoProto {
                name: out.clone(),
                ..Default::default()
            });
        }
    }

    let aug_bytes = {
        use prost::Message;
        let mut buf = Vec::with_capacity(aug.encoded_len());
        aug.encode(&mut buf)
            .map_err(|e| crate::KenosisError::InvalidModel(format!("encode: {e}")))?;
        buf
    };

    let mut session = Session::builder()
        .map_err(|e| crate::KenosisError::InvalidModel(format!("ort builder: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|e| crate::KenosisError::InvalidModel(format!("ort opt: {e}")))?
        .with_intra_threads(1)
        .map_err(|e| crate::KenosisError::InvalidModel(format!("ort threads: {e}")))?
        .commit_from_memory(&aug_bytes)
        .map_err(|e| crate::KenosisError::InvalidModel(format!("ort load: {e}")))?;

    let input_name = session.inputs()[0].name().to_string();
    let shape: Vec<usize> = session.inputs()[0]
        .dtype()
        .tensor_shape()
        .ok_or_else(|| crate::KenosisError::InvalidModel("no tensor shape".into()))?
        .iter()
        .map(|&d| if d < 0 { 1usize } else { d as usize })
        .collect();

    // Load external calibration data if directory provided
    let external_data: Option<Vec<Vec<f32>>> = if let Some(dir) = calib_dir {
        let total: usize = shape.iter().product();
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| crate::KenosisError::InvalidModel(format!("calib dir: {e}")))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "bin"))
            .collect();
        files.sort_by_key(|e| e.file_name());

        let mut samples = Vec::new();
        for entry in files.iter().take(n_calib) {
            let bytes = std::fs::read(entry.path())
                .map_err(|e| crate::KenosisError::InvalidModel(format!("read calib: {e}")))?;
            if bytes.len() != total * 4 {
                return Err(crate::KenosisError::InvalidModel(format!(
                    "calib file {} has {} bytes, expected {}",
                    entry.path().display(),
                    bytes.len(),
                    total * 4
                )));
            }
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            samples.push(floats);
        }
        tracing::info!(n = samples.len(), dir = %dir.display(), "Loaded external calibration data");
        Some(samples)
    } else {
        None
    };

    let actual_n = external_data.as_ref().map_or(n_calib, |d| d.len());
    tracing::info!(input = %input_name, shape = ?shape, n_calib = actual_n, "Starting calibration");


    // Collect info about all inputs (type, shape, name) for multi-input models
    struct InputInfo {
        name: String,
        shape: Vec<usize>,
        is_int: bool,
    }
    let all_inputs: Vec<InputInfo> = session
        .inputs()
        .iter()
        .map(|info| {
            let name = info.name().to_string();
            let dt = info.dtype();
            let desc = format!("{dt:?}");
            let is_int = desc.contains("Int64") || desc.contains("Int32");
            let sh: Vec<usize> = dt
                .tensor_shape()
                .into_iter()
                .flat_map(|s| s.iter())
                .map(|&d| if d < 0 { 1 } else { d as usize })
                .collect();
            InputInfo {
                name,
                shape: sh,
                is_int,
            }
        })
        .collect();

    let mut ranges: HashMap<String, ActivationRange> = HashMap::new();

    for i in 0..actual_n {
        let mut inputs_map = std::collections::HashMap::new();

        for inp in &all_inputs {
            let total: usize = inp.shape.iter().product();

            if inp.is_int {
                // Generate synthetic integer data for transformer inputs
                let mut rng = Xorshift64::new((i as u64 + 1) * 0xCAFE_BABE);
                let int_data: Vec<i64> = if inp.name.contains("attention_mask") {
                    // attention_mask: all 1s (all tokens are valid)
                    vec![1i64; total]
                } else if inp.name.contains("token_type") {
                    // token_type_ids: all 0s (single sentence)
                    vec![0i64; total]
                } else {
                    // input_ids or other int input: random token IDs [1, 30000)
                    (0..total)
                        .map(|_| {
                            let v = rng.next_f32();
                            (v * 29999.0) as i64 + 1
                        })
                        .collect()
                };
                let t = Tensor::from_array((inp.shape.clone(), int_data))
                    .map_err(|e| {
                        crate::KenosisError::InvalidModel(format!("int tensor: {e}"))
                    })?;
                inputs_map.insert(inp.name.clone(), ort::value::DynValue::from(t));
            } else {
                // Float data — use external calibration or synthetic Gaussian
                let data = if inp.name == input_name {
                    match &external_data {
                        Some(ext) => ext[i].clone(),
                        None => generate_calib_data(&inp.shape, i),
                    }
                } else {
                    // Extra float inputs (e.g. scale_factor for PP-YOLOE+)
                    let fill = if inp.name.contains("scale") {
                        1.0f32
                    } else {
                        0.0f32
                    };
                    vec![fill; total]
                };

                // Record input activation range for float inputs
                if inp.name == input_name {
                    let imin = data.iter().copied().fold(f32::MAX, f32::min);
                    let imax = data.iter().copied().fold(f32::MIN, f32::max);
                    let entry = ranges
                        .entry(input_name.clone())
                        .or_insert_with(ActivationRange::new);
                    entry.update(imin, imax);
                }

                let t = Tensor::from_array((inp.shape.clone(), data))
                    .map_err(|e| {
                        crate::KenosisError::InvalidModel(format!("tensor: {e}"))
                    })?;
                inputs_map.insert(inp.name.clone(), ort::value::DynValue::from(t));
            }
        }

        let outputs = session
            .run(inputs_map)
            .map_err(|e| crate::KenosisError::InvalidModel(format!("ort run: {e}")))?;

        for (name, value) in &outputs {
            if let Ok((_shape, values)) = value.try_extract_tensor::<f32>() {
                let vals: Vec<f32> = values.to_vec();
                let min = vals.iter().copied().fold(f32::MAX, f32::min);
                let max = vals.iter().copied().fold(f32::MIN, f32::max);
                let entry = ranges
                    .entry(name.to_string())
                    .or_insert_with(ActivationRange::new);
                entry.update(min, max);
            }
        }
    }

    tracing::info!(calibrated = ranges.len(), "Calibration complete");
    Ok(ranges)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Weight quantization helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Symmetric per-channel INT8 weight quantization.
///
/// `scale = max_abs / 127` per output channel, computed in f64 to match
/// ORT's internal precision.
fn quantize_weights_per_channel(values: &[f32], dims: &[i64]) -> (Vec<i8>, Vec<f32>) {
    let num_ch = dims[0] as usize;
    let ch_size = values.len() / num_ch;
    let mut scales = Vec::with_capacity(num_ch);
    let mut quantized = Vec::with_capacity(values.len());

    for c in 0..num_ch {
        let ch = &values[c * ch_size..(c + 1) * ch_size];
        let max_abs = ch.iter().map(|v| v.abs()).fold(0.0f32, f32::max) as f64;
        let scale = (max_abs / 127.0).max(f64::EPSILON);
        let scale_f32 = scale as f32;

        scales.push(scale_f32);
        for &v in ch {
            let q = ((v as f64) / scale).round().clamp(-127.0, 127.0) as i8;
            quantized.push(q);
        }
    }
    (quantized, scales)
}

/// Symmetric per-tensor INT8 weight quantization.
fn quantize_weights_per_tensor(values: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max) as f64;
    let scale = (max_abs / 127.0).max(f64::EPSILON);

    let quantized: Vec<i8> = values
        .iter()
        .map(|&v| ((v as f64) / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (quantized, scale as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Initializer helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_scalar_f32(name: &str, value: f32) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        data_type: data_type::FLOAT,
        float_data: vec![value],
        dims: vec![],
        ..Default::default()
    }
}

fn make_scalar_u8(name: &str, value: u8) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        data_type: data_type::UINT8,
        raw_data: vec![value],
        dims: vec![],
        ..Default::default()
    }
}

fn make_scalar_i8(name: &str, value: i8) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        data_type: data_type::INT8,
        raw_data: vec![value as u8],
        dims: vec![],
        ..Default::default()
    }
}

fn make_1d_f32(name: &str, values: &[f32]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        data_type: data_type::FLOAT,
        float_data: values.to_vec(),
        dims: vec![values.len() as i64],
        ..Default::default()
    }
}

fn make_1d_i8(name: &str, values: &[i8]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        data_type: data_type::INT8,
        raw_data: values.iter().map(|&v| v as u8).collect(),
        dims: vec![values.len() as i64],
        ..Default::default()
    }
}

#[allow(dead_code)]
fn make_1d_i32(name: &str, values: &[i32]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        data_type: data_type::INT32,
        raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        dims: vec![values.len() as i64],
        ..Default::default()
    }
}

fn make_scalar_i32(name: &str, value: i32) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        data_type: data_type::INT32,
        raw_data: value.to_le_bytes().to_vec(),
        dims: vec![],
        ..Default::default()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main entry point: QDQ-format static INT8 quantization
// ─────────────────────────────────────────────────────────────────────────────

/// Result of per-layer sensitivity analysis.
#[derive(Debug)]
pub struct LayerSensitivity {
    /// Node name in the ONNX graph.
    pub name: String,
    /// Op type (Conv or MatMul).
    pub op_type: String,
    /// Weight quantization signal-to-noise ratio in dB.
    /// Lower SNR → more distortion from quantization.
    pub weight_snr_db: f64,
    /// Whether this layer should be skipped (left in FP32).
    pub skip: bool,
}

/// Analyze quantization sensitivity for each Conv/MatMul layer.
///
/// Computes per-layer weight quantization SNR (signal-to-noise ratio):
/// `SNR = 10 * log10(signal_power / noise_power)` where noise is the
/// quantization error. Layers below `snr_threshold_db` are flagged for
/// skipping — they lose too much information when quantized to INT8.
///
/// A typical threshold is 30 dB (very conservative) to 20 dB (aggressive).
///
/// Returns per-layer results and a convenience `HashSet` of layer names to skip.
pub fn sensitivity_analysis(
    model: &OnnxModel,
    snr_threshold_db: f64,
) -> (Vec<LayerSensitivity>, HashSet<String>) {
    let graph = model.graph();
    let init_map: HashMap<String, usize> = graph
        .initializer
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.clone(), i))
        .collect();

    let mut results = Vec::new();
    let mut skip_set = HashSet::new();

    for node in &graph.node {
        let is_conv = node.op_type == "Conv" && node.domain.is_empty() && node.input.len() >= 2;
        let is_matmul = node.op_type == "MatMul" && node.domain.is_empty() && node.input.len() >= 2;

        if !is_conv && !is_matmul {
            continue;
        }

        // Weight is input[1] for both Conv and MatMul
        let w_name = &node.input[1];
        let w_idx = match init_map.get(w_name) {
            Some(&i) => i,
            None => continue, // Dynamic weight, skip analysis
        };
        let w_tensor = &graph.initializer[w_idx];
        let w_values = match OnnxModel::tensor_as_f32(w_tensor) {
            Some(v) if !v.is_empty() => v,
            _ => continue,
        };

        // Compute weight quantization SNR
        let (q, scale) = quantize_weights_per_tensor(&w_values);
        let mut signal_power = 0.0f64;
        let mut noise_power = 0.0f64;
        for (i, &original) in w_values.iter().enumerate() {
            let reconstructed = q[i] as f64 * scale as f64;
            let orig = original as f64;
            signal_power += orig * orig;
            noise_power += (orig - reconstructed) * (orig - reconstructed);
        }
        let snr_db = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f64::INFINITY // Perfect quantization
        };

        let should_skip = snr_db < snr_threshold_db;
        if should_skip {
            skip_set.insert(node.name.clone());
            tracing::info!(
                layer = %node.name,
                op = %node.op_type,
                snr_db = format!("{snr_db:.1}"),
                "Skipping layer (below sensitivity threshold)"
            );
        }

        results.push(LayerSensitivity {
            name: node.name.clone(),
            op_type: node.op_type.clone(),
            weight_snr_db: snr_db,
            skip: should_skip,
        });
    }

    tracing::info!(
        total_layers = results.len(),
        skipped = skip_set.len(),
        threshold_db = snr_threshold_db,
        "Sensitivity analysis complete"
    );

    (results, skip_set)
}

/// Run ORT's Level1 graph optimizer on the model before quantization.
///
/// This folds BatchNormalization into Conv, simplifies constant expressions,
/// and removes dead branches — producing a cleaner graph with fewer nodes
/// to quantize and better fusion opportunities.
///
/// Returns the optimized model, or the original model if optimization fails
/// (some models with unsupported ops may not survive the round-trip).
#[cfg(feature = "calibrate")]
pub fn pre_optimize_graph(model: OnnxModel) -> crate::Result<OnnxModel> {
    use ort::session::{builder::GraphOptimizationLevel, Session};

    // Serialize → run through ORT Level1 → save optimized → reload
    let tmp_dir = std::env::temp_dir().join("kenosis_preopt");
    std::fs::create_dir_all(&tmp_dir)
        .map_err(crate::KenosisError::Io)?;
    let input_path = tmp_dir.join("input.onnx");
    let output_path = tmp_dir.join("optimized.onnx");

    model.save(&input_path)?;

    let _session = Session::builder()
        .map_err(|e| crate::KenosisError::InvalidModel(format!("ort builder: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .map_err(|e| crate::KenosisError::InvalidModel(format!("ort opt: {e}")))?
        .with_optimized_model_path(&output_path)
        .map_err(|e| crate::KenosisError::InvalidModel(format!("ort path: {e}")))?
        .commit_from_file(&input_path)
        .map_err(|e| crate::KenosisError::InvalidModel(format!("ort load: {e}")))?;

    // Reload the optimized model
    let optimized = if output_path.exists() {
        let opt = OnnxModel::load(&output_path)?;
        tracing::info!(
            original_nodes = model.nodes().len(),
            optimized_nodes = opt.nodes().len(),
            "Graph pre-optimization complete"
        );
        opt
    } else {
        tracing::warn!("ORT did not produce optimized model, using original");
        model
    };

    // Clean up temp files
    let _ = std::fs::remove_file(&input_path);
    let _ = std::fs::remove_file(&output_path);
    let _ = std::fs::remove_dir(&tmp_dir);

    Ok(optimized)
}

/// Full static INT8 quantization using the QDQ (Quantize-Dequantize) format.
///
/// Instead of emitting `QLinearConv` nodes directly, this inserts
/// `QuantizeLinear → DequantizeLinear` pairs around each Conv's and MatMul's
/// weights and activations. ORT's graph optimizer fuses these into efficient
/// INT8 kernels at runtime, handling Conv+Activation fusion automatically.
///
/// When `pre_optimize` is true, runs ORT's Level1 graph optimizer first to
/// fold BatchNorm into Conv and simplify constants before quantizing.
///
/// This matches what ORT's own static quantizer produces and achieves
/// the best accuracy because ORT handles the fusion boundaries correctly.
#[cfg(feature = "calibrate")]
pub fn quantize_static_int8(
    mut model: OnnxModel,
    per_channel: bool,
    n_calib: usize,
    calib_dir: Option<&std::path::Path>,
    pre_optimize: bool,
    skip_layers: Option<HashSet<String>>,
) -> crate::Result<(OnnxModel, StaticInt8Stats)> {
    if pre_optimize {
        model = pre_optimize_graph(model)?;
    }

    let act_ranges = calibrate_activations(&model, n_calib, calib_dir)?;
    let skip = skip_layers.unwrap_or_default();
    let mut stats = StaticInt8Stats {
        activation_tensors_calibrated: act_ranges.len(),
        sensitivity_layers_skipped: skip.len(),
        ..Default::default()
    };

    let graph = model.graph_mut();

    // Build initializer lookup
    let init_map: HashMap<String, usize> = graph
        .initializer
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.clone(), i))
        .collect();

    // Build set of Conv outputs that feed directly into an activation function.
    // For these, we skip Conv output QDQ and let the activation pass add it instead.
    // This matches ORT's fused Conv+Activation QDQ placement patterns:
    //   Conv→Relu, Conv→Clip (ReLU6), Conv→LeakyRelu, Conv→Sigmoid, Conv→HardSwish
    let activation_ops: HashSet<&str> =
        ["Relu", "Clip", "LeakyRelu", "Sigmoid", "HardSwish"].iter().copied().collect();
    let conv_activation_set: HashSet<String> = {
        let conv_outputs: HashSet<String> = graph
            .node
            .iter()
            .filter(|n| n.op_type == "Conv")
            .flat_map(|n| n.output.iter().cloned())
            .collect();
        graph
            .node
            .iter()
            .filter(|n| activation_ops.contains(n.op_type.as_str()) && !n.input.is_empty())
            .filter(|n| conv_outputs.contains(&n.input[0]))
            .map(|n| n.input[0].clone())
            .collect()
    };

    // Build set of model output tensor names — never QDQ these
    let model_output_names: HashSet<String> = graph.output.iter().map(|o| o.name.clone()).collect();

    // Build set of tensors reachable from non-primary inputs (e.g. scale_factor path).
    // These carry metadata (box scales, reshape constants) that must not be quantized.
    //
    // NOTE: ONNX opset ≤ 12 lists weight initializers as graph.input entries.
    // We must exclude them — only genuine runtime inputs (like scale_factor
    // in PP-YOLOE+) should seed the non-vision flood fill.
    let non_vision_tensors: HashSet<String> = {
        let init_names_set: HashSet<&str> =
            graph.initializer.iter().map(|t| t.name.as_str()).collect();

        // Identify which graph inputs are the ONNX-level "model inputs" (not initializers).
        // For vision models like PP-YOLOE+, secondary float inputs (e.g. scale_factor)
        // carry metadata that must not be quantized. For transformer models, secondary
        // int64 inputs (attention_mask, token_type_ids) are legitimate — skip those.
        let input_names: Vec<String> = graph
            .input
            .iter()
            .skip(1) // skip primary input
            .filter(|i| !init_names_set.contains(i.name.as_str())) // exclude initializers
            .filter(|i| {
                // Only treat float-typed secondary inputs as non-vision metadata.
                // Int64 inputs (tokens, masks) are legitimate model inputs.
                if let Some(tp) = &i.r#type {
                    if let Some(crate::proto::r#type_proto::Value::TensorType(tt)) = &tp.value {
                        // ONNX elem_type: 1=FLOAT, 7=INT64, 6=INT32
                        return tt.elem_type == 1; // Only FLOAT triggers non-vision
                    }
                }
                false
            })
            .map(|i| i.name.clone())
            .collect();
        let mut reachable: HashSet<String> = input_names.into_iter().collect();
        if reachable.is_empty() {
            // Single-input model or all-integer secondary inputs — no non-vision paths
            reachable
        } else {
            // Walk graph forward from non-primary float inputs
            let mut changed = true;
            while changed {
                changed = false;
                for node in &graph.node {
                    if node.input.iter().any(|i| reachable.contains(i)) {
                        for o in &node.output {
                            if reachable.insert(o.clone()) {
                                changed = true;
                            }
                        }
                    }
                }
            }
            tracing::info!(
                non_vision = reachable.len(),
                "Identified non-vision tensors (scale_factor path etc.)"
            );
            reachable
        }
    };

    let mut new_initializers: Vec<TensorProto> = Vec::new();
    let mut new_nodes: Vec<NodeProto> = Vec::new();
    // Track which activation tensors already have QDQ wrappers
    let mut emitted_qdq: HashSet<String> = HashSet::new();
    // Track which tensors got Conv *output* QDQ (separate from input QDQ).
    // The second pass uses this to avoid double-wrapping, without being
    // contaminated by Conv input QDQ deduplication.
    let mut conv_output_qdq: HashSet<String> = HashSet::new();
    // Nodes to insert before each Conv (by Conv node index)
    let mut pre_conv_nodes: HashMap<usize, Vec<NodeProto>> = HashMap::new();
    // Original weight initializer names to remove
    let mut remove_initializers: HashSet<String> = HashSet::new();

    for (node_idx, node) in graph.node.iter().enumerate() {
        if node.op_type != "Conv" || !node.domain.is_empty() || node.input.len() < 2 {
            continue;
        }

        // Skip layers flagged by sensitivity analysis
        if skip.contains(&node.name) {
            continue;
        }

        let x_name = &node.input[0];
        let w_name = &node.input[1];
        let b_name = node.input.get(2).cloned();

        // Get weight tensor
        let w_idx = match init_map.get(w_name) {
            Some(&i) => i,
            None => continue,
        };
        let w_tensor = &graph.initializer[w_idx];
        let w_values = match OnnxModel::tensor_as_f32(w_tensor) {
            Some(v) if v.len() >= 16 => v,
            _ => continue,
        };

        // Get input activation range (skip if uncalibrated)
        let x_range = match act_ranges.get(x_name) {
            Some(r) => r,
            None => continue,
        };
        let (x_scale, x_zp) = x_range.to_uint8_params();

        // ── Weights: pre-quantize to INT8 + DequantizeLinear ────────
        // Store INT8 weights directly as initializers. Only need
        // DequantizeLinear (no QuantizeLinear since they're already quantized).
        // Skip if this weight was already quantized (shared weights in PaddlePaddle models)
        //
        // Note: depthwise Conv (group == out_channels) stays per-tensor.
        // Per-channel QDQ on depthwise causes ORT fusion failure (cosine → 0.02).
        // ORT's runtime optimizer cannot fuse per-channel DequantizeLinear
        // with grouped convolutions. This is a known ORT limitation.
        let is_depthwise = {
            let group = node
                .attribute
                .iter()
                .find(|a| a.name == "group")
                .map(|a| a.i)
                .unwrap_or(1);
            group > 1 && group == w_tensor.dims[0]
        };
        let use_pc = per_channel && w_tensor.dims.len() >= 2 && !is_depthwise;
        let w_q_name = format!("{w_name}_quantized");
        let w_dql_out = format!("{w_name}_DequantizeLinear_Output");
        let w_s_name = format!("{w_name}_scale");
        let w_zp_name = format!("{w_name}_zero_point");
        let weight_already_done = emitted_qdq.contains(w_name);

        // Track per-channel or per-tensor weight scale(s) for bias quantization
        // Always compute scale even if weight was already processed (needed for bias)
        let w_scale_value: f32 = if use_pc {
            let n_ch = w_tensor.dims[0] as usize;
            let ch_size = w_values.len() / n_ch;
            let scales_sum: f64 = (0..n_ch)
                .map(|c| {
                    let ch = &w_values[c * ch_size..(c + 1) * ch_size];
                    let max_abs = ch.iter().map(|v| v.abs() as f64).fold(0.0f64, f64::max);
                    (max_abs / 127.0).max(f64::EPSILON)
                })
                .sum();
            (scales_sum / n_ch as f64) as f32
        } else {
            let max_abs = w_values
                .iter()
                .map(|v| v.abs() as f64)
                .fold(0.0f64, f64::max);
            ((max_abs / 127.0).max(f64::EPSILON)) as f32
        };

        if !weight_already_done {
            emitted_qdq.insert(w_name.to_string());

            if use_pc {
                let (q, scales) = quantize_weights_per_channel(&w_values, &w_tensor.dims);
                let zps = vec![0i8; scales.len()];
                new_initializers.push(TensorProto {
                    name: w_q_name.clone(),
                    data_type: data_type::INT8,
                    raw_data: q.iter().map(|&v| v as u8).collect(),
                    dims: w_tensor.dims.clone(),
                    ..Default::default()
                });
                new_initializers.push(make_1d_f32(&w_s_name, &scales));
                new_initializers.push(make_1d_i8(&w_zp_name, &zps));
            } else {
                let (q, _scale) = quantize_weights_per_tensor(&w_values);
                new_initializers.push(TensorProto {
                    name: w_q_name.clone(),
                    data_type: data_type::INT8,
                    raw_data: q.iter().map(|&v| v as u8).collect(),
                    dims: w_tensor.dims.clone(),
                    ..Default::default()
                });
                new_initializers.push(make_scalar_f32(&w_s_name, w_scale_value));
                new_initializers.push(make_scalar_i8(&w_zp_name, 0));
            }

            // DequantizeLinear: INT8 weights -> FP32 for Conv
            new_nodes.push(NodeProto {
                op_type: "DequantizeLinear".into(),
                name: format!("{w_name}_DequantizeLinear"),
                input: vec![w_q_name.clone(), w_s_name.clone(), w_zp_name.clone()],
                output: vec![w_dql_out.clone()],
                attribute: if use_pc {
                    vec![proto::AttributeProto {
                        name: "axis".into(),
                        r#type: 2, // INT
                        i: 0,      // per output-channel (dim 0)
                        ..Default::default()
                    }]
                } else {
                    vec![]
                },
                domain: String::new(),
                ..Default::default()
            });
        }

        // ── Bias: quantize to INT32 + DequantizeLinear ──────────────
        // ORT quantizes bias to INT32 with scale = x_scale * w_scale, zp = 0.
        // This ensures accumulation precision in the INT8 Conv kernel.
        let b_dql_out = if let Some(ref b_name_str) = b_name {
            if let Some(&b_idx) = init_map.get(b_name_str) {
                let b_tensor = &graph.initializer[b_idx];
                if let Some(b_values) = OnnxModel::tensor_as_f32(b_tensor) {
                    let bias_scale = x_scale * w_scale_value;
                    let b_q_name = format!("{b_name_str}_quantized");
                    let b_dql_name = format!("{b_name_str}_DequantizeLinear_Output");
                    let b_s_name = format!("{b_name_str}_scale");
                    let b_zp_name = format!("{b_name_str}_zero_point");

                    // Quantize bias to INT32
                    let q_bias: Vec<i32> = b_values
                        .iter()
                        .map(|&v| {
                            let q = (v / bias_scale).round();
                            q.max(i32::MIN as f32).min(i32::MAX as f32) as i32
                        })
                        .collect();

                    new_initializers.push(TensorProto {
                        name: b_q_name.clone(),
                        data_type: data_type::INT32,
                        raw_data: q_bias.iter().flat_map(|v| v.to_le_bytes()).collect(),
                        dims: b_tensor.dims.clone(),
                        ..Default::default()
                    });
                    new_initializers.push(make_scalar_f32(&b_s_name, bias_scale));
                    new_initializers.push(make_scalar_i32(&b_zp_name, 0));

                    // DequantizeLinear for bias
                    new_nodes.push(NodeProto {
                        op_type: "DequantizeLinear".into(),
                        name: format!("{b_name_str}_DequantizeLinear"),
                        input: vec![b_q_name, b_s_name, b_zp_name],
                        output: vec![b_dql_name.clone()],
                        domain: String::new(),
                        ..Default::default()
                    });

                    remove_initializers.insert(b_name_str.clone());
                    Some(b_dql_name)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // ── Input activation QDQ (deduplicated) ─────────────────────
        let x_dql_out = format!("{x_name}_DequantizeLinear_Output");
        let x_s_name = format!("{x_name}_scale");
        let x_zp_name = format!("{x_name}_zero_point");

        let mut conv_pre = Vec::new();
        if emitted_qdq.insert(x_name.clone()) {
            new_initializers.push(make_scalar_f32(&x_s_name, x_scale));
            new_initializers.push(make_scalar_u8(&x_zp_name, x_zp));

            conv_pre.push(NodeProto {
                op_type: "QuantizeLinear".into(),
                name: format!("{x_name}_QuantizeLinear"),
                input: vec![x_name.clone(), x_s_name.clone(), x_zp_name.clone()],
                output: vec![format!("{x_name}_QuantizeLinear_Output")],
                domain: String::new(),
                ..Default::default()
            });
            conv_pre.push(NodeProto {
                op_type: "DequantizeLinear".into(),
                name: format!("{x_name}_DequantizeLinear"),
                input: vec![
                    format!("{x_name}_QuantizeLinear_Output"),
                    x_s_name.clone(),
                    x_zp_name.clone(),
                ],
                output: vec![x_dql_out.clone()],
                domain: String::new(),
                ..Default::default()
            });
        }

        // ── Rewrite Conv inputs to use dequantized versions ─────────
        // We'll modify the Conv node's inputs: x→x_dql, w→w_dql, b→b_dql
        // Store the replacement info; we'll apply after iteration
        let entry = pre_conv_nodes.entry(node_idx).or_default();
        // Store bias DQL output name for later rewrite
        if let Some(b_dql) = b_dql_out {
            entry.push(NodeProto {
                // Sentinel: store bias rewrite name in a dummy node we'll extract later
                op_type: "__bias_rewrite__".into(),
                name: b_dql,
                ..Default::default()
            });
        }
        entry.extend(conv_pre);

        remove_initializers.insert(w_name.clone());
        stats.conv_replaced += 1;
        stats.total_weights += 1;
    }

    // ── MatMul quantization ─────────────────────────────────────────
    // MatMul has two inputs and no bias. If input[1] is a static weight
    // (initializer), quantize it the same way as Conv weights. If both
    // inputs are activations, add QDQ to both sides.
    for (node_idx, node) in graph.node.iter().enumerate() {
        if node.op_type != "MatMul" || !node.domain.is_empty() || node.input.len() < 2 {
            continue;
        }

        let a_name = &node.input[0];
        let b_name = &node.input[1];

        // Skip if on non-vision path
        if non_vision_tensors.contains(a_name) || non_vision_tensors.contains(b_name) {
            continue;
        }
        // Skip if model output
        if node.output.first().is_some_and(|o| model_output_names.contains(o)) {
            continue;
        }
        // Skip layers flagged by sensitivity analysis
        if skip.contains(&node.name) {
            continue;
        }

        let b_is_weight = init_map.contains_key(b_name);

        if b_is_weight {
            // Static weight MatMul: quantize weight + activation QDQ
            let w_idx = init_map[b_name];
            let w_tensor = &graph.initializer[w_idx];
            let w_values = match OnnxModel::tensor_as_f32(w_tensor) {
                Some(v) if v.len() >= 16 => v,
                _ => continue,
            };

            // Get input activation range
            let a_range = match act_ranges.get(a_name) {
                Some(r) => r,
                None => continue,
            };
            let (a_scale, a_zp) = a_range.to_uint8_params();

            // Quantize weight per-tensor (MatMul weights are 2D, per-channel
            // is not standard for MatMul in ORT's QDQ fusion)
            let w_q_name = format!("{b_name}_quantized");
            let w_dql_out = format!("{b_name}_DequantizeLinear_Output");
            let w_s_name = format!("{b_name}_scale");
            let w_zp_name = format!("{b_name}_zero_point");

            if !emitted_qdq.contains(b_name) {
                emitted_qdq.insert(b_name.to_string());
                let (q, scale) = quantize_weights_per_tensor(&w_values);
                new_initializers.push(TensorProto {
                    name: w_q_name.clone(),
                    data_type: data_type::INT8,
                    raw_data: q.iter().map(|&v| v as u8).collect(),
                    dims: w_tensor.dims.clone(),
                    ..Default::default()
                });
                new_initializers.push(make_scalar_f32(&w_s_name, scale));
                new_initializers.push(make_scalar_i8(&w_zp_name, 0));

                new_nodes.push(NodeProto {
                    op_type: "DequantizeLinear".into(),
                    name: format!("{b_name}_DequantizeLinear"),
                    input: vec![w_q_name, w_s_name, w_zp_name],
                    output: vec![w_dql_out.clone()],
                    domain: String::new(),
                    ..Default::default()
                });
            }

            // Input activation QDQ
            let a_dql_out = format!("{a_name}_DequantizeLinear_Output");
            let a_s_name = format!("{a_name}_scale");
            let a_zp_name = format!("{a_name}_zero_point");

            let mut matmul_pre = Vec::new();
            if emitted_qdq.insert(a_name.clone()) {
                new_initializers.push(make_scalar_f32(&a_s_name, a_scale));
                new_initializers.push(make_scalar_u8(&a_zp_name, a_zp));

                matmul_pre.push(NodeProto {
                    op_type: "QuantizeLinear".into(),
                    name: format!("{a_name}_QuantizeLinear"),
                    input: vec![a_name.clone(), a_s_name.clone(), a_zp_name.clone()],
                    output: vec![format!("{a_name}_QuantizeLinear_Output")],
                    domain: String::new(),
                    ..Default::default()
                });
                matmul_pre.push(NodeProto {
                    op_type: "DequantizeLinear".into(),
                    name: format!("{a_name}_DequantizeLinear"),
                    input: vec![
                        format!("{a_name}_QuantizeLinear_Output"),
                        a_s_name,
                        a_zp_name,
                    ],
                    output: vec![a_dql_out.clone()],
                    domain: String::new(),
                    ..Default::default()
                });
            }

            let entry = pre_conv_nodes.entry(node_idx).or_default();
            entry.extend(matmul_pre);

            remove_initializers.insert(b_name.clone());
            stats.matmul_replaced += 1;
            stats.total_weights += 1;
        } else {
            // Dynamic MatMul: both inputs are activations.
            // Add QDQ to both sides if we have calibration data.
            let a_range = act_ranges.get(a_name);
            let b_range = act_ranges.get(b_name);

            if a_range.is_none() && b_range.is_none() {
                continue;
            }

            let mut matmul_pre = Vec::new();

            // QDQ for input A
            if let Some(range) = a_range {
                if emitted_qdq.insert(a_name.clone()) {
                    let (scale, zp) = range.to_uint8_params();
                    let s_name = format!("{a_name}_scale");
                    let zp_name = format!("{a_name}_zero_point");
                    new_initializers.push(make_scalar_f32(&s_name, scale));
                    new_initializers.push(make_scalar_u8(&zp_name, zp));

                    matmul_pre.push(NodeProto {
                        op_type: "QuantizeLinear".into(),
                        name: format!("{a_name}_QuantizeLinear"),
                        input: vec![a_name.clone(), s_name.clone(), zp_name.clone()],
                        output: vec![format!("{a_name}_QuantizeLinear_Output")],
                        domain: String::new(),
                        ..Default::default()
                    });
                    matmul_pre.push(NodeProto {
                        op_type: "DequantizeLinear".into(),
                        name: format!("{a_name}_DequantizeLinear"),
                        input: vec![format!("{a_name}_QuantizeLinear_Output"), s_name, zp_name],
                        output: vec![format!("{a_name}_DequantizeLinear_Output")],
                        domain: String::new(),
                        ..Default::default()
                    });
                }
            }

            // QDQ for input B
            if let Some(range) = b_range {
                if emitted_qdq.insert(b_name.clone()) {
                    let (scale, zp) = range.to_uint8_params();
                    let s_name = format!("{b_name}_scale");
                    let zp_name = format!("{b_name}_zero_point");
                    new_initializers.push(make_scalar_f32(&s_name, scale));
                    new_initializers.push(make_scalar_u8(&zp_name, zp));

                    matmul_pre.push(NodeProto {
                        op_type: "QuantizeLinear".into(),
                        name: format!("{b_name}_QuantizeLinear"),
                        input: vec![b_name.clone(), s_name.clone(), zp_name.clone()],
                        output: vec![format!("{b_name}_QuantizeLinear_Output")],
                        domain: String::new(),
                        ..Default::default()
                    });
                    matmul_pre.push(NodeProto {
                        op_type: "DequantizeLinear".into(),
                        name: format!("{b_name}_DequantizeLinear"),
                        input: vec![format!("{b_name}_QuantizeLinear_Output"), s_name, zp_name],
                        output: vec![format!("{b_name}_DequantizeLinear_Output")],
                        domain: String::new(),
                        ..Default::default()
                    });
                }
            }

            if !matmul_pre.is_empty() {
                let entry = pre_conv_nodes.entry(node_idx).or_default();
                entry.extend(matmul_pre);
                stats.matmul_replaced += 1;
            }
        }
    }

    // Remove original FP32 weight initializers and their graph.input entries
    graph
        .initializer
        .retain(|t| !remove_initializers.contains(&t.name));
    graph
        .input
        .retain(|vi| !remove_initializers.contains(&vi.name));

    // ── Apply Conv rewrites: input QDQ + output QDQ ─────────────────

    // Rebuild node list
    let mut final_nodes: Vec<NodeProto> = Vec::new();

    // First emit weight DequantizeLinear nodes (they only depend on initializers)
    final_nodes.extend(new_nodes);

    // Then walk original nodes, inserting activation QDQ around each Conv/MatMul
    for (idx, mut node) in graph.node.drain(..).enumerate() {
        if let Some(pre) = pre_conv_nodes.remove(&idx) {
            // Extract bias rewrite sentinel if present (Conv only)
            let bias_dql_name = pre
                .iter()
                .find(|n| n.op_type == "__bias_rewrite__")
                .map(|n| n.name.clone());
            // Emit activation QDQ nodes (skip sentinels)
            final_nodes.extend(pre.into_iter().filter(|n| n.op_type != "__bias_rewrite__"));

            if node.op_type == "Conv" {
                // Rewrite Conv inputs to use dequantized versions
                let x_name = &node.input[0];
                let w_name = &node.input[1];
                let x_dql = format!("{x_name}_DequantizeLinear_Output");
                let w_dql = format!("{w_name}_DequantizeLinear_Output");
                node.input[0] = x_dql;
                node.input[1] = w_dql;

                // Rewrite bias input if quantized
                if let Some(b_dql) = bias_dql_name {
                    if node.input.len() >= 3 {
                        node.input[2] = b_dql;
                    }
                }

                // ── Conv output QDQ (activation-aware placement) ────────
                // If an activation function immediately follows this Conv, skip
                // the output QDQ here — the activation pass will add QDQ after
                // the activation instead. This matches ORT's fused Conv+Activation
                // kernel selection (Relu, Clip/ReLU6, LeakyRelu, Sigmoid, HardSwish).
                let original_output = node.output[0].clone();
                let has_activation_consumer = conv_activation_set.contains(&original_output);

                if has_activation_consumer {
                    // Skip output QDQ — activation pass will handle it
                    final_nodes.push(node);
                } else if model_output_names.contains(&original_output)
                    || non_vision_tensors.contains(&original_output)
                {
                    // Skip QDQ — model output or non-vision path (scale_factor etc.)
                    final_nodes.push(node);
                } else if emitted_qdq.contains(&original_output) {
                    // Already have QDQ for this tensor (shared activation)
                    final_nodes.push(node);
                } else {
                    emitted_qdq.insert(original_output.clone());
                    conv_output_qdq.insert(original_output.clone());
                    // No activation follows — add output QDQ here
                    let conv_raw_out = format!("{original_output}_pre_qdq");
                    node.output[0] = conv_raw_out.clone();
                    final_nodes.push(node);

                    if let Some(y_range) = act_ranges.get(&original_output) {
                        let (y_scale, y_zp) = y_range.to_uint8_params();
                        let y_s_name = format!("{original_output}_scale");
                        let y_zp_name = format!("{original_output}_zero_point");
                        new_initializers.push(make_scalar_f32(&y_s_name, y_scale));
                        new_initializers.push(make_scalar_u8(&y_zp_name, y_zp));

                        final_nodes.push(NodeProto {
                            op_type: "QuantizeLinear".into(),
                            name: format!("{original_output}_QuantizeLinear"),
                            input: vec![conv_raw_out, y_s_name.clone(), y_zp_name.clone()],
                            output: vec![format!("{original_output}_QuantizeLinear_Output")],
                            domain: String::new(),
                            ..Default::default()
                        });
                        final_nodes.push(NodeProto {
                            op_type: "DequantizeLinear".into(),
                            name: format!("{original_output}_DequantizeLinear"),
                            input: vec![
                                format!("{original_output}_QuantizeLinear_Output"),
                                y_s_name,
                                y_zp_name,
                            ],
                            output: vec![original_output],
                            domain: String::new(),
                            ..Default::default()
                        });
                    } else {
                        final_nodes.last_mut().unwrap().output[0] = original_output;
                    }
                }
            } else if node.op_type == "MatMul" {
                // Rewrite MatMul inputs to use dequantized versions
                let a_name = node.input[0].clone();
                let b_name = node.input[1].clone();
                let a_dql = format!("{a_name}_DequantizeLinear_Output");
                let b_dql = format!("{b_name}_DequantizeLinear_Output");
                // Only rewrite if the dequantized version was emitted
                if emitted_qdq.contains(&a_name) {
                    node.input[0] = a_dql;
                }
                if emitted_qdq.contains(&b_name) {
                    node.input[1] = b_dql;
                }

                // MatMul output QDQ (no activation-aware skipping needed)
                let original_output = node.output[0].clone();
                if !model_output_names.contains(&original_output)
                    && !non_vision_tensors.contains(&original_output)
                    && !conv_output_qdq.contains(&original_output)
                {
                    if let Some(y_range) = act_ranges.get(&original_output) {
                        if emitted_qdq.insert(original_output.clone()) {
                            conv_output_qdq.insert(original_output.clone());
                            let (y_scale, y_zp) = y_range.to_uint8_params();
                            let y_s_name = format!("{original_output}_scale");
                            let y_zp_name = format!("{original_output}_zero_point");
                            new_initializers.push(make_scalar_f32(&y_s_name, y_scale));
                            new_initializers.push(make_scalar_u8(&y_zp_name, y_zp));

                            let raw_out = format!("{original_output}_pre_qdq");
                            node.output[0] = raw_out.clone();
                            final_nodes.push(node);

                            final_nodes.push(NodeProto {
                                op_type: "QuantizeLinear".into(),
                                name: format!("{original_output}_QuantizeLinear"),
                                input: vec![raw_out, y_s_name.clone(), y_zp_name.clone()],
                                output: vec![format!("{original_output}_QuantizeLinear_Output")],
                                domain: String::new(),
                                ..Default::default()
                            });
                            final_nodes.push(NodeProto {
                                op_type: "DequantizeLinear".into(),
                                name: format!("{original_output}_DequantizeLinear"),
                                input: vec![
                                    format!("{original_output}_QuantizeLinear_Output"),
                                    y_s_name,
                                    y_zp_name,
                                ],
                                output: vec![original_output],
                                domain: String::new(),
                                ..Default::default()
                            });
                            continue;
                        }
                    }
                }
                final_nodes.push(node);
            } else {
                final_nodes.push(node);
            }
        } else {
            final_nodes.push(node);
        }
    }
    graph.node = final_nodes;
    graph.initializer.extend(new_initializers);

    // ── Second pass: wrap non-Conv ops with output QDQ ────────────────
    // Activation outputs get QDQ here (after the activation), matching ORT's
    // fused Conv+Activation QDQ placement. The Conv output QDQ was skipped
    // for Conv→Activation pairs, so this is the only quantization point.
    //
    // IMPORTANT: We use a separate tracking set (output_qdq_done) instead
    // of reusing emitted_qdq. The emitted_qdq set tracks tensors with
    // *input-side* QDQ (QuantizeLinear before Conv), but the second pass
    // needs to check for *output-side* QDQ only. When an activation output
    // feeds into a downstream Conv, the Conv input QDQ marks that tensor in
    // emitted_qdq — but the activation output ALSO needs its own output QDQ
    // to enable Conv+Activation fusion.
    let graph = model.graph_mut();
    let qdq_ops = [
        "Relu",
        "LeakyRelu",  // YOLOv3/v4, Darknet architectures
        "Add",        // Residual connections (ResNet, EfficientNet) → QLinearAdd
        "Concat",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
        "Mul",        // SE-block channel scaling (MobileNetV3, EfficientNet)
        "Sigmoid",    // SE-block attention (MobileNetV3, EfficientNet)
        "Clip",       // ReLU6 in MobileNetV2/V3 (Clip(0,6))
        "HardSwish",  // MobileNetV3
    ];
    let mut extra_inits: Vec<TensorProto> = Vec::new();
    let mut wrapped_nodes: Vec<NodeProto> = Vec::new();

    for node in graph.node.drain(..) {
        if qdq_ops.contains(&node.op_type.as_str()) && !node.output.is_empty() {
            let out_name = node.output[0].clone();

            // Skip if this tensor already has Conv output QDQ from the first pass.
            // Uses conv_output_qdq (not emitted_qdq) to avoid false positives
            // from Conv *input* QDQ which shares the same tensor name.
            if conv_output_qdq.contains(&out_name) {
                wrapped_nodes.push(node);
                continue;
            }

            // Skip if this is a model output or on the non-vision path
            if model_output_names.contains(&out_name) || non_vision_tensors.contains(&out_name) {
                wrapped_nodes.push(node);
                continue;
            }

            if let Some(range) = act_ranges.get(&out_name) {
                emitted_qdq.insert(out_name.clone());
                let (scale, zp) = range.to_uint8_params();
                let s_name = format!("{out_name}_qdq_scale");
                let zp_name = format!("{out_name}_qdq_zero_point");

                extra_inits.push(make_scalar_f32(&s_name, scale));
                extra_inits.push(make_scalar_u8(&zp_name, zp));

                let raw_out = format!("{out_name}_raw");
                let mut modified = node;
                modified.output[0] = raw_out.clone();
                wrapped_nodes.push(modified);

                wrapped_nodes.push(NodeProto {
                    op_type: "QuantizeLinear".into(),
                    name: format!("{out_name}_post_QuantizeLinear"),
                    input: vec![raw_out, s_name.clone(), zp_name.clone()],
                    output: vec![format!("{out_name}_post_QL")],
                    domain: String::new(),
                    ..Default::default()
                });
                wrapped_nodes.push(NodeProto {
                    op_type: "DequantizeLinear".into(),
                    name: format!("{out_name}_post_DequantizeLinear"),
                    input: vec![format!("{out_name}_post_QL"), s_name, zp_name],
                    output: vec![out_name.clone()],
                    domain: String::new(),
                    ..Default::default()
                });
                continue;
            }
        }
        wrapped_nodes.push(node);
    }
    graph.node = wrapped_nodes;
    graph.initializer.extend(extra_inits);

    // Ensure opset >= 13 (per-channel DequantizeLinear axis attr needs opset 13)
    for opset in &mut model.proto.opset_import {
        if opset.domain.is_empty() && opset.version < 13 {
            opset.version = 13;
        }
    }

    // Upgrade Dropout nodes from opset-7 (ratio as attribute) to opset-13 (ratio as input)
    let graph = model.graph_mut();
    for node in &mut graph.node {
        if node.op_type == "Dropout" {
            // Extract ratio from attribute, default to 0.5
            let ratio = node
                .attribute
                .iter()
                .find(|a| a.name == "ratio")
                .map(|a| a.f)
                .unwrap_or(0.5);
            node.attribute.retain(|a| a.name != "ratio");

            // Create ratio initializer name
            let ratio_name = format!("{}_ratio", node.name);
            graph.initializer.push(TensorProto {
                name: ratio_name.clone(),
                data_type: data_type::FLOAT,
                float_data: vec![ratio],
                dims: vec![],
                ..Default::default()
            });

            // Add ratio as second input
            if node.input.len() < 2 {
                node.input.push(ratio_name);
            }
        }
        // BatchNormalization: remove deprecated 'spatial' attribute (removed in opset 9)
        if node.op_type == "BatchNormalization" {
            node.attribute.retain(|a| a.name != "spatial");
        }
        // Squeeze/Unsqueeze: 'axes' attribute → input tensor (changed in opset 13)
        if (node.op_type == "Squeeze" || node.op_type == "Unsqueeze")
            && node.attribute.iter().any(|a| a.name == "axes")
        {
            let axes: Vec<i64> = node
                .attribute
                .iter()
                .find(|a| a.name == "axes")
                .map(|a| a.ints.clone())
                .unwrap_or_default();
            node.attribute.retain(|a| a.name != "axes");

            let axes_name = format!("{}_axes", node.name);
            graph.initializer.push(TensorProto {
                name: axes_name.clone(),
                data_type: data_type::INT64,
                raw_data: axes.iter().flat_map(|v| v.to_le_bytes()).collect(),
                dims: vec![axes.len() as i64],
                ..Default::default()
            });
            node.input.push(axes_name);
        }
        // Split: 'split' attribute → input tensor (changed in opset 13)
        // Also 'axis' stays as attribute, only the split-sizes move to input.
        if node.op_type == "Split" && node.attribute.iter().any(|a| a.name == "split") {
            let split_sizes: Vec<i64> = node
                .attribute
                .iter()
                .find(|a| a.name == "split")
                .map(|a| a.ints.clone())
                .unwrap_or_default();
            node.attribute.retain(|a| a.name != "split");

            let split_name = format!("{}_split_sizes", node.name);
            graph.initializer.push(TensorProto {
                name: split_name.clone(),
                data_type: data_type::INT64,
                raw_data: split_sizes.iter().flat_map(|v| v.to_le_bytes()).collect(),
                dims: vec![split_sizes.len() as i64],
                ..Default::default()
            });
            // split sizes is the second input (after data)
            while node.input.len() < 2 {
                node.input.push(String::new());
            }
            if node.input.len() == 2 && node.input[1].is_empty() {
                node.input[1] = split_name;
            } else {
                node.input.push(split_name);
            }
        }
    }

    model.proto.producer_name = "kenosis".into();
    model
        .proto
        .metadata_props
        .push(proto::StringStringEntryProto {
            key: "kenosis.quantization".into(),
            value: "static-int8-qdq".into(),
        });

    tracing::info!(
        conv = stats.conv_replaced,
        matmul = stats.matmul_replaced,
        calibrated = stats.activation_tensors_calibrated,
        "Static INT8 QDQ quantization complete"
    );
    Ok((model, stats))
}
