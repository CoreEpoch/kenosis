// Copyright (c) 2026 Core Epoch. Licensed under Apache-2.0.
//! `kenosis validate` — compare quantized model accuracy and speed vs FP32 baseline.
#![allow(clippy::format_in_format_args)]

use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use owo_colors::OwoColorize;

#[derive(Args)]
pub struct ValidateArgs {
    /// Path to the FP32 baseline ONNX model.
    baseline: PathBuf,

    /// Path to the quantized ONNX model to validate.
    quantized: PathBuf,

    /// Number of random test inputs (default: 50).
    #[arg(short = 'n', long, default_value = "50")]
    samples: usize,

    /// Number of warmup runs for latency measurement (default: 20).
    #[arg(long, default_value = "20")]
    warmup: usize,

    /// Number of timed runs for latency measurement (default: 200).
    #[arg(long, default_value = "200")]
    timed: usize,
}

pub fn run(args: ValidateArgs) -> kenosis_core::Result<()> {
    use ort::session::{builder::GraphOptimizationLevel, Session};
    use ort::value::Tensor;

    println!("\n  {} Loading models...", "▸".cyan(),);

    // Load both models
    let mut baseline = Session::builder()
        .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("ort: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("ort: {e}")))?
        .with_intra_threads(1)
        .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("ort: {e}")))?
        .commit_from_file(&args.baseline)
        .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("ort: {e}")))?;

    let mut quantized = Session::builder()
        .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("ort: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("ort: {e}")))?
        .with_intra_threads(1)
        .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("ort: {e}")))?
        .commit_from_file(&args.quantized)
        .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("ort: {e}")))?;

    // Get input metadata
    let input_name = baseline.inputs()[0].name().to_string();
    let shape: Vec<usize> = baseline.inputs()[0]
        .dtype()
        .tensor_shape()
        .ok_or_else(|| kenosis_core::KenosisError::InvalidModel("no tensor shape".into()))?
        .iter()
        .map(|&d| if d < 0 { 1usize } else { d as usize })
        .collect();

    // Collect extra inputs for multi-input models (e.g. PP-YOLOE+ scale_factor)
    let extra_inputs: Vec<(String, Vec<usize>, f32)> = baseline
        .inputs()
        .iter()
        .skip(1)
        .filter_map(|info| {
            let name = info.name().to_string();
            let sh = info.dtype().tensor_shape()?;
            let shape_vec: Vec<usize> = sh
                .iter()
                .map(|&d| if d < 0 { 1 } else { d as usize })
                .collect();
            let fill = if name.contains("scale") {
                1.0f32
            } else {
                0.0f32
            };
            Some((name, shape_vec, fill))
        })
        .collect();
    let has_extra = !extra_inputs.is_empty();

    let is_nchw = shape.len() == 4 && shape[1] == 3;

    println!(
        "  {} Input: {} {:?}{}",
        "▸".cyan(),
        input_name.green(),
        shape,
        if is_nchw { " (NCHW vision)" } else { "" },
    );

    // ── Accuracy test ──────────────────────────────────────────────
    println!(
        "  {} Running {} accuracy samples...",
        "▸".cyan(),
        args.samples,
    );

    let mut cosines: Vec<f64> = Vec::with_capacity(args.samples);
    let mut top1_matches: usize = 0;

    // ImageNet normalization constants
    const MEANS: [f32; 3] = [0.485, 0.456, 0.406];
    const STDS: [f32; 3] = [0.229, 0.224, 0.225];

    for i in 0..args.samples {
        let data = generate_test_input(&shape, i, is_nchw, &MEANS, &STDS);

        let input_b = Tensor::from_array((shape.clone(), data.clone()))
            .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("tensor: {e}")))?;
        let input_q = Tensor::from_array((shape.clone(), data))
            .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("tensor: {e}")))?;

        let out_b = if has_extra {
            let mut m = std::collections::HashMap::new();
            m.insert(input_name.clone(), ort::value::DynValue::from(input_b));
            for (n, s, f) in &extra_inputs {
                let t =
                    Tensor::from_array((s.clone(), vec![*f; s.iter().product()])).map_err(|e| {
                        kenosis_core::KenosisError::InvalidModel(format!("tensor: {e}"))
                    })?;
                m.insert(n.clone(), ort::value::DynValue::from(t));
            }
            baseline
                .run(m)
                .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("run: {e}")))?
        } else {
            baseline
                .run(ort::inputs![input_name.as_str() => input_b])
                .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("run: {e}")))?
        };
        let out_q = if has_extra {
            let mut m = std::collections::HashMap::new();
            m.insert(input_name.clone(), ort::value::DynValue::from(input_q));
            for (n, s, f) in &extra_inputs {
                let t =
                    Tensor::from_array((s.clone(), vec![*f; s.iter().product()])).map_err(|e| {
                        kenosis_core::KenosisError::InvalidModel(format!("tensor: {e}"))
                    })?;
                m.insert(n.clone(), ort::value::DynValue::from(t));
            }
            quantized
                .run(m)
                .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("run: {e}")))?
        } else {
            quantized
                .run(ort::inputs![input_name.as_str() => input_q])
                .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("run: {e}")))?
        };

        // Extract first output as f32
        let a = extract_f32_output(&out_b);
        let b = extract_f32_output(&out_q);

        if let (Some(a), Some(b)) = (a, b) {
            let cos = cosine_similarity(&a, &b);
            cosines.push(cos);

            if argmax(&a) == argmax(&b) {
                top1_matches += 1;
            }
        }
    }

    // ── Latency test ───────────────────────────────────────────────
    println!(
        "  {} Benchmarking latency ({} warmup, {} timed)...",
        "▸".cyan(),
        args.warmup,
        args.timed,
    );

    let bench_data = generate_test_input(&shape, 9999, is_nchw, &MEANS, &STDS);

    // Warmup + bench quantized
    for _ in 0..args.warmup {
        let t = Tensor::from_array((shape.clone(), bench_data.clone()))
            .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("tensor: {e}")))?;
        let _ = if has_extra {
            let mut m = std::collections::HashMap::new();
            m.insert(input_name.clone(), ort::value::DynValue::from(t));
            for (n, s, f) in &extra_inputs {
                let et = Tensor::from_array((s.clone(), vec![*f; s.iter().product()])).unwrap();
                m.insert(n.clone(), ort::value::DynValue::from(et));
            }
            quantized.run(m)
        } else {
            quantized.run(ort::inputs![input_name.as_str() => t])
        };
    }
    let t0 = Instant::now();
    for _ in 0..args.timed {
        let t = Tensor::from_array((shape.clone(), bench_data.clone()))
            .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("tensor: {e}")))?;
        let _ = if has_extra {
            let mut m = std::collections::HashMap::new();
            m.insert(input_name.clone(), ort::value::DynValue::from(t));
            for (n, s, f) in &extra_inputs {
                let et = Tensor::from_array((s.clone(), vec![*f; s.iter().product()])).unwrap();
                m.insert(n.clone(), ort::value::DynValue::from(et));
            }
            quantized.run(m)
        } else {
            quantized.run(ort::inputs![input_name.as_str() => t])
        };
    }
    let ms_q = t0.elapsed().as_secs_f64() / args.timed as f64 * 1000.0;

    // Warmup + bench baseline
    for _ in 0..args.warmup {
        let t = Tensor::from_array((shape.clone(), bench_data.clone()))
            .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("tensor: {e}")))?;
        let _ = if has_extra {
            let mut m = std::collections::HashMap::new();
            m.insert(input_name.clone(), ort::value::DynValue::from(t));
            for (n, s, f) in &extra_inputs {
                let et = Tensor::from_array((s.clone(), vec![*f; s.iter().product()])).unwrap();
                m.insert(n.clone(), ort::value::DynValue::from(et));
            }
            baseline.run(m)
        } else {
            baseline.run(ort::inputs![input_name.as_str() => t])
        };
    }
    let t0 = Instant::now();
    for _ in 0..args.timed {
        let t = Tensor::from_array((shape.clone(), bench_data.clone()))
            .map_err(|e| kenosis_core::KenosisError::InvalidModel(format!("tensor: {e}")))?;
        let _ = if has_extra {
            let mut m = std::collections::HashMap::new();
            m.insert(input_name.clone(), ort::value::DynValue::from(t));
            for (n, s, f) in &extra_inputs {
                let et = Tensor::from_array((s.clone(), vec![*f; s.iter().product()])).unwrap();
                m.insert(n.clone(), ort::value::DynValue::from(et));
            }
            baseline.run(m)
        } else {
            baseline.run(ort::inputs![input_name.as_str() => t])
        };
    }
    let ms_b = t0.elapsed().as_secs_f64() / args.timed as f64 * 1000.0;

    // ── File sizes ─────────────────────────────────────────────────
    let size_b = std::fs::metadata(&args.baseline)
        .map(|m| m.len())
        .unwrap_or(0);
    let size_q = std::fs::metadata(&args.quantized)
        .map(|m| m.len())
        .unwrap_or(0);

    // ── Report ─────────────────────────────────────────────────────
    let mean_cos = if cosines.is_empty() {
        0.0
    } else {
        cosines.iter().sum::<f64>() / cosines.len() as f64
    };
    let min_cos = cosines.iter().copied().fold(f64::MAX, f64::min);

    println!("\n  {}", "═".repeat(56).dimmed());
    println!("  {}  Kenosis Validation Report", "📊".green());
    println!("  {}", "═".repeat(56).dimmed());
    println!(
        "  {} Cosine similarity:  {} (min {:.4})",
        "▸".cyan(),
        format!("{mean_cos:.6}").green(),
        min_cos,
    );
    println!(
        "  {} Top-1 agreement:    {}/{} ({})",
        "▸".cyan(),
        top1_matches.to_string().green(),
        args.samples,
        format!("{:.0}%", top1_matches as f64 / args.samples as f64 * 100.0).green(),
    );
    println!(
        "  {} Latency:            {} vs {} ({})",
        "▸".cyan(),
        format!("{ms_q:.2}ms").green(),
        format!("{ms_b:.2}ms"),
        format!("{:.2}× speedup", ms_b / ms_q).yellow(),
    );
    println!(
        "  {} Size:               {} vs {} ({})",
        "▸".cyan(),
        format_size(size_q).green(),
        format_size(size_b),
        format!("{:.1}× smaller", size_b as f64 / size_q as f64).yellow(),
    );

    // Quality verdict
    let verdict = if mean_cos >= 0.995 {
        "EXCELLENT — production ready".green().to_string()
    } else if mean_cos >= 0.98 {
        "GOOD — minor accuracy loss".yellow().to_string()
    } else if mean_cos >= 0.95 {
        "ACCEPTABLE — noticeable degradation".yellow().to_string()
    } else {
        "POOR — significant accuracy loss".red().to_string()
    };
    println!("  {} Verdict:             {}", "▸".cyan(), verdict,);
    println!("  {}\n", "═".repeat(56).dimmed());

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Xorshift64 PRNG for deterministic test inputs.
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

fn generate_test_input(
    shape: &[usize],
    sample_idx: usize,
    is_nchw: bool,
    means: &[f32; 3],
    stds: &[f32; 3],
) -> Vec<f32> {
    let total: usize = shape.iter().product();
    let mut rng = Xorshift64::new((sample_idx as u64 + 42) * 0xCAFE_BABE);

    if is_nchw {
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let mut data = Vec::with_capacity(total);
        for _batch in 0..n {
            for ch in 0..c {
                let mean = means[ch % 3];
                let std = stds[ch % 3];
                for _ in 0..(h * w) {
                    let pixel = rng.next_f32(); // [0, 1) — simulates image pixel
                    data.push((pixel - mean) / std);
                }
            }
        }
        data
    } else {
        (0..total).map(|_| rng.next_normal()).collect()
    }
}

fn extract_f32_output(outputs: &ort::session::SessionOutputs<'_>) -> Option<Vec<f32>> {
    // Get first output key
    let first_key = outputs.keys().next()?;
    let value = outputs.get(first_key)?;
    let (_shape, data) = value.try_extract_tensor::<f32>().ok()?;
    Some(data.to_vec())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum();
    let norm_a: f64 = a
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    let norm_b: f64 = b
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    dot / (norm_a * norm_b + 1e-10)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}
