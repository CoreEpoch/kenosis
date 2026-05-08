// Copyright (c) 2026 Core Epoch. Licensed under Apache-2.0.
//! `kenosis quantize` — static INT8 QDQ quantization.

use std::path::PathBuf;

use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use owo_colors::OwoColorize;

use kenosis_core::{inspect, OnnxModel};

#[derive(Args)]
pub struct QuantizeArgs {
    /// Path to the input ONNX model.
    model: PathBuf,

    /// Output path for the quantized model.
    #[arg(short, long)]
    output: PathBuf,

    /// Enable static INT8 QDQ quantization with calibration.
    #[arg(long)]
    static_int8: bool,

    /// Use per-channel quantization for Conv weights (one scale per output channel).
    /// Improves accuracy on high-channel models like ResNet50.
    #[arg(long)]
    per_channel: bool,

    /// Number of calibration samples (default: 20).
    #[arg(long, default_value = "20")]
    n_calib: usize,

    /// Directory containing raw f32 binary calibration files.
    /// Each file must be named *.bin and contain product(input_shape)*4 bytes.
    #[arg(long)]
    calib_dir: Option<PathBuf>,

    /// Extract Constant nodes to initializers before quantizing.
    /// Required for models that embed weights as Constant ops (e.g. PaddlePaddle).
    #[arg(long)]
    extract_constants: bool,
}

pub fn run(args: QuantizeArgs) -> kenosis_core::Result<()> {
    if !args.static_int8 {
        return Err(kenosis_core::KenosisError::UnsupportedOp(
            "please specify --static-int8 for quantization".to_string(),
        ));
    }

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );

    pb.set_message("Loading model...");
    let mut model = OnnxModel::load(&args.model)?;
    let original_size = model.byte_size() as u64;

    // Extract Constant nodes if requested
    if args.extract_constants {
        let extracted = model.extract_constants();
        if extracted > 0 {
            println!(
                "  {} Extracted {} Constant nodes to initializers",
                "▸".cyan(),
                extracted.to_string().green(),
            );
        }
    }

    pb.set_message(format!(
        "Static INT8 QDQ: calibrating activations ({} samples)...",
        args.n_calib
    ));

    let (model, stats) = kenosis_core::static_int8::quantize_static_int8(
        model,
        args.per_channel,
        args.n_calib,
        args.calib_dir.as_deref(),
    )?;

    pb.set_message("Saving...");
    model.save(&args.output)?;
    let new_size = model.byte_size() as u64;
    pb.finish_and_clear();

    let label = format!(
        "Static INT8 QDQ ({})",
        if args.per_channel {
            "per-channel"
        } else {
            "per-tensor"
        }
    );

    let ratio = original_size as f64 / new_size.max(1) as f64;
    println!(
        "\n  ✨ {} → {} ({:.1}× smaller)  [{}]",
        inspect::format_bytes(original_size),
        inspect::format_bytes(new_size).green(),
        ratio,
        label.bold(),
    );
    println!("  {} Saved to {}", "→".dimmed(), args.output.display());
    println!(
        "  {} Wrapped {} Conv nodes with QDQ ({} activation tensors calibrated)",
        ">".cyan(),
        stats.conv_replaced.to_string().green(),
        stats.activation_tensors_calibrated,
    );
    println!(
        "  {} Pure Rust — no Python dependency. Runs on stock ORT.\n",
        ">".cyan(),
    );

    Ok(())
}
