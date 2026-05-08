// Copyright (c) 2026 Core Epoch. Licensed under Apache-2.0.
//! `kenosis cast` — precision casting.

use std::path::PathBuf;

use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use owo_colors::OwoColorize;

use kenosis_core::{cast, inspect, OnnxModel, Precision};

#[derive(Args)]
pub struct CastArgs {
    /// Path to the input ONNX model.
    model: PathBuf,

    /// Output path for the casted model.
    #[arg(short, long)]
    output: PathBuf,

    /// Target precision: fp16 or bf16.
    #[arg(short, long, default_value = "fp16")]
    precision: String,
}

pub fn run(args: CastArgs) -> kenosis_core::Result<()> {
    let target = match args.precision.to_lowercase().as_str() {
        "fp16" | "float16" => Precision::Float16,
        "bf16" | "bfloat16" => Precision::BFloat16,
        other => {
            return Err(kenosis_core::KenosisError::UnsupportedOp(format!(
                "unknown precision '{other}', expected fp16 or bf16"
            )));
        }
    };

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );

    pb.set_message("Loading model...");
    let model = OnnxModel::load(&args.model)?;
    let original_size = model.byte_size() as u64;

    pb.set_message(format!("Casting to {target}..."));
    let model = cast::cast_precision(model, target)?;

    pb.set_message("Saving...");
    model.save(&args.output)?;
    let new_size = model.byte_size() as u64;

    pb.finish_and_clear();

    let ratio = original_size as f64 / new_size as f64;
    println!(
        "\n  {} {} → {} ({:.1}× smaller)",
        "✨".green(),
        inspect::format_bytes(original_size),
        inspect::format_bytes(new_size).green(),
        ratio,
    );
    println!("  {} Saved to {}\n", "→".dimmed(), args.output.display());
    Ok(())
}
