// Copyright (c) 2026 Core Epoch. Licensed under Apache-2.0.
//! CLI subcommands.

pub mod cast;
pub mod diff;
pub mod inspect;
pub mod quantize;
pub mod validate;

use clap::Subcommand;

#[derive(Subcommand)]
pub enum Command {
    /// Inspect an ONNX model: show stats, op distribution, and data types.
    Inspect(inspect::InspectArgs),
    /// Cast model weights to a different precision (FP32 → FP16/BF16).
    Cast(cast::CastArgs),
    /// Quantize an ONNX model (static INT8 QDQ).
    Quantize(quantize::QuantizeArgs),
    /// Compare two ONNX models side-by-side.
    Diff(diff::DiffArgs),
    /// Validate a quantized model vs FP32 baseline: accuracy, latency, and compression.
    Validate(validate::ValidateArgs),
}

pub fn run(cmd: Command) -> kenosis_core::Result<()> {
    match cmd {
        Command::Inspect(args) => inspect::run(args),
        Command::Cast(args) => cast::run(args),
        Command::Quantize(args) => quantize::run(args),
        Command::Diff(args) => diff::run(args),
        Command::Validate(args) => validate::run(args),
    }
}
