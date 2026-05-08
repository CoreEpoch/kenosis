// Copyright (c) 2026 Core Epoch. Licensed under Apache-2.0.
//! Kenosis CLI — ONNX model optimization toolkit.

use clap::Parser;
use tracing_subscriber::EnvFilter;

mod commands;

/// Kenosis — fast, calibration-free ONNX model optimization.
///
/// Inspect, cast, quantize, and compare ONNX models from the command line.
#[derive(Parser)]
#[command(name = "kenosis", version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: commands::Command,
}

fn main() {
    // Initialize tracing (controlled via RUST_LOG env var)
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let cli = Cli::parse();

    if let Err(e) = commands::run(cli.command) {
        eprintln!("\x1b[31merror\x1b[0m: {e}");
        std::process::exit(1);
    }
}
