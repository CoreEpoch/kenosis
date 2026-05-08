// Copyright (c) 2026 Core Epoch. Licensed under Apache-2.0.
//! `kenosis inspect` — model analysis and profiling.

use std::path::PathBuf;

use clap::Args;
use comfy_table::{modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL, ContentArrangement, Table};
use owo_colors::OwoColorize;

use kenosis_core::{inspect, OnnxModel};

#[derive(Args)]
pub struct InspectArgs {
    /// Path to the ONNX model file.
    model: PathBuf,
}

pub fn run(args: InspectArgs) -> kenosis_core::Result<()> {
    let model = OnnxModel::load(&args.model)?;
    let stats = inspect::analyze(&model);
    let filename = args
        .model
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    // Header
    println!();
    println!("  {} {}", "⬡".cyan(), filename.bold());
    println!("  {}", "─".repeat(56).dimmed());

    // Summary table
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic);

    table.set_header(vec!["Metric", "Value"]);
    table.add_row(vec!["Opset Version", &stats.opset_version.to_string()]);
    table.add_row(vec![
        "Parameters",
        &inspect::format_count(stats.total_params),
    ]);
    table.add_row(vec![
        "Model Size",
        &inspect::format_bytes(stats.total_size_bytes),
    ]);
    table.add_row(vec![
        "Weight Size",
        &inspect::format_bytes(stats.weight_size_bytes),
    ]);
    table.add_row(vec!["Nodes", &stats.node_count.to_string()]);
    table.add_row(vec!["Inputs", &stats.input_count.to_string()]);
    table.add_row(vec!["Outputs", &stats.output_count.to_string()]);
    table.add_row(vec!["Initializers", &stats.initializer_count.to_string()]);
    if stats.constant_weight_count > 0 {
        table.add_row(vec![
            "Constant Weights",
            &format!(
                "{} (inline — use --extract-constants to convert)",
                stats.constant_weight_count
            ),
        ]);
    }
    println!("\n{table}");

    // Operator distribution
    if !stats.op_counts.is_empty() {
        println!("\n  {} {}", "▸".cyan(), "Operator Distribution".bold());
        let mut op_table = Table::new();
        op_table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic);
        op_table.set_header(vec!["Operator", "Count"]);
        for (op, count) in &stats.op_counts {
            op_table.add_row(vec![op.as_str(), &count.to_string()]);
        }
        println!("{op_table}");
    }

    // Data type breakdown
    if !stats.dtype_breakdown.is_empty() {
        println!("\n  {} {}", "▸".cyan(), "Data Type Breakdown".bold());
        let mut dtype_table = Table::new();
        dtype_table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic);
        dtype_table.set_header(vec!["Type", "Size"]);
        for (dtype, bytes) in &stats.dtype_breakdown {
            dtype_table.add_row(vec![dtype.as_str(), &inspect::format_bytes(*bytes)]);
        }
        println!("{dtype_table}");
    }

    // Top tensors
    if !stats.top_tensors.is_empty() {
        println!("\n  {} {}", "▸".cyan(), "Largest Tensors".bold());
        let mut tensor_table = Table::new();
        tensor_table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic);
        tensor_table.set_header(vec!["Name", "Shape", "Type", "Size"]);
        for t in &stats.top_tensors {
            let shape = format!("{:?}", t.shape);
            tensor_table.add_row(vec![
                &t.name,
                &shape,
                &t.dtype,
                &inspect::format_bytes(t.size_bytes),
            ]);
        }
        println!("{tensor_table}");
    }

    println!();
    Ok(())
}
