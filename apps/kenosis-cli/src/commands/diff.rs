// Copyright (c) 2026 Core Epoch. Licensed under Apache-2.0.
//! `kenosis diff` — compare two ONNX models.

use std::path::PathBuf;

use clap::Args;
use comfy_table::{
    modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL, Cell, Color, ContentArrangement, Table,
};
use owo_colors::OwoColorize;

use kenosis_core::diff;
use kenosis_core::inspect;
use kenosis_core::OnnxModel;

#[derive(Args)]
pub struct DiffArgs {
    /// Path to the first (original) ONNX model.
    model_a: PathBuf,

    /// Path to the second (modified) ONNX model.
    model_b: PathBuf,
}

pub fn run(args: DiffArgs) -> kenosis_core::Result<()> {
    let model_a = OnnxModel::load(&args.model_a)?;
    let model_b = OnnxModel::load(&args.model_b)?;

    let d = diff::compare(&model_a, &model_b);

    let name_a = args
        .model_a
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "Model A".into());
    let name_b = args
        .model_b
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "Model B".into());

    println!();
    println!("  {} Model Comparison", "⬡".cyan());
    println!("  {}", "─".repeat(56).dimmed());

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic);

    table.set_header(vec!["Metric", &name_a, &name_b]);
    table.add_row(vec![
        Cell::new("Total Size"),
        Cell::new(inspect::format_bytes(d.size_a)),
        Cell::new(inspect::format_bytes(d.size_b)),
    ]);
    table.add_row(vec![
        Cell::new("Weight Size"),
        Cell::new(inspect::format_bytes(d.weight_size_a)),
        Cell::new(inspect::format_bytes(d.weight_size_b)),
    ]);
    table.add_row(vec![
        Cell::new("Parameters"),
        Cell::new(inspect::format_count(d.params_a)),
        Cell::new(inspect::format_count(d.params_b)),
    ]);
    table.add_row(vec![
        Cell::new("Nodes"),
        Cell::new(d.nodes_a.to_string()),
        Cell::new(d.nodes_b.to_string()),
    ]);
    println!("\n{table}");

    // Compression headline
    if d.compression_ratio > 1.0 {
        println!(
            "\n  {} {:.1}× compression ({} → {})",
            "✨".green(),
            d.compression_ratio,
            inspect::format_bytes(d.size_a),
            inspect::format_bytes(d.size_b).green(),
        );
    }

    // Data type shifts
    let all_dtypes: std::collections::BTreeSet<&String> =
        d.dtypes_a.keys().chain(d.dtypes_b.keys()).collect();

    let has_shift = all_dtypes
        .iter()
        .any(|dt| d.dtypes_a.get(*dt) != d.dtypes_b.get(*dt));

    if has_shift {
        println!("\n  {} {}", "▸".cyan(), "Data Type Shift".bold());
        let mut dt_table = Table::new();
        dt_table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic);
        dt_table.set_header(vec!["Type", &name_a, &name_b]);

        for dtype in &all_dtypes {
            let a = d.dtypes_a.get(*dtype).copied().unwrap_or(0);
            let b = d.dtypes_b.get(*dtype).copied().unwrap_or(0);
            if a != b {
                dt_table.add_row(vec![
                    Cell::new(dtype.as_str()),
                    Cell::new(inspect::format_bytes(a)),
                    Cell::new(inspect::format_bytes(b)).fg(if b < a {
                        Color::Green
                    } else {
                        Color::Yellow
                    }),
                ]);
            }
        }
        println!("{dt_table}");
    }

    println!();
    Ok(())
}
