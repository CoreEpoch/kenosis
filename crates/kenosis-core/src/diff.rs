// Copyright (c) 2026 Core Epoch. Licensed under Apache-2.0.
//! Model comparison (diff).

use std::collections::BTreeMap;
use std::fmt;

use crate::inspect::{self, format_bytes, format_count};
use crate::model::OnnxModel;

/// Comparison between two ONNX models.
#[derive(Debug, Clone)]
pub struct ModelDiff {
    /// Total serialized size of model A.
    pub size_a: u64,
    /// Total serialized size of model B.
    pub size_b: u64,
    /// Compression ratio (size_a / size_b).
    pub compression_ratio: f64,
    /// Total parameters in model A.
    pub params_a: u64,
    /// Total parameters in model B.
    pub params_b: u64,
    /// Weight size in model A.
    pub weight_size_a: u64,
    /// Weight size in model B.
    pub weight_size_b: u64,
    /// Node count in model A.
    pub nodes_a: usize,
    /// Node count in model B.
    pub nodes_b: usize,
    /// Data type breakdown for model A.
    pub dtypes_a: BTreeMap<String, u64>,
    /// Data type breakdown for model B.
    pub dtypes_b: BTreeMap<String, u64>,
}

/// Compare two ONNX models and produce a diff summary.
#[must_use = "returns the diff result"]
pub fn compare(model_a: &OnnxModel, model_b: &OnnxModel) -> ModelDiff {
    let stats_a = inspect::analyze(model_a);
    let stats_b = inspect::analyze(model_b);

    let size_a = stats_a.total_size_bytes;
    let size_b = stats_b.total_size_bytes;
    let compression_ratio = if size_b > 0 {
        size_a as f64 / size_b as f64
    } else {
        0.0
    };

    ModelDiff {
        size_a,
        size_b,
        compression_ratio,
        params_a: stats_a.total_params,
        params_b: stats_b.total_params,
        weight_size_a: stats_a.weight_size_bytes,
        weight_size_b: stats_b.weight_size_bytes,
        nodes_a: stats_a.node_count,
        nodes_b: stats_b.node_count,
        dtypes_a: stats_a.dtype_breakdown,
        dtypes_b: stats_b.dtype_breakdown,
    }
}

impl fmt::Display for ModelDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model Comparison")?;
        writeln!(f, "  {}", "─".repeat(60))?;
        writeln!(f, "  {:<24} {:>14} {:>14}", "Metric", "Model A", "Model B")?;
        writeln!(f, "  {}", "─".repeat(60))?;
        writeln!(
            f,
            "  {:<24} {:>14} {:>14}",
            "Total size",
            format_bytes(self.size_a),
            format_bytes(self.size_b),
        )?;
        writeln!(
            f,
            "  {:<24} {:>14} {:>14}",
            "Weight size",
            format_bytes(self.weight_size_a),
            format_bytes(self.weight_size_b),
        )?;
        writeln!(
            f,
            "  {:<24} {:>14} {:>14}",
            "Parameters",
            format_count(self.params_a),
            format_count(self.params_b),
        )?;
        writeln!(
            f,
            "  {:<24} {:>14} {:>14}",
            "Nodes", self.nodes_a, self.nodes_b,
        )?;

        writeln!(f)?;
        if self.compression_ratio > 1.0 {
            writeln!(
                f,
                "  Compression: {:.1}x smaller ({} → {})",
                self.compression_ratio,
                format_bytes(self.size_a),
                format_bytes(self.size_b),
            )?;
        } else if self.compression_ratio > 0.0 {
            writeln!(
                f,
                "  Expansion: {:.1}x larger ({} → {})",
                1.0 / self.compression_ratio,
                format_bytes(self.size_a),
                format_bytes(self.size_b),
            )?;
        }

        // Data type shift
        let all_dtypes: BTreeMap<&String, ()> = self
            .dtypes_a
            .keys()
            .chain(self.dtypes_b.keys())
            .map(|k| (k, ()))
            .collect();

        if !all_dtypes.is_empty() {
            writeln!(f, "\n  Data Type Shift:")?;
            for dtype in all_dtypes.keys() {
                let a = self.dtypes_a.get(*dtype).copied().unwrap_or(0);
                let b = self.dtypes_b.get(*dtype).copied().unwrap_or(0);
                if a != b {
                    writeln!(
                        f,
                        "    {:<12} {} → {}",
                        dtype,
                        format_bytes(a),
                        format_bytes(b),
                    )?;
                }
            }
        }

        Ok(())
    }
}
