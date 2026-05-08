// Copyright (c) 2026 Core Epoch. Licensed under Apache-2.0.
//! Model inspection and statistics.

use std::collections::BTreeMap;
use std::fmt;

use crate::model::OnnxModel;
use crate::proto::data_type;

/// Comprehensive statistics about an ONNX model.
#[derive(Debug, Clone)]
pub struct ModelStats {
    /// ONNX opset version.
    pub opset_version: i64,
    /// Total number of weight parameters.
    pub total_params: u64,
    /// Total model size in bytes (serialized protobuf).
    pub total_size_bytes: u64,
    /// Total weight data in bytes (sum of initializer sizes).
    pub weight_size_bytes: u64,
    /// Operator frequency map (op_type → count).
    pub op_counts: BTreeMap<String, usize>,
    /// Byte breakdown by data type.
    pub dtype_breakdown: BTreeMap<String, u64>,
    /// Information about the largest tensors.
    pub top_tensors: Vec<TensorInfo>,
    /// Number of graph nodes.
    pub node_count: usize,
    /// Number of graph inputs.
    pub input_count: usize,
    /// Number of graph outputs.
    pub output_count: usize,
    /// Number of initializer tensors.
    pub initializer_count: usize,
    /// Number of Constant-node weights detected.
    pub constant_weight_count: usize,
}

/// Summary information about a single tensor.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// Shape dimensions.
    pub shape: Vec<i64>,
    /// Data type name.
    pub dtype: String,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Number of elements.
    pub numel: u64,
}

/// Analyze a model and produce comprehensive statistics.
#[must_use = "returns model statistics"]
pub fn analyze(model: &OnnxModel) -> ModelStats {
    let graph = model.graph();

    // Operator frequency
    let mut op_counts = BTreeMap::new();
    for node in &graph.node {
        *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;
    }

    // Tensor analysis — covers both initializers and Constant-node weights
    let all_tensors = model.all_weight_tensors();
    let constant_weight_count = all_tensors.len().saturating_sub(graph.initializer.len());

    let mut total_params: u64 = 0;
    let mut weight_size_bytes: u64 = 0;
    let mut dtype_breakdown: BTreeMap<String, u64> = BTreeMap::new();
    let mut tensor_infos: Vec<TensorInfo> = Vec::new();

    for tensor in &all_tensors {
        let numel = OnnxModel::tensor_numel(tensor);
        let size = OnnxModel::tensor_byte_size(tensor);
        let dtype_name = data_type::name(tensor.data_type).to_string();

        total_params += numel;
        weight_size_bytes += size;
        *dtype_breakdown.entry(dtype_name.clone()).or_insert(0) += size;

        tensor_infos.push(TensorInfo {
            name: tensor.name.clone(),
            shape: tensor.dims.clone(),
            dtype: dtype_name,
            size_bytes: size,
            numel,
        });
    }

    // Sort by size descending, take top 10
    tensor_infos.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
    tensor_infos.truncate(10);

    ModelStats {
        opset_version: model.opset_version(),
        total_params,
        total_size_bytes: model.byte_size() as u64,
        weight_size_bytes,
        op_counts,
        dtype_breakdown,
        top_tensors: tensor_infos,
        node_count: graph.node.len(),
        input_count: graph.input.len(),
        output_count: graph.output.len(),
        initializer_count: graph.initializer.len(),
        constant_weight_count,
    }
}

impl fmt::Display for ModelStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model Statistics")?;
        writeln!(f, "  Opset version:   {}", self.opset_version)?;
        writeln!(f, "  Total params:    {}", format_count(self.total_params))?;
        writeln!(
            f,
            "  Model size:      {}",
            format_bytes(self.total_size_bytes)
        )?;
        writeln!(
            f,
            "  Weight size:     {}",
            format_bytes(self.weight_size_bytes)
        )?;
        writeln!(f, "  Nodes:           {}", self.node_count)?;
        writeln!(f, "  Inputs:          {}", self.input_count)?;
        writeln!(f, "  Outputs:         {}", self.output_count)?;
        writeln!(f, "  Initializers:    {}", self.initializer_count)?;

        if !self.op_counts.is_empty() {
            writeln!(f, "\nOperator Distribution:")?;
            for (op, count) in &self.op_counts {
                writeln!(f, "  {op:<24} {count:>5}")?;
            }
        }

        if !self.dtype_breakdown.is_empty() {
            writeln!(f, "\nData Type Breakdown:")?;
            for (dtype, bytes) in &self.dtype_breakdown {
                writeln!(f, "  {dtype:<12} {}", format_bytes(*bytes))?;
            }
        }

        if !self.top_tensors.is_empty() {
            writeln!(f, "\nLargest Tensors:")?;
            for t in &self.top_tensors {
                writeln!(
                    f,
                    "  {:<40} {:>10}  {:>8}  {:?}",
                    truncate_name(&t.name, 40),
                    format_bytes(t.size_bytes),
                    t.dtype,
                    t.shape,
                )?;
            }
        }

        Ok(())
    }
}

/// Format a byte count as a human-readable string.
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Format a parameter count with K/M/B suffixes.
pub fn format_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{n}")
    }
}

pub fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        name.to_string()
    } else {
        format!("...{}", &name[name.len() - (max_len - 3)..])
    }
}
