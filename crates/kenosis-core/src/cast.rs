// Copyright (c) 2026 Core Epoch. Licensed under Apache-2.0.
//! Precision casting — FP32 → FP16/BF16 conversion.

use half::{bf16, f16};

use crate::error::KenosisError;
use crate::model::OnnxModel;
use crate::proto::data_type;
use crate::Precision;

/// Cast all float32 initializer tensors to the target precision.
///
/// This is a straightforward type conversion — no quantization or binning.
/// Each FP32 value is directly converted to FP16 or BF16.
///
/// # Errors
///
/// Returns an error if the target precision is not FP16 or BF16.
#[must_use = "returns the cast model"]
pub fn cast_precision(mut model: OnnxModel, target: Precision) -> crate::Result<OnnxModel> {
    let target_dtype = match target {
        Precision::Float16 => data_type::FLOAT16,
        Precision::BFloat16 => data_type::BFLOAT16,
        other => {
            return Err(KenosisError::UnsupportedOp(format!(
                "cast_precision only supports FP16/BF16, got {other}"
            )));
        }
    };

    let graph = model.graph_mut();
    let mut converted = 0usize;

    for tensor in &mut graph.initializer {
        if tensor.data_type != data_type::FLOAT {
            continue;
        }

        let f32_values = match extract_f32(tensor) {
            Some(v) => v,
            None => continue,
        };

        let raw_bytes: Vec<u8> = match target {
            Precision::Float16 => f32_values
                .iter()
                .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                .collect(),
            Precision::BFloat16 => f32_values
                .iter()
                .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
                .collect(),
            _ => unreachable!(),
        };

        tensor.raw_data = raw_bytes;
        tensor.float_data.clear();
        tensor.data_type = target_dtype;
        converted += 1;
    }

    tracing::info!(converted, target = %target, "precision cast complete");
    Ok(model)
}

fn extract_f32(tensor: &crate::proto::TensorProto) -> Option<Vec<f32>> {
    OnnxModel::tensor_as_f32(tensor)
}
