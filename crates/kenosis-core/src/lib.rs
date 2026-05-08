// Copyright (c) 2026 Core Epoch. Licensed under Apache-2.0.
//! Kenosis — Pure-Rust ONNX model optimization toolkit.
//!
//! Kenosis provides static INT8 quantization for ONNX models, producing
//! QDQ-format models that run on stock ONNX Runtime with full kernel fusion.
//!
//! Key capabilities:
//!
//! - **Static INT8 QDQ Quantization**: Calibration-based quantization with
//!   ReLU-aware QDQ placement for optimal Conv+ReLU fusion. Achieves 2×+
//!   speedup over FP32 on standard Conv architectures.
//!
//! - **Model Inspection**: Operator distribution, data type breakdown,
//!   and parameter count analysis.
//!
//! - **Precision Casting**: FP32 → FP16/BF16 conversion.
//!
//! - **Model Comparison**: Side-by-side diff of two ONNX models.

pub mod cast;
pub mod diff;
pub mod error;
pub mod inspect;
pub mod model;
#[allow(clippy::enum_variant_names)] // Names match the ONNX protobuf spec
pub(crate) mod proto;
#[cfg(feature = "calibrate")]
pub mod static_int8;

pub use error::KenosisError;
pub use model::OnnxModel;
pub use proto::data_type;

/// Target precision for casting and quantization operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Precision {
    /// 32-bit floating point (no conversion).
    Float32,
    /// 16-bit IEEE 754 floating point.
    Float16,
    /// 16-bit brain floating point.
    BFloat16,
    /// 8-bit signed integer.
    Int8,
    /// 8-bit unsigned integer.
    Uint8,
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Precision::Float32 => write!(f, "FP32"),
            Precision::Float16 => write!(f, "FP16"),
            Precision::BFloat16 => write!(f, "BF16"),
            Precision::Int8 => write!(f, "INT8"),
            Precision::Uint8 => write!(f, "UINT8"),
        }
    }
}

/// Result type alias using [`KenosisError`].
pub type Result<T> = std::result::Result<T, KenosisError>;
