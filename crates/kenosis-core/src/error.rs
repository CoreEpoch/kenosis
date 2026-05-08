// Copyright (c) 2026 Core Epoch. Licensed under Apache-2.0.
//! Error types for Kenosis operations.

/// All errors produced by Kenosis operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum KenosisError {
    /// I/O error reading or writing a model file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to decode the ONNX protobuf.
    #[error("invalid ONNX model: {0}")]
    InvalidModel(String),

    /// The model contains an unsupported operator or feature.
    #[error("unsupported operation: {0}")]
    UnsupportedOp(String),

    /// Quantization failed for a specific reason.
    #[error("quantization failed: {0}")]
    QuantizationFailed(String),

    /// Protobuf decode error.
    #[error("protobuf decode error: {0}")]
    ProtoDecode(#[from] prost::DecodeError),

    /// Protobuf encode error.
    #[error("protobuf encode error: {0}")]
    ProtoEncode(#[from] prost::EncodeError),
}
