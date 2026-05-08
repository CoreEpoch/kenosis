// Copyright (c) 2026 Core Epoch. Licensed under Apache-2.0.
//! ONNX model loading, saving, and graph traversal.

use std::path::Path;

use prost::Message;

use crate::error::KenosisError;
use crate::proto::{self, data_type, ModelProto, TensorProto};

/// A loaded ONNX model with convenience accessors.
#[derive(Clone, Debug)]
pub struct OnnxModel {
    /// The underlying protobuf model.
    pub proto: ModelProto,
}

impl OnnxModel {
    /// Load an ONNX model from a file path.
    ///
    /// # Errors
    ///
    /// Returns [`KenosisError::Io`] if the file cannot be read, or
    /// [`KenosisError::ProtoDecode`] if the protobuf is malformed.
    pub fn load(path: impl AsRef<Path>) -> crate::Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        let proto = ModelProto::decode(bytes.as_slice())?;
        if proto.graph.is_none() {
            return Err(KenosisError::InvalidModel("model has no graph".into()));
        }
        Ok(Self { proto })
    }

    /// Save the model to a file path.
    ///
    /// # Errors
    ///
    /// Returns [`KenosisError::Io`] if the file cannot be written.
    pub fn save(&self, path: impl AsRef<Path>) -> crate::Result<()> {
        let mut buf = Vec::with_capacity(self.proto.encoded_len());
        self.proto.encode(&mut buf)?;
        std::fs::write(path.as_ref(), &buf)?;
        Ok(())
    }

    /// Returns the graph, panicking if absent (validated at load time).
    pub fn graph(&self) -> &proto::GraphProto {
        self.proto.graph.as_ref().expect("graph validated at load")
    }

    /// Returns a mutable reference to the graph.
    pub fn graph_mut(&mut self) -> &mut proto::GraphProto {
        self.proto.graph.as_mut().expect("graph validated at load")
    }

    /// Returns the opset version for the default ONNX domain.
    pub fn opset_version(&self) -> i64 {
        self.proto
            .opset_import
            .iter()
            .find(|op| op.domain.is_empty() || op.domain == "ai.onnx")
            .map(|op| op.version)
            .unwrap_or(0)
    }

    /// Returns all weight initializer tensors in the graph.
    pub fn initializers(&self) -> &[TensorProto] {
        &self.graph().initializer
    }

    /// Returns the computation nodes in the graph.
    pub fn nodes(&self) -> &[proto::NodeProto] {
        &self.graph().node
    }

    /// Total byte size of the serialized model.
    pub fn byte_size(&self) -> usize {
        self.proto.encoded_len()
    }

    /// Extract the raw float32 values from a tensor, regardless of storage format.
    ///
    /// Handles both `float_data` and `raw_data` storage.
    pub fn tensor_as_f32(tensor: &TensorProto) -> Option<Vec<f32>> {
        if tensor.data_type != data_type::FLOAT {
            return None;
        }
        if !tensor.float_data.is_empty() {
            return Some(tensor.float_data.clone());
        }
        if !tensor.raw_data.is_empty() {
            let count = tensor.raw_data.len() / 4;
            let mut values = Vec::with_capacity(count);
            for chunk in tensor.raw_data.chunks_exact(4) {
                values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            return Some(values);
        }
        None
    }

    /// Compute the total number of elements in a tensor from its dimensions.
    pub fn tensor_numel(tensor: &TensorProto) -> u64 {
        if tensor.dims.is_empty() {
            return 0;
        }
        tensor.dims.iter().map(|&d| d.max(0) as u64).product()
    }

    /// Compute the byte size of a tensor based on its dims and data type.
    pub fn tensor_byte_size(tensor: &TensorProto) -> u64 {
        Self::tensor_numel(tensor) * data_type::byte_size(tensor.data_type) as u64
    }

    /// Extract `Constant` op nodes into graph initializers.
    ///
    /// Many ONNX exporters (e.g. PaddlePaddle, some PyTorch exports) embed
    /// weight tensors as `Constant` nodes rather than `initializer` entries.
    /// This pass moves those tensors into `graph.initializer` so they become
    /// visible to quantization, casting, and inspection.
    ///
    /// Returns the number of constants extracted.
    pub fn extract_constants(&mut self) -> usize {
        let graph = self.graph_mut();
        let mut extracted = 0usize;

        // Collect constant nodes to extract
        let mut tensors_to_add: Vec<TensorProto> = Vec::new();
        let mut nodes_to_remove: Vec<usize> = Vec::new();

        for (idx, node) in graph.node.iter().enumerate() {
            if node.op_type != "Constant" {
                continue;
            }

            // A Constant node has a single output and a "value" attribute
            // containing a TensorProto.
            let output_name = match node.output.first() {
                Some(name) if !name.is_empty() => name.clone(),
                _ => continue,
            };

            // Look for the "value" attribute (type = TENSOR = 4)
            let tensor_attr = node.attribute.iter().find(|a| a.name == "value");
            if let Some(attr) = tensor_attr {
                if let Some(mut tensor) = attr.t.clone() {
                    // Set the tensor name to the node's output name so
                    // downstream nodes find it as an initializer.
                    tensor.name = output_name;
                    tensors_to_add.push(tensor);
                    nodes_to_remove.push(idx);
                    extracted += 1;
                }
            }
        }

        // Remove Constant nodes in reverse order to preserve indices.
        for &idx in nodes_to_remove.iter().rev() {
            graph.node.remove(idx);
        }

        // Add extracted tensors as initializers.
        graph.initializer.extend(tensors_to_add);

        tracing::info!(extracted, "constant nodes extracted to initializers");
        extracted
    }

    /// Returns all weight tensors — both graph initializers and any remaining
    /// `Constant` node tensors — without mutating the model.
    ///
    /// This is a read-only alternative to [`extract_constants`] for inspection.
    pub fn all_weight_tensors(&self) -> Vec<&TensorProto> {
        let graph = self.graph();
        let mut tensors: Vec<&TensorProto> = graph.initializer.iter().collect();

        // Also collect Constant node tensors
        for node in &graph.node {
            if node.op_type == "Constant" {
                for attr in &node.attribute {
                    if attr.name == "value" {
                        if let Some(ref t) = attr.t {
                            tensors.push(t);
                        }
                    }
                }
            }
        }

        tensors
    }
}
