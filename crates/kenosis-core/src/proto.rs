// Copyright (c) 2026 Core Epoch. Licensed under Apache-2.0.
//! ONNX protobuf type definitions.
//!
//! Hand-written prost structs matching the ONNX IR spec. This avoids requiring
//! `protoc` at build time and gives us full control over the type surface.

/// Top-level model container.
#[derive(Clone, prost::Message)]
pub struct ModelProto {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(string, tag = "3")]
    pub producer_version: String,
    #[prost(string, tag = "4")]
    pub domain: String,
    #[prost(int64, tag = "5")]
    pub model_version: i64,
    #[prost(string, tag = "6")]
    pub doc_string: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "14")]
    pub metadata_props: Vec<StringStringEntryProto>,
    #[prost(message, repeated, tag = "20")]
    pub training_info: Vec<TrainingInfoProto>,
    #[prost(message, repeated, tag = "25")]
    pub functions: Vec<FunctionProto>,
}

/// Computation graph.
#[derive(Clone, prost::Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "15")]
    pub sparse_initializer: Vec<SparseTensorProto>,
    #[prost(string, tag = "10")]
    pub doc_string: String,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "13")]
    pub value_info: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "14")]
    pub quantization_annotation: Vec<TensorAnnotation>,
    #[prost(message, repeated, tag = "16")]
    pub metadata_props: Vec<StringStringEntryProto>,
}

/// A single computation node.
#[derive(Clone, prost::Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(string, tag = "7")]
    pub domain: String,
    #[prost(string, tag = "8")]
    pub overload: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
    #[prost(string, tag = "6")]
    pub doc_string: String,
    #[prost(message, repeated, tag = "9")]
    pub metadata_props: Vec<StringStringEntryProto>,
}

/// Serialized tensor value.
#[derive(Clone, prost::Message)]
pub struct TensorProto {
    #[prost(int64, repeated, packed = "true", tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(float, repeated, packed = "true", tag = "4")]
    pub float_data: Vec<f32>,
    #[prost(int32, repeated, packed = "true", tag = "5")]
    pub int32_data: Vec<i32>,
    #[prost(bytes = "vec", repeated, tag = "6")]
    pub string_data: Vec<Vec<u8>>,
    #[prost(int64, repeated, packed = "true", tag = "7")]
    pub int64_data: Vec<i64>,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(string, tag = "12")]
    pub doc_string: String,
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,
    #[prost(message, repeated, tag = "13")]
    pub external_data: Vec<StringStringEntryProto>,
    #[prost(int32, tag = "14")]
    pub data_location: i32,
    #[prost(double, repeated, packed = "true", tag = "10")]
    pub double_data: Vec<f64>,
    #[prost(uint64, repeated, packed = "true", tag = "11")]
    pub uint64_data: Vec<u64>,
    #[prost(message, repeated, tag = "16")]
    pub metadata_props: Vec<StringStringEntryProto>,
}

/// Standard ONNX data type constants.
#[allow(dead_code)]
pub mod data_type {
    pub const UNDEFINED: i32 = 0;
    pub const FLOAT: i32 = 1;
    pub const UINT8: i32 = 2;
    pub const INT8: i32 = 3;
    pub const UINT16: i32 = 4;
    pub const INT16: i32 = 5;
    pub const INT32: i32 = 6;
    pub const INT64: i32 = 7;
    pub const STRING: i32 = 8;
    pub const BOOL: i32 = 9;
    pub const FLOAT16: i32 = 10;
    pub const DOUBLE: i32 = 11;
    pub const UINT32: i32 = 12;
    pub const UINT64: i32 = 13;
    pub const BFLOAT16: i32 = 16;

    /// Returns a human-readable name for an ONNX data type constant.
    pub fn name(dt: i32) -> &'static str {
        match dt {
            FLOAT => "FLOAT",
            UINT8 => "UINT8",
            INT8 => "INT8",
            UINT16 => "UINT16",
            INT16 => "INT16",
            INT32 => "INT32",
            INT64 => "INT64",
            STRING => "STRING",
            BOOL => "BOOL",
            FLOAT16 => "FLOAT16",
            DOUBLE => "DOUBLE",
            UINT32 => "UINT32",
            UINT64 => "UINT64",
            BFLOAT16 => "BFLOAT16",
            _ => "UNKNOWN",
        }
    }

    /// Returns the byte size of a single element of the given data type.
    pub fn byte_size(dt: i32) -> usize {
        match dt {
            FLOAT | INT32 | UINT32 => 4,
            DOUBLE | INT64 | UINT64 => 8,
            FLOAT16 | BFLOAT16 | INT16 | UINT16 => 2,
            UINT8 | INT8 | BOOL => 1,
            _ => 0,
        }
    }
}

/// Named attribute on a node.
#[derive(Clone, prost::Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(string, tag = "21")]
    pub ref_attr_name: String,
    #[prost(string, tag = "13")]
    pub doc_string: String,
    #[prost(int32, tag = "20")]
    pub r#type: i32,
    #[prost(float, tag = "2")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(bytes = "vec", tag = "4")]
    pub s: Vec<u8>,
    #[prost(message, optional, tag = "5")]
    pub t: Option<TensorProto>,
    #[prost(message, optional, tag = "6")]
    pub g: Option<GraphProto>,
    #[prost(float, repeated, packed = "true", tag = "7")]
    pub floats: Vec<f32>,
    #[prost(int64, repeated, packed = "true", tag = "8")]
    pub ints: Vec<i64>,
    #[prost(bytes = "vec", repeated, tag = "9")]
    pub strings: Vec<Vec<u8>>,
    #[prost(message, repeated, tag = "10")]
    pub tensors: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub graphs: Vec<GraphProto>,
}

/// Value type and shape information.
#[derive(Clone, prost::Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
    #[prost(string, tag = "3")]
    pub doc_string: String,
    #[prost(message, repeated, tag = "4")]
    pub metadata_props: Vec<StringStringEntryProto>,
}

/// Type information for values.
#[derive(Clone, prost::Message)]
pub struct TypeProto {
    #[prost(string, tag = "6")]
    pub denotation: String,
    #[prost(oneof = "type_proto::Value", tags = "1, 4, 5, 8, 9")]
    pub value: Option<type_proto::Value>,
}

pub mod type_proto {
    #[derive(Clone, prost::Oneof)]
    pub enum Value {
        #[prost(message, tag = "1")]
        TensorType(super::type_proto_tensor::Tensor),
        #[prost(message, tag = "4")]
        SequenceType(super::type_proto_sequence::Sequence),
        #[prost(message, tag = "5")]
        MapType(super::type_proto_map::Map),
        #[prost(message, tag = "8")]
        SparseTensorType(super::type_proto_sparse::SparseTensor),
        #[prost(message, tag = "9")]
        OptionalType(super::type_proto_optional::Optional),
    }
}

pub mod type_proto_tensor {
    #[derive(Clone, prost::Message)]
    pub struct Tensor {
        #[prost(int32, tag = "1")]
        pub elem_type: i32,
        #[prost(message, optional, tag = "2")]
        pub shape: Option<super::TensorShapeProto>,
    }
}

pub mod type_proto_sequence {
    #[derive(Clone, prost::Message)]
    pub struct Sequence {
        #[prost(message, optional, boxed, tag = "1")]
        pub elem_type: Option<Box<super::TypeProto>>,
    }
}

pub mod type_proto_map {
    #[derive(Clone, prost::Message)]
    pub struct Map {
        #[prost(int32, tag = "1")]
        pub key_type: i32,
        #[prost(message, optional, boxed, tag = "2")]
        pub value_type: Option<Box<super::TypeProto>>,
    }
}

pub mod type_proto_sparse {
    #[derive(Clone, prost::Message)]
    pub struct SparseTensor {
        #[prost(int32, tag = "1")]
        pub elem_type: i32,
        #[prost(message, optional, tag = "2")]
        pub shape: Option<super::TensorShapeProto>,
    }
}

pub mod type_proto_optional {
    #[derive(Clone, prost::Message)]
    pub struct Optional {
        #[prost(message, optional, boxed, tag = "1")]
        pub elem_type: Option<Box<super::TypeProto>>,
    }
}

/// Tensor shape description.
#[derive(Clone, prost::Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<tensor_shape_proto::Dimension>,
}

pub mod tensor_shape_proto {
    #[derive(Clone, prost::Message)]
    pub struct Dimension {
        #[prost(string, tag = "3")]
        pub denotation: String,
        #[prost(oneof = "dimension::Value", tags = "1, 2")]
        pub value: Option<dimension::Value>,
    }
    pub mod dimension {
        #[derive(Clone, prost::Oneof)]
        pub enum Value {
            #[prost(int64, tag = "1")]
            DimValue(i64),
            #[prost(string, tag = "2")]
            DimParam(String),
        }
    }
}

/// Operator set identifier.
#[derive(Clone, prost::Message)]
pub struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}

/// Key-value string pair.
#[derive(Clone, prost::Message)]
pub struct StringStringEntryProto {
    #[prost(string, tag = "1")]
    pub key: String,
    #[prost(string, tag = "2")]
    pub value: String,
}

/// Tensor quantization annotation.
#[derive(Clone, prost::Message)]
pub struct TensorAnnotation {
    #[prost(string, tag = "1")]
    pub tensor_name: String,
    #[prost(message, repeated, tag = "2")]
    pub quant_parameter_tensor_names: Vec<StringStringEntryProto>,
}

/// Sparse tensor representation.
#[derive(Clone, prost::Message)]
pub struct SparseTensorProto {
    #[prost(message, optional, tag = "1")]
    pub values: Option<TensorProto>,
    #[prost(message, optional, tag = "2")]
    pub indices: Option<TensorProto>,
    #[prost(int64, repeated, tag = "3")]
    pub dims: Vec<i64>,
}

/// Training information (preserved but not used by Kenosis).
#[derive(Clone, prost::Message)]
pub struct TrainingInfoProto {
    #[prost(message, optional, tag = "1")]
    pub initialization: Option<GraphProto>,
    #[prost(message, optional, tag = "2")]
    pub algorithm: Option<GraphProto>,
    #[prost(message, repeated, tag = "3")]
    pub initialization_binding: Vec<StringStringEntryProto>,
    #[prost(message, repeated, tag = "4")]
    pub update_binding: Vec<StringStringEntryProto>,
}

/// Model-local function definition.
#[derive(Clone, prost::Message)]
pub struct FunctionProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(string, repeated, tag = "4")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "5")]
    pub output: Vec<String>,
    #[prost(string, repeated, tag = "6")]
    pub attribute: Vec<String>,
    #[prost(message, repeated, tag = "11")]
    pub attribute_proto: Vec<AttributeProto>,
    #[prost(message, repeated, tag = "7")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "8")]
    pub doc_string: String,
    #[prost(message, repeated, tag = "9")]
    pub opset_import: Vec<OperatorSetIdProto>,
    #[prost(string, tag = "10")]
    pub domain: String,
    #[prost(string, tag = "13")]
    pub overload: String,
    #[prost(message, repeated, tag = "12")]
    pub value_info: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "14")]
    pub metadata_props: Vec<StringStringEntryProto>,
}
