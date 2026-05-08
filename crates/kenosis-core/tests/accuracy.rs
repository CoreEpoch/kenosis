// Copyright (c) 2026 Core Epoch. Licensed under Apache-2.0.
//! Accuracy tests for kenosis-core.
//!
//! Run with: cargo test --release -p kenosis-core -- --nocapture accuracy

#[cfg(test)]
mod accuracy_tests {
    use kenosis_core::data_type;
    use kenosis_core::model::OnnxModel;
    use std::path::Path;

    fn benchmark_model(name: &str) -> String {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root");
        workspace
            .join("benchmarks")
            .join(name)
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn model_load_save_roundtrip() {
        let path = benchmark_model("squeezenet1.1.onnx");
        if !Path::new(&path).exists() {
            println!("  SKIP: {} not found", path);
            return;
        }

        let model = OnnxModel::load(&path).expect("load");
        let original_size = model.byte_size();

        // Save to temp file and reload
        let tmp = std::env::temp_dir().join("kenosis_roundtrip_test.onnx");
        model.save(&tmp).expect("save");
        let reloaded = OnnxModel::load(&tmp).expect("reload");

        assert_eq!(
            model.graph().node.len(),
            reloaded.graph().node.len(),
            "Node count mismatch after roundtrip"
        );
        assert_eq!(
            model.initializers().len(),
            reloaded.initializers().len(),
            "Initializer count mismatch after roundtrip"
        );
        // Size should be very close (protobuf re-encoding may differ by a few bytes)
        let diff = (original_size as i64 - reloaded.byte_size() as i64).unsigned_abs();
        assert!(
            diff < 100,
            "Size difference too large after roundtrip: {diff}"
        );

        std::fs::remove_file(&tmp).ok();
        println!(
            "  Roundtrip test passed: {} nodes preserved",
            model.graph().node.len()
        );
    }

    #[test]
    fn inspect_analysis() {
        let path = benchmark_model("squeezenet1.1.onnx");
        if !Path::new(&path).exists() {
            println!("  SKIP: {} not found", path);
            return;
        }

        let model = OnnxModel::load(&path).expect("load");
        let stats = kenosis_core::inspect::analyze(&model);

        assert!(stats.total_params > 0, "Should have parameters");
        assert!(stats.total_size_bytes > 0, "Should have non-zero size");
        assert!(stats.node_count > 0, "Should have nodes");
        assert!(!stats.op_counts.is_empty(), "Should have op counts");

        println!(
            "  SqueezeNet: {} params, {} nodes, {} ops",
            stats.total_params,
            stats.node_count,
            stats.op_counts.len()
        );
    }

    #[test]
    fn constant_extraction() {
        let path = benchmark_model("squeezenet1.1.onnx");
        if !Path::new(&path).exists() {
            println!("  SKIP: {} not found", path);
            return;
        }

        let mut model = OnnxModel::load(&path).expect("load");
        let initial_inits = model.initializers().len();
        let extracted = model.extract_constants();

        // SqueezeNet may or may not have Constant nodes, but the function
        // should not panic and should return a count >= 0
        assert!(
            model.initializers().len() >= initial_inits,
            "Initializer count should not decrease"
        );
        println!(
            "  Extracted {} constants ({} -> {} initializers)",
            extracted,
            initial_inits,
            model.initializers().len()
        );
    }

    #[test]
    fn tensor_f32_extraction() {
        let path = benchmark_model("squeezenet1.1.onnx");
        if !Path::new(&path).exists() {
            println!("  SKIP: {} not found", path);
            return;
        }

        let model = OnnxModel::load(&path).expect("load");
        let fp32_tensors: Vec<_> = model
            .all_weight_tensors()
            .into_iter()
            .filter(|t| t.data_type == data_type::FLOAT)
            .collect();

        assert!(!fp32_tensors.is_empty(), "Should have FP32 tensors");

        for tensor in &fp32_tensors {
            if let Some(values) = OnnxModel::tensor_as_f32(tensor) {
                assert!(!values.is_empty(), "Extracted tensor should not be empty");
                // Verify values are finite
                for &v in &values {
                    assert!(v.is_finite(), "Tensor values should be finite");
                }
            }
        }

        println!(
            "  Verified {} FP32 tensors with finite values",
            fp32_tensors.len()
        );
    }

    #[test]
    fn cast_fp16_roundtrip() {
        let path = benchmark_model("squeezenet1.1.onnx");
        if !Path::new(&path).exists() {
            println!("  SKIP: {} not found", path);
            return;
        }

        let model = OnnxModel::load(&path).expect("load");
        let original_size = model.byte_size();

        let casted = kenosis_core::cast::cast_precision(model, kenosis_core::Precision::Float16)
            .expect("cast to FP16");

        // FP16 model should be smaller than FP32
        assert!(
            casted.byte_size() < original_size,
            "FP16 model should be smaller than FP32"
        );

        // Verify the cast actually changed data types
        let fp16_count = casted
            .all_weight_tensors()
            .iter()
            .filter(|t| t.data_type == data_type::FLOAT16)
            .count();
        assert!(fp16_count > 0, "Should have FP16 tensors after casting");

        let ratio = original_size as f64 / casted.byte_size() as f64;
        println!(
            "  FP16 cast: {} -> {} ({:.1}x smaller, {} FP16 tensors)",
            original_size,
            casted.byte_size(),
            ratio,
            fp16_count
        );
    }

    #[test]
    fn diff_identical_models() {
        let path = benchmark_model("squeezenet1.1.onnx");
        if !Path::new(&path).exists() {
            println!("  SKIP: {} not found", path);
            return;
        }

        let model_a = OnnxModel::load(&path).expect("load A");
        let model_b = OnnxModel::load(&path).expect("load B");

        let d = kenosis_core::diff::compare(&model_a, &model_b);

        assert_eq!(d.size_a, d.size_b, "Same model should have same size");
        assert_eq!(d.nodes_a, d.nodes_b, "Same model should have same nodes");
        assert_eq!(d.params_a, d.params_b, "Same model should have same params");
        assert!(
            (d.compression_ratio - 1.0).abs() < 0.01,
            "Compression ratio of same model should be ~1.0"
        );

        println!("  Diff test passed: identical models report 1.0x ratio");
    }
}
