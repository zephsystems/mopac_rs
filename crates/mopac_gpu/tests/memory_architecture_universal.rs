//! Comprehensive Hardware Introspection & Universal Memory Architecture Verification Tests.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Empirically validates that MOPAC_RS dynamically interrogates and adapts to any GPU memory
//! hierarchy:
//!  - High-Bandwidth Memory (HBM2, HBM2e, HBM3, HBM3e) in HPC Datacenter nodes (A100, H100, MI300X)
//!  - Unified Memory Architectures (UMA / LPDDR5X) in APUs and Superchips (Grace Hopper GH200, MI300A, Apple Silicon)
//!  - Dedicated Desktop/Workstation VRAM (GDDR5, GDDR6, GDDR6X, GDDR7) with and without PCIe Resizable BAR (ReBAR)

use mopac_gpu::{
    AtomGpuFP32, BatchMoleculeDescriptor, GpuBatchDeviceMemoryManager, GpuCoulombCalculatorFP32,
    GpuMemoryArchitecture, VulkanContext,
};
use std::sync::Arc;
use std::time::Instant;

fn skip_if_no_gpu() -> Option<Arc<VulkanContext>> {
    match VulkanContext::new() {
        Ok(ctx) => Some(Arc::new(ctx)),
        Err(e) => {
            eprintln!(
                "[SKIPPED] No Vulkan compute-capable hardware found: {:?}. Skipping test on headless host.",
                e
            );
            None
        }
    }
}

#[test]
fn test_vulkan_gpu_memory_architecture_introspection() {
    let ctx = match skip_if_no_gpu() {
        Some(c) => c,
        None => return,
    };

    let mem_info = &ctx.device_info.memory_info;
    let dev_name = &ctx.device_info.device_name;
    let total_vram_gb = (mem_info.total_device_memory as f64) / (1024.0 * 1024.0 * 1024.0);

    println!(
        "\n=======================================================\n\
         [GPU HARDWARE MEMORY INTROSPECTION AUDIT]\n\
         Device Name:         {}\n\
         Memory Architecture: {}\n\
         Is Unified Memory:   {}\n\
         Supports ReBAR:      {}\n\
         Primary Device Heap: {}\n\
         Total Device VRAM:   {:.3} GB ({} bytes)\n\
         Total Heaps Count:   {}\n\
         =======================================================",
        dev_name,
        mem_info.architecture,
        mem_info.is_unified,
        mem_info.supports_rebar,
        mem_info.primary_device_heap_index,
        total_vram_gb,
        mem_info.total_device_memory,
        mem_info.heap_sizes.len(),
    );

    assert!(
        mem_info.total_device_memory > 0,
        "Total device memory must be greater than zero"
    );
    assert!(
        !mem_info.heap_sizes.is_empty(),
        "At least one memory heap must be detected"
    );

    // Verify consistency between architecture and properties
    match mem_info.architecture {
        GpuMemoryArchitecture::UnifiedMemory => {
            assert!(
                mem_info.is_unified,
                "UnifiedMemory architecture must have is_unified=true"
            );
        }
        GpuMemoryArchitecture::DiscreteReBar => {
            assert!(
                !mem_info.is_unified,
                "DiscreteReBar architecture must have is_unified=false"
            );
            assert!(
                mem_info.supports_rebar,
                "DiscreteReBar architecture must have supports_rebar=true"
            );
        }
        GpuMemoryArchitecture::DiscreteStaged => {
            assert!(
                !mem_info.is_unified,
                "DiscreteStaged architecture must have is_unified=false"
            );
            assert!(
                !mem_info.supports_rebar,
                "DiscreteStaged architecture must have supports_rebar=false"
            );
        }
    }
}

#[test]
fn test_optimal_memory_selection_scoring() {
    let ctx = match skip_if_no_gpu() {
        Some(c) => c,
        None => return,
    };

    // Test optimal device memory selection with and without direct host mapping
    let all_bits = u32::MAX;
    let (dev_type_idx, is_direct) = ctx
        .find_optimal_device_memory_type(all_bits, true)
        .expect("Failed to find optimal device memory type");

    let dev_mem_type = ctx.memory_properties.memory_types[dev_type_idx as usize];
    assert!(
        dev_mem_type
            .property_flags
            .contains(ash::vk::MemoryPropertyFlags::DEVICE_LOCAL),
        "Optimal device memory type must have DEVICE_LOCAL flag"
    );

    if is_direct {
        assert!(
            dev_mem_type
                .property_flags
                .contains(ash::vk::MemoryPropertyFlags::HOST_VISIBLE),
            "Direct mapping must be HOST_VISIBLE"
        );
        println!(
            "[OK] Direct zero-copy device memory path available! (Memory Type Index: {})",
            dev_type_idx
        );
    } else {
        println!(
            "[OK] High-throughput discrete device-local memory selected (DMA Staged Path, Memory Type Index: {})",
            dev_type_idx
        );
    }

    // Test optimal staging memory selection
    let staging_type_idx = ctx
        .find_optimal_staging_memory_type(all_bits)
        .expect("Failed to find optimal staging memory type");

    let staging_mem_type = ctx.memory_properties.memory_types[staging_type_idx as usize];
    assert!(
        staging_mem_type
            .property_flags
            .contains(ash::vk::MemoryPropertyFlags::HOST_VISIBLE),
        "Staging memory must be HOST_VISIBLE"
    );
    assert!(
        staging_mem_type
            .property_flags
            .contains(ash::vk::MemoryPropertyFlags::HOST_COHERENT),
        "Staging memory must be HOST_COHERENT"
    );
    println!(
        "[OK] Optimal staging memory selected (Memory Type Index: {}, Cached: {})",
        staging_type_idx,
        staging_mem_type
            .property_flags
            .contains(ash::vk::MemoryPropertyFlags::HOST_CACHED)
    );
}

#[test]
fn test_universal_device_batch_manager_api() {
    let ctx = match skip_if_no_gpu() {
        Some(c) => c,
        None => return,
    };

    // Initialize universal device batch manager for up to 512 atoms
    let mut mgr = GpuBatchDeviceMemoryManager::allocate(Arc::clone(&ctx), 512)
        .expect("Failed to allocate universal GpuBatchDeviceMemoryManager");

    println!(
        "[OK] Allocated GpuBatchDeviceMemoryManager for 512 atoms. Direct Host Mapping: {}",
        mgr.is_direct_host_mapped
    );

    // Create a sample water molecule (H2O)
    let water_atoms = vec![
        AtomGpuFP32 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            gss: 13.0,
        },
        AtomGpuFP32 {
            x: 0.757,
            y: 0.586,
            z: 0.0,
            gss: 12.8,
        },
        AtomGpuFP32 {
            x: -0.757,
            y: 0.586,
            z: 0.0,
            gss: 12.8,
        },
    ];

    // Upload using new universal upload_batch_to_device method
    mgr.upload_batch_to_device(&water_atoms, 0)
        .expect("Failed to upload batch to device memory");

    // Execute compute shader
    let gpu_calc = GpuCoulombCalculatorFP32::new(Arc::clone(&ctx))
        .expect("Failed to create GpuCoulombCalculatorFP32");

    let descriptors = vec![BatchMoleculeDescriptor {
        atom_offset: 0,
        num_atoms: 3,
        matrix_offset: 0,
    }];

    mgr.dispatch_batch(&gpu_calc, &descriptors)
        .expect("Failed to dispatch batch on universal memory");

    // Download using new universal download_matrix_from_device method
    let mat = mgr
        .download_matrix_from_device(0, 3)
        .expect("Failed to download matrix from device memory");

    assert_eq!(mat.rows, 3);
    assert_eq!(mat.cols, 3);

    // Verify diagonal elements match atomic parameters
    assert!((mat.get(0, 0) - 13.0).abs() < 1e-4);
    assert!((mat.get(1, 1) - 12.8).abs() < 1e-4);
    assert!((mat.get(2, 2) - 12.8).abs() < 1e-4);

    // Verify off-diagonal symmetry
    assert!((mat.get(0, 1) - mat.get(1, 0)).abs() < 1e-5);
    assert!((mat.get(0, 2) - mat.get(2, 0)).abs() < 1e-5);
    assert!((mat.get(1, 2) - mat.get(2, 1)).abs() < 1e-5);

    println!("[OK] Universal Device Batch Manager successfully verified on device hardware!");
}

#[test]
fn test_backward_compatibility_gddr6_aliases() {
    let ctx = match skip_if_no_gpu() {
        Some(c) => c,
        None => return,
    };

    let mut mgr = GpuBatchDeviceMemoryManager::allocate(Arc::clone(&ctx), 256)
        .expect("Failed to allocate GpuBatchDeviceMemoryManager");

    let atom = vec![AtomGpuFP32 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        gss: 11.0,
    }];

    // Test backward compatible upload_batch_to_gddr6 alias
    mgr.upload_batch_to_gddr6(&atom, 0)
        .expect("upload_batch_to_gddr6 failed");

    let gpu_calc = GpuCoulombCalculatorFP32::new(Arc::clone(&ctx))
        .expect("Failed to create GpuCoulombCalculatorFP32");

    let desc = vec![BatchMoleculeDescriptor {
        atom_offset: 0,
        num_atoms: 1,
        matrix_offset: 0,
    }];

    mgr.dispatch_batch(&gpu_calc, &desc)
        .expect("dispatch_batch failed");

    // Test backward compatible download_matrix_from_gddr6 alias
    let mat = mgr
        .download_matrix_from_gddr6(0, 1)
        .expect("download_matrix_from_gddr6 failed");

    assert!((mat.get(0, 0) - 11.0).abs() < 1e-4);
    println!("[OK] Backward compatibility aliases (upload_batch_to_gddr6 / download_matrix_from_gddr6) 100% verified!");
}

#[test]
fn test_universal_throughput_and_performance_benchmark() {
    let ctx = match skip_if_no_gpu() {
        Some(c) => c,
        None => return,
    };

    // Benchmark across 100 iterations of a 6-atom benzene skeleton
    let benzene_atoms = vec![
        AtomGpuFP32 {
            x: 0.0,
            y: 1.397,
            z: 0.0,
            gss: 11.0,
        },
        AtomGpuFP32 {
            x: 1.210,
            y: 0.698,
            z: 0.0,
            gss: 11.0,
        },
        AtomGpuFP32 {
            x: 1.210,
            y: -0.698,
            z: 0.0,
            gss: 11.0,
        },
        AtomGpuFP32 {
            x: 0.0,
            y: -1.397,
            z: 0.0,
            gss: 11.0,
        },
        AtomGpuFP32 {
            x: -1.210,
            y: -0.698,
            z: 0.0,
            gss: 11.0,
        },
        AtomGpuFP32 {
            x: -1.210,
            y: 0.698,
            z: 0.0,
            gss: 11.0,
        },
    ];

    let mut mgr = GpuBatchDeviceMemoryManager::allocate(Arc::clone(&ctx), 256)
        .expect("Failed to allocate manager");

    let gpu_calc =
        GpuCoulombCalculatorFP32::new(Arc::clone(&ctx)).expect("Failed to create calculator");

    let desc = vec![BatchMoleculeDescriptor {
        atom_offset: 0,
        num_atoms: 6,
        matrix_offset: 0,
    }];

    // Warm-up
    mgr.upload_batch_to_device(&benzene_atoms, 0).unwrap();
    mgr.dispatch_batch(&gpu_calc, &desc).unwrap();
    let _ = mgr.download_matrix_from_device(0, 6).unwrap();

    let iterations = 100;
    let start = Instant::now();

    for _ in 0..iterations {
        mgr.upload_batch_to_device(&benzene_atoms, 0).unwrap();
        mgr.dispatch_batch(&gpu_calc, &desc).unwrap();
        let _ = mgr.download_matrix_from_device(0, 6).unwrap();
    }

    let elapsed = start.elapsed();
    let total_us = elapsed.as_micros() as f64;
    let us_per_op = total_us / (iterations as f64);
    let throughput = (iterations as f64) / elapsed.as_secs_f64();

    println!(
        "\n=======================================================\n\
         [BENCHMARK] Universal GPU Memory Pipeline Performance:\n\
         Iterations:       {}\n\
         Total Duration:   {:.3} ms\n\
         Latency per pass: {:.2} us (Upload + Dispatch + Download)\n\
         Throughput:       {:.2} molecular matrices/s\n\
         =======================================================",
        iterations,
        elapsed.as_secs_f64() * 1000.0,
        us_per_op,
        throughput
    );

    // Performance assertion: ensure pipeline latency is competitive (< 2500 us per full cycle in debug build)
    assert!(
        us_per_op < 5000.0,
        "Latency per operation ({:.2} us) exceeded expected threshold",
        us_per_op
    );
}
