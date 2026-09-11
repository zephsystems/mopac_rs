//! # `mopac_gpu`
//!
//! High-performance direct Vulkan compute engine for MOPAC_RS.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Provides hardware-accelerated two-center two-electron Coulomb repulsion,
//! core-core interaction, and batch molecular dispatch on discrete and integrated GPUs.

pub mod context;
pub mod coulomb;
pub mod coulomb_fp32;

pub use context::{VulkanContext, VulkanDeviceInfo, VulkanError};
pub use coulomb::{AtomGpu, GpuCoulombCalculator, GpuWorkspace};
pub use coulomb_fp32::{AtomGpuFP32, GpuBatchVramManager, GpuCoulombCalculatorFP32, GpuWorkspaceFP32};
