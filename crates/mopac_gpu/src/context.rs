//! High-Performance Vulkan Hardware Context for MOPAC_RS.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Manages direct Vulkan 1.3/1.4 instance initialization, device selection,
//! compute queue routing, and coherent memory management.

use ash::vk;
use std::ffi::CStr;

/// Error conditions during Vulkan operations.
#[derive(Debug)]
pub enum VulkanError {
    LoadingError(ash::LoadingError),
    VkError(vk::Result),
    NoComputeQueue,
    NoSuitableMemoryType,
    DeviceLost,
}

impl From<ash::LoadingError> for VulkanError {
    fn from(e: ash::LoadingError) -> Self {
        VulkanError::LoadingError(e)
    }
}

impl From<vk::Result> for VulkanError {
    fn from(e: vk::Result) -> Self {
        VulkanError::VkError(e)
    }
}

impl std::fmt::Display for VulkanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VulkanError::LoadingError(e) => write!(f, "Vulkan loading error: {:?}", e),
            VulkanError::VkError(e) => write!(f, "Vulkan API error: {:?}", e),
            VulkanError::NoComputeQueue => write!(f, "No compute-capable queue family found"),
            VulkanError::NoSuitableMemoryType => write!(f, "No suitable GPU memory type found"),
            VulkanError::DeviceLost => write!(f, "Vulkan device was lost"),
        }
    }
}

impl std::error::Error for VulkanError {}

/// Classification of the GPU physical memory architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMemoryArchitecture {
    /// Dedicated device-local VRAM or HBM without direct host mapping.
    /// Transfers use PCIe DMA copy via host-visible staging buffers.
    DiscreteStaged,
    /// Dedicated device-local VRAM or HBM with PCIe Resizable BAR (ReBAR / SAM).
    /// CPU can directly write to or map device-local memory without staging.
    DiscreteReBar,
    /// Unified / Heterogeneous shared memory architecture (e.g. NVIDIA Grace Hopper GH200,
    /// AMD Instinct MI300A, Apple Silicon M-series, or integrated APUs).
    UnifiedMemory,
}

impl std::fmt::Display for GpuMemoryArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DiscreteStaged => write!(f, "Discrete VRAM/HBM (DMA Staged)"),
            Self::DiscreteReBar => write!(f, "Discrete VRAM/HBM (PCIe Resizable BAR Zero-Copy)"),
            Self::UnifiedMemory => write!(f, "Unified System/Device Coherent Memory (UMA)"),
        }
    }
}

/// Comprehensive hardware memory properties detected for the Vulkan device.
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub architecture: GpuMemoryArchitecture,
    pub total_device_memory: u64,
    pub is_unified: bool,
    pub supports_rebar: bool,
    pub primary_device_heap_index: u32,
    pub heap_sizes: Vec<u64>,
}

/// Information regarding a detected Vulkan physical device.
#[derive(Debug, Clone)]
pub struct VulkanDeviceInfo {
    pub device_name: String,
    pub is_discrete: bool,
    pub api_version: (u32, u32, u32),
    pub supports_float64: bool,
    pub memory_info: GpuMemoryInfo,
}

/// Encapsulates a live Vulkan compute context with a dedicated compute queue.
pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub compute_queue: vk::Queue,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    pub device_info: VulkanDeviceInfo,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl VulkanContext {
    /// Initialize a new Vulkan compute context.
    ///
    /// Automatically prefers discrete GPUs (e.g. NVIDIA/AMD dedicated GPUs) over integrated graphics.
    pub fn new() -> Result<Self, VulkanError> {
        let entry = unsafe { ash::Entry::load()? };

        let app_name = c"mopac_rs";
        let engine_name = c"mopac_compute";

        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_3);

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&create_info, None)? };

        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        if physical_devices.is_empty() {
            return Err(VulkanError::NoComputeQueue);
        }

        // Select discrete GPU if available, otherwise take first device with compute support
        let mut selected_device = None;
        let mut selected_qf = None;

        for &pdev in &physical_devices {
            let props = unsafe { instance.get_physical_device_properties(pdev) };
            let qf_props = unsafe { instance.get_physical_device_queue_family_properties(pdev) };

            if let Some((qf_idx, _)) = qf_props
                .iter()
                .enumerate()
                .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            {
                let is_discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
                if is_discrete {
                    selected_device = Some((pdev, props, is_discrete));
                    selected_qf = Some(qf_idx as u32);
                    break; // Prefer discrete GPU immediately
                } else if selected_device.is_none() {
                    selected_device = Some((pdev, props, is_discrete));
                    selected_qf = Some(qf_idx as u32);
                }
            }
        }

        let (pdev, dev_props, is_discrete) = selected_device.ok_or(VulkanError::NoComputeQueue)?;
        let qf_index = selected_qf.ok_or(VulkanError::NoComputeQueue)?;

        let dev_name = unsafe {
            CStr::from_ptr(dev_props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        let features = unsafe { instance.get_physical_device_features(pdev) };
        let supports_float64 = features.shader_float64 == vk::TRUE;

        let memory_properties = unsafe { instance.get_physical_device_memory_properties(pdev) };

        // Analyze hardware memory architecture (HBM, UMA, ReBAR, GDDR)
        let mut total_device_memory = 0u64;
        let mut primary_device_heap_index = 0u32;
        let mut max_device_heap_size = 0u64;
        let mut heap_sizes = Vec::new();

        for (idx, heap) in memory_properties.memory_heaps
            [..memory_properties.memory_heap_count as usize]
            .iter()
            .enumerate()
        {
            heap_sizes.push(heap.size);
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                total_device_memory += heap.size;
                if heap.size > max_device_heap_size {
                    max_device_heap_size = heap.size;
                    primary_device_heap_index = idx as u32;
                }
            }
        }

        let mut supports_rebar = false;
        for mem_type in
            memory_properties.memory_types[..memory_properties.memory_type_count as usize].iter()
        {
            let flags = mem_type.property_flags;
            if flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                && flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                && flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            {
                supports_rebar = true;
                break;
            }
        }

        let is_unified = !is_discrete;
        let architecture = if is_unified {
            GpuMemoryArchitecture::UnifiedMemory
        } else if supports_rebar {
            GpuMemoryArchitecture::DiscreteReBar
        } else {
            GpuMemoryArchitecture::DiscreteStaged
        };

        let memory_info = GpuMemoryInfo {
            architecture,
            total_device_memory,
            is_unified,
            supports_rebar,
            primary_device_heap_index,
            heap_sizes,
        };

        let device_info = VulkanDeviceInfo {
            device_name: dev_name,
            is_discrete,
            api_version: (
                vk::api_version_major(dev_props.api_version),
                vk::api_version_minor(dev_props.api_version),
                vk::api_version_patch(dev_props.api_version),
            ),
            supports_float64,
            memory_info,
        };

        // Create logical device with compute queue and float64 support
        let priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qf_index)
            .queue_priorities(&priorities);

        let req_features = vk::PhysicalDeviceFeatures::default().shader_float64(supports_float64);
        let dev_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_features(&req_features);

        let device = unsafe { instance.create_device(pdev, &dev_create_info, None)? };
        let compute_queue = unsafe { device.get_device_queue(qf_index, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(qf_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None)? };

        Ok(Self {
            entry,
            instance,
            physical_device: pdev,
            device,
            compute_queue,
            queue_family_index: qf_index,
            command_pool,
            device_info,
            memory_properties,
        })
    }

    /// Finds a compatible memory type index matching the required filter and flags.
    pub fn find_memory_type(
        &self,
        type_filter: u32,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<u32, VulkanError> {
        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (self.memory_properties.memory_types[i as usize].property_flags & flags) == flags
            {
                return Ok(i);
            }
        }
        Err(VulkanError::NoSuitableMemoryType)
    }

    /// Finds the optimal device-local memory type, prioritizing high-speed device-local heaps (HBM/GDDR/LPDDR5X).
    ///
    /// If `prefer_direct_host_mapping` is true and PCIe Resizable BAR (ReBAR) or Unified Memory (UMA)
    /// is detected, it selects a memory type that is both `DEVICE_LOCAL` and `HOST_VISIBLE | HOST_COHERENT`.
    ///
    /// Returns a tuple `(memory_type_index, is_host_visible)`.
    pub fn find_optimal_device_memory_type(
        &self,
        type_filter: u32,
        prefer_direct_host_mapping: bool,
    ) -> Result<(u32, bool), VulkanError> {
        // 1. Direct host-accessible VRAM (ReBAR / UMA zero-copy)
        if prefer_direct_host_mapping {
            let direct_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT;
            for i in 0..self.memory_properties.memory_type_count {
                if (type_filter & (1 << i)) != 0
                    && (self.memory_properties.memory_types[i as usize].property_flags
                        & direct_flags)
                        == direct_flags
                    && self.memory_properties.memory_types[i as usize].heap_index
                        == self.device_info.memory_info.primary_device_heap_index
                {
                    return Ok((i, true));
                }
            }
        }

        // 2. Primary high-speed device-local heap (HBM / dedicated VRAM)
        let dev_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (self.memory_properties.memory_types[i as usize].property_flags & dev_flags)
                    == dev_flags
                && self.memory_properties.memory_types[i as usize].heap_index
                    == self.device_info.memory_info.primary_device_heap_index
            {
                let is_host_vis = (self.memory_properties.memory_types[i as usize].property_flags
                    & vk::MemoryPropertyFlags::HOST_VISIBLE)
                    == vk::MemoryPropertyFlags::HOST_VISIBLE;
                return Ok((i, is_host_vis));
            }
        }

        // 3. Fallback to any device-local memory type
        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (self.memory_properties.memory_types[i as usize].property_flags & dev_flags)
                    == dev_flags
            {
                let is_host_vis = (self.memory_properties.memory_types[i as usize].property_flags
                    & vk::MemoryPropertyFlags::HOST_VISIBLE)
                    == vk::MemoryPropertyFlags::HOST_VISIBLE;
                return Ok((i, is_host_vis));
            }
        }

        // 4. Fallback to standard find_memory_type
        let idx = self.find_memory_type(type_filter, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        Ok((idx, false))
    }

    /// Finds the optimal host-visible and host-coherent staging memory type.
    pub fn find_optimal_staging_memory_type(&self, type_filter: u32) -> Result<u32, VulkanError> {
        let staging_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let cached_flags = staging_flags | vk::MemoryPropertyFlags::HOST_CACHED;

        // Prefer cached memory for host downloads if available
        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (self.memory_properties.memory_types[i as usize].property_flags & cached_flags)
                    == cached_flags
            {
                return Ok(i);
            }
        }

        self.find_memory_type(type_filter, staging_flags)
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

unsafe impl Send for VulkanContext {}
unsafe impl Sync for VulkanContext {}
