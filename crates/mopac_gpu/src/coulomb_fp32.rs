//! High-Throughput FP32 GPU-Accelerated Dewar-Klopman Coulomb Repulsion Engine.
//!
//! Evaluates the NxN pairwise Coulomb repulsion matrix on Vulkan compute cores in FP32,
//! unlocking the peak 18 TFLOPS capability of consumer NVIDIA Ada Lovelace / Ampere GPUs
//! and utilizing dedicated GDDR6 device-local VRAM for molecular batching.

use ash::vk;
use std::sync::Arc;
use crate::context::{VulkanContext, VulkanError};
use mopac_core::constants::codata2018::EV_ANGSTROM_FACTOR;
use mopac_core::types::{AlignedMatrix, MolecularBatch};
use mopac_core::parameters::ParameterModel;

/// GPU representation of an atom center in 32-bit single precision.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtomGpuFP32 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub gss: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct PushConstantsFP32 {
    num_atoms: u32,
    batch_offset: u32,
    ev_angstrom_factor: f32,
    pad0: f32,
}

/// Vulkan FP32 compute pipeline for high-throughput evaluation of Coulomb repulsion matrices.
pub struct GpuCoulombCalculatorFP32 {
    ctx: Arc<VulkanContext>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    shader_module: vk::ShaderModule,
}

impl GpuCoulombCalculatorFP32 {
    /// Create a new FP32 GPU Coulomb calculator using pre-compiled SPIR-V bytecode.
    pub fn new(ctx: Arc<VulkanContext>) -> Result<Self, VulkanError> {
        let spv_bytes = include_bytes!("../shaders/coulomb_fp32.spv");
        let spv_words = ash::util::read_spv(&mut std::io::Cursor::new(spv_bytes))
            .map_err(|_| VulkanError::NoComputeQueue)?;

        let shader_info = vk::ShaderModuleCreateInfo::default().code(&spv_words);
        let shader_module = unsafe { ctx.device.create_shader_module(&shader_info, None)? };

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe { ctx.device.create_descriptor_set_layout(&layout_info, None)? };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<PushConstantsFP32>() as u32);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let pipeline_layout = unsafe { ctx.device.create_pipeline_layout(&pipeline_layout_info, None)? };

        let entry_point = c"main";
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| err)?[0]
        };

        Ok(Self {
            ctx,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            shader_module,
        })
    }

    /// Evaluates pairwise Coulomb repulsion matrix in FP32 from a slice of atom coordinates and parameters.
    pub fn compute_pairwise(&self, atoms: &[AtomGpuFP32]) -> Result<AlignedMatrix<f32>, VulkanError> {
        let n = atoms.len();
        let atom_buf_size = std::mem::size_of_val(atoms) as u64;
        let out_buf_size = (n * n * std::mem::size_of::<f32>()) as u64;
        let device = &self.ctx.device;

        // 1. In buffer
        let buf_info_in = vk::BufferCreateInfo::default()
            .size(atom_buf_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
        let buf_in = unsafe { device.create_buffer(&buf_info_in, None)? };
        let req_in = unsafe { device.get_buffer_memory_requirements(buf_in) };
        let mem_type_in = self.ctx.find_memory_type(
            req_in.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let mem_in = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req_in.size)
                    .memory_type_index(mem_type_in),
                None,
            )?
        };
        unsafe {
            device.bind_buffer_memory(buf_in, mem_in, 0)?;
            let ptr = device.map_memory(mem_in, 0, atom_buf_size, vk::MemoryMapFlags::empty())? as *mut AtomGpuFP32;
            std::ptr::copy_nonoverlapping(atoms.as_ptr(), ptr, n);
            device.unmap_memory(mem_in);
        }

        // 2. Out buffer
        let buf_info_out = vk::BufferCreateInfo::default()
            .size(out_buf_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
        let buf_out = unsafe { device.create_buffer(&buf_info_out, None)? };
        let req_out = unsafe { device.get_buffer_memory_requirements(buf_out) };
        let mem_type_out = self.ctx.find_memory_type(
            req_out.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let mem_out = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req_out.size)
                    .memory_type_index(mem_type_out),
                None,
            )?
        };
        unsafe { device.bind_buffer_memory(buf_out, mem_out, 0)? };

        // 3. Descriptor set
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(2);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(std::slice::from_ref(&pool_size))
            .max_sets(1);
        let desc_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));
        let desc_set = unsafe { device.allocate_descriptor_sets(&alloc_info)?[0] };

        let d_buf0 = vk::DescriptorBufferInfo::default().buffer(buf_in).offset(0).range(atom_buf_size);
        let d_buf1 = vk::DescriptorBufferInfo::default().buffer(buf_out).offset(0).range(out_buf_size);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&d_buf0)),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&d_buf1)),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        // 4. Command buffer dispatch
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&cmd_alloc)?[0] };

        let pc = PushConstantsFP32 {
            num_atoms: n as u32,
            batch_offset: 0,
            ev_angstrom_factor: EV_ANGSTROM_FACTOR as f32,
            pad0: 0.0,
        };

        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[desc_set],
                &[],
            );
            let pc_bytes = std::slice::from_raw_parts(
                &pc as *const _ as *const u8,
                std::mem::size_of::<PushConstantsFP32>(),
            );
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );

            let group_x = (n as u32).div_ceil(16);
            let group_y = (n as u32).div_ceil(16);
            device.cmd_dispatch(cmd, group_x, group_y, 1);
            device.end_command_buffer(cmd)?;

            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            device.queue_submit(self.ctx.compute_queue, &[submit_info], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
        }

        // 5. Read back results into AlignedMatrix<f32>
        let mut result_matrix = AlignedMatrix::zeroed(n, n);
        unsafe {
            let ptr = device.map_memory(mem_out, 0, out_buf_size, vk::MemoryMapFlags::empty())? as *const f32;
            std::ptr::copy_nonoverlapping(ptr, result_matrix.data.as_mut_ptr(), n * n);
            device.unmap_memory(mem_out);

            device.destroy_descriptor_pool(desc_pool, None);
            device.destroy_buffer(buf_in, None);
            device.free_memory(mem_in, None);
            device.destroy_buffer(buf_out, None);
            device.free_memory(mem_out, None);
        }

        Ok(result_matrix)
    }

    /// Evaluates the pairwise Coulomb matrix directly from a MolecularBatch in FP32.
    pub fn compute_batch(&self, batch: &MolecularBatch, model: &dyn ParameterModel) -> Result<AlignedMatrix<f32>, VulkanError> {
        let n = batch.natoms;
        let mut atoms = Vec::with_capacity(n);
        for i in 0..n {
            let z = batch.atomic_numbers[i];
            let param = model.get_element(z).expect("Unsupported atomic element in batch");
            atoms.push(AtomGpuFP32 {
                x: batch.x[i] as f32,
                y: batch.y[i] as f32,
                z: batch.z[i] as f32,
                gss: param.gss as f32,
            });
        }
        self.compute_pairwise(&atoms)
    }

    /// Allocates a pre-allocated FP32 scratch workspace for zero-allocation SCF loops.
    pub fn allocate_workspace(&self, max_atoms: usize) -> Result<GpuWorkspaceFP32, VulkanError> {
        let atom_buf_size = (max_atoms * std::mem::size_of::<AtomGpuFP32>()) as u64;
        let out_buf_size = (max_atoms * max_atoms * std::mem::size_of::<f32>()) as u64;
        let device = &self.ctx.device;

        let buf_info_in = vk::BufferCreateInfo::default()
            .size(atom_buf_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
        let buf_in = unsafe { device.create_buffer(&buf_info_in, None)? };
        let req_in = unsafe { device.get_buffer_memory_requirements(buf_in) };
        let mem_type_in = self.ctx.find_memory_type(
            req_in.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let mem_in = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req_in.size)
                    .memory_type_index(mem_type_in),
                None,
            )?
        };
        let mapped_in = unsafe {
            device.bind_buffer_memory(buf_in, mem_in, 0)?;
            device.map_memory(mem_in, 0, atom_buf_size, vk::MemoryMapFlags::empty())? as *mut AtomGpuFP32
        };

        let buf_info_out = vk::BufferCreateInfo::default()
            .size(out_buf_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
        let buf_out = unsafe { device.create_buffer(&buf_info_out, None)? };
        let req_out = unsafe { device.get_buffer_memory_requirements(buf_out) };
        let mem_type_out = self.ctx.find_memory_type(
            req_out.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let mem_out = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req_out.size)
                    .memory_type_index(mem_type_out),
                None,
            )?
        };
        let mapped_out = unsafe {
            device.bind_buffer_memory(buf_out, mem_out, 0)?;
            device.map_memory(mem_out, 0, out_buf_size, vk::MemoryMapFlags::empty())? as *const f32
        };

        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(2);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(std::slice::from_ref(&pool_size))
            .max_sets(1);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));
        let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info)?[0] };

        let d_buf0 = vk::DescriptorBufferInfo::default().buffer(buf_in).offset(0).range(atom_buf_size);
        let d_buf1 = vk::DescriptorBufferInfo::default().buffer(buf_out).offset(0).range(out_buf_size);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&d_buf0)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&d_buf1)),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&cmd_alloc)?[0] };
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };

        Ok(GpuWorkspaceFP32 {
            ctx: Arc::clone(&self.ctx),
            max_atoms,
            buf_in,
            mem_in,
            mapped_in,
            buf_out,
            mem_out,
            mapped_out,
            descriptor_pool,
            descriptor_set,
            fence,
            command_buffer,
        })
    }

    /// Computes the pairwise Coulomb matrix in-place into `out` with **ZERO dynamic allocations** in FP32.
    pub fn compute_pairwise_in_workspace(
        &self,
        atoms: &[AtomGpuFP32],
        ws: &mut GpuWorkspaceFP32,
        out: &mut AlignedMatrix<f32>,
    ) -> Result<(), VulkanError> {
        let n = atoms.len();
        assert!(n <= ws.max_atoms, "Atom count {} exceeds max workspace capacity {}", n, ws.max_atoms);
        assert_eq!(out.rows, n);
        assert_eq!(out.cols, n);

        let device = &self.ctx.device;

        unsafe {
            // 1. Copy atoms to pre-mapped GPU input buffer (0 allocation)
            std::ptr::copy_nonoverlapping(atoms.as_ptr(), ws.mapped_in, n);

            // 2. Record and dispatch compute command buffer
            device.reset_command_buffer(ws.command_buffer, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                ws.command_buffer,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_bind_pipeline(ws.command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                ws.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[ws.descriptor_set],
                &[],
            );

            let pc = PushConstantsFP32 {
                num_atoms: n as u32,
                batch_offset: 0,
                ev_angstrom_factor: EV_ANGSTROM_FACTOR as f32,
                pad0: 0.0,
            };
            let pc_bytes = std::slice::from_raw_parts(
                &pc as *const _ as *const u8,
                std::mem::size_of::<PushConstantsFP32>(),
            );
            device.cmd_push_constants(
                ws.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );

            let group_x = (n as u32).div_ceil(16);
            let group_y = (n as u32).div_ceil(16);
            device.cmd_dispatch(ws.command_buffer, group_x, group_y, 1);
            device.end_command_buffer(ws.command_buffer)?;

            device.reset_fences(&[ws.fence])?;
            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&ws.command_buffer));
            device.queue_submit(self.ctx.compute_queue, &[submit_info], ws.fence)?;
            device.wait_for_fences(&[ws.fence], true, u64::MAX)?;

            // 3. Copy back from pre-mapped GPU output into AlignedMatrix (0 allocation)
            std::ptr::copy_nonoverlapping(ws.mapped_out, out.data.as_mut_ptr(), n * n);
        }

        Ok(())
    }
}

/// Pre-allocated FP32 GPU scratch workspace for zero-allocation iterative compute dispatches.
pub struct GpuWorkspaceFP32 {
    ctx: Arc<VulkanContext>,
    pub max_atoms: usize,
    buf_in: vk::Buffer,
    mem_in: vk::DeviceMemory,
    pub mapped_in: *mut AtomGpuFP32,
    buf_out: vk::Buffer,
    mem_out: vk::DeviceMemory,
    pub mapped_out: *const f32,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
}

unsafe impl Send for GpuWorkspaceFP32 {}
unsafe impl Sync for GpuWorkspaceFP32 {}

impl Drop for GpuWorkspaceFP32 {
    fn drop(&mut self) {
        unsafe {
            let device = &self.ctx.device;
            let _ = device.device_wait_idle();
            device.destroy_fence(self.fence, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.unmap_memory(self.mem_in);
            device.unmap_memory(self.mem_out);
            device.destroy_buffer(self.buf_in, None);
            device.free_memory(self.mem_in, None);
            device.destroy_buffer(self.buf_out, None);
            device.free_memory(self.mem_out, None);
        }
    }
}

impl Drop for GpuCoulombCalculatorFP32 {
    fn drop(&mut self) {
        unsafe {
            let device = &self.ctx.device;
            let _ = device.device_wait_idle();
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_shader_module(self.shader_module, None);
        }
    }
}

/// Dedicated GDDR6 Device-Local VRAM Memory & Batch Manager.
///
/// Stores multiple molecular systems or large coordinate sets directly in high-speed
/// VRAM (192 GB/s on RTX 4050 GDDR6), avoiding per-iteration PCIe bus transfers.
pub struct GpuBatchVramManager {
    ctx: Arc<VulkanContext>,
    pub vram_capacity_atoms: usize,
    vram_atom_buf: vk::Buffer,
    vram_atom_mem: vk::DeviceMemory,
    vram_matrix_buf: vk::Buffer,
    vram_matrix_mem: vk::DeviceMemory,
    staging_buf: vk::Buffer,
    staging_mem: vk::DeviceMemory,
    pub mapped_staging: *mut AtomGpuFP32,
}

impl GpuBatchVramManager {
    /// Allocate dedicated GDDR6 VRAM buffers for storing up to `vram_capacity_atoms`.
    pub fn allocate(ctx: Arc<VulkanContext>, vram_capacity_atoms: usize) -> Result<Self, VulkanError> {
        let device = &ctx.device;
        let atom_size = (vram_capacity_atoms * std::mem::size_of::<AtomGpuFP32>()) as u64;
        let matrix_size = (vram_capacity_atoms * vram_capacity_atoms * std::mem::size_of::<f32>()) as u64;

        // 1. Device-Local (GDDR6 VRAM) atom buffer
        let vram_atom_info = vk::BufferCreateInfo::default()
            .size(atom_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
        let vram_atom_buf = unsafe { device.create_buffer(&vram_atom_info, None)? };
        let req_atom = unsafe { device.get_buffer_memory_requirements(vram_atom_buf) };
        let mem_type_vram = ctx.find_memory_type(
            req_atom.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let vram_atom_mem = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req_atom.size)
                    .memory_type_index(mem_type_vram),
                None,
            )?
        };
        unsafe { device.bind_buffer_memory(vram_atom_buf, vram_atom_mem, 0)? };

        // 2. Device-Local (GDDR6 VRAM) matrix output buffer
        let vram_mat_info = vk::BufferCreateInfo::default()
            .size(matrix_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC);
        let vram_matrix_buf = unsafe { device.create_buffer(&vram_mat_info, None)? };
        let req_mat = unsafe { device.get_buffer_memory_requirements(vram_matrix_buf) };
        let vram_matrix_mem = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req_mat.size)
                    .memory_type_index(mem_type_vram),
                None,
            )?
        };
        unsafe { device.bind_buffer_memory(vram_matrix_buf, vram_matrix_mem, 0)? };

        // 3. Host-visible staging buffer for PCIe upload to GDDR6
        let staging_info = vk::BufferCreateInfo::default()
            .size(atom_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC);
        let staging_buf = unsafe { device.create_buffer(&staging_info, None)? };
        let req_staging = unsafe { device.get_buffer_memory_requirements(staging_buf) };
        let mem_type_staging = ctx.find_memory_type(
            req_staging.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let staging_mem = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req_staging.size)
                    .memory_type_index(mem_type_staging),
                None,
            )?
        };
        let mapped_staging = unsafe {
            device.bind_buffer_memory(staging_buf, staging_mem, 0)?;
            device.map_memory(staging_mem, 0, atom_size, vk::MemoryMapFlags::empty())? as *mut AtomGpuFP32
        };

        Ok(Self {
            ctx,
            vram_capacity_atoms,
            vram_atom_buf,
            vram_atom_mem,
            vram_matrix_buf,
            vram_matrix_mem,
            staging_buf,
            staging_mem,
            mapped_staging,
        })
    }

    /// Upload a molecular batch to GDDR6 VRAM via DMA copy.
    pub fn upload_batch_to_gddr6(&mut self, atoms: &[AtomGpuFP32], vram_offset_atoms: usize) -> Result<(), VulkanError> {
        let count = atoms.len();
        assert!(vram_offset_atoms + count <= self.vram_capacity_atoms);
        let byte_size = std::mem::size_of_val(atoms) as u64;
        let dst_offset = (vram_offset_atoms * std::mem::size_of::<AtomGpuFP32>()) as u64;

        let device = &self.ctx.device;
        unsafe {
            // Write to host-visible staging buffer
            std::ptr::copy_nonoverlapping(atoms.as_ptr(), self.mapped_staging, count);

            // Execute DMA copy on compute/transfer queue
            let cmd_alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.ctx.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = device.allocate_command_buffers(&cmd_alloc)?[0];

            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            let copy_region = vk::BufferCopy::default().src_offset(0).dst_offset(dst_offset).size(byte_size);
            device.cmd_copy_buffer(cmd, self.staging_buf, self.vram_atom_buf, &[copy_region]);
            device.end_command_buffer(cmd)?;

            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmd_slice = [cmd];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_slice);
            device.queue_submit(self.ctx.compute_queue, &[submit_info], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.ctx.command_pool, &[cmd]);
        }

        Ok(())
    }
}

unsafe impl Send for GpuBatchVramManager {}
unsafe impl Sync for GpuBatchVramManager {}

impl Drop for GpuBatchVramManager {
    fn drop(&mut self) {
        unsafe {
            let device = &self.ctx.device;
            let _ = device.device_wait_idle();
            device.unmap_memory(self.staging_mem);
            device.destroy_buffer(self.staging_buf, None);
            device.free_memory(self.staging_mem, None);
            device.destroy_buffer(self.vram_atom_buf, None);
            device.free_memory(self.vram_atom_mem, None);
            device.destroy_buffer(self.vram_matrix_buf, None);
            device.free_memory(self.vram_matrix_mem, None);
        }
    }
}
