//! GPU-Accelerated Dewar-Klopman Two-Center Two-Electron Repulsion Engine.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Evaluates the NxN pairwise Coulomb repulsion matrix on Vulkan compute shader cores.

use ash::vk;
use std::ffi::CStr;
use std::sync::Arc;
use crate::context::{VulkanContext, VulkanError};
use mopac_core::constants::codata2018::EV_ANGSTROM_FACTOR;
use mopac_core::types::{AlignedMatrix, MolecularBatch};
use mopac_core::parameters::ParameterModel;

/// GPU representation of an atom center for Coulomb repulsion shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtomGpu {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub gss: f64,
}

/// Vulkan compute pipeline for parallel evaluation of two-center two-electron integrals.
pub struct GpuCoulombCalculator {
    ctx: Arc<VulkanContext>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    shader_module: vk::ShaderModule,
}

impl GpuCoulombCalculator {
    /// Create a new GPU Coulomb calculator using pre-compiled SPIR-V bytecode.
    pub fn new(ctx: Arc<VulkanContext>) -> Result<Self, VulkanError> {
        let spv_bytes = include_bytes!("../shaders/coulomb.spv");
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
            .size(16); // 2x u32 (8 bytes) + 1x f64 (8 bytes)

        let playout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let pipeline_layout = unsafe { ctx.device.create_pipeline_layout(&playout_info, None)? };

        let entry_point = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_point);

        let pipe_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipe_info], None)
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

    /// Computes the complete NxN pairwise two-electron repulsion matrix on GPU.
    pub fn compute_pairwise_coulomb(&self, atoms: &[AtomGpu]) -> Result<AlignedMatrix<f64>, VulkanError> {
        let n = atoms.len();
        if n == 0 {
            return Ok(AlignedMatrix::zeroed(0, 0));
        }

        let device = &self.ctx.device;
        let atom_buf_size = (n * std::mem::size_of::<AtomGpu>()) as u64;
        let out_buf_size = (n * n * std::mem::size_of::<f64>()) as u64;

        // 1. Allocate input buffer
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
            let ptr = device.map_memory(mem_in, 0, atom_buf_size, vk::MemoryMapFlags::empty())? as *mut AtomGpu;
            std::ptr::copy_nonoverlapping(atoms.as_ptr(), ptr, n);
            device.unmap_memory(mem_in);
        }

        // 2. Allocate output buffer
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

        let mut push_constants = [0u8; 16];
        push_constants[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push_constants[8..16].copy_from_slice(&EV_ANGSTROM_FACTOR.to_ne_bytes());

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
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push_constants,
            );

            let group_x = (n as u32 + 15) / 16;
            let group_y = (n as u32 + 15) / 16;
            device.cmd_dispatch(cmd, group_x, group_y, 1);
            device.end_command_buffer(cmd)?;

            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            device.queue_submit(self.ctx.compute_queue, &[submit_info], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
        }

        // 5. Read back results into AlignedMatrix
        let mut result_matrix = AlignedMatrix::zeroed(n, n);
        unsafe {
            let ptr = device.map_memory(mem_out, 0, out_buf_size, vk::MemoryMapFlags::empty())? as *const f64;
            std::ptr::copy_nonoverlapping(ptr, result_matrix.data.as_mut_ptr(), n * n);
            device.unmap_memory(mem_out);

            // Clean up temporary execution resources
            device.destroy_descriptor_pool(desc_pool, None);
            device.destroy_buffer(buf_in, None);
            device.free_memory(mem_in, None);
            device.destroy_buffer(buf_out, None);
            device.free_memory(mem_out, None);
        }

        Ok(result_matrix)
    }

    /// Evaluates the pairwise Coulomb matrix directly from a MolecularBatch.
    pub fn compute_batch(&self, batch: &MolecularBatch, model: &dyn ParameterModel) -> Result<AlignedMatrix<f64>, VulkanError> {
        let n = batch.natoms;
        let mut atoms = Vec::with_capacity(n);
        for i in 0..n {
            let z = batch.atomic_numbers[i];
            let param = model.get_element(z).expect("Unsupported atomic element in batch");
            atoms.push(AtomGpu {
                x: batch.x[i],
                y: batch.y[i],
                z: batch.z[i],
                gss: param.gss,
            });
        }
        self.compute_pairwise_coulomb(&atoms)
    }

    /// Allocate a pre-allocated GPU workspace capable of evaluating up to `max_atoms` atoms
    /// with zero dynamic heap allocations during iterative dispatches.
    pub fn allocate_workspace(&self, max_atoms: usize) -> Result<GpuWorkspace, VulkanError> {
        let device = &self.ctx.device;
        let atom_buf_size = (max_atoms * std::mem::size_of::<AtomGpu>()) as u64;
        let out_buf_size = (max_atoms * max_atoms * std::mem::size_of::<f64>()) as u64;

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
        let mapped_in = unsafe {
            device.bind_buffer_memory(buf_in, mem_in, 0)?;
            device.map_memory(mem_in, 0, atom_buf_size, vk::MemoryMapFlags::empty())? as *mut AtomGpu
        };

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
        let mapped_out = unsafe {
            device.bind_buffer_memory(buf_out, mem_out, 0)?;
            device.map_memory(mem_out, 0, out_buf_size, vk::MemoryMapFlags::empty())? as *const f64
        };

        // 3. Descriptor pool & set
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

        // 4. Command buffer & Fence
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&cmd_alloc)?[0] };
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };

        Ok(GpuWorkspace {
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

    /// Computes the pairwise Coulomb matrix in-place into `out` with **ZERO dynamic allocations**.
    pub fn compute_pairwise_in_workspace(
        &self,
        atoms: &[AtomGpu],
        ws: &mut GpuWorkspace,
        out: &mut AlignedMatrix<f64>,
    ) -> Result<(), VulkanError> {
        let n = atoms.len();
        assert!(n <= ws.max_atoms, "System atom count {} exceeds pre-allocated GPU workspace {}", n, ws.max_atoms);
        assert_eq!(out.rows, n);
        assert_eq!(out.cols, n);

        let device = &self.ctx.device;

        // 1. Direct copy into pre-mapped host-coherent GPU buffer (0 allocation)
        unsafe {
            std::ptr::copy_nonoverlapping(atoms.as_ptr(), ws.mapped_in, n);
        }

        // 2. Record and submit compute commands
        let mut push_constants = [0u8; 16];
        push_constants[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push_constants[8..16].copy_from_slice(&EV_ANGSTROM_FACTOR.to_ne_bytes());

        unsafe {
            device.reset_fences(&[ws.fence])?;
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
            device.cmd_push_constants(
                ws.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push_constants,
            );

            let group_x = (n as u32 + 15) / 16;
            let group_y = (n as u32 + 15) / 16;
            device.cmd_dispatch(ws.command_buffer, group_x, group_y, 1);
            device.end_command_buffer(ws.command_buffer)?;

            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&ws.command_buffer));
            device.queue_submit(self.ctx.compute_queue, &[submit_info], ws.fence)?;
            device.wait_for_fences(&[ws.fence], true, u64::MAX)?;

            // 3. Copy back from pre-mapped GPU output into AlignedMatrix (0 allocation)
            std::ptr::copy_nonoverlapping(ws.mapped_out, out.data.as_mut_ptr(), n * n);
        }

        Ok(())
    }
}

/// Pre-allocated GPU scratch workspace for zero-allocation iterative compute dispatches.
pub struct GpuWorkspace {
    ctx: Arc<VulkanContext>,
    pub max_atoms: usize,
    buf_in: vk::Buffer,
    mem_in: vk::DeviceMemory,
    pub mapped_in: *mut AtomGpu,
    buf_out: vk::Buffer,
    mem_out: vk::DeviceMemory,
    pub mapped_out: *const f64,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
}

unsafe impl Send for GpuWorkspace {}
unsafe impl Sync for GpuWorkspace {}

impl Drop for GpuWorkspace {
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

impl Drop for GpuCoulombCalculator {
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
