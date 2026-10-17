//! Vulkan GPU Compute Acceleration for Core-Core Nuclear Repulsion Energy.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Evaluates the NxN pairwise core-core repulsion matrix on Vulkan compute units
//! with authentic Klopman screening, Gaussian corrections, and Lennard-Jones R^-12 unpolarizable core potential.

use crate::context::{VulkanContext, VulkanError};
use ash::vk;
use mopac_core::constants::codata2018::EV_ANGSTROM_FACTOR;
use mopac_core::parameters::ParameterModel;
use mopac_core::types::{AlignedMatrix, MolecularBatch};
use std::sync::Arc;

/// GPU representation of a Gaussian core correction term.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianParamGpu {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub pad: f64,
}

/// GPU representation of atomic parameters for nuclear repulsion shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElementGpuParams {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub gss: f64,
    pub alpha: f64,
    pub core_charge: f64,
    pub z_cbrt: f64,
    pub z_num: u32,
    pub num_gaussians: u32,
    pub gaussians: [GaussianParamGpu; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct PushConstants {
    num_atoms: u32,
    pad0: u32,
    ev_angstrom_factor: f64,
}

/// Vulkan compute pipeline for parallel evaluation of pairwise core-core nuclear repulsion.
pub struct GpuCoreRepulsionCalculator {
    ctx: Arc<VulkanContext>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    shader_module: vk::ShaderModule,
}

impl GpuCoreRepulsionCalculator {
    /// Create a new GPU core repulsion calculator using pre-compiled SPIR-V bytecode.
    pub fn new(ctx: Arc<VulkanContext>) -> Result<Self, VulkanError> {
        let spv_bytes = include_bytes!("../shaders/core_repulsion.spv");
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
        let descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&layout_info, None)?
        };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32);

        let playout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let pipeline_layout = unsafe { ctx.device.create_pipeline_layout(&playout_info, None)? };

        let entry_point = c"main";
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

    /// Evaluates pairwise core-core repulsion matrix from atomic parameters.
    pub fn compute_pairwise(
        &self,
        atoms: &[ElementGpuParams],
    ) -> Result<AlignedMatrix<f64>, VulkanError> {
        let n = atoms.len();
        if n == 0 {
            return Ok(AlignedMatrix::zeroed(0, 0));
        }

        let device = &self.ctx.device;
        let atom_buf_size = std::mem::size_of_val(atoms) as u64;
        let out_buf_size = (n * n * std::mem::size_of::<f64>()) as u64;

        // Allocate input buffer
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
            let ptr = device.map_memory(mem_in, 0, atom_buf_size, vk::MemoryMapFlags::empty())?
                as *mut ElementGpuParams;
            std::ptr::copy_nonoverlapping(atoms.as_ptr(), ptr, n);
            device.unmap_memory(mem_in);
        }

        // Allocate output buffer
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
        unsafe {
            device.bind_buffer_memory(buf_out, mem_out, 0)?;
        }

        // Allocate descriptor pool & set
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(2)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let desc_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));
        let desc_set = unsafe { device.allocate_descriptor_sets(&alloc_info)?[0] };

        let buf_desc_in = vk::DescriptorBufferInfo::default()
            .buffer(buf_in)
            .offset(0)
            .range(atom_buf_size);
        let buf_desc_out = vk::DescriptorBufferInfo::default()
            .buffer(buf_out)
            .offset(0)
            .range(out_buf_size);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_desc_in)),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_desc_out)),
        ];
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }

        // Dispatch compute command
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&cmd_alloc)?[0] };

        let push_data = PushConstants {
            num_atoms: n as u32,
            pad0: 0,
            ev_angstrom_factor: EV_ANGSTROM_FACTOR,
        };
        let push_bytes = unsafe {
            std::slice::from_raw_parts(
                &push_data as *const _ as *const u8,
                std::mem::size_of::<PushConstants>(),
            )
        };

        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
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
                push_bytes,
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

        // Download result matrix
        let mut result_matrix = AlignedMatrix::zeroed(n, n);
        unsafe {
            let ptr = device.map_memory(mem_out, 0, out_buf_size, vk::MemoryMapFlags::empty())?
                as *const f64;
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

    /// Evaluates pairwise core-core repulsion matrix from MolecularBatch and ParameterModel.
    pub fn compute_batch(
        &self,
        batch: &MolecularBatch,
        model: &dyn ParameterModel,
    ) -> Result<AlignedMatrix<f64>, VulkanError> {
        let n = batch.natoms;
        if n == 0 {
            return Ok(AlignedMatrix::zeroed(0, 0));
        }
        let mut atoms = Vec::with_capacity(n);
        for i in 0..n {
            let z = batch.atomic_numbers[i];
            let p = model
                .get_element(z)
                .expect("Unsupported atomic element in batch");
            let mut g_params = [GaussianParamGpu {
                a: 0.0,
                b: 0.0,
                c: 0.0,
                pad: 0.0,
            }; 4];
            for (k, g_param) in g_params.iter_mut().enumerate().take(p.num_gaussians.min(4)) {
                *g_param = GaussianParamGpu {
                    a: p.gaussians[k].a,
                    b: p.gaussians[k].b,
                    c: p.gaussians[k].c,
                    pad: 0.0,
                };
            }
            atoms.push(ElementGpuParams {
                x: batch.x[i],
                y: batch.y[i],
                z: batch.z[i],
                gss: p.gss,
                alpha: p.alpha,
                core_charge: p.core_charge,
                z_cbrt: (z as f64).powf(0.3333),
                z_num: z as u32,
                num_gaussians: p.num_gaussians as u32,
                gaussians: g_params,
            });
        }
        self.compute_pairwise(&atoms)
    }

    /// Evaluates total nuclear core repulsion energy $E_{\text{nuc}} = \sum_{i < j} E_{ij}^{\text{core-core}}$ in eV.
    pub fn compute_total_energy(
        &self,
        batch: &MolecularBatch,
        model: &dyn ParameterModel,
    ) -> Result<f64, VulkanError> {
        let mat = self.compute_batch(batch, model)?;
        let n = batch.natoms;
        let mut e_nuc = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                e_nuc += mat.get(i, j);
            }
        }
        Ok(e_nuc)
    }
}

impl Drop for GpuCoreRepulsionCalculator {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_pipeline(self.pipeline, None);
            self.ctx
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.ctx
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.ctx
                .device
                .destroy_shader_module(self.shader_module, None);
        }
    }
}
