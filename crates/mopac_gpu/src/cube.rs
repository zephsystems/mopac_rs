//! Vulkan Compute Acceleration for Gaussian Cube Volumetric Field Generation.
//!
//! Generates 3D regular scalar grids of Molecular Orbitals (HOMO, LUMO)
//! and total electron density rho(r) on GPU compute units using pre-compiled SPIR-V shaders.

use crate::context::{VulkanContext, VulkanError};
use ash::vk;
use mopac_core::export::cube::{CubeGridConfig, ANGSTROM_TO_BOHR};
use mopac_core::parameters::ParameterModel;
use mopac_core::types::{AlignedMatrix, BasisType, MolecularBatch};
use std::f64::consts::PI;
use std::sync::Arc;

/// GPU representation of an atom for volumetric compute shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtomDataGpu {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub zs: f32,
    pub zp: f32,
    pub norm_s: f32,
    pub norm_p: f32,
    pub z_num: u32,
    pub offset: u32,
    pub basis_type: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CubePushConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    natoms: u32,
    norbs: u32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    step_bohr: f32,
}

#[inline(always)]
fn compute_sto_norms_f32(z: u8, zs: f64, zp: f64) -> (f32, f32) {
    if z == 1 {
        let norm_s = (zs * zs * zs / PI).sqrt();
        (norm_s as f32, 0.0f32)
    } else if z <= 10 {
        let norm_s = (zs.powi(5) / (3.0 * PI)).sqrt();
        let norm_p = (zp.powi(5) / PI).sqrt();
        (norm_s as f32, norm_p as f32)
    } else {
        let norm_s = (2.0 * zs.powi(7) / (45.0 * PI)).sqrt();
        let norm_p = (2.0 * zp.powi(7) / (15.0 * PI)).sqrt();
        (norm_s as f32, norm_p as f32)
    }
}

/// Vulkan-accelerated regular 3D grid evaluator for Gaussian Cube fields.
pub struct GpuCubeEvaluator {
    ctx: Arc<VulkanContext>,
    orbital_pipeline: vk::Pipeline,
    orbital_pipeline_layout: vk::PipelineLayout,
    orbital_desc_layout: vk::DescriptorSetLayout,
    orbital_shader_module: vk::ShaderModule,
    density_pipeline: vk::Pipeline,
    density_pipeline_layout: vk::PipelineLayout,
    density_desc_layout: vk::DescriptorSetLayout,
    density_shader_module: vk::ShaderModule,
}

impl GpuCubeEvaluator {
    /// Initialize GPU Gaussian Cube compute pipelines.
    pub fn new(ctx: Arc<VulkanContext>) -> Result<Self, VulkanError> {
        let device = &ctx.device;

        // 1. Orbital Shader Pipeline
        let orb_spv_bytes = include_bytes!("../shaders/orbital_grid.spv");
        let orb_spv_words = ash::util::read_spv(&mut std::io::Cursor::new(orb_spv_bytes))
            .map_err(|_| VulkanError::NoComputeQueue)?;
        let orb_sm_info = vk::ShaderModuleCreateInfo::default().code(&orb_spv_words);
        let orbital_shader_module = unsafe { device.create_shader_module(&orb_sm_info, None)? };

        let bindings = [
            // Binding 0: Atoms buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 1: MO Coeffs buffer or Density Matrix buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 2: Output Grid buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let d_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let orbital_desc_layout =
            unsafe { device.create_descriptor_set_layout(&d_layout_info, None)? };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<CubePushConstants>() as u32);

        let playout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&orbital_desc_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let orbital_pipeline_layout =
            unsafe { device.create_pipeline_layout(&playout_info, None)? };

        let entry_point = c"main";
        let orb_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(orbital_shader_module)
            .name(entry_point);

        let orb_pipe_info = vk::ComputePipelineCreateInfo::default()
            .stage(orb_stage_info)
            .layout(orbital_pipeline_layout);

        let orbital_pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[orb_pipe_info], None)
                .map_err(|(_, err)| err)?[0]
        };

        // 2. Density Shader Pipeline
        let dens_spv_bytes = include_bytes!("../shaders/density_grid.spv");
        let dens_spv_words = ash::util::read_spv(&mut std::io::Cursor::new(dens_spv_bytes))
            .map_err(|_| VulkanError::NoComputeQueue)?;
        let dens_sm_info = vk::ShaderModuleCreateInfo::default().code(&dens_spv_words);
        let density_shader_module = unsafe { device.create_shader_module(&dens_sm_info, None)? };

        let density_desc_layout =
            unsafe { device.create_descriptor_set_layout(&d_layout_info, None)? };
        let dens_playout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&density_desc_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let density_pipeline_layout =
            unsafe { device.create_pipeline_layout(&dens_playout_info, None)? };

        let dens_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(density_shader_module)
            .name(entry_point);

        let dens_pipe_info = vk::ComputePipelineCreateInfo::default()
            .stage(dens_stage_info)
            .layout(density_pipeline_layout);

        let density_pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[dens_pipe_info], None)
                .map_err(|(_, err)| err)?[0]
        };

        Ok(Self {
            ctx,
            orbital_pipeline,
            orbital_pipeline_layout,
            orbital_desc_layout,
            orbital_shader_module,
            density_pipeline,
            density_pipeline_layout,
            density_desc_layout,
            density_shader_module,
        })
    }

    fn prepare_gpu_atoms(
        &self,
        batch: &MolecularBatch,
        model: &dyn ParameterModel,
    ) -> Vec<AtomDataGpu> {
        let mut atoms = Vec::with_capacity(batch.natoms);
        for a in 0..batch.natoms {
            let z = batch.atomic_numbers[a];
            let p = model
                .get_element(z)
                .expect("Element params missing in model");
            let ax = (batch.x[a] * ANGSTROM_TO_BOHR) as f32;
            let ay = (batch.y[a] * ANGSTROM_TO_BOHR) as f32;
            let az = (batch.z[a] * ANGSTROM_TO_BOHR) as f32;
            let (norm_s, norm_p) = compute_sto_norms_f32(z, p.zs, p.zp);

            let basis_type = match batch.basis_types[a] {
                BasisType::S => 0,
                BasisType::SP => 1,
                BasisType::SPD => 2,
            };

            atoms.push(AtomDataGpu {
                x: ax,
                y: ay,
                z: az,
                zs: p.zs as f32,
                zp: p.zp as f32,
                norm_s,
                norm_p,
                z_num: z as u32,
                offset: batch.orbital_offsets[a] as u32,
                basis_type,
            });
        }
        atoms
    }

    /// Computes the 3D scalar grid for a Molecular Orbital on GPU.
    pub fn compute_orbital_grid(
        &self,
        batch: &MolecularBatch,
        model: &dyn ParameterModel,
        mo_coefficients: &[f64],
        origin_bohr: [f64; 3],
        step_bohr: f64,
        dimensions: (usize, usize, usize),
    ) -> Result<Vec<f64>, VulkanError> {
        let (nx, ny, nz) = dimensions;
        let total_voxels = nx * ny * nz;
        if total_voxels == 0 || batch.natoms == 0 {
            return Ok(vec![0.0; total_voxels]);
        }

        let atoms = self.prepare_gpu_atoms(batch, model);
        let mo_coeffs_f32: Vec<f32> = mo_coefficients.iter().map(|&v| v as f32).collect();

        let device = &self.ctx.device;
        let atom_buf_size = (atoms.len() * std::mem::size_of::<AtomDataGpu>()) as u64;
        let mo_buf_size = (mo_coeffs_f32.len() * std::mem::size_of::<f32>()) as u64;
        let out_buf_size = (total_voxels * std::mem::size_of::<f32>()) as u64;

        // 1. Input Atom Buffer
        let buf_info_atom = vk::BufferCreateInfo::default()
            .size(atom_buf_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
        let buf_atom = unsafe { device.create_buffer(&buf_info_atom, None)? };
        let req_atom = unsafe { device.get_buffer_memory_requirements(buf_atom) };
        let mem_type_atom = self.ctx.find_memory_type(
            req_atom.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let mem_atom = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req_atom.size)
                    .memory_type_index(mem_type_atom),
                None,
            )?
        };
        unsafe {
            device.bind_buffer_memory(buf_atom, mem_atom, 0)?;
            let ptr = device.map_memory(mem_atom, 0, atom_buf_size, vk::MemoryMapFlags::empty())?
                as *mut AtomDataGpu;
            std::ptr::copy_nonoverlapping(atoms.as_ptr(), ptr, atoms.len());
            device.unmap_memory(mem_atom);
        }

        // 2. Input MO Coeffs Buffer
        let buf_info_mo = vk::BufferCreateInfo::default()
            .size(mo_buf_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
        let buf_mo = unsafe { device.create_buffer(&buf_info_mo, None)? };
        let req_mo = unsafe { device.get_buffer_memory_requirements(buf_mo) };
        let mem_type_mo = self.ctx.find_memory_type(
            req_mo.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let mem_mo = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req_mo.size)
                    .memory_type_index(mem_type_mo),
                None,
            )?
        };
        unsafe {
            device.bind_buffer_memory(buf_mo, mem_mo, 0)?;
            let ptr =
                device.map_memory(mem_mo, 0, mo_buf_size, vk::MemoryMapFlags::empty())? as *mut f32;
            std::ptr::copy_nonoverlapping(mo_coeffs_f32.as_ptr(), ptr, mo_coeffs_f32.len());
            device.unmap_memory(mem_mo);
        }

        // 3. Output Grid Buffer
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

        // 4. Descriptor Set
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(std::slice::from_ref(&pool_size))
            .max_sets(1);
        let desc_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(std::slice::from_ref(&self.orbital_desc_layout));
        let desc_set = unsafe { device.allocate_descriptor_sets(&alloc_info)?[0] };

        let d_buf0 = vk::DescriptorBufferInfo::default()
            .buffer(buf_atom)
            .offset(0)
            .range(atom_buf_size);
        let d_buf1 = vk::DescriptorBufferInfo::default()
            .buffer(buf_mo)
            .offset(0)
            .range(mo_buf_size);
        let d_buf2 = vk::DescriptorBufferInfo::default()
            .buffer(buf_out)
            .offset(0)
            .range(out_buf_size);

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
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&d_buf2)),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        // 5. Command Buffer
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&cmd_alloc)?[0] };

        let pc = CubePushConstants {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            natoms: batch.natoms as u32,
            norbs: batch.norbs as u32,
            origin_x: origin_bohr[0] as f32,
            origin_y: origin_bohr[1] as f32,
            origin_z: origin_bohr[2] as f32,
            step_bohr: step_bohr as f32,
        };

        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.orbital_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.orbital_pipeline_layout,
                0,
                &[desc_set],
                &[],
            );
            let pc_bytes = std::slice::from_raw_parts(
                &pc as *const CubePushConstants as *const u8,
                std::mem::size_of::<CubePushConstants>(),
            );
            device.cmd_push_constants(
                cmd,
                self.orbital_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );

            let workgroups = (total_voxels as u32).div_ceil(64);
            device.cmd_dispatch(cmd, workgroups, 1, 1);
            device.end_command_buffer(cmd)?;

            let fence_info = vk::FenceCreateInfo::default();
            let fence = device.create_fence(&fence_info, None)?;

            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            device.queue_submit(self.ctx.compute_queue, &[submit_info], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;

            device.destroy_fence(fence, None);
            device.free_command_buffers(self.ctx.command_pool, &[cmd]);
        }

        // 6. Read back results
        let mut out_f64 = vec![0.0f64; total_voxels];
        unsafe {
            let ptr = device.map_memory(mem_out, 0, out_buf_size, vk::MemoryMapFlags::empty())?
                as *const f32;
            for (i, val) in out_f64.iter_mut().enumerate() {
                *val = *ptr.add(i) as f64;
            }
            device.unmap_memory(mem_out);
        }

        // 7. Cleanup
        unsafe {
            device.destroy_descriptor_pool(desc_pool, None);
            device.destroy_buffer(buf_atom, None);
            device.free_memory(mem_atom, None);
            device.destroy_buffer(buf_mo, None);
            device.free_memory(mem_mo, None);
            device.destroy_buffer(buf_out, None);
            device.free_memory(mem_out, None);
        }

        Ok(out_f64)
    }

    /// Computes the 3D scalar grid for Total Electron Density on GPU.
    pub fn compute_density_grid(
        &self,
        batch: &MolecularBatch,
        model: &dyn ParameterModel,
        density_matrix: &AlignedMatrix<f64>,
        origin_bohr: [f64; 3],
        step_bohr: f64,
        dimensions: (usize, usize, usize),
    ) -> Result<Vec<f64>, VulkanError> {
        let (nx, ny, nz) = dimensions;
        let total_voxels = nx * ny * nz;
        if total_voxels == 0 || batch.natoms == 0 {
            return Ok(vec![0.0; total_voxels]);
        }

        let atoms = self.prepare_gpu_atoms(batch, model);
        let mut density_f32 = Vec::with_capacity(batch.norbs * batch.norbs);
        for mu in 0..batch.norbs {
            for nu in 0..batch.norbs {
                density_f32.push(density_matrix.get(mu, nu) as f32);
            }
        }

        let device = &self.ctx.device;
        let atom_buf_size = (atoms.len() * std::mem::size_of::<AtomDataGpu>()) as u64;
        let dens_buf_size = (density_f32.len() * std::mem::size_of::<f32>()) as u64;
        let out_buf_size = (total_voxels * std::mem::size_of::<f32>()) as u64;

        // 1. Input Atom Buffer
        let buf_info_atom = vk::BufferCreateInfo::default()
            .size(atom_buf_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
        let buf_atom = unsafe { device.create_buffer(&buf_info_atom, None)? };
        let req_atom = unsafe { device.get_buffer_memory_requirements(buf_atom) };
        let mem_type_atom = self.ctx.find_memory_type(
            req_atom.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let mem_atom = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req_atom.size)
                    .memory_type_index(mem_type_atom),
                None,
            )?
        };
        unsafe {
            device.bind_buffer_memory(buf_atom, mem_atom, 0)?;
            let ptr = device.map_memory(mem_atom, 0, atom_buf_size, vk::MemoryMapFlags::empty())?
                as *mut AtomDataGpu;
            std::ptr::copy_nonoverlapping(atoms.as_ptr(), ptr, atoms.len());
            device.unmap_memory(mem_atom);
        }

        // 2. Input Density Matrix Buffer
        let buf_info_dens = vk::BufferCreateInfo::default()
            .size(dens_buf_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
        let buf_dens = unsafe { device.create_buffer(&buf_info_dens, None)? };
        let req_dens = unsafe { device.get_buffer_memory_requirements(buf_dens) };
        let mem_type_dens = self.ctx.find_memory_type(
            req_dens.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let mem_dens = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req_dens.size)
                    .memory_type_index(mem_type_dens),
                None,
            )?
        };
        unsafe {
            device.bind_buffer_memory(buf_dens, mem_dens, 0)?;
            let ptr = device.map_memory(mem_dens, 0, dens_buf_size, vk::MemoryMapFlags::empty())?
                as *mut f32;
            std::ptr::copy_nonoverlapping(density_f32.as_ptr(), ptr, density_f32.len());
            device.unmap_memory(mem_dens);
        }

        // 3. Output Grid Buffer
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

        // 4. Descriptor Set
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(std::slice::from_ref(&pool_size))
            .max_sets(1);
        let desc_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(std::slice::from_ref(&self.density_desc_layout));
        let desc_set = unsafe { device.allocate_descriptor_sets(&alloc_info)?[0] };

        let d_buf0 = vk::DescriptorBufferInfo::default()
            .buffer(buf_atom)
            .offset(0)
            .range(atom_buf_size);
        let d_buf1 = vk::DescriptorBufferInfo::default()
            .buffer(buf_dens)
            .offset(0)
            .range(dens_buf_size);
        let d_buf2 = vk::DescriptorBufferInfo::default()
            .buffer(buf_out)
            .offset(0)
            .range(out_buf_size);

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
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&d_buf2)),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        // 5. Command Buffer
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&cmd_alloc)?[0] };

        let pc = CubePushConstants {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            natoms: batch.natoms as u32,
            norbs: batch.norbs as u32,
            origin_x: origin_bohr[0] as f32,
            origin_y: origin_bohr[1] as f32,
            origin_z: origin_bohr[2] as f32,
            step_bohr: step_bohr as f32,
        };

        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.density_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.density_pipeline_layout,
                0,
                &[desc_set],
                &[],
            );
            let pc_bytes = std::slice::from_raw_parts(
                &pc as *const CubePushConstants as *const u8,
                std::mem::size_of::<CubePushConstants>(),
            );
            device.cmd_push_constants(
                cmd,
                self.density_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );

            let workgroups = (total_voxels as u32).div_ceil(64);
            device.cmd_dispatch(cmd, workgroups, 1, 1);
            device.end_command_buffer(cmd)?;

            let fence_info = vk::FenceCreateInfo::default();
            let fence = device.create_fence(&fence_info, None)?;

            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            device.queue_submit(self.ctx.compute_queue, &[submit_info], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;

            device.destroy_fence(fence, None);
            device.free_command_buffers(self.ctx.command_pool, &[cmd]);
        }

        // 6. Read back results
        let mut out_f64 = vec![0.0f64; total_voxels];
        unsafe {
            let ptr = device.map_memory(mem_out, 0, out_buf_size, vk::MemoryMapFlags::empty())?
                as *const f32;
            for (i, val) in out_f64.iter_mut().enumerate() {
                *val = *ptr.add(i) as f64;
            }
            device.unmap_memory(mem_out);
        }

        // 7. Cleanup
        unsafe {
            device.destroy_descriptor_pool(desc_pool, None);
            device.destroy_buffer(buf_atom, None);
            device.free_memory(mem_atom, None);
            device.destroy_buffer(buf_dens, None);
            device.free_memory(mem_dens, None);
            device.destroy_buffer(buf_out, None);
            device.free_memory(mem_out, None);
        }

        Ok(out_f64)
    }

    /// Generates a complete Gaussian Cube formatted string for a Molecular Orbital on GPU.
    pub fn generate_molecular_orbital_cube(
        &self,
        batch: &MolecularBatch,
        model: &dyn ParameterModel,
        mo_coefficients: &[f64],
        orbital_index: usize,
        orbital_energy_ev: f64,
        config: &CubeGridConfig,
    ) -> Result<String, VulkanError> {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;

        for a in 0..batch.natoms {
            min_x = min_x.min(batch.x[a]);
            max_x = max_x.max(batch.x[a]);
            min_y = min_y.min(batch.y[a]);
            max_y = max_y.max(batch.y[a]);
            min_z = min_z.min(batch.z[a]);
            max_z = max_z.max(batch.z[a]);
        }

        let pad = config.padding_angstrom;
        let step_a = config.resolution_angstrom;

        min_x -= pad;
        max_x += pad;
        min_y -= pad;
        max_y += pad;
        min_z -= pad;
        max_z += pad;

        let nx = ((max_x - min_x) / step_a).ceil() as usize + 1;
        let ny = ((max_y - min_y) / step_a).ceil() as usize + 1;
        let nz = ((max_z - min_z) / step_a).ceil() as usize + 1;

        let origin_bohr = [
            min_x * ANGSTROM_TO_BOHR,
            min_y * ANGSTROM_TO_BOHR,
            min_z * ANGSTROM_TO_BOHR,
        ];
        let step_bohr = step_a * ANGSTROM_TO_BOHR;

        let grid_values = self.compute_orbital_grid(
            batch,
            model,
            mo_coefficients,
            origin_bohr,
            step_bohr,
            (nx, ny, nz),
        )?;

        let mut out = String::with_capacity(1024 + nx * ny * nz * 14);

        out.push_str("MOPAC_RS Molecular Orbital Cube File (Vulkan GPU Accelerated)\n");
        out.push_str(&format!(
            "Orbital {} Energy = {:.4} eV\n",
            orbital_index, orbital_energy_ev
        ));

        out.push_str(&format!(
            "{:5} {:12.6} {:12.6} {:12.6}\n",
            -(batch.natoms as i64),
            origin_bohr[0],
            origin_bohr[1],
            origin_bohr[2]
        ));

        out.push_str(&format!(
            "{:5} {:12.6} {:12.6} {:12.6}\n",
            nx, step_bohr, 0.0, 0.0
        ));
        out.push_str(&format!(
            "{:5} {:12.6} {:12.6} {:12.6}\n",
            ny, 0.0, step_bohr, 0.0
        ));
        out.push_str(&format!(
            "{:5} {:12.6} {:12.6} {:12.6}\n",
            nz, 0.0, 0.0, step_bohr
        ));

        for a in 0..batch.natoms {
            let z = batch.atomic_numbers[a];
            let p = model
                .get_element(z)
                .expect("Element params missing in model");
            let ax = batch.x[a] * ANGSTROM_TO_BOHR;
            let ay = batch.y[a] * ANGSTROM_TO_BOHR;
            let az = batch.z[a] * ANGSTROM_TO_BOHR;
            out.push_str(&format!(
                "{:5} {:12.6} {:12.6} {:12.6} {:12.6}\n",
                z, p.core_charge, ax, ay, az
            ));
        }

        out.push_str(&format!("{:5} {:5}\n", 1, orbital_index));

        for ix in 0..nx {
            for iy in 0..ny {
                let mut line_count = 0;
                for iz in 0..nz {
                    let idx = ix * (ny * nz) + iy * nz + iz;
                    let val = grid_values[idx];
                    out.push_str(&format!(" {:12.5E}", val));
                    line_count += 1;
                    if line_count == 6 {
                        out.push('\n');
                        line_count = 0;
                    }
                }
                if line_count > 0 {
                    out.push('\n');
                }
            }
        }

        Ok(out)
    }

    /// Generates a complete Gaussian Cube formatted string for Total Electron Density on GPU.
    pub fn generate_density_cube(
        &self,
        batch: &MolecularBatch,
        model: &dyn ParameterModel,
        density_matrix: &AlignedMatrix<f64>,
        config: &CubeGridConfig,
    ) -> Result<String, VulkanError> {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;

        for a in 0..batch.natoms {
            min_x = min_x.min(batch.x[a]);
            max_x = max_x.max(batch.x[a]);
            min_y = min_y.min(batch.y[a]);
            max_y = max_y.max(batch.y[a]);
            min_z = min_z.min(batch.z[a]);
            max_z = max_z.max(batch.z[a]);
        }

        let pad = config.padding_angstrom;
        let step_a = config.resolution_angstrom;

        min_x -= pad;
        max_x += pad;
        min_y -= pad;
        max_y += pad;
        min_z -= pad;
        max_z += pad;

        let nx = ((max_x - min_x) / step_a).ceil() as usize + 1;
        let ny = ((max_y - min_y) / step_a).ceil() as usize + 1;
        let nz = ((max_z - min_z) / step_a).ceil() as usize + 1;

        let origin_bohr = [
            min_x * ANGSTROM_TO_BOHR,
            min_y * ANGSTROM_TO_BOHR,
            min_z * ANGSTROM_TO_BOHR,
        ];
        let step_bohr = step_a * ANGSTROM_TO_BOHR;

        let grid_values = self.compute_density_grid(
            batch,
            model,
            density_matrix,
            origin_bohr,
            step_bohr,
            (nx, ny, nz),
        )?;

        let mut out = String::with_capacity(1024 + nx * ny * nz * 14);

        out.push_str("MOPAC_RS Total Electron Density Cube File (Vulkan GPU Accelerated)\n");
        out.push_str("Total SCF Electron Density rho(r) in e/Bohr^3\n");

        out.push_str(&format!(
            "{:5} {:12.6} {:12.6} {:12.6}\n",
            batch.natoms, origin_bohr[0], origin_bohr[1], origin_bohr[2]
        ));

        out.push_str(&format!(
            "{:5} {:12.6} {:12.6} {:12.6}\n",
            nx, step_bohr, 0.0, 0.0
        ));
        out.push_str(&format!(
            "{:5} {:12.6} {:12.6} {:12.6}\n",
            ny, 0.0, step_bohr, 0.0
        ));
        out.push_str(&format!(
            "{:5} {:12.6} {:12.6} {:12.6}\n",
            nz, 0.0, 0.0, step_bohr
        ));

        for a in 0..batch.natoms {
            let z = batch.atomic_numbers[a];
            let p = model
                .get_element(z)
                .expect("Element params missing in model");
            let ax = batch.x[a] * ANGSTROM_TO_BOHR;
            let ay = batch.y[a] * ANGSTROM_TO_BOHR;
            let az = batch.z[a] * ANGSTROM_TO_BOHR;
            out.push_str(&format!(
                "{:5} {:12.6} {:12.6} {:12.6} {:12.6}\n",
                z, p.core_charge, ax, ay, az
            ));
        }

        for ix in 0..nx {
            for iy in 0..ny {
                let mut line_count = 0;
                for iz in 0..nz {
                    let idx = ix * (ny * nz) + iy * nz + iz;
                    let val = grid_values[idx];
                    out.push_str(&format!(" {:12.5E}", val));
                    line_count += 1;
                    if line_count == 6 {
                        out.push('\n');
                        line_count = 0;
                    }
                }
                if line_count > 0 {
                    out.push('\n');
                }
            }
        }

        Ok(out)
    }
}

impl Drop for GpuCubeEvaluator {
    fn drop(&mut self) {
        let device = &self.ctx.device;
        unsafe {
            device.destroy_pipeline(self.orbital_pipeline, None);
            device.destroy_pipeline_layout(self.orbital_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.orbital_desc_layout, None);
            device.destroy_shader_module(self.orbital_shader_module, None);

            device.destroy_pipeline(self.density_pipeline, None);
            device.destroy_pipeline_layout(self.density_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.density_desc_layout, None);
            device.destroy_shader_module(self.density_shader_module, None);
        }
    }
}
