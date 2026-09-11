//! High-performance data-oriented memory structures and types for MOPAC_RS.
//!
//! Licensed under the Apache License, Version 2.0 (the "License").
//! Employs strict 64-byte alignment matching modern cache lines and SIMD registers.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ops::{Deref, DerefMut};

/// Cache line alignment boundary in bytes (AVX-512 and standard x86_64 cache line)
pub const CACHE_LINE_ALIGNMENT: usize = 64;

/// A contiguous heap buffer guaranteed to be aligned to 64 bytes.
#[derive(Debug)]
pub struct AlignedVec64<T: Copy + Default> {
    ptr: *mut T,
    len: usize,
    capacity: usize,
}

unsafe impl<T: Copy + Default + Send> Send for AlignedVec64<T> {}
unsafe impl<T: Copy + Default + Sync> Sync for AlignedVec64<T> {}

impl<T: Copy + Default> AlignedVec64<T> {
    /// Create a zero-initialized aligned buffer of length `len`.
    pub fn zeroed(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                capacity: 0,
            };
        }
        let layout = Layout::from_size_align(len * std::mem::size_of::<T>(), CACHE_LINE_ALIGNMENT)
            .expect("Invalid layout for AlignedVec64");
        let ptr = unsafe { alloc_zeroed(layout) as *mut T };
        if ptr.is_null() {
            panic!("Memory allocation of size {} failed", len * std::mem::size_of::<T>());
        }
        Self {
            ptr,
            len,
            capacity: len,
        }
    }

    /// Number of elements.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is the buffer empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return raw pointer.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Return mutable raw pointer.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Reset all elements to zero without reallocating.
    #[inline(always)]
    pub fn fill_zero(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.ptr, 0, self.len);
        }
    }
}

impl<T: Copy + Default> Deref for AlignedVec64<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

impl<T: Copy + Default> DerefMut for AlignedVec64<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }
}

impl<T: Copy + Default> Drop for AlignedVec64<T> {
    fn drop(&mut self) {
        if self.capacity > 0 && !self.ptr.is_null() {
            let layout = Layout::from_size_align(
                self.capacity * std::mem::size_of::<T>(),
                CACHE_LINE_ALIGNMENT,
            )
            .expect("Invalid layout in drop");
            unsafe {
                dealloc(self.ptr as *mut u8, layout);
            }
        }
    }
}

impl<T: Copy + Default> Clone for AlignedVec64<T> {
    fn clone(&self) -> Self {
        let mut new_vec = Self::zeroed(self.len);
        new_vec.copy_from_slice(self);
        new_vec
    }
}

/// 2D dense row-major matrix backed by a contiguous 64-byte aligned buffer.
#[derive(Debug, Clone)]
pub struct AlignedMatrix<T: Copy + Default> {
    pub rows: usize,
    pub cols: usize,
    pub data: AlignedVec64<T>,
}

impl<T: Copy + Default> AlignedMatrix<T> {
    /// Create a zero-initialized matrix of dimension `rows x cols`.
    pub fn zeroed(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: AlignedVec64::zeroed(rows * cols),
        }
    }

    /// Read element at (i, j).
    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> T {
        debug_assert!(r < self.rows && c < self.cols);
        self.data[r * self.cols + c]
    }

    /// Set element at (i, j).
    #[inline(always)]
    pub fn set(&mut self, r: usize, c: usize, val: T) {
        debug_assert!(r < self.rows && c < self.cols);
        self.data[r * self.cols + c] = val;
    }

    /// As mutable slice for row `r`.
    #[inline(always)]
    pub fn row(&self, r: usize) -> &[T] {
        debug_assert!(r < self.rows);
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }

    /// As mutable slice for row `r`.
    #[inline(always)]
    pub fn row_mut(&mut self, r: usize) -> &mut [T] {
        debug_assert!(r < self.rows);
        let start = r * self.cols;
        &mut self.data[start..start + self.cols]
    }

    /// Reset all elements to zero.
    #[inline(always)]
    pub fn fill_zero(&mut self) {
        self.data.fill_zero();
    }
}

/// Orbital basis set type of an atomic site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BasisType {
    /// Only s orbital (1 basis function: H)
    S = 1,
    /// s and p orbitals (4 basis functions: s, px, py, pz; C, N, O, F, etc.)
    SP = 4,
    /// s, p, and d orbitals (9 basis functions: transition metals, etc.)
    SPD = 9,
}

impl BasisType {
    /// Number of atomic orbitals for this basis type.
    #[inline(always)]
    pub fn num_orbitals(self) -> usize {
        self as usize
    }
}

/// Struct of Arrays (SoA) representation of molecular coordinates and atomic identities.
#[derive(Debug, Clone)]
pub struct MolecularBatch {
    pub natoms: usize,
    pub norbs: usize,
    /// X coordinates in Ångströms (aligned to 64 bytes)
    pub x: AlignedVec64<f64>,
    /// Y coordinates in Ångströms (aligned to 64 bytes)
    pub y: AlignedVec64<f64>,
    /// Z coordinates in Ångströms (aligned to 64 bytes)
    pub z: AlignedVec64<f64>,
    /// Atomic numbers (Z in 1..118)
    pub atomic_numbers: Vec<u8>,
    /// Basis type for each atom
    pub basis_types: Vec<BasisType>,
    /// Starting orbital index for each atom in the molecular secular matrix
    pub orbital_offsets: Vec<usize>,
}

impl MolecularBatch {
    /// Construct a new MolecularBatch from vectors of atomic numbers and coordinates.
    pub fn new(atomic_numbers: Vec<u8>, coords_angstrom: &[[f64; 3]]) -> Self {
        let natoms = atomic_numbers.len();
        assert_eq!(natoms, coords_angstrom.len(), "Atomic numbers and coordinates length mismatch");

        let mut x = AlignedVec64::zeroed(natoms);
        let mut y = AlignedVec64::zeroed(natoms);
        let mut z = AlignedVec64::zeroed(natoms);

        for (i, coord) in coords_angstrom.iter().enumerate() {
            x[i] = coord[0];
            y[i] = coord[1];
            z[i] = coord[2];
        }

        let mut basis_types = Vec::with_capacity(natoms);
        let mut orbital_offsets = Vec::with_capacity(natoms);
        let mut norbs = 0;

        for &z_num in &atomic_numbers {
            orbital_offsets.push(norbs);
            let b_type = match z_num {
                1 => BasisType::S,
                2..=10 => BasisType::SP,
                11..=18 => BasisType::SP, // Standard MNDO/AM1/PM3 main group
                _ => BasisType::SPD,
            };
            norbs += b_type.num_orbitals();
            basis_types.push(b_type);
        }

        Self {
            natoms,
            norbs,
            x,
            y,
            z,
            atomic_numbers,
            basis_types,
            orbital_offsets,
        }
    }

    /// Compute distance between two atoms in Ångströms.
    #[inline(always)]
    pub fn distance(&self, a: usize, b: usize) -> f64 {
        let dx = self.x[a] - self.x[b];
        let dy = self.y[a] - self.y[b];
        let dz = self.z[a] - self.z[b];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Pre-allocated workspace for the Self-Consistent Field (SCF) loop.
///
/// Ensures 0 heap allocations during SCF iterations.
#[derive(Debug)]
pub struct ScfWorkspace {
    pub norbs: usize,
    /// Fock matrix $F$
    pub fock: AlignedMatrix<f64>,
    /// Density matrix $P$
    pub density: AlignedMatrix<f64>,
    /// Core Hamiltonian $H^{\text{core}}$
    pub h_core: AlignedMatrix<f64>,
    /// Eigenvectors (Molecular Orbital coefficients $C$)
    pub eigenvectors: AlignedMatrix<f64>,
    /// Eigenvalues (Orbital energies $\epsilon$)
    pub eigenvalues: AlignedVec64<f64>,
    /// DIIS error matrix $[F, P] = FP - PF$
    pub diis_error: AlignedMatrix<f64>,
    /// Temporary matrix buffer 1
    pub tmp1: AlignedMatrix<f64>,
    /// Temporary matrix buffer 2
    pub tmp2: AlignedMatrix<f64>,
}

impl ScfWorkspace {
    /// Allocate reusable workspace for a system of `norbs` basis functions.
    pub fn allocate(norbs: usize) -> Self {
        Self {
            norbs,
            fock: AlignedMatrix::zeroed(norbs, norbs),
            density: AlignedMatrix::zeroed(norbs, norbs),
            h_core: AlignedMatrix::zeroed(norbs, norbs),
            eigenvectors: AlignedMatrix::zeroed(norbs, norbs),
            eigenvalues: AlignedVec64::zeroed(norbs),
            diis_error: AlignedMatrix::zeroed(norbs, norbs),
            tmp1: AlignedMatrix::zeroed(norbs, norbs),
            tmp2: AlignedMatrix::zeroed(norbs, norbs),
        }
    }

    /// Reset all computational matrices to zero.
    pub fn reset(&mut self) {
        self.fock.fill_zero();
        self.density.fill_zero();
        self.h_core.fill_zero();
        self.eigenvectors.fill_zero();
        self.eigenvalues.fill_zero();
        self.diis_error.fill_zero();
        self.tmp1.fill_zero();
        self.tmp2.fill_zero();
    }
}
