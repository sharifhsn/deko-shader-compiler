//! GM20B target boundary for the NAK-derived machine backend.
//!
//! The optimizer, allocator, scheduler, and encoder are being extracted behind this
//! API. Until that extraction is complete, compilation returns an explicit error.

use thiserror::Error;

mod bindings;
mod compiler;
pub mod debug;
pub mod sph;

pub(crate) use debug as api;

#[rustfmt::skip]
mod assign_regs;
#[rustfmt::skip]
mod builder;
#[rustfmt::skip]
mod calc_instr_deps;
#[rustfmt::skip]
mod const_tracker;
pub mod ir;
#[rustfmt::skip]
mod legalize;
#[rustfmt::skip]
mod liveness;
#[rustfmt::skip]
mod lower_copy_swap;
#[rustfmt::skip]
mod lower_par_copies;
#[rustfmt::skip]
mod opt_bar_prop;
#[rustfmt::skip]
mod opt_copy_prop;
#[rustfmt::skip]
mod opt_crs;
#[rustfmt::skip]
mod opt_dce;
#[rustfmt::skip]
mod opt_instr_sched_common;
#[rustfmt::skip]
mod opt_instr_sched_postpass;
#[rustfmt::skip]
mod opt_instr_sched_prepass;
#[rustfmt::skip]
mod opt_lop;
#[rustfmt::skip]
mod opt_out;
#[rustfmt::skip]
mod opt_prmt;
#[rustfmt::skip]
mod opt_uniform_instrs;
#[rustfmt::skip]
mod reg_tracker;
#[rustfmt::skip]
mod repair_ssa;
#[rustfmt::skip]
mod sm50;
#[rustfmt::skip]
mod spill_values;
#[rustfmt::skip]
mod ssa_value;
#[rustfmt::skip]
mod to_cssa;
#[rustfmt::skip]
mod union_find;

pub use bindings::{FsKey, MeshTopology, TransformFeedbackInfo};
pub use compiler::compile_ir;

/// Maxwell target properties needed by the standalone backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Target {
    /// NVIDIA shader-model number (`53` for GM20B).
    pub shader_model: u8,
    /// Maximum resident warps per multiprocessor.
    pub max_warps_per_multiprocessor: u8,
}

impl Target {
    /// Nintendo Switch Tegra X1 GM20B target.
    pub const GM20B: Self = Self {
        shader_model: 53,
        max_warps_per_multiprocessor: 64,
    };
}

/// Native shader emitted by the machine backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderBinary {
    /// Maxwell machine code and embedded constant data.
    pub code: Vec<u8>,
    /// Number of general-purpose registers used.
    pub num_gprs: u32,
    /// Scratch bytes required per warp.
    pub per_warp_scratch_size: u32,
    /// NAK shader program header. For GM20B the first 20 words are meaningful `SPHv3`.
    pub sph: [u32; 32],
}

/// Machine-backend failure.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum Error {
    /// The NAK extraction has not reached an executable backend yet.
    #[error("the standalone NAK backend is not implemented yet")]
    BackendUnavailable,
    /// The standalone backend deliberately targets only the Switch's GM20B GPU.
    #[error("shader model {0} is unsupported; deko-nak targets SM53")]
    UnsupportedShaderModel(u8),
}

/// Confirm that a target is within the deliberately narrow initial support envelope.
///
/// # Errors
///
/// Returns [`Error::BackendUnavailable`] for targets other than GM20B until additional
/// target descriptors and backend implementations are added deliberately.
pub fn validate_target(target: Target) -> Result<(), Error> {
    if target == Target::GM20B {
        Ok(())
    } else {
        Err(Error::BackendUnavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_is_exactly_gm20b() {
        assert_eq!(Target::GM20B.shader_model, 53);
        assert_eq!(Target::GM20B.max_warps_per_multiprocessor, 64);
        assert_eq!(validate_target(Target::GM20B), Ok(()));
    }
}
