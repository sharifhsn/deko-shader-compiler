//! Standalone construction helpers layered over the imported NAK IR.

use crate::ir::{
    BasicBlock, ComputeShaderInfo, FragmentIoInfo, FragmentShaderInfo, Function, Instr,
    LabelAllocator, PhiAllocator, SSAValueAllocator, ShaderInfo, ShaderIoInfo, ShaderStageInfo,
    SysValInfo, VertexShaderInfo, VtgIoInfo,
};
use crate::sph::PixelImap;
use compiler::cfg::CFGBuilder;
use std::hash::RandomState;

impl Function {
    /// Construct a one-block function with fresh SSA and phi allocators.
    #[must_use]
    pub fn single_block(instrs: Vec<Instr>) -> Self {
        let mut labels = LabelAllocator::new();
        let mut cfg = CFGBuilder::<_, _, RandomState>::new();
        cfg.add_node(
            0,
            BasicBlock {
                label: labels.alloc(),
                uniform: false,
                instrs,
            },
        );
        Self {
            ssa_alloc: SSAValueAllocator::new(),
            phi_alloc: PhiAllocator::new(),
            blocks: cfg.as_cfg(false),
        }
    }
}

impl ShaderInfo {
    fn base(stage: ShaderStageInfo, io: ShaderIoInfo) -> Self {
        Self {
            max_warps_per_sm: 0,
            num_gprs: 0,
            num_control_barriers: 0,
            num_instrs: 0,
            num_static_cycles: 0,
            num_spills_to_mem: 0,
            num_fills_from_mem: 0,
            num_spills_to_reg: 0,
            num_fills_from_reg: 0,
            slm_size: 0,
            max_crs_depth: 0,
            uses_global_mem: false,
            writes_global_mem: false,
            uses_fp64: false,
            stage,
            io,
        }
    }

    /// Construct initial metadata for a compute shader before NAK's gather pass.
    #[must_use]
    pub fn compute(local_size: [u16; 3], shared_memory_size: u16) -> Self {
        Self::base(
            ShaderStageInfo::Compute(ComputeShaderInfo {
                local_size,
                smem_size: shared_memory_size,
            }),
            ShaderIoInfo::None,
        )
    }

    /// Construct initial metadata for a vertex shader.
    #[must_use]
    pub fn vertex() -> Self {
        Self::base(
            ShaderStageInfo::Vertex(VertexShaderInfo {
                isbe_space_sharing_enable: false,
            }),
            ShaderIoInfo::Vtg(VtgIoInfo {
                sysvals_in: SysValInfo::default(),
                sysvals_in_d: 0,
                sysvals_out: SysValInfo::default(),
                sysvals_out_d: 0,
                attr_in: [0; 4],
                attr_out: [0; 4],
                store_req_start: u8::MAX,
                store_req_end: 0,
                clip_enable: 0,
                cull_enable: 0,
                xfb: None,
            }),
        )
    }

    /// Construct initial metadata for a fragment shader.
    #[must_use]
    pub fn fragment(
        early_fragment_tests: bool,
        post_depth_coverage: bool,
        uses_sample_shading: bool,
    ) -> Self {
        Self::base(
            ShaderStageInfo::Fragment(FragmentShaderInfo {
                uses_kill: false,
                does_interlock: false,
                post_depth_coverage,
                early_fragment_tests,
                uses_sample_shading,
            }),
            ShaderIoInfo::Fragment(FragmentIoInfo {
                sysvals_in: SysValInfo {
                    // Required on fragment shaders; omitting it traps on Maxwell.
                    ab: 1 << 31,
                    c: 0,
                },
                sysvals_in_d: [PixelImap::Unused; 8],
                attr_in: [PixelImap::Unused; 128],
                barycentric_attr_in: [0; 4],
                reads_sample_mask: false,
                writes_color: 0,
                writes_sample_mask: false,
                writes_depth: false,
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::OpExit;

    #[test]
    fn single_block_function_has_one_exit() {
        let function = Function::single_block(vec![Instr::new(OpExit {})]);
        assert_eq!(function.blocks.len(), 1);
        assert_eq!(function.blocks[0].instrs.len(), 1);
    }
}
