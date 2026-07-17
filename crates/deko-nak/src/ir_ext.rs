//! Standalone construction helpers layered over the imported NAK IR.

use crate::ir::{
    BasicBlock, ComputeShaderInfo, Function, Instr, LabelAllocator, PhiAllocator,
    SSAValueAllocator, ShaderInfo, ShaderIoInfo, ShaderStageInfo,
};
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
    /// Construct initial metadata for a compute shader before NAK's gather pass.
    #[must_use]
    pub fn compute(local_size: [u16; 3], shared_memory_size: u16) -> Self {
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
            stage: ShaderStageInfo::Compute(ComputeShaderInfo {
                local_size,
                smem_size: shared_memory_size,
            }),
            io: ShaderIoInfo::None,
        }
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
