//! Standalone SM50 pass pipeline and machine-code emission.

use crate::bindings::FsKey;
use crate::ir::{Shader, ShaderModel, ShaderStageInfo};
use crate::{Error, ShaderBinary};

/// Optimize, allocate, schedule, and encode one Maxwell NAK shader.
///
/// The input is already-lowered NAK IR. This is the machine-backend boundary used by
/// the Naga lowering layer; it intentionally contains no NIR or Mesa C API dependency.
///
/// # Errors
///
/// Returns [`Error::UnsupportedShaderModel`] unless the shader targets GM20B/SM53.
pub fn compile_ir(mut shader: Shader<'_>, fs_key: Option<&FsKey>) -> Result<ShaderBinary, Error> {
    let sm = shader.sm;
    if sm.sm() != 53 {
        return Err(Error::UnsupportedShaderModel(sm.sm()));
    }

    shader.opt_bar_prop();
    shader.opt_uniform_instrs();
    shader.opt_copy_prop();
    shader.opt_prmt();
    shader.opt_lop();
    shader.opt_copy_prop();
    shader.opt_dce();
    shader.opt_out();
    shader.legalize();
    shader.opt_dce();
    shader.opt_instr_sched_prepass();
    shader.assign_regs();
    shader.lower_par_copies();
    shader.lower_copy_swap();
    shader.opt_crs();
    shader.remove_annotations();
    shader.opt_instr_sched_postpass();
    shader.calc_instr_deps();
    shader.gather_info();

    let sph = crate::sph::encode_header(sm, &shader.info, fs_key);
    let words = sm.encode_shader(&shader);
    let code = words
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();

    let crs_size = sm.crs_size(shader.info.max_crs_depth);
    let is_compute = matches!(shader.info.stage, ShaderStageInfo::Compute(_));
    let shared_memory_size = match shader.info.stage {
        ShaderStageInfo::Compute(ref info) => u32::from(info.smem_size),
        _ => 0,
    };
    let per_warp_scratch_size = if is_compute {
        shader
            .info
            .slm_size
            .saturating_mul(32)
            .max(crs_size)
            .max(0x800)
    } else {
        shader.info.slm_size.saturating_mul(32)
    };

    Ok(ShaderBinary {
        code,
        num_gprs: (u32::from(shader.info.num_gprs) + sm.hw_reserved_gprs()).max(4),
        per_warp_scratch_size,
        local_memory_size: shader.info.slm_size,
        shared_memory_size,
        crs_size: if is_compute {
            crs_size.max(0x800)
        } else {
            crs_size
        },
        num_control_barriers: u32::from(shader.info.num_control_barriers),
        sph,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        BasicBlock, Function, Instr, LabelAllocator, OpExit, PhiAllocator, SSAValueAllocator,
        ShaderInfo, ShaderIoInfo, ShaderModelInfo, ShaderStageInfo, VertexShaderInfo, VtgIoInfo,
    };
    use compiler::cfg::CFGBuilder;
    use std::hash::RandomState;

    fn minimal_vertex_shader<'a>(sm: &'a ShaderModelInfo) -> Shader<'a> {
        let mut labels = LabelAllocator::new();
        let mut cfg = CFGBuilder::<_, _, RandomState>::new();
        cfg.add_node(
            0,
            BasicBlock {
                label: labels.alloc(),
                uniform: false,
                instrs: vec![Instr::new(OpExit {})],
            },
        );

        Shader {
            sm,
            info: ShaderInfo {
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
                stage: ShaderStageInfo::Vertex(VertexShaderInfo {
                    isbe_space_sharing_enable: false,
                }),
                io: ShaderIoInfo::Vtg(VtgIoInfo {
                    sysvals_in: Default::default(),
                    sysvals_in_d: 0,
                    sysvals_out: Default::default(),
                    sysvals_out_d: 0,
                    attr_in: [0; 4],
                    attr_out: [0; 4],
                    store_req_start: 0,
                    store_req_end: 0,
                    clip_enable: 0,
                    cull_enable: 0,
                    xfb: None,
                }),
            },
            functions: vec![Function {
                ssa_alloc: SSAValueAllocator::new(),
                phi_alloc: PhiAllocator::new(),
                blocks: cfg.as_cfg(false),
            }],
        }
    }

    #[test]
    fn synthetic_vertex_ir_encodes_deterministically() {
        let sm = ShaderModelInfo::new(53, 64);
        let first = compile_ir(minimal_vertex_shader(&sm), None).unwrap();
        let second = compile_ir(minimal_vertex_shader(&sm), None).unwrap();

        assert_eq!(first, second);
        assert!(!first.code.is_empty());
        assert_eq!(first.code.len() % 32, 0);
        assert_eq!(first.sph[0] & 0x3fff, 0x0461);
    }
}
