// Copyright © 2022 Collabora, Ltd.
// SPDX-License-Identifier: MIT

//! Rust-native replacement for the generated `nak_bindings` surface used by NAK's
//! machine IR and shader-program-header encoder.

#![allow(dead_code, non_camel_case_types, non_upper_case_globals)]

pub(crate) const NAK_TS_DOMAIN_ISOLINE: u8 = 0;
pub(crate) const NAK_TS_DOMAIN_TRIANGLE: u8 = 1;
pub(crate) const NAK_TS_DOMAIN_QUAD: u8 = 2;

pub(crate) const NAK_TS_SPACING_INTEGER: u8 = 0;
pub(crate) const NAK_TS_SPACING_FRACT_ODD: u8 = 1;
pub(crate) const NAK_TS_SPACING_FRACT_EVEN: u8 = 2;

pub(crate) type nak_mesh_topology = u32;
pub(crate) const NAK_MESH_TOPOLOGY_POINTS: nak_mesh_topology = 0;
pub(crate) const NAK_MESH_TOPOLOGY_LINES: nak_mesh_topology = 1;
pub(crate) const NAK_MESH_TOPOLOGY_TRIANGLES: nak_mesh_topology = 4;

/// Fragment pipeline state consumed while generating the NAK SPH.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub struct FsKey {
    /// Depth/stencil self-dependency requires kill behavior.
    pub zs_self_dep: bool,
    /// API-forced sample shading.
    pub force_sample_shading: bool,
    /// Fragment interlock uses underestimate mode.
    pub uses_underestimate: bool,
    /// Explicit zero padding retained for stable representation.
    pub pad: u8,
}

pub(crate) type nak_fs_key = FsKey;

/// Mesh primitive topology exposed by the compiler boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum MeshTopology {
    /// Point list.
    Points = NAK_MESH_TOPOLOGY_POINTS,
    /// Line list.
    Lines = NAK_MESH_TOPOLOGY_LINES,
    /// Triangle list.
    Triangles = NAK_MESH_TOPOLOGY_TRIANGLES,
}

/// Transform-feedback layout carried by NAK's vertex/geometry IO model.
#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct TransformFeedbackInfo {
    /// Byte stride for each transform-feedback buffer.
    pub stride: [u32; 4],
    /// Stream assigned to each buffer.
    pub stream: [u8; 4],
    /// Number of attributes written to each buffer.
    pub attribute_count: [u8; 4],
    /// Attribute index table for each buffer.
    pub attribute_index: [[u8; 128]; 4],
}

impl Default for TransformFeedbackInfo {
    fn default() -> Self {
        Self {
            stride: [0; 4],
            stream: [0; 4],
            attribute_count: [0; 4],
            attribute_index: [[0; 128]; 4],
        }
    }
}

pub(crate) type nak_xfb_info = TransformFeedbackInfo;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_values_match_public_nak_headers() {
        assert_eq!(size_of::<FsKey>(), 4);
        assert_eq!(MeshTopology::Points as u32, 0);
        assert_eq!(MeshTopology::Lines as u32, 1);
        assert_eq!(MeshTopology::Triangles as u32, 4);
        assert_eq!(NAK_TS_DOMAIN_QUAD, 2);
        assert_eq!(NAK_TS_SPACING_FRACT_EVEN, 2);
        assert_eq!(size_of::<TransformFeedbackInfo>(), 16 + 4 + 4 + 512);
    }
}
