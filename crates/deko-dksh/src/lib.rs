//! Safe, deterministic support for `Deko3D`'s DKSH shader container.
//!
//! DKSH has one aligned control section followed by one aligned machine-code section.
//! wgpu's `Deko3D` backend additionally appends versioned resource-binding metadata after
//! those sections. This crate reads and writes both without casting byte slices to Rust
//! structs.

use core::fmt;

use thiserror::Error;

/// The little-endian value of the `DKSH` file signature.
pub const MAGIC: u32 = u32::from_le_bytes(*b"DKSH");
/// Required alignment of the control and code sections.
pub const SECTION_ALIGNMENT: usize = 256;
const SECTION_ALIGNMENT_U32: u32 = 256;
/// Size of the fixed DKSH container header.
pub const HEADER_SIZE: usize = 24;
const HEADER_SIZE_U32: u32 = 24;
/// Size of one DKSH program-table entry.
pub const PROGRAM_HEADER_SIZE: usize = 64;
/// Signature of the wgpu `Deko3D` binding metadata extension.
pub const BINDING_METADATA_MAGIC: &[u8; 8] = b"DKRBv001";

/// Shader program type stored in a DKSH program-table entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum ProgramType {
    /// Vertex shader.
    Vertex = 0,
    /// Tessellation-control shader.
    TessellationControl = 1,
    /// Tessellation-evaluation shader.
    TessellationEvaluation = 2,
    /// Geometry shader.
    Geometry = 3,
    /// Fragment shader.
    Fragment = 4,
    /// Compute shader.
    Compute = 5,
}

impl TryFrom<u32> for ProgramType {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Vertex),
            1 => Ok(Self::TessellationControl),
            2 => Ok(Self::TessellationEvaluation),
            3 => Ok(Self::Geometry),
            4 => Ok(Self::Fragment),
            5 => Ok(Self::Compute),
            other => Err(Error::InvalidProgramType(other)),
        }
    }
}

/// Resource class encoded in the wgpu `Deko3D` binding metadata extension.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum BindingKind {
    /// Uniform/constant buffer.
    Uniform = 0,
    /// Sampled texture.
    Texture = 1,
    /// Sampler.
    Sampler = 2,
    /// Storage buffer.
    Storage = 3,
    /// Storage texture/image.
    StorageTexture = 4,
}

impl TryFrom<u32> for BindingKind {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Uniform),
            1 => Ok(Self::Texture),
            2 => Ok(Self::Sampler),
            3 => Ok(Self::Storage),
            4 => Ok(Self::StorageTexture),
            other => Err(Error::InvalidBindingKind(other)),
        }
    }
}

/// One logical wgpu resource binding and its Deko target slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Binding {
    /// Bind-group index.
    pub group: u32,
    /// Binding number within the group.
    pub binding: u32,
    /// Deko constant-buffer, texture, sampler, or storage slot.
    pub target: u32,
    /// Resource class.
    pub kind: BindingKind,
}

/// Stage metadata required by one DKSH program-table entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Program {
    /// Shader stage.
    pub program_type: ProgramType,
    /// Byte offset of the first instruction within the code section.
    pub entrypoint: u32,
    /// Number of general-purpose registers used by the program.
    pub num_gprs: u32,
    /// Optional `(offset, size)` of constant buffer 1 data in the code section.
    pub constbuf1: Option<(u32, u32)>,
    /// Scratch bytes required per warp.
    pub per_warp_scratch_size: u32,
    /// Native 36-byte Maxwell shader-program-header payload.
    pub payload: [u8; 36],
}

/// A validated view of a DKSH container.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Container<'a> {
    /// Program metadata.
    pub program: Program,
    /// Entire aligned code section, including trailing zero padding.
    pub code: &'a [u8],
    /// Parsed binding metadata.
    pub bindings: Vec<Binding>,
    /// Size of the aligned control section.
    pub control_size: u32,
}

/// DKSH encoding or validation failure.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum Error {
    /// Input is shorter than the fixed header.
    #[error("DKSH input is shorter than its header")]
    TruncatedHeader,
    /// Header magic is not `DKSH`.
    #[error("DKSH input has invalid magic 0x{0:08x}")]
    InvalidMagic(u32),
    /// A fixed header field or section relationship is invalid.
    #[error("DKSH input has an invalid header")]
    InvalidHeader,
    /// A size calculation overflowed the supported address space.
    #[error("DKSH size calculation overflowed")]
    SizeOverflow,
    /// A program table entry names an unknown stage.
    #[error("DKSH program type {0} is invalid")]
    InvalidProgramType(u32),
    /// A program table field points outside the code section.
    #[error("DKSH program entry is invalid")]
    InvalidProgram,
    /// Machine-code input is empty.
    #[error("DKSH code must not be empty")]
    EmptyCode,
    /// Binding metadata has an invalid signature or length.
    #[error("DKSH binding metadata is invalid")]
    InvalidBindingMetadata,
    /// Binding metadata names an unknown resource class.
    #[error("DKSH binding kind {0} is invalid")]
    InvalidBindingKind(u32),
}

/// Encode one program and its resource map as a deterministic DKSH container.
///
/// # Errors
///
/// Returns an error when code is empty, program offsets exceed the unpadded code, or
/// an encoded size cannot be represented safely.
pub fn encode(program: Program, code: &[u8], bindings: &[Binding]) -> Result<Vec<u8>, Error> {
    if code.is_empty() {
        return Err(Error::EmptyCode);
    }
    let code_len = u32::try_from(code.len()).map_err(|_| Error::SizeOverflow)?;
    validate_program(program, code_len)?;

    let control_size = align_up(HEADER_SIZE + PROGRAM_HEADER_SIZE, SECTION_ALIGNMENT)?;
    let code_size = align_up(code.len(), SECTION_ALIGNMENT)?;
    let metadata_size = if bindings.is_empty() {
        0
    } else {
        BINDING_METADATA_MAGIC
            .len()
            .checked_add(4)
            .and_then(|value| value.checked_add(bindings.len().checked_mul(16)?))
            .ok_or(Error::SizeOverflow)?
    };
    let total_size = control_size
        .checked_add(code_size)
        .and_then(|value| value.checked_add(metadata_size))
        .ok_or(Error::SizeOverflow)?;
    let control_size_u32 = u32::try_from(control_size).map_err(|_| Error::SizeOverflow)?;
    let code_size_u32 = u32::try_from(code_size).map_err(|_| Error::SizeOverflow)?;

    let mut output = Vec::with_capacity(total_size);
    push_u32(&mut output, MAGIC);
    push_u32(&mut output, HEADER_SIZE_U32);
    push_u32(&mut output, control_size_u32);
    push_u32(&mut output, code_size_u32);
    push_u32(&mut output, HEADER_SIZE_U32);
    push_u32(&mut output, 1);

    push_u32(&mut output, program.program_type as u32);
    push_u32(&mut output, program.entrypoint);
    push_u32(&mut output, program.num_gprs);
    let (constbuf1_offset, constbuf1_size) = program.constbuf1.unwrap_or((0, 0));
    push_u32(&mut output, constbuf1_offset);
    push_u32(&mut output, constbuf1_size);
    push_u32(&mut output, program.per_warp_scratch_size);
    output.extend_from_slice(&program.payload);
    push_u32(&mut output, 0);

    output.resize(control_size, 0);
    output.extend_from_slice(code);
    output.resize(control_size + code_size, 0);
    if !bindings.is_empty() {
        output.extend_from_slice(BINDING_METADATA_MAGIC);
        push_u32(
            &mut output,
            u32::try_from(bindings.len()).map_err(|_| Error::SizeOverflow)?,
        );
        for binding in bindings {
            push_u32(&mut output, binding.group);
            push_u32(&mut output, binding.binding);
            push_u32(&mut output, binding.target);
            push_u32(&mut output, binding.kind as u32);
        }
    }
    debug_assert_eq!(output.len(), total_size);
    Ok(output)
}

/// Parse and validate a single-program DKSH container.
///
/// # Errors
///
/// Returns an error when the input is truncated, malformed, overflows checked size
/// calculations, contains an invalid program entry, or has malformed binding metadata.
pub fn parse(bytes: &[u8]) -> Result<Container<'_>, Error> {
    if bytes.len() < HEADER_SIZE {
        return Err(Error::TruncatedHeader);
    }
    let magic = read_u32(bytes, 0).ok_or(Error::TruncatedHeader)?;
    if magic != MAGIC {
        return Err(Error::InvalidMagic(magic));
    }
    let header_size = read_u32(bytes, 4).ok_or(Error::TruncatedHeader)?;
    let control_size = read_u32(bytes, 8).ok_or(Error::TruncatedHeader)?;
    let code_size = read_u32(bytes, 12).ok_or(Error::TruncatedHeader)?;
    let programs_offset = read_u32(bytes, 16).ok_or(Error::TruncatedHeader)?;
    let program_count = read_u32(bytes, 20).ok_or(Error::TruncatedHeader)?;
    if header_size != HEADER_SIZE_U32
        || control_size < header_size
        || code_size == 0
        || program_count != 1
        || programs_offset < header_size
        || control_size % SECTION_ALIGNMENT_U32 != 0
        || code_size % SECTION_ALIGNMENT_U32 != 0
    {
        return Err(Error::InvalidHeader);
    }

    let control_len = usize::try_from(control_size).map_err(|_| Error::SizeOverflow)?;
    let code_len = usize::try_from(code_size).map_err(|_| Error::SizeOverflow)?;
    let code_end = control_len
        .checked_add(code_len)
        .ok_or(Error::SizeOverflow)?;
    if code_end > bytes.len() {
        return Err(Error::InvalidHeader);
    }
    let program_offset = usize::try_from(programs_offset).map_err(|_| Error::SizeOverflow)?;
    let program_end = program_offset
        .checked_add(PROGRAM_HEADER_SIZE)
        .ok_or(Error::SizeOverflow)?;
    if program_offset % 4 != 0 || program_end > control_len {
        return Err(Error::InvalidHeader);
    }

    let program_type =
        ProgramType::try_from(read_u32(bytes, program_offset).ok_or(Error::InvalidProgram)?)?;
    let entrypoint = read_u32(bytes, program_offset + 4).ok_or(Error::InvalidProgram)?;
    let num_gprs = read_u32(bytes, program_offset + 8).ok_or(Error::InvalidProgram)?;
    let constbuf1_offset = read_u32(bytes, program_offset + 12).ok_or(Error::InvalidProgram)?;
    let constbuf1_size = read_u32(bytes, program_offset + 16).ok_or(Error::InvalidProgram)?;
    let per_warp_scratch_size =
        read_u32(bytes, program_offset + 20).ok_or(Error::InvalidProgram)?;
    let payload: [u8; 36] = bytes[program_offset + 24..program_offset + 60]
        .try_into()
        .map_err(|_| Error::InvalidProgram)?;
    let reserved = read_u32(bytes, program_offset + 60).ok_or(Error::InvalidProgram)?;
    if reserved != 0 {
        return Err(Error::InvalidProgram);
    }
    let constbuf1 = (constbuf1_size != 0).then_some((constbuf1_offset, constbuf1_size));
    let program = Program {
        program_type,
        entrypoint,
        num_gprs,
        constbuf1,
        per_warp_scratch_size,
        payload,
    };
    validate_program(program, code_size)?;

    let bindings = parse_bindings(&bytes[code_end..])?;
    Ok(Container {
        program,
        code: &bytes[control_len..code_end],
        bindings,
        control_size,
    })
}

fn validate_program(program: Program, code_size: u32) -> Result<(), Error> {
    if program.entrypoint >= code_size {
        return Err(Error::InvalidProgram);
    }
    if let Some((offset, size)) = program.constbuf1 {
        let end = offset.checked_add(size).ok_or(Error::InvalidProgram)?;
        if size == 0 || end > code_size {
            return Err(Error::InvalidProgram);
        }
    }
    Ok(())
}

fn parse_bindings(bytes: &[u8]) -> Result<Vec<Binding>, Error> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let header_size = BINDING_METADATA_MAGIC.len() + 4;
    if bytes.len() < header_size || bytes.get(..8) != Some(BINDING_METADATA_MAGIC) {
        return Err(Error::InvalidBindingMetadata);
    }
    let count = usize::try_from(read_u32(bytes, 8).ok_or(Error::InvalidBindingMetadata)?)
        .map_err(|_| Error::SizeOverflow)?;
    let expected = count
        .checked_mul(16)
        .and_then(|value| header_size.checked_add(value))
        .ok_or(Error::SizeOverflow)?;
    if expected != bytes.len() {
        return Err(Error::InvalidBindingMetadata);
    }
    bytes[header_size..]
        .chunks_exact(16)
        .map(|entry| {
            Ok(Binding {
                group: read_u32(entry, 0).ok_or(Error::InvalidBindingMetadata)?,
                binding: read_u32(entry, 4).ok_or(Error::InvalidBindingMetadata)?,
                target: read_u32(entry, 8).ok_or(Error::InvalidBindingMetadata)?,
                kind: BindingKind::try_from(
                    read_u32(entry, 12).ok_or(Error::InvalidBindingMetadata)?,
                )?,
            })
        })
        .collect()
}

fn align_up(value: usize, alignment: usize) -> Result<usize, Error> {
    debug_assert!(alignment.is_power_of_two());
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(Error::SizeOverflow)
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    bytes
        .get(offset..offset.checked_add(4)?)?
        .try_into()
        .ok()
        .map(u32::from_le_bytes)
}

impl fmt::Display for ProgramType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Vertex => "vertex",
            Self::TessellationControl => "tessellation-control",
            Self::TessellationEvaluation => "tessellation-evaluation",
            Self::Geometry => "geometry",
            Self::Fragment => "fragment",
            Self::Compute => "compute",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program(program_type: ProgramType) -> Program {
        Program {
            program_type,
            entrypoint: 0,
            num_gprs: 12,
            constbuf1: Some((8, 8)),
            per_warp_scratch_size: 32,
            payload: [0x5a; 36],
        }
    }

    #[test]
    fn deterministic_round_trip() {
        let code = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let bindings = [
            Binding {
                group: 0,
                binding: 2,
                target: 1,
                kind: BindingKind::Uniform,
            },
            Binding {
                group: 1,
                binding: 4,
                target: 7,
                kind: BindingKind::Storage,
            },
        ];
        let first = encode(program(ProgramType::Compute), &code, &bindings).unwrap();
        let second = encode(program(ProgramType::Compute), &code, &bindings).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 256 + 256 + 12 + 2 * 16);

        let decoded = parse(&first).unwrap();
        assert_eq!(decoded.program, program(ProgramType::Compute));
        assert_eq!(&decoded.code[..code.len()], code);
        assert!(decoded.code[code.len()..].iter().all(|byte| *byte == 0));
        assert_eq!(decoded.bindings, bindings);
        assert_eq!(decoded.control_size, 256);
    }

    #[test]
    fn rejects_invalid_sections_and_program_ranges() {
        assert_eq!(
            encode(program(ProgramType::Vertex), &[], &[]),
            Err(Error::EmptyCode)
        );
        let mut invalid_program = program(ProgramType::Vertex);
        invalid_program.entrypoint = 8;
        assert_eq!(
            encode(invalid_program, &[0; 8], &[]),
            Err(Error::InvalidProgram)
        );

        let mut bytes = encode(program(ProgramType::Vertex), &[0; 16], &[]).unwrap();
        bytes[0] = b'X';
        assert!(matches!(parse(&bytes), Err(Error::InvalidMagic(_))));
    }

    #[test]
    fn rejects_trailing_or_unknown_binding_metadata() {
        let mut bytes = encode(program(ProgramType::Fragment), &[0; 16], &[]).unwrap();
        bytes.extend_from_slice(b"unexpected");
        assert_eq!(parse(&bytes), Err(Error::InvalidBindingMetadata));

        let mut bytes = encode(
            program(ProgramType::Fragment),
            &[0; 16],
            &[Binding {
                group: 0,
                binding: 0,
                target: 0,
                kind: BindingKind::Texture,
            }],
        )
        .unwrap();
        let kind = bytes.len() - 4;
        bytes[kind..].copy_from_slice(&99_u32.to_le_bytes());
        assert_eq!(parse(&bytes), Err(Error::InvalidBindingKind(99)));
    }
}
