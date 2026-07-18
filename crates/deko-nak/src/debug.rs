//! Standalone debug settings consumed by the extracted Mesa NAK passes.

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Debug;

/// Deterministic standalone configuration: all Mesa debug flags are disabled.
pub static DEBUG: Debug = Debug;

pub trait GetDebugFlags {
    fn annotate(&self) -> bool;
    fn cycles(&self) -> bool;
    fn print(&self) -> bool;
    fn serial(&self) -> bool;
    fn spill(&self) -> bool;
}

impl GetDebugFlags for Debug {
    #[inline]
    fn annotate(&self) -> bool {
        false
    }
    #[inline]
    fn cycles(&self) -> bool {
        false
    }
    #[inline]
    fn print(&self) -> bool {
        false
    }
    #[inline]
    fn serial(&self) -> bool {
        false
    }
    #[inline]
    fn spill(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::{DEBUG, Debug, GetDebugFlags};

    #[test]
    fn standalone_debug_flags_are_disabled() {
        for flags in [&DEBUG, &Debug] {
            assert!(!flags.annotate());
            assert!(!flags.cycles());
            assert!(!flags.print());
            assert!(!flags.serial());
            assert!(!flags.spill());
        }
    }
}
