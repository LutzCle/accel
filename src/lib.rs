//! acc: Write GPGPU code in Rust
//! ==============================
//!

#[macro_use]
extern crate procedurals;
extern crate glob;
extern crate cuda_sys as ffi;

pub mod ptx_builder;
pub mod uvec;
pub mod error;
