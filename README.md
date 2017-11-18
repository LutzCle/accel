Accel: GPGPU Framework for Rust
================================

[![pipeline status](https://gitlab.com/termoshtt/accel/badges/master/pipeline.svg)](https://gitlab.com/termoshtt/accel/commits/master)

CUDA-based GPGPU framework for Rust

Features
---------

- Compile PTX Kernel from Rust using NVPTX backend of LLVM (demonstrated in [japaric/nvptx](https://github.com/japaric/nvptx))
- [proc-macro-attribute](https://github.com/rust-lang/rust/issues/38356)-based approach like [futures-await](https://github.com/alexcrichton/futures-await)
- Simple memory management using [Unified Memory](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)

Example
--------

```rust
#![feature(proc_macro)]

extern crate accel;
extern crate accel_derive;

use accel_derive::kernel;
use accel::*;

#[kernel]
pub unsafe fn add(a: *const f64, b: *const f64, c: *mut f64, n: usize) {
    let i = accel_core::index();
    if (i as usize) < n {
        *c.offset(i) = *a.offset(i) + *b.offset(i);
    }
}

fn main() {
    let n = 8;
    let mut a = UVec::new(n).unwrap();
    let mut b = UVec::new(n).unwrap();
    let mut c = UVec::new(n).unwrap();

    for i in 0..n {
        a[i] = i as f64;
        b[i] = 2.0 * i as f64;
    }
    println!("a = {:?}", a.as_slice());
    println!("b = {:?}", b.as_slice());

    let grid = Grid::x(64);
    let block = Block::x(64);
    add(grid, block, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n);

    device::sync().unwrap();
    println!("c = {:?}", c.as_slice());
}
```

Licence
--------
MIT-License