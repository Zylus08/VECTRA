# Hardware Profile: Host Reconnaissance Report

## System Identification
- **Operating System**: Microsoft Windows 11 Home Single Language (64-bit, Version 10.0.26200)
- **Host Architecture**: x86_64 / AMD64

## CPU Specifications & Topology
- **Processor Model**: 12th Gen Intel(R) Core(TM) i7-12650H
- **Base / Max Clock Speed**: 2.30 GHz (Base), up to 4.70 GHz (Turbo)
- **Physical Cores**: 10 Cores
  - 6 Performance-cores (P-cores, hyperthreaded: 12 logical processors)
  - 4 Efficient-cores (E-cores, single-threaded: 4 logical processors)
- **Logical Processors**: 16

## Instruction Set Capabilities
- **AVX Support**: Yes (AVX, AVX2)
- **FMA Support**: Yes (FMA3)
- **F16C Support**: Yes
- **AVX-512 Support**: **No** (Intel Alder Lake / Golden Cove consumer hybrid disables AVX-512)
  - *Engineering Note*: All vectorized kernels must target AVX2 / 256-bit registers exclusively. AVX-512 instructions are strictly omitted.

## Memory Architecture & Budget
- **Total Physical RAM**: 16,437,236 KB (~16.0 GB)
- **Free Physical Memory**: ~2.9 GB available during active system workload
- **Allocation Strategy**:
  - Tiny test configurations: < 1 MB memory footprint
  - Target model (Qwen2.5-1.5B in 4-bit): ~1.2 GB weight footprint + KV cache
  - Strictly avoid unbudgeted multi-gigabyte in-memory duplicates

## Storage Geometry
- **Primary Volume (E:)**:
  - File System: NTFS
  - Total Capacity: ~350.5 GB
  - Used Space: ~239.2 GB
  - Free Space: **111.3 GB**
  - Role: Dedicated project directory, build cache, test fixtures, and artifacts
- **System Volume (C:)**:
  - Free Space: **~6.5 GB** (Constrained; forbidden for large model downloads or temporary artifacts)
