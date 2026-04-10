# 🚀 gemini-cuda

![C++17](https://img.shields.io/badge/C++-17-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-lightgrey)

**Eliminating O(n³) bottlenecks at the silicon level: Automated architectural auditing for next-generation GPU-accelerated solvers.**

`gemini-cuda` is a high-performance C++ utility bridging frontier LLM reasoning with GPU systems engineering. It performs deep architectural audits of NVIDIA `.cu` source code to identify synchronization errors, unoptimized memory patterns, and hardware-level bottlenecks in parallel solvers that traditional static analysis misses.

---

## 🎯 Goal

As GPU compute becomes the primary line item in AI infrastructure TCO, code efficiency at the kernel level is critical. `gemini-cuda` leverages frontier reasoning models with massive context windows to help engineering teams:

* **Reduce Warp Divergence:** Identify branch-heavy logic that degrades streaming multiprocessor (SM) throughput.
* **Eliminate Race Conditions:** Detect missing `__syncthreads()` in complex reduction algorithms across multi-file dependencies.
* **Optimize Memory Coalescing:** Ensure global memory access patterns are aligned for maximum memory bandwidth.

---

## 🚀 Quick Start

### 1. Install Dependencies (Ubuntu/Debian)
Ensure you have the required networking libraries to communicate with the API.
```bash
sudo apt-get update
sudo apt-get install libcurl4-openssl-dev cmake g++
```

### 2. Build 
We support two frameworks - Gemini and Claude, you can configure which one should be used on compile time using a make variable.

#### 1. Gemini

```bash
git clone [https://github.com/abokov/gemini-cuda.git](https://github.com/abokov/gemini-cuda.git)
cd gemini-cuda
mkdir build && cd build
cmake ..
make
```

#### 2. Claude

git clone [https://github.com/abokov/gemini-cuda.git](https://github.com/abokov/gemini-cuda.git)
cd gemini-cuda
mkdir build_claude && cd build_claude
cmake -DUSE_CLAUDE=ON ..
make


### 3. Configure the Environment
Copy the example environment file and add your Google AI Studio API key. 
```bash
cp ../.env.example ../.env
# Edit .env to add your keys, then source it:
export $(grep -v '^#' ../.env | xargs)
```

### 4. Run an Audit
Run the tool against any CUDA kernel. A broken reduction sample is included for testing.
```bash
export GEMINI_API_KEY="your_api_key_here"
./gemini-cuda ../samples/broken_reduction.cu
```

## 📊 Sample Output

When running against a kernel with hidden synchronization flaws, the engine outputs actionable, architecturally-aware fixes:

```text
🚀 Dispatching audit to: gemini-pro-latest...

--- AUDIT REPORT ---
[CRITICAL] Race Condition Detected:
Kernel `buggy_sum_reduction` accesses shared memory `sdata` without proper synchronization. 

[ANALYSIS]:
Threads are entering the reduction loop before all memory loads from `input` to `sdata` are complete across the block.

[RESOLUTION]:
Insert `__syncthreads()` at line 14, immediately before the `for` loop, to ensure all threads have finished writing to shared memory.
```

## 🧬 Architectural Bug Samples

The `samples/` directory contains deliberately flawed CUDA kernels designed to evaluate the engine's ability to detect deep silicon-level bottlenecks. You can run `gemini-cuda` against any of these to test the LLM's diagnostic accuracy:

* **`broken_reduction.cu`**: Demonstrates critical race conditions (missing `__syncthreads()`) and severe warp divergence caused by branching within a warp.
* **`uncoalesced_transpose.cu`**: Highlights global memory transaction overhead caused by strided, uncoalesced memory writes.
* **`bank_conflict_matmul.cu`**: Simulates n-way shared memory bank conflicts during column-wise reads in a tiled matrix multiplication kernel.
* **`atomic_bottleneck.cu`**: Shows extreme execution serialization by forcing an entire grid of threads to queue for a single global atomic counter.
* **`naive_softmax.cu`**: Exposes severe global memory bandwidth thrashing common in unoptimized GenAI/Attention mechanisms (missing kernel fusion).
* **`blocking_streams.cu`**: Simulates PCIe pipeline stalls caused by synchronous Host-to-Device memory transfers on the default stream.
* **`tail_effect_imbalance.cu`**: Demonstrates SM resource waste due to extreme thread-level workload imbalance and warp divergence.


## 📬 Contact & License

**Author:** Alexey Bokov  
**Contact:** [alex@bokov.net](mailto:alex@bokov.net)  
**License:** [Apache 2.0](LICENSE)



