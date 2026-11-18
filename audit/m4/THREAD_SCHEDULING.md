# Project VECTRA — Milestone M4: Hardware-Aware Thread Scheduling & Pool Architecture

**Component:** `vectra::runtime::ThreadPool`  
**Header:** `include/vectra/runtime/thread_pool.hpp`  
**Source:** `src/runtime/thread_pool.cpp`

---

## 1. Thread Pool Architecture

To eliminate the operating system overhead of repeated thread creation and destruction (which costs $\sim 50\text{ }\mu\text{s}$ to $100\text{ }\mu\text{s}$ per spawn on Windows 11), VECTRA maintains a persistent worker thread pool initialized at process startup:

```
                          Main Dispatcher Thread
                                    |
          [Parallel Task Submission: parallel_for(0, N, chunk_fn)]
                                    |
          +-------------------------+-------------------------+
          |                         |                         |
          v                         v                         v
   Worker Thread 0           Worker Thread 1           Worker Thread N-1
  [Slice 0 .. N/T]          [Slice N/T .. 2N/T]       [Slice (T-1)N/T .. N]
          |                         |                         |
          +-------------------------+-------------------------+
                                    |
            [Atomic active_count_ Decrement & Completion Signal]
                                    |
                        Return to Main Thread
```

### Generation-Counter Synchronization Contract
To prevent ABA race conditions and spurious wakeups without dynamic heap memory allocation:
1. Each worker thread maintains its local generation tracker `worker_generations_[thread_idx]`.
2. When the main thread dispatches a task, it sets `current_task_`, partitions contiguous index ranges `worker_ranges_[i] = {c_start, c_end}`, stores `active_count_ = n_workers`, and increments `current_generation_`.
3. Workers sleep on `std::condition_variable cv_start_` with predicate `(worker_generations_[thread_idx] < current_generation_)`, wake up, execute their slice, decrement `active_count_`, and signal `cv_done_` when the last worker completes.

---

## 2. Hardware Scaling Analysis on Alder Lake (Core i7-12650H)

The Intel Core i7-12650H contains 6 Performance cores (P-cores, 12 logical threads with Hyperthreading) and 4 Efficient cores (E-cores, 4 single-threaded cores).

We benchmarked matrix-vector multiplication scaling across thread counts ($T \in \{1, 2, 4, 6, 8, 10, 12, 16\}$):

| Thread Count | Microarchitecture Mapping | Matrix Latency ($1536 \times 1536$) | Scaling Efficiency | Microarchitectural Observations |
| :---: | :--- | :---: | :---: | :--- |
| **1** | 1 P-Core | $47.97\text{ ms}$ | $1.00\times$ (Ref) | Single thread baseline, zero inter-core synchronization. |
| **2** | 2 P-Cores | $24.81\text{ ms}$ | $1.93\times$ | Near-linear scaling across dedicated P-core L2 caches. |
| **4** | 4 P-Cores | $13.20\text{ ms}$ | $3.63\times$ | High scaling efficiency, L3 bandwidth scaling smoothly. |
| **6** | 6 P-Cores (Physical) | **$9.15\text{ ms}$** | **$5.24\times$** | **Optimal P-Core Utilization**: No SMT port contention. |
| **8** | 6 P-Cores + 2 E-Cores | $7.82\text{ ms}$ | $6.13\times$ | Heterogeneous scheduling: E-cores assist chunk processing. |
| **10** | **All 10 Physical Cores** | **$6.94\text{ ms}$** | **$6.91\times$** | **Optimal Full-Chip Configuration**: Default VECTRA setting. |
| **12** | 6 P-Cores (Hyperthreaded) | $8.45\text{ ms}$ | $5.68\times$ | SMT degradation: AVX2 port 0/1 contention on P-cores. |
| **16** | 16 Logical Threads | $9.20\text{ ms}$ | $5.21\times$ | Tail latency penalty: E-cores throttle slower logical threads. |

---

## 3. Findings & Engineering Conclusions

1. **Physical vs Logical Threading:** Running 10 threads (matching physical core count) outperforms 16 logical threads by **$1.32\times$**, because AVX2 dense integer and floating-point pipelines heavily saturate execution ports, causing severe SMT thrashing when sibling hyperthreads compete for the same execution units.
2. **Deterministic Partitioning:** VECTRA's static contiguous range partitioning ensures $100\%$ bit-exact reproducibility across repeated executions, certified by regression suite `test_thread_pool_determinism`.
