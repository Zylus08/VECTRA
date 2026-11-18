#include "vectra/runtime/thread_pool.hpp"
#include <algorithm>
#include <immintrin.h>

namespace vectra::runtime {

static size_t g_configured_threads = 0;

ThreadPool &ThreadPool::get_default_pool() {
  static ThreadPool pool(g_configured_threads);
  return pool;
}

void ThreadPool::set_default_threads(size_t n) { g_configured_threads = n; }

ThreadPool::ThreadPool(size_t num_threads) {
  if (num_threads == 0) {
    unsigned int hw = std::thread::hardware_concurrency();
    num_threads = hw > 0 ? hw : 4;
    if (num_threads > 10) {
      num_threads = 10; // Default to 10 for Core i7-12650H (6P + 4E)
    }
  }

  size_t n_workers = (num_threads > 1) ? (num_threads - 1) : 0;
  workers_.reserve(n_workers);
  worker_ranges_.resize(n_workers, {0, 0});
  worker_generations_.resize(n_workers, 0);

  for (size_t i = 0; i < n_workers; ++i) {
    workers_.emplace_back(&ThreadPool::worker_loop, this, i);
  }
}

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    stop_.store(true, std::memory_order_relaxed);
    cv_start_.notify_all();
  }

  for (auto &w : workers_) {
    if (w.joinable()) {
      w.join();
    }
  }
}

void ThreadPool::worker_loop(size_t thread_idx) {
  while (true) {
    uint64_t my_gen = worker_generations_[thread_idx];
    bool has_work = false;

    // Fast path: bounded active spin-pause before OS sleep
    for (int spin = 0; spin < 5000; ++spin) {
      if (stop_.load(std::memory_order_relaxed)) {
        return;
      }
      if (current_generation_.load(std::memory_order_acquire) > my_gen) {
        has_work = true;
        break;
      }
      _mm_pause();
    }

    // Slow path: condition variable wait
    if (!has_work) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_start_.wait(lock, [this, thread_idx]() {
        return stop_.load(std::memory_order_relaxed) ||
               (worker_generations_[thread_idx] <
                current_generation_.load(std::memory_order_relaxed));
      });

      if (stop_.load(std::memory_order_relaxed)) {
        return;
      }
    }

    worker_generations_[thread_idx] =
        current_generation_.load(std::memory_order_acquire);

    // Execute assigned slice
    size_t r_start = worker_ranges_[thread_idx].first;
    size_t r_end = worker_ranges_[thread_idx].second;
    if (r_start < r_end && current_task_) {
      current_task_(r_start, r_end);
    }

    // Signal completion
    if (active_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_done_.notify_one();
    }
  }
}

void ThreadPool::parallel_for(
    size_t start, size_t end,
    const std::function<void(size_t r_start, size_t r_end)> &chunk_fn) {
  if (start >= end)
    return;
  size_t count = end - start;

  size_t n_workers = workers_.size();
  if (n_workers == 0 || count <= 1) {
    chunk_fn(start, end);
    return;
  }

  size_t total_threads = n_workers + 1;
  size_t slice = (count + total_threads - 1) / total_threads;

  size_t c_start_0 = start;
  size_t c_end_0 = std::min(start + slice, end);

  {
    std::unique_lock<std::mutex> lock(mutex_);
    current_task_ = chunk_fn;
    for (size_t i = 0; i < n_workers; ++i) {
      size_t c_start = start + (i + 1) * slice;
      size_t c_end = (c_start < end) ? std::min(c_start + slice, end) : end;
      worker_ranges_[i] = {c_start, c_end};
    }
    active_count_.store(static_cast<int32_t>(n_workers),
                        std::memory_order_release);
    current_generation_.fetch_add(1, std::memory_order_release);
    cv_start_.notify_all();
  }

  // Caller executes slice 0 immediately with zero dispatch delay
  if (c_start_0 < c_end_0) {
    chunk_fn(c_start_0, c_end_0);
  }

  // Wait until all workers finish their assigned slices
  // Fast path: bounded active spin-pause before OS sleep
  bool done = false;
  for (int spin = 0; spin < 10000; ++spin) {
    if (active_count_.load(std::memory_order_acquire) == 0) {
      done = true;
      break;
    }
    _mm_pause();
  }

  if (!done) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [this]() {
      return active_count_.load(std::memory_order_acquire) == 0;
    });
  }
}

} // namespace vectra::runtime