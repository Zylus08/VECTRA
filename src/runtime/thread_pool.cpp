#include "vectra/runtime/thread_pool.hpp"
#include <algorithm>

namespace vectra::runtime {

static size_t g_configured_threads = 0;

ThreadPool& ThreadPool::get_default_pool() {
    static ThreadPool pool(g_configured_threads);
    return pool;
}

void ThreadPool::set_default_threads(size_t n) {
    g_configured_threads = n;
}

ThreadPool::ThreadPool(size_t num_threads) {
    if (num_threads == 0) {
        unsigned int hw = std::thread::hardware_concurrency();
        num_threads = hw > 0 ? hw : 4;
        if (num_threads > 10) {
            num_threads = 10; // Default to 10 for Core i7-12650H (6P + 4E)
        }
    }

    workers_.reserve(num_threads);
    worker_ranges_.resize(num_threads, {0, 0});
    worker_generations_.resize(num_threads, 0);

    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&ThreadPool::worker_loop, this, i);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        stop_ = true;
        cv_start_.notify_all();
    }

    for (auto& w : workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
}

void ThreadPool::worker_loop(size_t thread_idx) {
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_start_.wait(lock, [this, thread_idx]() {
                return stop_ || (worker_generations_[thread_idx] < current_generation_);
            });

            if (stop_) {
                return;
            }

            worker_generations_[thread_idx] = current_generation_;
        }

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

void ThreadPool::parallel_for(size_t start, size_t end, const std::function<void(size_t r_start, size_t r_end)>& chunk_fn) {
    if (start >= end) return;
    size_t count = end - start;

    size_t n_workers = workers_.size();
    if (n_workers <= 1 || count <= 1) {
        chunk_fn(start, end);
        return;
    }

    size_t slice = (count + n_workers - 1) / n_workers;

    {
        std::unique_lock<std::mutex> lock(mutex_);
        current_task_ = chunk_fn;
        for (size_t i = 0; i < n_workers; ++i) {
            size_t c_start = start + i * slice;
            size_t c_end = std::min(c_start + slice, end);
            worker_ranges_[i] = {c_start, c_end};
        }
        active_count_.store(static_cast<int32_t>(n_workers), std::memory_order_release);
        current_generation_++;
        cv_start_.notify_all();
    }

    // Wait until all workers finish their assigned slices
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_done_.wait(lock, [this]() {
            return active_count_.load(std::memory_order_acquire) == 0;
        });
    }
}

} // namespace vectra::runtime