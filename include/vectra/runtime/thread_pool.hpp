#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <cstdint>
#include <utility>

namespace vectra::runtime {

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    size_t num_threads() const { return workers_.size(); }

    // Deterministically partition [start, end) across all worker threads
    void parallel_for(size_t start, size_t end, const std::function<void(size_t r_start, size_t r_end)>& chunk_fn);

    static ThreadPool& get_default_pool();
    static void set_default_threads(size_t n);

private:
    void worker_loop(size_t thread_idx);

    std::vector<std::thread> workers_;
    std::vector<std::pair<size_t, size_t>> worker_ranges_;
    std::vector<uint64_t> worker_generations_;

    std::mutex mutex_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;

    std::function<void(size_t, size_t)> current_task_;
    uint64_t current_generation_ = 0;
    std::atomic<int32_t> active_count_{0};
    bool stop_ = false;
};

} // namespace vectra::runtime