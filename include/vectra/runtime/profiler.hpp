#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <iostream>
#include <iomanip>

namespace vectra::runtime {

struct ProfileRecord {
    std::string name;
    uint64_t call_count = 0;
    double total_time_ms = 0.0;
    uint64_t bytes_processed = 0;
};

class Profiler {
public:
    static Profiler& instance() {
        static Profiler inst;
        return inst;
    }

    void reset() {
        records_.clear();
    }

    void record(const std::string& name, double elapsed_ms, uint64_t bytes = 0) {
        auto& r = records_[name];
        r.name = name;
        r.call_count++;
        r.total_time_ms += elapsed_ms;
        r.bytes_processed += bytes;
    }

    void print_summary(std::ostream& os = std::cout) const {
        double total_ms = 0.0;
        for (const auto& [name, rec] : records_) {
            total_ms += rec.total_time_ms;
        }

        os << "========================================================================================\n";
        os << "                                OPERATOR EXECUTION PROFILE                              \n";
        os << "========================================================================================\n";
        os << std::left << std::setw(24) << "Operator"
           << std::right << std::setw(10) << "Calls"
           << std::setw(16) << "Total Time (ms)"
           << std::setw(16) << "Avg Time (ms)"
           << std::setw(12) << "% Total"
           << std::setw(16) << "Throughput (GB/s)\n";
        os << "----------------------------------------------------------------------------------------\n";

        for (const auto& [name, rec] : records_) {
            double avg_ms = rec.call_count ? (rec.total_time_ms / rec.call_count) : 0.0;
            double pct = total_ms > 0 ? (rec.total_time_ms / total_ms * 100.0) : 0.0;
            double gb_per_sec = (rec.total_time_ms > 0 && rec.bytes_processed > 0)
                ? (static_cast<double>(rec.bytes_processed) / (rec.total_time_ms * 1e6))
                : 0.0;

            os << std::left << std::setw(24) << name
               << std::right << std::setw(10) << rec.call_count
               << std::fixed << std::setprecision(2)
               << std::setw(16) << rec.total_time_ms
               << std::setw(16) << avg_ms
               << std::setprecision(1)
               << std::setw(11) << pct << "%"
               << std::setprecision(2)
               << std::setw(16) << gb_per_sec << "\n";
        }
        os << "----------------------------------------------------------------------------------------\n";
        os << "Total Profiled Time: " << std::fixed << std::setprecision(2) << total_ms << " ms\n";
        os << "========================================================================================\n";
    }

    const std::unordered_map<std::string, ProfileRecord>& records() const { return records_; }

private:
    Profiler() = default;
    std::unordered_map<std::string, ProfileRecord> records_;
};

class ScopeTimer {
public:
    ScopeTimer(const std::string& name, uint64_t bytes = 0)
        : name_(name), bytes_(bytes), start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopeTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        Profiler::instance().record(name_, ms, bytes_);
    }

private:
    std::string name_;
    uint64_t bytes_;
    std::chrono::high_resolution_clock::time_point start_;
};

} // namespace vectra::runtime