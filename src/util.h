#pragma once
// Small shared helpers: file IO, JSON, timing, and a minimal parallel-for.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "nlohmann/json.hpp"

namespace exl3 {

inline int64_t now_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

inline bool read_file_text(const std::string& path, std::string& out, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { err = "cannot open " + path; return false; }
  f.seekg(0, std::ios::end);
  std::streamoff n = f.tellg();
  f.seekg(0);
  out.resize((size_t)n);
  f.read(out.data(), n);
  if (!f) { err = "short read on " + path; return false; }
  return true;
}

inline bool parse_json_file(const std::string& path, nlohmann::json& out, std::string& err) {
  std::string text;
  if (!read_file_text(path, text, err)) return false;
  try {
    out = nlohmann::json::parse(text);
  } catch (const std::exception& e) {
    err = std::string("invalid JSON in ") + path + ": " + e.what();
    return false;
  }
  return true;
}

inline int default_threads() {
  unsigned n = std::thread::hardware_concurrency();
  return n ? (int)n : 1;
}

// Run f(i) for i in [0, n) across `threads` workers.
template <typename F>
void parallel_for(int64_t n, int threads, F&& f) {
  if (n <= 0) return;
  if (threads <= 1 || n == 1) {
    for (int64_t i = 0; i < n; ++i) f(i);
    return;
  }
  std::atomic<int64_t> next{0};
  auto worker = [&]() {
    for (;;) {
      int64_t i = next.fetch_add(1, std::memory_order_relaxed);
      if (i >= n) break;
      f(i);
    }
  };
  std::vector<std::thread> pool;
  pool.reserve((size_t)threads - 1);
  for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
  worker();
  for (auto& t : pool) t.join();
}

} // namespace exl3
