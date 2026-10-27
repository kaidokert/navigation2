// Copyright 2026 Jetacker contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef NAV2_UTIL__CONTROL_TIMING_HPP_
#define NAV2_UTIL__CONTROL_TIMING_HPP_

#include <algorithm>
#include <atomic>
#include <iomanip>
#include <stdexcept>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <utility>
#include <ctime>
#include <cstdio>
#include <cstring>
#ifdef __linux__
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace nav2_util
{
inline int64_t timingThreadId()
{
#ifdef __linux__
  return syscall(SYS_gettid);
#else
  return 0;
#endif
}
inline void nameTimingThread(const char * name, bool log_id = true)
{
#ifdef __linux__
  const std::string short_name = std::string(name).substr(0, 15);
  char previous[16]{};
  pthread_getname_np(pthread_self(), previous, sizeof(previous));
  if (short_name != previous && pthread_setname_np(pthread_self(), short_name.c_str()) == 0) {
    if (log_id) {
      std::fprintf(stderr, "[nav-thread] name=%s tid=%s\n",
      short_name.c_str(), std::to_string(timingThreadId()).c_str());
    }
  }
#else
  (void)name;
  (void)log_id;
#endif
}

// One bounded collector per producer. No additional threads. Timer serialization
// occurs outside the producer lock; each active call contributes one fixed row.
class ControlTiming
{
public:
  using Clock = std::chrono::steady_clock;
  static constexpr size_t capacity = 1024;
  enum class Stage : size_t
  {
    Elapsed, ThreadCpu, PublicationInterval, CostmapReadyWait, PathTf, CostmapLockWait,
    Optimizer, Visualization, ParameterLockWait, ComputePublish, GlobalPathUpdate, Count
  };
  enum class Outcome {Exception, Completed, Cancelled, Inactive};
  static constexpr size_t fields = static_cast<size_t>(Stage::Count);
  using Row = std::array<double, fields>;
  static double seconds(Clock::duration d)
  {return std::chrono::duration<double>(d).count();}
  static double cpuSeconds()
  {
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
  }
  template<class Node>
  void configure(Node node, const std::string & topic, double budget)
  {
    cleanup();
    if (!node->has_parameter("timing_enabled")) {
      node->declare_parameter("timing_enabled", false);
    }
    enabled_ = node->get_parameter("timing_enabled").as_bool();
    setBudget(budget);
    if (!enabled_) {return;}
    publisher_ = rclcpp::create_publisher<std_msgs::msg::String>(
      node, topic, rclcpp::QoS(4).reliable());
    {
      std::lock_guard<std::mutex> lock(mutex_);
      window_ = std::make_unique<Window>();
      spare_ = std::make_unique<Window>();
      last_publish_ = Clock::time_point{};
    }
    window_start_ = Clock::now();
    timer_ = node->create_wall_timer(std::chrono::seconds(1), [this]() {publish();});
  }
  bool enabled() const {return enabled_;}
  void setBudget(double budget)
  {
    if (!std::isfinite(budget) || budget <= 0) {
      throw std::invalid_argument("Timing budget must be finite and positive");
    }
    budget_.store(budget);
  }
  void cleanup()
  {
    timer_.reset(); publisher_.reset(); enabled_ = false;
  }

  class Cycle;
  // Defined in nav2_util_core: shared across controller and plugin DSOs.
  static Cycle * & currentCycle();
  static void stageCurrent(Stage index, Clock::time_point start);

  class Cycle
  {
public:
    explicit Cycle(ControlTiming & owner)
    : owner_(owner)
    {
      previous_ = currentCycle();
      currentCycle() = this;
      row_.fill(-1.0);
      if (owner_.enabled_) {
        start_ = Clock::now(); cpu_ = cpuSeconds(); tid_ = timingThreadId();
        cycle_budget_ = owner_.budget_.load();
      }
    }
    ~Cycle() {finish(); currentCycle() = previous_;}
    void finish()
    {
      if (!owner_.enabled_ || finished_) {return;}
      finished_ = true;
      endMeasurement();
      owner_.add(row_, tid_, outcome_, cycle_budget_, missed_);
    }
    void endMeasurement()
    {
      if (owner_.enabled_ && row_[0] < 0) {
        row_[0] = seconds(Clock::now() - start_);
        row_[1] = cpuSeconds() - cpu_;
      }
    }
    void stage(Stage index, Clock::time_point start)
    {if (owner_.enabled_) {row_.at(static_cast<size_t>(index)) = seconds(Clock::now() - start);}}
    void outcome(Outcome value) {outcome_ = value;}
    void missed(bool value) {missed_ = value;}
    void published()
    {
      if (!owner_.enabled_) {return;}
      const auto now = Clock::now();
      // Called by the same producer thread. Reset between action executions.
      if (owner_.last_publish_ != Clock::time_point{}) {
        row_[2] = seconds(now - owner_.last_publish_);
      }
      owner_.last_publish_ = now;
    }

private:
    Cycle * previous_{nullptr};
    ControlTiming & owner_;
    Row row_;
    Clock::time_point start_{};
    double cpu_{0};
    int64_t tid_{0};
    Outcome outcome_{Outcome::Exception};
    double cycle_budget_{0};
    bool missed_{false};
    bool finished_{false};
  };
  void resetPublicationInterval() {last_publish_ = Clock::time_point{};}

private:
  struct Window
  {
    // Rows beyond stored are never read; reusing a window resets counters only.
    std::array<Row, capacity> rows;
    size_t count, stored, misses, completed, cancelled, exceptions, inactive;
    size_t scheduler_misses;
    int64_t tid;
    double budget_min, budget_max;
    Window() {reset();}
    void reset()
    {
      count = stored = misses = completed = cancelled = exceptions = inactive = 0;
      scheduler_misses = 0; tid = 0; budget_min = budget_max = 0;
    }
  };
  void add(
    const Row & row, int64_t tid, Outcome outcome,
    double budget, bool scheduler_missed)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto & w = *window_;
    if (w.count == 0) {w.budget_min = w.budget_max = budget;}
    w.budget_min = std::min(w.budget_min, budget);
    w.budget_max = std::max(w.budget_max, budget);
    ++w.count;
    if (w.stored < capacity) {w.rows[w.stored++] = row;}
    w.misses += row[0] > budget;
    w.scheduler_misses += scheduler_missed;
    w.completed += outcome == Outcome::Completed;
    w.cancelled += outcome == Outcome::Cancelled;
    w.exceptions += outcome == Outcome::Exception;
    w.inactive += outcome == Outcome::Inactive;
    w.tid = tid;
  }
  void publish()
  {
    auto snapshot = std::move(spare_);
    const auto now = Clock::now();
    double duration;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      snapshot.swap(window_);  // O(1): no rows copied while blocking the producer.
      duration = seconds(now - window_start_); window_start_ = now;
    }
    if (snapshot->count == 0) {
      snapshot->budget_min = snapshot->budget_max = budget_.load();
    }
    static constexpr const char * names[fields] = {
      "elapsed", "thread_cpu", "publication_interval", "costmap_ready_wait",
      "path_tf", "costmap_lock_wait", "optimizer", "visualization",
      "parameter_lock_wait", "compute_publish", "global_path_update"};
    std::ostringstream out;
    out << std::fixed << std::setprecision(9);
    out << "{\"schema\":1,\"window_seconds\":" << duration
        << ",\"monotonic_seconds\":" << seconds(now.time_since_epoch())
        << ",\"thread_id\":" << snapshot->tid
        << ",\"samples\":" << snapshot->count
        << ",\"dropped_samples\":" << snapshot->count - snapshot->stored
        << ",\"completed\":" << snapshot->completed
        << ",\"cancelled\":" << snapshot->cancelled
        << ",\"exceptions\":" << snapshot->exceptions
        << ",\"inactive\":" << snapshot->inactive
        << ",\"deadline_budget_min_seconds\":" << snapshot->budget_min
        << ",\"deadline_budget_max_seconds\":" << snapshot->budget_max
        << ",\"deadline_budget_seconds\":";
    if (snapshot->budget_min == snapshot->budget_max) {out << snapshot->budget_min;} else {
      out << "null";
    }
    out
        << ",\"scheduler_misses\":" << snapshot->scheduler_misses
        << ",\"deadline_misses\":" << snapshot->misses
        << ",\"deadline_miss_fraction\":"
        << (snapshot->count ? static_cast<double>(snapshot->misses) / snapshot->count : 0.0)
        << ",\"metrics_seconds\":{";
    for (size_t f = 0; f < fields; ++f) {
      std::vector<double> values;
      values.reserve(snapshot->stored);
      for (size_t i = 0; i < snapshot->stored; ++i) {
        if (snapshot->rows[i][f] >= 0) {values.push_back(snapshot->rows[i][f]);}
      }
      std::sort(values.begin(), values.end());
      if (f) {out << ',';}
      out << '\"' << names[f] << "\":{\"samples\":" << values.size();
      if (!values.empty()) {
        auto percentile = [&](double q) {
            return values[static_cast<size_t>(std::ceil(q * values.size())) - 1];
          };
        out << ",\"p50\":" << percentile(.50) << ",\"p95\":" << percentile(.95)
            << ",\"p99\":" << percentile(.99) << ",\"max\":" << values.back();
      }
      out << '}';
    }
    out << "}}";
    std_msgs::msg::String message;
    message.data = out.str(); publisher_->publish(message);
    snapshot->reset();
    spare_ = std::move(snapshot);
  }
  bool enabled_{false};
  std::atomic<double> budget_{0};
  std::mutex mutex_;
  std::unique_ptr<Window> window_, spare_;
  Clock::time_point window_start_{}, last_publish_{};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};
}  // namespace nav2_util
#endif  // NAV2_UTIL__CONTROL_TIMING_HPP_
