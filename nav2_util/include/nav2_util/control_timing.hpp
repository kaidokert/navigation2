// Copyright 2026 Jetacker contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef NAV2_UTIL__CONTROL_TIMING_HPP_
#define NAV2_UTIL__CONTROL_TIMING_HPP_

#include <algorithm>
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
inline void nameTimingThread(const char * name)
{
#ifdef __linux__
  const std::string short_name = std::string(name).substr(0, 15);
  char previous[16]{};
  pthread_getname_np(pthread_self(), previous, sizeof(previous));
  if (short_name != previous && pthread_setname_np(pthread_self(), short_name.c_str()) == 0) {
    std::fprintf(stderr, "[nav-thread] name=%s tid=%ld\n",
      short_name.c_str(), static_cast<long>(timingThreadId()));
  }
#else
  (void)name;
#endif
}

// One bounded collector per producer. No additional threads. Timer serialization
// occurs outside the producer lock; each active call contributes one fixed row.
class ControlTiming
{
public:
  using Clock = std::chrono::steady_clock;
  static constexpr size_t capacity = 1024;
  static constexpr size_t fields = 11;
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
    budget_ = budget;
    if (!enabled_) {return;}
    publisher_ = rclcpp::create_publisher<std_msgs::msg::String>(
      node, topic, rclcpp::QoS(4).reliable());
    {
      std::lock_guard<std::mutex> lock(mutex_);
      window_ = Window{};
      last_publish_ = Clock::time_point{};
    }
    window_start_ = Clock::now();
    timer_ = node->create_wall_timer(std::chrono::seconds(1), [this]() {publish();});
  }
  bool enabled() const {return enabled_;}
  void cleanup()
  {
    timer_.reset(); publisher_.reset(); enabled_ = false;
  }

  class Cycle;
  // Defined in nav2_util_core: shared across controller and plugin DSOs.
  static Cycle * & currentCycle();
  static void stageCurrent(size_t index, Clock::time_point start);

  class Cycle
  {
  public:
    explicit Cycle(ControlTiming & owner) : owner_(owner)
    {
      previous_ = currentCycle();
      currentCycle() = this;
      row_.fill(-1.0);
      if (owner_.enabled_) {
        start_ = Clock::now(); cpu_ = cpuSeconds(); tid_ = timingThreadId();
      }
    }
    ~Cycle() {finish(); currentCycle() = previous_;}
    void finish()
    {
      if (!owner_.enabled_ || finished_) {return;}
      finished_ = true;
      endMeasurement();
      owner_.add(row_, tid_, outcome_, row_[0] > owner_.budget_, missed_);
    }
    void endMeasurement()
    {
      if (owner_.enabled_ && row_[0] < 0) {
        row_[0] = seconds(Clock::now() - start_);
        row_[1] = cpuSeconds() - cpu_;
      }
    }
    void stage(size_t index, Clock::time_point start)
    {if (owner_.enabled_) {row_.at(index) = seconds(Clock::now() - start);}}
    void outcome(const char * value) {outcome_ = value;}
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
    const char * outcome_{"exception"};
    bool missed_{false};
    bool finished_{false};
  };
  void resetPublicationInterval() {last_publish_ = Clock::time_point{};}

private:
  struct Window
  {
    std::array<Row, capacity> rows{};
    size_t count{0}, stored{0}, misses{0}, completed{0}, cancelled{0}, exceptions{0};
    size_t scheduler_misses{0};
    int64_t tid{0};
  };
  void add(const Row & row, int64_t tid, const std::string & outcome, bool missed, bool scheduler_missed)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++window_.count;
    if (window_.stored < capacity) {window_.rows[window_.stored++] = row;}
    window_.misses += missed;
    window_.scheduler_misses += scheduler_missed;
    window_.completed += outcome == "completed";
    window_.cancelled += outcome == "cancelled";
    window_.exceptions += outcome == "exception";
    window_.tid = tid;
  }
  void publish()
  {
    auto snapshot = std::make_unique<Window>();
    const auto now = Clock::now();
    double duration;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      std::swap(*snapshot, window_);
      duration = seconds(now - window_start_); window_start_ = now;
    }
    static constexpr const char * names[fields] = {
      "elapsed", "thread_cpu", "publication_interval", "costmap_ready_wait",
      "path_tf", "costmap_lock_wait", "optimizer", "visualization",
      "parameter_lock_wait", "compute_publish", "global_path_update"};
    std::ostringstream out;
    out.precision(9);
    out << "{\"schema\":1,\"window_seconds\":" << duration
        << ",\"monotonic_seconds\":" << seconds(now.time_since_epoch())
        << ",\"thread_id\":" << snapshot->tid
        << ",\"samples\":" << snapshot->count
        << ",\"dropped_samples\":" << snapshot->count - snapshot->stored
        << ",\"completed\":" << snapshot->completed
        << ",\"cancelled\":" << snapshot->cancelled
        << ",\"exceptions\":" << snapshot->exceptions
        << ",\"deadline_budget_seconds\":" << budget_
        << ",\"scheduler_misses\":" << snapshot->scheduler_misses
        << ",\"deadline_misses\":" << snapshot->misses
        << ",\"deadline_miss_fraction\":"
        << (snapshot->count ? double(snapshot->misses) / snapshot->count : 0.0)
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
  }
  bool enabled_{false};
  double budget_{0};
  std::mutex mutex_;
  Window window_;
  Clock::time_point window_start_{}, last_publish_{};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};
}  // namespace nav2_util
#endif  // NAV2_UTIL__CONTROL_TIMING_HPP_
