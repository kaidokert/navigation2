// Copyright 2026 Kaido Kert. Apache-2.0.
#ifndef NAV2_MPPI_CONTROLLER__TOOLS__DECISION_RECORDER_HPP_
#define NAV2_MPPI_CONTROLLER__TOOLS__DECISION_RECORDER_HPP_

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <fcntl.h>
#include <optional>
#include <string>
#include <unistd.h>
#include <nlohmann/json.hpp>

namespace mppi::recorder
{
using Json = nlohmann::json;
inline uint64_t monotonicNs()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline bool finite(double value)
{
  uint64_t bits; std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
template<typename Tensor> Json tensor(const Tensor & t)
{
  Json data = Json::array(), shape = Json::array();
  if (t.size() > 2048 * 128) {throw std::runtime_error("tensor capture bound");}
  for (auto n : t.shape()) {shape.push_back(n);}
  for (auto x : t) {
    if (!finite(static_cast<double>(x))) {throw std::runtime_error("nonfinite tensor");}
    data.push_back(x);
  }
  return Json{{"shape", shape}, {"data", data}};
}
template<typename Sequence> Json sequence(const Sequence & s)
{return Json{{"vx", tensor(s.vx)}, {"vy", tensor(s.vy)}, {"wz", tensor(s.wz)}};}
template<typename History> Json history(const History & h)
{
  Json j = Json::array();
  for (const auto & c : h) {j.push_back({c.vx, c.vy, c.wz});}
  return j;
}
template<typename Pose> Json pose(const Pose & p)
{
  return Json{{"position", {p.position.x, p.position.y, p.position.z}},
    {"quaternion", {p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w}}};
}
template<typename Path> Json rosPath(const Path & p)
{
  Json points = Json::array();
  if (p.poses.size() > 8192) {throw std::runtime_error("path capture bound");}
  for (const auto & x : p.poses) {points.push_back(pose(x.pose));}
  return Json{{"frame", p.header.frame_id}, {"poses", points}};
}
template<typename Path> Json path(const Path & p)
{return Json{{"x", tensor(p.x)}, {"y", tensor(p.y)}, {"yaw", tensor(p.yaws)}};}

struct Config
{
  size_t pre{3}, post{2}, maximum_invocations{128}, maximum_bytes{256 * 1024 * 1024};
  unsigned collapse_cycles{20}, reversal_count{4};
  double collapse_speed{0.02}, progress_floor{0.005}, moving_command{0.05};
};
class Recorder
{
public:
  explicit Recorder(Json manifest, Config config = {})
  : manifest_(std::move(manifest)), config_(config)
  {
    if (!config_.pre || !config_.maximum_invocations || !config_.maximum_bytes ||
      !config_.collapse_cycles || manifest_.dump().size() > 65536)
    {throw std::invalid_argument("invalid recorder bounds or oversized manifest");}
  }
  bool open() const {return current_.has_value();}
  void begin(int64_t stamp_ns) noexcept
  {
    try {
      if (last_stamp_ && stamp_ns < *last_stamp_) {++epoch_; ++clock_resets_; low_=0; reversals_=0; previous_sign_=0;}
      last_stamp_ = stamp_ns;
      retry = iteration = bank = 0;
      current_ = Json{{"invocation_id", ++next_id_}, {"clock_epoch", epoch_},
        {"stamp_ns", stamp_ns}, {"host_start_ns", monotonicNs()},
        {"mode", mode}, {"events", Json::array()}, {"complete", true}};
    } catch (...) {++errors_; current_.reset();}
  }
  template<typename Builder> void event(const char * kind, Builder build) noexcept
  {
    if (!current_) {return;}
    try {
      if ((*current_)["events"].size() >= 10000) {throw std::runtime_error("event bound");}
      (*current_)["events"].push_back(Json{{"sequence", ++event_id_},
        {"kind", kind}, {"retry_id", retry}, {"iteration_id", iteration},
        {"bank_id", bank}, {"host_ns", monotonicNs()}, {"payload", build()}});
    } catch (...) {++errors_; (*current_)["complete"] = false;}
  }
  void end() noexcept
  {
    if (!current_) {return;}
    try {
      (*current_)["host_end_ns"] = monotonicNs();
      Json frame = std::move(*current_); current_.reset();
      bool eligible = frame["mode"] == "active";
      bool has_command = false; double vx = 0;
      for (const auto & e : frame["events"]) {
        if (e["kind"] == "issued") {vx = e["payload"]["vx"]; has_command = true;}
      }
      low_ = eligible && has_command && std::abs(vx) < config_.collapse_speed ? low_ + 1 : 0;
      const int sign = vx > config_.collapse_speed ? 1 : vx < -config_.collapse_speed ? -1 : 0;
      if (eligible && sign && previous_sign_ && sign != previous_sign_) {++reversals_;}
      if (eligible && sign) {previous_sign_ = sign;}
      if (!eligible) {reversals_ = 0; previous_sign_ = 0;}
      if (eligible && low_ >= config_.collapse_cycles) {trigger("command_speed_collapse");}
      if (eligible && reversals_ >= config_.reversal_count) {trigger("repeated_reversals");}
      if (eligible && independent_progress && delivered_speed &&
        std::abs(*delivered_speed) >= config_.moving_command &&
        std::abs(*independent_progress) < config_.progress_floor)
      {trigger("delivered_command_without_progress");}
      if (frame.contains("exception")) {trigger("abort");}
      const auto bytes = frame.dump().size();
      if (bytes > config_.maximum_bytes) {++overflow_; return;}
      if (!triggered_) {
        while (frames_.size() >= config_.pre && !frames_.empty()) {evict();}
      } else if (post_seen_ >= config_.post && frozen_) {++after_window_; return;}
      while (!frames_.empty() && (bytes_ + bytes > config_.maximum_bytes ||
        frames_.size() >= config_.maximum_invocations))
      {if (triggered_) {++overflow_;} evict();}
      if (frames_.size() >= config_.maximum_invocations) {++overflow_; return;}
      frames_.push_back(std::move(frame)); sizes_.push_back(bytes); bytes_ += bytes;
      if (triggered_ && ++post_seen_ > config_.post) {frozen_ = true;}
    } catch (...) {++errors_; current_.reset();}
  }
  void trigger(const std::string & reason) noexcept
  {
    if (!triggered_) {triggered_ = true; reason_ = reason; post_seen_ = 0;}
  }
  void exception(const char * message) noexcept
  {try {if (current_) {(*current_)["exception"] = message;}} catch (...) {++errors_;}}
  Json document() const
  {
    return Json{{"schema", "jetacker-decision-trace/1.0.0"}, {"manifest", manifest_},
      {"termination", reason_}, {"triggered", triggered_}, {"frozen", frozen_},
      {"health", {{"errors", errors_}, {"overflow", overflow_}, {"pre_evictions", evictions_},
        {"after_window", after_window_}, {"clock_resets", clock_resets_},
        {"retained_serialized_bytes", bytes_}, {"maximum_bytes", config_.maximum_bytes}}},
      {"invocations", frames_}};
  }
  void writeExclusive(const std::string & filename) const
  {
    const auto content = document().dump() + "\n";
    int fd = ::open(filename.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {throw std::runtime_error("exclusive trace open failed");}
    size_t written = 0;
    while (written < content.size()) {
      auto n = ::write(fd, content.data() + written, content.size() - written);
      if (n <= 0) {::close(fd); throw std::runtime_error("trace write failed");}
      written += static_cast<size_t>(n);
    }
    if (::close(fd)) {throw std::runtime_error("trace close failed");}
  }
  size_t retry{0}, iteration{0}, bank{0};
  std::string mode{"active"};
  std::optional<double> independent_progress, delivered_speed;
private:
  void evict() {bytes_ -= sizes_.front(); sizes_.pop_front(); frames_.pop_front(); ++evictions_;}
  Json manifest_; Config config_; std::optional<Json> current_;
  std::deque<Json> frames_; std::deque<size_t> sizes_;
  std::optional<int64_t> last_stamp_;
  uint64_t epoch_{0}, next_id_{0}, event_id_{0};
  size_t bytes_{0}, errors_{0}, overflow_{0}, evictions_{0}, after_window_{0}, clock_resets_{0};
  unsigned low_{0}, reversals_{0}; int previous_sign_{0};
  bool triggered_{false}, frozen_{false}; size_t post_seen_{0};
  std::string reason_{"not_triggered"};
};
inline thread_local Recorder * active = nullptr;
struct Session
{
  Recorder * previous;
  explicit Session(Recorder & recorder) : previous(active) {active = &recorder;}
  ~Session() {active = previous;}
  Session(const Session &) = delete;
  Session & operator=(const Session &) = delete;
};
struct Invocation
{
  Recorder * r{active}; bool owns{false}; int exceptions;
  explicit Invocation(int64_t stamp_ns) : exceptions(std::uncaught_exceptions())
  {if (r && !r->open()) {owns = true; r->begin(stamp_ns);}}
  ~Invocation()
  {if (r && owns) {if (std::uncaught_exceptions() > exceptions) {r->exception("native_exception");} r->end();}}
};
template<typename Builder> void observe(const char * kind, Builder build) noexcept
{if (active) {active->event(kind, build);}}
}  // namespace mppi::recorder
#endif
