// Copyright 2026 Kaido Kert. Apache-2.0.
#ifndef NAV2_MPPI_CONTROLLER__TOOLS__LIVE_DECISION_RECORDER_HPP_
#define NAV2_MPPI_CONTROLLER__TOOLS__LIVE_DECISION_RECORDER_HPP_
#include "nav2_mppi_controller/tools/decision_recorder.hpp"
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

namespace mppi::recorder
{
// Opt-in observation adapter. The control thread owns each mutable Recorder.
// Only detached recorders are passed to the bounded serialization worker.
class LiveRecorder
{
  struct Job
  {
    std::string folder, name;
    Json row;
    std::unique_ptr<Recorder> capture;
  };
public:
  explicit LiveRecorder(std::string root, Json manifest)
  : root_(std::move(root)), manifest_(std::move(manifest)), worker_([this] {work();}) {}
  ~LiveRecorder()
  {
    try {std::lock_guard<std::mutex> lock(mutex_); detach("shutdown");}
    catch (...) {++worker_errors_;}
    {std::lock_guard<std::mutex> lock(queue_mutex_); shutdown_ = true;}
    wake_.notify_one(); worker_.join();
  }
  LiveRecorder(const LiveRecorder &) = delete;
  LiveRecorder & operator=(const LiveRecorder &) = delete;

  void command(const Json & input)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto op = input.at("operation").get<std::string>();
    if (op == "arm") {
      if (armed_) {throw std::runtime_error("close current recorder session before rearming");}
      const auto id = input.at("run_id").get<std::string>();
      if (id.empty() || id.size() > 120 || !std::all_of(id.begin(),id.end(),[](char c) {
          return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-';}))
      {throw std::invalid_argument("invalid recorder session identifier");}
      folder_ = (std::filesystem::path(root_) / id).string();
      // Callback thread, never the control call; exclusive creation prevents reuse.
      if (!std::filesystem::create_directory(folder_)) {throw std::runtime_error("session exists");}
      session_ = input; run_ = id; goal_.clear(); motion_=nullptr; episode_ = 0; captured_ = false;
      drop_start_=drops_.load();error_start_=worker_errors_.load();
      dense_ = input.value("dense",true); armed_ = true;
      enqueue({folder_,"event",{{"kind","session_armed"},{"host_ns",monotonicNs()},
        {"session",session_},{"manifest",manifest_}},nullptr});
    } else if (!armed_ || input.value("run_id",std::string())!=run_) {
      throw std::invalid_argument("command does not identify the active session");
    } else if (op == "trigger") {
      if (recorder_) {recorder_->trigger(input.value("reason",std::string("manual")));}
    } else if (op == "close") {
      detach("session_closed"); armed_ = false;
      enqueue({folder_,"closed",{{"kind","session_closed"},{"run_id",run_},
        {"host_ns",monotonicNs()},{"queue_drops",drops_.load()-drop_start_},
        {"worker_errors",worker_errors_.load()-error_start_},
        {"drop_start",drop_start_},{"error_start",error_start_}},nullptr});
    } else if (op == "motion") {
      const auto displacement=input.at("displacement_m").get<double>();
      const auto window=input.at("window_s").get<double>();
      if(input.at("source")!="gazebo_ground_truth" || input.at("frame")!="world" ||
        input.at("clock_epoch").get<int64_t>()<0 || !finite(displacement) || displacement<0 ||
        !finite(window) || window<1.0 || window>5.0 || input.at("stamp_ns").get<int64_t>()<0)
      {throw std::invalid_argument("invalid independent motion evidence");}
      if(!input.at("minimum_delivered_speed").is_null()) {
        const auto speed=input.at("minimum_delivered_speed").get<double>();
        if(!finite(speed)||speed<0||speed>2) {throw std::invalid_argument("invalid delivered-speed evidence");}
      }
      motion_ = input;
    } else {throw std::invalid_argument("unknown recorder command");}
  }
  // Must be used in the actual ControllerServer worker, not an executor callback.
  class Scope
  {
  public:
    Scope(LiveRecorder * owner, std::string goal, int64_t stamp,
      const std::string & mode="active", const std::string & reason="control_tick")
    : owner_(owner), start_(monotonicNs()), exceptions_(std::uncaught_exceptions())
    {
      if (!owner_) {return;}
      try {
      lock_ = std::unique_lock<std::mutex>(owner_->mutex_);
      if (!owner_->armed_) {owner_ = nullptr; return;}
      if (!goal.empty() && owner_->goal_ != goal) {
        owner_->detach("controller_goal_changed"); owner_->goal_=std::move(goal);
        owner_->captured_=false;
      }
      if (owner_->dense_ && !owner_->captured_ && !owner_->recorder_) {
        Json m=owner_->manifest_;m["run_id"]=owner_->run_;
        m["goal_uuid"]=owner_->goal_.empty()?"no_controller_goal":owner_->goal_;
        m["mission_uuid"]=owner_->session_.value("mission_uuid",std::string("unknown"));
        m["episode_id"]=++owner_->episode_;
        {std::lock_guard<std::mutex> queue_lock(owner_->queue_mutex_);
          if(owner_->pending_captures_>=2) {owner_->captured_=true;++owner_->drops_;}}
        if(!owner_->captured_) {owner_->recorder_=std::make_unique<Recorder>(std::move(m));}
      }
      current_scope_=owner_;
      owner_->tick_++;
      owner_->context_={{"run_id",owner_->run_},{"goal_uuid",owner_->goal_},
        {"episode_id",owner_->episode_},{"server_tick",owner_->tick_},
        {"stamp_ns",stamp},{"mode",mode},{"reason",reason}};
      if (owner_->recorder_) {
        auto & r=*owner_->recorder_;r.mode=mode;
        r.independent_progress.reset();r.delivered_speed.reset();
        session_scope_=std::make_unique<Session>(r);
        invocation_=std::make_unique<Invocation>(stamp);
        if (!owner_->motion_.is_null() && owner_->motion_.at("run_id")==owner_->run_) {
          const auto age=stamp-owner_->motion_.value("stamp_ns",int64_t(0));
          if (age>=0 && age<=250000000 && owner_->motion_.value("window_s",0.0)>=1.0 &&
            owner_->motion_.at("clock_epoch")==r.currentIdentity().at("clock_epoch")) {
            r.independent_progress=owner_->motion_.at("displacement_m").get<double>();
            if (!owner_->motion_.at("minimum_delivered_speed").is_null()) {
              r.delivered_speed=owner_->motion_.at("minimum_delivered_speed").get<double>();
            }
          }
        }

        owner_->context_["capture_identity"]=r.currentIdentity();
        observe("server_context",[&] {return owner_->context_;});
      }
      owner_->journal("tick_begin",owner_->context_);
      } catch (...) {
        ++owner_->worker_errors_;invocation_.reset();session_scope_.reset();
        current_scope_=nullptr;owner_=nullptr;
      }
    }
    ~Scope()
    {
      if (!owner_) {return;}
      try {
      const bool exception=std::uncaught_exceptions()>exceptions_;
      if (exception && owner_->recorder_) {owner_->recorder_->exception("controller_server_exception");}
      invocation_.reset();session_scope_.reset();
      auto row=owner_->context_;row["duration_ns"]=monotonicNs()-start_;
      row["exception"]=exception;owner_->journal("tick_end",row);
      if (owner_->recorder_ && (exception || owner_->recorder_->frozen())) {
        owner_->detach(exception?"abort":"symptom_window_complete");
        owner_->captured_=true;
      }
      } catch (...) {++owner_->worker_errors_;}
      current_scope_=nullptr;
    }
    Scope(const Scope &)=delete;Scope & operator=(const Scope &)=delete;
  private:
    LiveRecorder * owner_; uint64_t start_; int exceptions_;
    std::unique_lock<std::mutex> lock_;
    std::unique_ptr<Session> session_scope_;
    std::unique_ptr<Invocation> invocation_;
  };
  void noteError() noexcept {++worker_errors_;}
  // Called while Scope owns mutex_. The payload is the actual publish boundary.
  void publication(Json payload)
  {
    if (current_scope_!=this) {return;}
    payload["context"]=context_;
    observe("server_dispatch",[&] {return payload;});
    journal("server_dispatch",std::move(payload));
  }
private:
  void journal(const char * kind, Json payload)
  {
    payload["kind"]=kind;payload["host_ns"]=monotonicNs();
    enqueue({folder_,"event",std::move(payload),nullptr});
  }
  void detach(const std::string & reason)
  {
    if (!recorder_) {return;}
    recorder_->trigger(reason);
    enqueue({folder_,"episode_"+std::to_string(episode_)+".trace.json",{},std::move(recorder_)});
  }
  void enqueue(Job job)
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (queue_.size()>=4096 || (job.capture && pending_captures_>=2)) {++drops_;return;}
    const bool capture=bool(job.capture);
    queue_.push_back(std::move(job));if(capture) {++pending_captures_;} wake_.notify_one();
  }
  void work() noexcept
  {
    while (true) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        wake_.wait(lock,[&] {return shutdown_||!queue_.empty();});
        if (shutdown_&&queue_.empty()) {return;}
        job=std::move(queue_.front());queue_.pop_front();
      }
      try {
        if (job.capture) {job.capture->writeExclusive(job.folder+"/"+job.name);}
        else if (job.name=="closed") {
          job.row["lifetime_queue_drops"]=drops_.load();job.row["lifetime_worker_errors"]=worker_errors_.load();
          job.row["queue_drops"]=drops_.load()-job.row.at("drop_start").get<size_t>();
          job.row["worker_errors"]=worker_errors_.load()-job.row.at("error_start").get<size_t>();
          const auto filename=job.folder+"/session.closed.json";
          const int fd=::open(filename.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
          if(fd<0) {throw std::runtime_error("exclusive closure receipt failed");}
          const auto value=job.row.dump()+"\n";
          const auto n=::write(fd,value.data(),value.size());::close(fd);
          if(n!=static_cast<ssize_t>(value.size())) {throw std::runtime_error("closure receipt write failed");}
        } else {
          std::ofstream file(job.folder+"/server.events.jsonl",std::ios::app);
          file<<job.row.dump()<<'\n';if(!file) {throw std::runtime_error("journal write failed");}
        }
      } catch (...) {++worker_errors_;}
      if (job.capture) {std::lock_guard<std::mutex> lock(queue_mutex_);--pending_captures_;}
    }
  }
  inline static thread_local LiveRecorder * current_scope_=nullptr;
  std::string root_,folder_,run_,goal_;
  Json manifest_,session_,motion_,context_;
  std::mutex mutex_,queue_mutex_;std::condition_variable wake_;
  std::deque<Job> queue_;std::unique_ptr<Recorder> recorder_;
  bool armed_{false},dense_{true},captured_{false},shutdown_{false};
  size_t episode_{0},pending_captures_{0},drop_start_{0},error_start_{0};uint64_t tick_{0};
  std::atomic<size_t> drops_{0},worker_errors_{0};
  // Declared last so the thread cannot observe uninitialized synchronization state.
  std::thread worker_;
};
}  // namespace mppi::recorder
#endif
