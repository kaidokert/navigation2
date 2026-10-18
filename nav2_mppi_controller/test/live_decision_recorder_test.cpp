// Copyright 2026 Kaido Kert. Apache-2.0.
#include "nav2_mppi_controller/tools/live_decision_recorder.hpp"
#include "gtest/gtest.h"
#include <future>
#include <chrono>

namespace rec=mppi::recorder;
namespace
{
std::filesystem::path directory(const std::string & name)
{
  const char * output=std::getenv("RECORDER_PROOF_OUTPUT");
  auto path=std::filesystem::path(output?output:"/tmp")/name;
  if(!std::filesystem::create_directory(path)) {throw std::runtime_error("test output exists");}
  return path;
}
rec::Json read(const std::filesystem::path & path)
{std::ifstream file(path);rec::Json value;file>>value;return value;}
void tick(rec::LiveRecorder & live,int64_t stamp)
{
  rec::LiveRecorder::Scope scope(&live,"actual-native-goal",stamp);
  rec::observe("issued",[] {return rec::Json{{"vx",.1},{"vy",0.0},{"wz",0.0}};});
  live.publication({{"published",true},{"stamp_ns",stamp+1},{"vx",.1},{"vy",0.0},{"wz",0.0}});
}
}
TEST(LiveRecorder,DetachedFlushAndDispatchIdentity)
{
  auto root=directory("live_adapter_healthy");
  {
    rec::LiveRecorder live(root.string(),{{"native_baseline","fixture"}});
    live.command({{"operation","arm"},{"run_id","healthy"},{"mission_uuid","native-mission"}});
    tick(live,1000);
    live.command({{"operation","close"},{"run_id","healthy"}});
  }
  const auto trace=read(root/"healthy/episode_1.trace.json");
  const auto closed=read(root/"healthy/session.closed.json");
  EXPECT_EQ(closed["worker_errors"],0);EXPECT_EQ(closed["queue_drops"],0);
  EXPECT_EQ(trace["manifest"]["goal_uuid"],"actual-native-goal");
  EXPECT_EQ(trace["manifest"]["mission_uuid"],"native-mission");
  const auto & frame=trace["invocations"][0];bool found=false;
  for(const auto & event:frame["events"]) {
    if(event["kind"]!="server_dispatch") {continue;}
    const auto & identity=event["payload"]["context"]["capture_identity"];
    EXPECT_EQ(identity["invocation_id"],frame["invocation_id"]);
    EXPECT_EQ(identity["clock_epoch"],frame["clock_epoch"]);found=true;
  }
  EXPECT_TRUE(found);
}
TEST(LiveRecorder,RejectCrossRunCommandsAndClearMotion)
{
  auto root=directory("live_adapter_identity");
  {
    rec::LiveRecorder live(root.string(),{{"native_baseline","fixture"}});
    EXPECT_THROW(live.command({{"operation","close"},{"run_id","old"}}),std::invalid_argument);
    live.command({{"operation","arm"},{"run_id","first"}});
    const rec::Json motion={{"operation","motion"},{"run_id","first"},{"source","gazebo_ground_truth"},
      {"frame","world"},{"clock_epoch",0},{"window_s",1.0},{"stamp_ns",1000},
      {"displacement_m",0.0},{"max_excursion_m",0.0},{"consumer_coverage",true},{"minimum_delivered_speed",.1}};
    live.command(motion);
    EXPECT_THROW(live.command({{"operation","close"},{"run_id","old"}}),std::invalid_argument);
    auto unproven=motion;unproven["consumer_coverage"]=false;EXPECT_THROW(live.command(unproven),std::invalid_argument);
    auto bad=motion;bad["window_s"]=-1;EXPECT_THROW(live.command(bad),std::invalid_argument);
    live.command({{"operation","close"},{"run_id","first"}});
    live.command({{"operation","arm"},{"run_id","second"}});
    EXPECT_THROW(live.command(motion),std::invalid_argument);
    tick(live,2000);
    live.command({{"operation","close"},{"run_id","second"}});
  }
  const auto trace=read(root/"second/episode_1.trace.json");
  EXPECT_TRUE(trace["invocations"][0]["trigger_inputs"]["independent_progress"].is_null());
  EXPECT_EQ(trace["termination"],"session_closed");
}
TEST(LiveRecorder,SerializationFailureIsUnhealthyWithoutChangingControl)
{
  auto root=directory("live_adapter_failure");int calls=0;
  {
    rec::LiveRecorder live(root.string(),{{"native_baseline","fixture"}});
    live.command({{"operation","arm"},{"run_id","failure"}});
    std::ofstream existing(root/"failure/episode_1.trace.json");existing<<"preserve me";existing.close();
    EXPECT_NO_THROW(tick(live,1000));++calls;
    live.command({{"operation","close"},{"run_id","failure"}});
  }
  EXPECT_EQ(calls,1);
  EXPECT_EQ(read(root/"failure/session.closed.json")["worker_errors"],1);
  std::ifstream original(root/"failure/episode_1.trace.json");std::string text;
  std::getline(original,text);EXPECT_EQ(text,"preserve me");
}

TEST(LiveRecorder,RecurringCaptureCapacityBudgetAndRearmIsolation)
{
  using namespace std::chrono_literals;
  auto root=directory("live_adapter_recurring");
  std::promise<void> started,release;auto started_future=started.get_future();
  auto released=release.get_future().share();std::atomic<unsigned> flushes{0};
  {
    rec::LiveRecorder live(root.string(),{{"native_baseline","fixture"}},[&] {
      if(++flushes==1) {started.set_value();released.wait_for(5s);}
    });
    live.command({{"operation","arm"},{"run_id","recurring"},{"maximum_episodes",2}});
    int64_t stamp=1000;
    auto call=[&] {tick(live,stamp++);};
    call();live.command({{"operation","trigger"},{"run_id","recurring"},{"reason","test_first"}});
    for(int i=0;i<3;++i) {call();}
    EXPECT_EQ(started_future.wait_for(2s),std::future_status::ready);
    for(int i=0;i<10;++i) {call();}
    EXPECT_FALSE(std::filesystem::exists(root/"recurring/episode_2.trace.json"));
    release.set_value();
    for(int i=0;i<2000 && !std::filesystem::exists(root/"recurring/episode_2.trace.json");++i) {
      call();live.command({{"operation","trigger"},{"run_id","recurring"},{"reason","test_repeat"}});
      std::this_thread::sleep_for(1ms);
    }
    for(int i=0;i<10;++i) {call();}
    live.command({{"operation","close"},{"run_id","recurring"}});
    // Rearm before old closure is processed: queued jobs must retain their folder/identity.
    live.command({{"operation","arm"},{"run_id","next"},{"maximum_episodes",1}});
    for(int i=0;i<2000 && !std::filesystem::exists(root/"recurring/session.closed.json");++i) {
      call();std::this_thread::sleep_for(1ms);
    }
    call();live.command({{"operation","close"},{"run_id","next"}});
  }
  const auto receipt=read(root/"recurring/session.closed.json");
  EXPECT_EQ(receipt["episodes_started"],2);EXPECT_GT(receipt["capture_wait_ticks"],0);
  EXPECT_GT(receipt["budget_exhausted_ticks"],0);EXPECT_EQ(receipt["queue_drops"],0);
  EXPECT_EQ(receipt["worker_errors"],0);
  EXPECT_EQ(read(root/"recurring/episode_1.trace.json")["manifest"]["episode_id"],1);
  EXPECT_EQ(read(root/"recurring/episode_2.trace.json")["manifest"]["episode_id"],2);
  EXPECT_EQ(read(root/"next/episode_1.trace.json")["manifest"]["run_id"],"next");
  EXPECT_EQ(read(root/"next/session.closed.json")["worker_errors"],0);
}
