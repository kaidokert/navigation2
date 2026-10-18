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
bool armAfterClose(rec::LiveRecorder & live,const rec::Json & request)
{
  for(int i=0;i<2000;++i) {
    try {live.command(request);return true;} catch(const std::runtime_error &) {}
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}
void tick(rec::LiveRecorder & live,int64_t stamp,double vx=.1)
{
  rec::LiveRecorder::Scope scope(&live,"actual-native-goal",stamp);
  rec::observe("issued",[&] {return rec::Json{{"vx",vx},{"vy",0.0},{"wz",0.0}};});
  live.publication({{"published",true},{"stamp_ns",stamp+1},{"vx",vx},{"vy",0.0},{"wz",0.0}});
}
}
TEST(RecorderResource,EscapedKeyChargeRemainsConservativeWithoutSixfoldAsciiCharge)
{
  const std::vector<rec::Json> cases={
    {{std::string(200,'k'),0}},
    {{"quote\"\\\n\x1f",rec::Json{nullptr,true,false,-1.2e300}}},
    {{u8"µ",u8"µ"}},rec::Json::binary({0,1,2,255})};
  for(const auto & value:cases) {
    auto encoded=value;rec::encodeBinary(encoded);
    EXPECT_GE(rec::serializedUpperBound(value),encoded.dump().size());
  }
  EXPECT_LT(rec::serializedUpperBound(cases.front()),2*cases.front().dump().size());
}
TEST(RecorderResource,StreamedWriterMatchesCanonicalDocumentAndRejectsOverwrite)
{
  auto root=directory("resource_streamed_canonical");
  rec::Config config;config.pre=1;config.post=0;
  rec::Recorder recorder({{"quoted","a\"b\\c\n"},{"unicode",u8"µ"}},config);
  for(int i=0;i<2;++i) {
    if(i==1) {recorder.trigger("quote\"\\\n");}
    recorder.begin(i+1);
    recorder.event("payload",[] {return rec::Json{{"encoding","f32le-base64"},
        {"data",rec::Json::binary({0,1,2,255})}};});
    recorder.end();
  }
  const auto expected=recorder.document().dump()+"\n";
  const auto path=(root/"trace.json").string();recorder.writeExclusive(path);
  std::ifstream file(path);const std::string actual((std::istreambuf_iterator<char>(file)),{});
  EXPECT_EQ(actual,expected);
  EXPECT_THROW(recorder.writeExclusive(path),std::runtime_error);
  EXPECT_EQ(read(path),recorder.document());
}
TEST(RecorderResource,EmptyStreamedWriterMatchesCanonicalDocument)
{
  auto root=directory("resource_streamed_empty");rec::Recorder recorder({{"empty",true}});
  const auto expected=recorder.document().dump()+"\n";
  const auto path=(root/"trace.json").string();recorder.writeExclusive(path);
  std::ifstream file(path);const std::string actual((std::istreambuf_iterator<char>(file)),{});
  EXPECT_EQ(actual,expected);
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
    for(int i=0;i<2000 && !std::filesystem::exists(root/"first/session.closed.json");++i) {std::this_thread::sleep_for(std::chrono::milliseconds(1));}
    ASSERT_TRUE(armAfterClose(live,{{"operation","arm"},{"run_id","second"}}));
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
    live.command({{"operation","arm"},{"run_id","recurring"},{"maximum_episodes",3}});
    int64_t stamp=1000;
    auto call=[&] {tick(live,stamp++);};
    call();live.command({{"operation","trigger"},{"run_id","recurring"},{"reason","test_first"}});
    for(int i=0;i<3;++i) {call();}
    EXPECT_EQ(started_future.wait_for(2s),std::future_status::ready);
    call();live.command({{"operation","trigger"},{"run_id","recurring"},{"reason","test_second_while_writer_blocked"}});
    for(int i=0;i<10;++i) {call();}
    EXPECT_FALSE(std::filesystem::exists(root/"recurring/episode_2.trace.json"));
    release.set_value();
    for(int i=0;i<2000 && !std::filesystem::exists(root/"recurring/episode_3.trace.json");++i) {
      call();live.command({{"operation","trigger"},{"run_id","recurring"},{"reason","test_repeat"}});
      std::this_thread::sleep_for(1ms);
    }
    for(int i=0;i<10;++i) {call();}
    live.command({{"operation","close"},{"run_id","recurring"}});
    // Closure is an actual fence; do not attribute old worker errors to a new run.

    for(int i=0;i<2000 && !std::filesystem::exists(root/"recurring/session.closed.json");++i) {
      std::this_thread::sleep_for(1ms);
    }
    ASSERT_TRUE(armAfterClose(live,{{"operation","arm"},{"run_id","next"},{"maximum_episodes",1}}));
    call();live.command({{"operation","close"},{"run_id","next"}});
  }
  const auto receipt=read(root/"recurring/session.closed.json");
  EXPECT_EQ(receipt["episodes_started"],3);EXPECT_GT(receipt["capture_wait_ticks"],0);
  EXPECT_GT(receipt["budget_exhausted_ticks"],0);EXPECT_EQ(receipt["queue_drops"],0);
  EXPECT_EQ(receipt["worker_errors"],0);
  EXPECT_EQ(read(root/"recurring/episode_1.trace.json")["manifest"]["episode_id"],1);
  EXPECT_EQ(read(root/"recurring/episode_2.trace.json")["manifest"]["episode_id"],2);
  EXPECT_EQ(read(root/"next/episode_1.trace.json")["manifest"]["run_id"],"next");
  EXPECT_EQ(read(root/"next/session.closed.json")["worker_errors"],0);
}

TEST(LiveRecorder,RefinedCollapseCapturedWhileFirstWriterBlocked)
{
  using namespace std::chrono_literals;
  auto root=directory("live_adapter_refined");
  std::promise<void> started,release;auto ready=started.get_future();auto released=release.get_future().share();std::atomic<unsigned> flushes{0};
  {
    rec::LiveRecorder live(root.string(),{{"native_baseline","fixture"}},[&] {if(++flushes==1) {started.set_value();released.wait_for(5s);}});
    live.command({{"operation","arm"},{"run_id","refined"},{"maximum_episodes",2}});
    int64_t stamp=1000;
    for(int i=0;i<24;++i) {tick(live,stamp++,.01);}
    EXPECT_EQ(ready.wait_for(2s),std::future_status::ready);
    for(int i=0;i<30;++i) {tick(live,stamp++,.01);}
    for(int i=0;i<24;++i) {tick(live,stamp++,.001);}
    live.command({{"operation","close"},{"run_id","refined"}});
    EXPECT_THROW(live.command({{"operation","arm"},{"run_id","too_early"}}),std::runtime_error);
    release.set_value();
  }
  auto first=read(root/"refined/episode_1.trace.json");auto second=read(root/"refined/episode_2.trace.json");
  EXPECT_EQ(first["manifest"]["recording_policy"]["collapse_speed_mps"],.02);
  EXPECT_EQ(second["manifest"]["recording_policy"]["collapse_speed_mps"],.005);
  bool coarse=false,fine=false;
  for(const auto & f:second["invocations"]) {
    coarse=coarse||f["trigger_inputs"]["command_vx"]==.01;
    fine=fine||f["trigger_inputs"]["command_vx"]==.001;
  }
  EXPECT_TRUE(coarse);EXPECT_TRUE(fine);
  auto receipt=read(root/"refined/session.closed.json");
  EXPECT_EQ(receipt["episodes_started"],2);EXPECT_EQ(receipt["queue_drops"],0);EXPECT_EQ(receipt["worker_errors"],0);
  std::ifstream journal(root/"refined/server.events.jsonl");std::string line;unsigned ends=0;
  while(std::getline(journal,line)) {
    auto e=rec::Json::parse(line);
    if(e["kind"]=="tick_end") {EXPECT_EQ(e["run_id"],"refined");++ends;}
  }
  EXPECT_EQ(ends,78);
}
