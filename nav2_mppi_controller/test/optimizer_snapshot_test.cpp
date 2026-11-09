// Copyright (c) 2026
// Licensed under the Apache License, Version 2.0.
#include <chrono>
#include <map>
#include <vector>
#include <xtensor/xrandom.hpp>
#include "gtest/gtest.h"
#include "nav2_mppi_controller/optimizer.hpp"

namespace
{
std::map<std::string, std::vector<double>> readSnapshot(const std::string & file)
{
  std::ifstream stream(file, std::ios::binary);
  std::string magic;
  std::getline(stream, magic);
  EXPECT_EQ(magic, "MPPI_SNAPSHOT_LE_V1");
  std::map<std::string, std::vector<double>> records;
  std::string name, type;
  size_t count;
  while (stream >> name >> type >> count) {
    EXPECT_EQ(stream.get(), '\n');
    std::vector<double> values(count);
    if (type == "f32") {
      std::vector<float> raw(count);
      stream.read(reinterpret_cast<char *>(raw.data()), count * sizeof(float));
      std::copy(raw.begin(), raw.end(), values.begin());
    } else if (type == "f64") {
      stream.read(reinterpret_cast<char *>(values.data()), count * sizeof(double));
    } else {
      EXPECT_EQ(type, "u8");
      std::vector<uint8_t> raw(count);
      stream.read(reinterpret_cast<char *>(raw.data()), count);
      std::copy(raw.begin(), raw.end(), values.begin());
    }
    EXPECT_EQ(stream.get(), '\n');
    EXPECT_EQ(records.count(name), 0u);
    records.emplace(name, values);
  }
  return records;
}

std::string temporaryFile()
{
  return (std::filesystem::temp_directory_path() / ("mppi_snapshot_" + std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()))).string();
}

class SnapshotOptimizer : public mppi::Optimizer
{
public:
  bool snapshotCleared() const {return critics_data_.diagnostic_snapshot == nullptr;}
  std::vector<float> evaluate(mppi::OptimizerSnapshot * snapshot)
  {
    settings_.batch_size = 3;
    settings_.time_steps = 5;
    settings_.model_dt = .05;
    settings_.temperature = .3;
    settings_.gamma = .01;
    settings_.sampling_std.vx = .1;
    settings_.sampling_std.wz = .1;
    settings_.constraints.vx_max = .5;
    settings_.constraints.vx_min = -.3;
    settings_.constraints.wz = 1.9;
    settings_.constraints.ax_max = 10;
    settings_.constraints.ax_min = -10;
    settings_.constraints.az_max = 10;
    state_.reset(3, 5);
    state_.speed.linear.x = .1;
    control_sequence_.reset(5);
    control_sequence_.vx.fill(.1);
    costs_ = xt::xtensor<float, 1>{2, 1, 0};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 5; ++j) {
        state_.cvx(i, j) = .08f + .02f * i;
        state_.cwz(i, j) = -.1f + .1f * i;
      }
    }
    motion_model_ = std::make_shared<mppi::DiffDriveMotionModel>();
    motion_model_->initialize(settings_.constraints, settings_.model_dt);
    critics_data_.diagnostic_snapshot = snapshot;
    updateControlSequence();
    critics_data_.diagnostic_snapshot = nullptr;
    return {control_sequence_.vx(1), control_sequence_.wz(1)};
  }
};

class ConstantCritic : public mppi::critics::CriticFunction
{
public:
  void initialize() override {}
  void score(mppi::CriticData & data) override {data.costs += 3.25f;}
};

class SnapshotManager : public mppi::CriticManager
{
public:
  SnapshotManager()
  {
    critic_names_.push_back("constant");
    critics_.push_back(std::make_unique<ConstantCritic>());
  }
};
}  // namespace

namespace
{
struct EvalFixture
{
  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> map;
  std::unique_ptr<mppi::ParametersHandler> parameters;
  SnapshotOptimizer optimizer;
  std::string directory;
  geometry_msgs::msg::PoseStamped robot;
  nav_msgs::msg::Path path;
  geometry_msgs::msg::Pose goal;

  EvalFixture(bool capture, bool blocked, bool obstacle_critic)
  {
    directory = temporaryFile();
    node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("snapshot_test");
    node->declare_parameter("controller_frequency", 20.0);
    node->declare_parameter("mppi.batch_size", 4);
    node->declare_parameter("mppi.time_steps", 41);
    node->declare_parameter("mppi.iteration_count", 2);
    node->declare_parameter("mppi.motion_model", "Ackermann");
    node->declare_parameter("mppi.retry_attempt_limit", 1);
    node->declare_parameter("mppi.PathFollowCritic.threshold_to_consider", .01);
    node->declare_parameter("mppi.PathFollowCritic.cost_weight", 66.05);
    node->declare_parameter("mppi.optimizer_snapshot_directory", capture ? directory : "");
    node->declare_parameter("mppi.optimizer_snapshot_max_ticks", 1);
    node->declare_parameter("mppi.critics", std::vector<std::string>{
        obstacle_critic ? "ObstaclesCritic" : "PathFollowCritic"});
    map = std::make_shared<nav2_costmap_2d::Costmap2DROS>("snapshot_map", "", "snapshot_map", true);
    map->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
    map->set_parameter(rclcpp::Parameter("footprint",
        "[[.05,.05],[.05,-.05],[-.05,-.05],[-.05,.05]]"));
    rclcpp_lifecycle::State state;
    map->on_configure(state);
    map->getCostmap()->resizeMap(200, 200, .02, -2, -2);
    std::fill_n(map->getCostmap()->getCharMap(), 40000, blocked ? 254 : 0);
    parameters = std::make_unique<mppi::ParametersHandler>(node);
    xt::random::seed(1234);
    optimizer.initialize(node, "mppi", map, parameters.get());
    robot.header.frame_id = "map";
    robot.header.stamp.sec = 12;
    robot.pose.orientation.w = 1;
    goal.orientation.w = 1;
    goal.position.x = .2;
    path.header = robot.header;
    path.poses.push_back(robot);
    auto end = robot;
    end.pose = goal;
    path.poses.push_back(end);
  }

  geometry_msgs::msg::TwistStamped evaluate()
  {
    geometry_msgs::msg::Twist speed;
    speed.linear.x = .1;
    return optimizer.evalControl(robot, speed, path, goal, nullptr);
  }

  std::map<std::string, std::vector<double>> records() const
  {
    std::vector<std::string> files;
    for (const auto & entry : std::filesystem::directory_iterator(directory)) {
      files.push_back(entry.path().string());
    }
    EXPECT_EQ(files.size(), 1u);
    return readSnapshot(files.at(0));
  }

  ~EvalFixture()
  {
    optimizer.shutdown();
    if (std::filesystem::exists(directory)) {std::filesystem::remove_all(directory);}
  }
};
}  // namespace

TEST(OptimizerSnapshot, RoundTripAndHotPathDoesNotChangeWeightedOutput)
{
  SnapshotOptimizer optimizer;
  const auto baseline = optimizer.evaluate(nullptr);
  mppi::OptimizerSnapshot snapshot;
  const auto captured = optimizer.evaluate(&snapshot);
  EXPECT_EQ(baseline, captured);
  const std::string file = temporaryFile();
  ASSERT_TRUE(snapshot.save(file));
  const auto records = readSnapshot(file);
  EXPECT_EQ(records.at("weights").size(), 3u);
  EXPECT_EQ(records.at("mean.wz").size(), 5u);
  EXPECT_FLOAT_EQ(records.at("mean.wz")[1], captured[1]);
  EXPECT_GT(captured[1], .08f);
  EXPECT_EQ(records.at("constrained.vx")[1], captured[0]);
  EXPECT_EQ(records.at("costs.regularized").size(), 3u);
  std::filesystem::remove(file);
}

TEST(OptimizerSnapshot, CapturesActualCriticDeltaWithoutStatsPublisher)
{
  mppi::models::State state;
  mppi::models::Trajectories trajectories;
  mppi::models::Path path;
  geometry_msgs::msg::Pose goal;
  xt::xtensor<float, 1> costs{1, 2, 3};
  float dt = .05;
  mppi::CriticData data{state, trajectories, path, goal, costs, dt, false,
    nullptr, nullptr, std::nullopt, std::nullopt};
  SnapshotManager manager;
  mppi::OptimizerSnapshot snapshot;
  data.diagnostic_snapshot = &snapshot;
  manager.evalTrajectoriesScores(data);
  EXPECT_FLOAT_EQ(costs(0), 4.25f);
  const std::string file = temporaryFile();
  ASSERT_TRUE(snapshot.save(file));
  const auto records = readSnapshot(file);
  EXPECT_EQ(records.at("critic.constant"), (std::vector<double>{3.25, 3.25, 3.25}));
  std::filesystem::remove(file);
}

TEST(OptimizerSnapshot, EvalControlCapturesActualMultiIterationInputsAndOutput)
{
  rclcpp::init(0, nullptr);
  {
    EvalFixture off(false, false, false);
    const auto baseline = off.evaluate();
    EvalFixture on(true, false, false);
    const auto captured = on.evaluate();
    EXPECT_EQ(baseline.twist.linear.x, captured.twist.linear.x);
    EXPECT_EQ(baseline.twist.angular.z, captured.twist.angular.z);
    const auto records = on.records();
    EXPECT_EQ(records.at("completed")[0], 1);
    EXPECT_EQ(records.at("input")[0], 12);
    EXPECT_EQ(records.at("map.costs").size(), 40000u);
    EXPECT_EQ(records.at("path.x").size(), 2u);
    for (const std::string iteration : {"retry.0.iter.0.", "retry.0.iter.1."}) {
      EXPECT_EQ(records.at(iteration + "cvx").size(), 164u);
      EXPECT_EQ(records.at(iteration + "trajectory.x").size(), 164u);
      EXPECT_EQ(records.at(iteration + "critic.PathFollowCritic").size(), 4u);
      EXPECT_EQ(records.at(iteration + "weights").size(), 4u);
      const auto & critic = records.at(iteration + "critic.PathFollowCritic");
      const auto & tx = records.at(iteration + "trajectory.x");
      const auto & ty = records.at(iteration + "trajectory.y");
      for (size_t i = 0; i < 4; ++i) {
        EXPECT_GT(critic[i], 0);
        EXPECT_NEAR(critic[i], 66.05 * std::hypot(tx[i * 41 + 40] - .2, ty[i * 41 + 40]), 1e-4);
      }
      const auto & weights = records.at(iteration + "weights");
      const auto & cvx = records.at(iteration + "cvx");
      const auto & cwz = records.at(iteration + "cwz");
      const auto & pv = records.at(iteration + "prior.vx");
      const auto & pw = records.at(iteration + "prior.wz");
      const auto & total = records.at(iteration + "costs.regularized");
      const auto & config = records.at("settings");
      double weight_sum = 0, w_mean = 0;
      for (size_t i = 0; i < 4; ++i) {
        double reg = 0;
        for (size_t j = 0; j < 41; ++j) {
          reg += config[2] / (config[3] * config[3]) * pv[j] * (cvx[i * 41 + j] - pv[j]);
          reg += config[2] / (config[4] * config[4]) * pw[j] * (cwz[i * 41 + j] - pw[j]);
        }
        EXPECT_NEAR(total[i], critic[i] + reg, 1e-4);
        weight_sum += weights[i];
        w_mean += weights[i] * cwz[i * 41 + 1];
      }
      EXPECT_NEAR(weight_sum, 1, 1e-6);
      EXPECT_NEAR(w_mean, records.at(iteration + "mean.wz")[1], 1e-6);
      const double minimum = *std::min_element(total.begin(), total.end());
      double denominator = 0;
      for (double cost : total) {
        denominator += std::exp(-(cost - minimum) / config[1]);
      }
      for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(weights[i], std::exp(-(total[i] - minimum) / config[1]) / denominator, 1e-5);
      }
    }
    EXPECT_EQ(records.at("emitted")[0], captured.twist.linear.x);
    EXPECT_EQ(records.at("validated.vx")[1], captured.twist.linear.x);
    EXPECT_EQ(records.at("validated.wz")[1], captured.twist.angular.z);
    EXPECT_TRUE(on.optimizer.snapshotCleared());
  }
  rclcpp::shutdown();
}

TEST(OptimizerSnapshot, EvalControlPreservesFinalVetoEvidenceAndClearsPointer)
{
  rclcpp::init(0, nullptr);
  {
    EvalFixture fixture(true, true, false);
    EXPECT_THROW(fixture.evaluate(), nav2_core::NoValidControl);
    const auto records = fixture.records();
    EXPECT_EQ(records.at("completed")[0], 0);
    EXPECT_EQ(records.at("retry.0.iter.1.weights").size(), 4u);
    EXPECT_TRUE(fixture.optimizer.snapshotCleared());
  }
  rclcpp::shutdown();
}

TEST(OptimizerSnapshot, EvalControlPreservesDistinctFailedFallbackBanks)
{
  rclcpp::init(0, nullptr);
  {
    EvalFixture fixture(true, true, true);
    EXPECT_THROW(fixture.evaluate(), nav2_core::NoValidControl);
    const auto records = fixture.records();
    EXPECT_EQ(records.at("completed")[0], 0);
    EXPECT_EQ(records.at("retry.0.iter.0.fail_flag")[0], 1);
    EXPECT_EQ(records.at("retry.1.iter.0.fail_flag")[0], 1);
    EXPECT_TRUE(fixture.optimizer.snapshotCleared());
  }
  rclcpp::shutdown();
}
