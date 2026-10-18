cmake_minimum_required(VERSION 3.16)
project(nav2_collision_regression)

# Compile actual production critic/optimizer code and their real dependencies.
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_gtest REQUIRED)
find_package(xtensor REQUIRED)
find_package(xsimd REQUIRED)
find_package(nlohmann_json REQUIRED)
set(dependencies
  rclcpp nav2_common nav2_core nav2_costmap_2d nav2_util pluginlib
  nav2_critics_msgs geometry_msgs nav_msgs visualization_msgs tf2 tf2_eigen tf2_ros tf2_geometry_msgs)
foreach(dependency IN LISTS dependencies)
  find_package(${dependency} REQUIRED)
endforeach()

set(mppi_dir "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_library(collision_critic_under_test STATIC
  "${mppi_dir}/src/critics/obstacles_critic.cpp"
  "${mppi_dir}/src/parameters_handler.cpp")
target_include_directories(collision_critic_under_test PUBLIC "${mppi_dir}/include")
target_compile_features(collision_critic_under_test PUBLIC cxx_std_17)
# Match the production floating-point optimization policy.
target_compile_options(collision_critic_under_test PUBLIC -O3 -ffast-math -ffp-contract=fast)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(collision_critic_under_test PUBLIC -fconcepts)
endif()
ament_target_dependencies(collision_critic_under_test ${dependencies})
target_link_libraries(collision_critic_under_test xtensor xsimd nlohmann_json::nlohmann_json)

enable_testing()
ament_add_gtest(collision_critic_regressions ../collision_critic_tests.cpp TIMEOUT 90)
ament_target_dependencies(collision_critic_regressions ${dependencies})
target_link_libraries(collision_critic_regressions collision_critic_under_test)

add_library(final_optimizer_under_test STATIC
  "${mppi_dir}/src/optimizer.cpp"
  "${mppi_dir}/src/critic_manager.cpp"
  "${mppi_dir}/src/noise_generator.cpp")
target_link_libraries(final_optimizer_under_test collision_critic_under_test)
ament_target_dependencies(final_optimizer_under_test ${dependencies})
ament_add_gtest(final_output_regressions ../final_output_collision_tests.cpp TIMEOUT 90)
ament_target_dependencies(final_output_regressions ${dependencies})
target_link_libraries(final_output_regressions final_optimizer_under_test)
