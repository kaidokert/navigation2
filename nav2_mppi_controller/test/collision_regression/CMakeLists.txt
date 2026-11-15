cmake_minimum_required(VERSION 3.16)
project(nav2_collision_regression)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_definitions(-DXTENSOR_ENABLE_XSIMD)
add_definitions(-DXTENSOR_USE_XSIMD)
set(XTENSOR_USE_TBB 0)
set(XTENSOR_USE_OPENMP 0)
set(XTENSOR_USE_XSIMD 1)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mno-avx512f" COMPILER_SUPPORTS_AVX512)
check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE4)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
if(COMPILER_SUPPORTS_AVX512)
  add_compile_options(-mno-avx512f)
endif()
if(COMPILER_SUPPORTS_SSE4)
  add_compile_options(-msse4.2)
endif()
if(COMPILER_SUPPORTS_AVX2)
  add_compile_options(-mavx2)
endif()
if(COMPILER_SUPPORTS_FMA)
  add_compile_options(-mfma)
endif()
add_compile_options(-O3 -finline-limit=10000000 -ffp-contract=fast -ffast-math -mtune=generic)

# Compile actual production critic/optimizer code and their real dependencies.
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_gtest REQUIRED)
find_package(xtensor REQUIRED)
find_package(xsimd REQUIRED)
set(dependencies
  rclcpp rclcpp_lifecycle nav2_common nav2_core nav2_costmap_2d nav2_util pluginlib
  nav2_critics_msgs geometry_msgs nav_msgs std_msgs visualization_msgs
  angles tf2 tf2_eigen tf2_ros tf2_geometry_msgs)
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
target_link_libraries(collision_critic_under_test xtensor xsimd)

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

# Reuse this existing lightweight native build, without a Gazebo/planner stack.
find_package(yaml_cpp_vendor REQUIRED)
find_package(yaml-cpp REQUIRED)
add_library(controller_scenario_code STATIC
  "${mppi_dir}/src/controller.cpp"
  "${mppi_dir}/src/trajectory_visualizer.cpp"
  "${mppi_dir}/src/critics/path_align_critic.cpp"
  "${mppi_dir}/src/critics/path_follow_critic.cpp"
  "${mppi_dir}/src/critics/terminal_approach_critic.cpp"
  "${mppi_dir}/src/path_handler.cpp"
  "${mppi_dir}/../nav2_controller/plugins/simple_goal_checker.cpp"
  "${mppi_dir}/../nav2_controller/plugins/stopped_goal_checker.cpp")
target_include_directories(controller_scenario_code PUBLIC
  "${mppi_dir}/../nav2_controller/include")
target_link_libraries(controller_scenario_code final_optimizer_under_test)
ament_target_dependencies(controller_scenario_code ${dependencies})
ament_add_gtest(controller_scenario_tests ../controller_scenario_tests.cpp TIMEOUT 120)
ament_target_dependencies(controller_scenario_tests ${dependencies})
target_link_libraries(controller_scenario_tests controller_scenario_code yaml-cpp)
target_compile_definitions(controller_scenario_tests PRIVATE
  CONTROLLER_SCENARIO_FILE="${mppi_dir}/test/fixtures/approach_atlas.json")

add_library(all_critics_under_test SHARED
  "${mppi_dir}/src/critics/obstacles_critic.cpp"
  "${mppi_dir}/src/critics/cost_critic.cpp"
  "${mppi_dir}/src/critics/goal_critic.cpp"
  "${mppi_dir}/src/critics/approach_angle_critic.cpp"
  "${mppi_dir}/src/critics/goal_angle_critic.cpp"
  "${mppi_dir}/src/critics/path_align_critic.cpp"
  "${mppi_dir}/src/critics/path_follow_critic.cpp"
  "${mppi_dir}/src/critics/path_angle_critic.cpp"
  "${mppi_dir}/src/critics/prefer_forward_critic.cpp"
  "${mppi_dir}/src/critics/twirling_critic.cpp"
  "${mppi_dir}/src/critics/constraint_critic.cpp"
  "${mppi_dir}/src/critics/velocity_deadband_critic.cpp"
  "${mppi_dir}/src/critics/terminal_approach_critic.cpp")
target_include_directories(all_critics_under_test PUBLIC "${mppi_dir}/include")
target_compile_features(all_critics_under_test PUBLIC cxx_std_17)
target_compile_options(all_critics_under_test PUBLIC -O3 -ffast-math -ffp-contract=fast)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(all_critics_under_test PUBLIC -fconcepts)
endif()
ament_target_dependencies(all_critics_under_test ${dependencies})
target_link_libraries(all_critics_under_test final_optimizer_under_test xtensor xsimd)
set(test_prefix "${CMAKE_CURRENT_BINARY_DIR}/test_prefix")
file(MAKE_DIRECTORY "${test_prefix}/lib")
set_target_properties(all_critics_under_test PROPERTIES
  OUTPUT_NAME "mppi_critics"
  LIBRARY_OUTPUT_DIRECTORY "${test_prefix}/lib")

# Stage a self-contained ament/pluginlib prefix in the build tree so critic_manager_test
# loads this checkout's critics.xml and libmppi_critics.so in standalone CI.
file(MAKE_DIRECTORY "${test_prefix}/share/ament_index/resource_index/packages")
file(WRITE "${test_prefix}/share/ament_index/resource_index/packages/nav2_mppi_controller" "")
file(MAKE_DIRECTORY
  "${test_prefix}/share/ament_index/resource_index/nav2_mppi_controller__pluginlib__plugin")
file(WRITE
  "${test_prefix}/share/ament_index/resource_index/nav2_mppi_controller__pluginlib__plugin/nav2_mppi_controller"
  "share/nav2_mppi_controller/critics.xml\n")
file(MAKE_DIRECTORY "${test_prefix}/share/nav2_mppi_controller")
configure_file(
  "${mppi_dir}/critics.xml"
  "${test_prefix}/share/nav2_mppi_controller/critics.xml"
  COPYONLY)
configure_file(
  "${mppi_dir}/package.xml"
  "${test_prefix}/share/nav2_mppi_controller/package.xml"
  COPYONLY)

ament_add_gtest(critics_tests ../critics_tests.cpp TIMEOUT 90
  ENV
    LD_LIBRARY_PATH=${test_prefix}/lib:$ENV{LD_LIBRARY_PATH})
ament_target_dependencies(critics_tests ${dependencies})
target_link_libraries(critics_tests all_critics_under_test final_optimizer_under_test)
if(UNIX AND NOT APPLE)
  target_link_options(critics_tests PRIVATE "-Wl,--disable-new-dtags")
endif()

ament_add_gtest(critic_manager_test ../critic_manager_test.cpp TIMEOUT 90
  ENV
    AMENT_PREFIX_PATH=${test_prefix}:$ENV{AMENT_PREFIX_PATH}
    LD_LIBRARY_PATH=${test_prefix}/lib:$ENV{LD_LIBRARY_PATH})
ament_target_dependencies(critic_manager_test ${dependencies})
target_link_libraries(critic_manager_test controller_scenario_code)
add_dependencies(critic_manager_test all_critics_under_test)
