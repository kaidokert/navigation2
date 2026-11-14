set(TEST_NAMES
  control_timing_benchmark
  final_output_collision_tests
  optimizer_smoke_test
  controller_state_transition_test
  models_test
  noise_generator_test
  parameter_handler_test
  motion_model_tests
  trajectory_visualizer_tests
  utils_test
  path_handler_test
  critic_manager_test
  optimizer_unit_tests
)

foreach(name IN LISTS TEST_NAMES)
  ament_add_gtest(${name}
    ${name}.cpp
  )

  ament_target_dependencies(${name}
    ${dependencies_pkgs}
  )

  target_link_libraries(${name}
    mppi_controller
  )

  if(${TEST_DEBUG_INFO})
    target_compile_definitions(${name} PUBLIC -DTEST_DEBUG_INFO)
  endif()

endforeach()

# This is a special case requiring linking against the critics library
ament_add_gtest(critics_tests critics_tests.cpp)
ament_target_dependencies(critics_tests ${dependencies_pkgs})
target_link_libraries(critics_tests mppi_controller mppi_critics)
if(${TEST_DEBUG_INFO})
  target_compile_definitions(critics_tests PUBLIC -DTEST_DEBUG_INFO)
endif()

ament_add_gtest(collision_critic_tests collision_critic_tests.cpp)
ament_target_dependencies(collision_critic_tests ${dependencies_pkgs})
target_link_libraries(collision_critic_tests mppi_controller mppi_critics)

find_package(nav2_controller REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(yaml-cpp REQUIRED)
ament_add_gtest(controller_scenario_tests controller_scenario_tests.cpp TIMEOUT 120)
ament_target_dependencies(controller_scenario_tests ${dependencies_pkgs} nav2_controller)
target_link_libraries(controller_scenario_tests mppi_controller mppi_critics yaml-cpp)
target_compile_definitions(controller_scenario_tests PRIVATE
  CONTROLLER_SCENARIO_FILE="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/approach_atlas.json")
