set(TEST_NAMES
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
  set_tests_properties(${name} PROPERTIES LABELS "gtest;office_regression" TIMEOUT 120)

  if(${TEST_DEBUG_INFO})
    target_compile_definitions(${name} PUBLIC -DTEST_DEBUG_INFO)
  endif()

endforeach()
target_sources(optimizer_unit_tests PRIVATE early_cusp_family_test.cpp)
set_tests_properties(optimizer_unit_tests PROPERTIES ENVIRONMENT
  "GTEST_FILTER=-OptimizerTests.ExactFinalCollisionCaptureIsOptionalAndDoesNotAlterCommands:OptimizerTests.FailedBatchCaptureOccursBeforeControlUpdateAndPreservesFailure")

# This is a special case requiring linking against the critics library
ament_add_gtest(critics_tests critics_tests.cpp)
ament_target_dependencies(critics_tests ${dependencies_pkgs})
target_link_libraries(critics_tests mppi_controller mppi_critics)
set_tests_properties(critics_tests PROPERTIES LABELS "gtest;office_regression" TIMEOUT 120)
if(${TEST_DEBUG_INFO})
  target_compile_definitions(critics_tests PUBLIC -DTEST_DEBUG_INFO)
endif()
