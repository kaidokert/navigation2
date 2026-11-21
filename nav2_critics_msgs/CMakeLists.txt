cmake_minimum_required(VERSION 3.5)
project(nav2_critics_msgs)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CriticsStats.msg"
  "msg/MppiCycle.msg"
  "msg/MppiRejection.msg"
  "msg/TerminalApproachCheck.msg"
  "msg/TerminalApproachDiagnostics.msg"
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
