cmake_minimum_required(VERSION 3.22)
project(behavior_cloning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -------------------------
# Find dependencies
# -------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# -------------------------
# Generate Action interfaces
# -------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
    "action/MoveToJoint.action"
    "action/ExecutePlan.action"
)

# -------------------------
# C++ executable
# -------------------------
add_executable(moveit_action_server
    src/moveit_action_server.cpp
)

ament_target_dependencies(moveit_action_server
    rclcpp
    rclcpp_action
    sensor_msgs
    moveit_ros_planning_interface
)

# Link generated action interfaces
rosidl_target_interfaces(moveit_action_server
    ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

# -------------------------
# Install
# -------------------------
install(TARGETS
    moveit_action_server
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY action
    DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
    behavior_cloning/infer_node.py
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
)

ament_package()