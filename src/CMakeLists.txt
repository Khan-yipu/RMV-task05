cmake_minimum_required(VERSION 3.8)

project(hik_camera)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# 添加这行 - 查找 cURL 库
find_package(CURL REQUIRED)

add_executable(hik_camera_node source/hik_camera_node.cpp)
target_include_directories(hik_camera_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/hik_camera>
  $<INSTALL_INTERFACE:include>)
target_compile_features(hik_camera_node PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  hik_camera_node
  "rclcpp"
  "sensor_msgs"
  "cv_bridge"
  "OpenCV"
)

# Link against the Hikvision SDK libraries installed in /opt/MVS
set(HIK_SDK_DIR "/opt/MVS")
set(HIK_SDK_INCLUDE_DIR "${HIK_SDK_DIR}/include")
set(HIK_SDK_LIB_DIR "${HIK_SDK_DIR}/lib/64")

if(EXISTS ${HIK_SDK_INCLUDE_DIR} AND EXISTS ${HIK_SDK_LIB_DIR})
    message(STATUS "Found Hikvision SDK in ${HIK_SDK_DIR}")
    target_include_directories(hik_camera_node PUBLIC ${HIK_SDK_INCLUDE_DIR})
    target_link_libraries(hik_camera_node
    ${HIK_SDK_LIB_DIR}/libMvCameraControl.so
    ${HIK_SDK_LIB_DIR}/libMvCameraControlWrapper.so
    /usr/lib/x86_64-linux-gnu/libcurl.so
  )
    # ensure runtime linker can find the .so when running from build tree
    set_target_properties(hik_camera_node PROPERTIES BUILD_RPATH "${HIK_SDK_LIB_DIR}" INSTALL_RPATH "${HIK_SDK_LIB_DIR}")
else()
    message(WARNING "Hikvision SDK not found in ${HIK_SDK_DIR}")
endif()

# ============ 装甲板识别节点 ============
add_executable(armor_detector_node source/armor_detector_node.cpp)
target_include_directories(armor_detector_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(armor_detector_node PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  armor_detector_node
  "rclcpp"
  "sensor_msgs"
  "cv_bridge"
  "OpenCV"
)

# 安装模型文件
# install(DIRECTORY models/
#   DESTINATION share/${PROJECT_NAME}/models
# )

install(TARGETS
  hik_camera_node
  armor_detector_node
  DESTINATION lib/${PROJECT_NAME})

# Install launch files so ros2 launch can find them
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.py"
)

# Install configuration files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)

# 安装RViz配置文件
install(
  FILES config/hik_camera.rviz
  FILES config/armor_detect.rviz
  DESTINATION share/${PROJECT_NAME}/config
)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # comment the line when a copyright and license is added to all source files
    set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # comment the line when this package is in a git repo and when
    # a copyright and license is added to all source files
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()
