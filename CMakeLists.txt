cmake_minimum_required(VERSION 3.20)
project(kalman_filter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Turn this ON once kalman.hpp exists and compiles
option(KF_WITH_KALMAN "Build Kalman filter tests and demo" OFF)

add_library(kf INTERFACE)
target_include_directories(kf INTERFACE include)
target_compile_options(kf INTERFACE -Wall -Wextra -Wpedantic -Wconversion -Werror)

# Sanitizers in Debug: catch out-of-bounds, use-after-free, UB early.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(kf INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(kf INTERFACE -fsanitize=address,undefined)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

add_executable(test_matrix tests/test_matrix.cpp)
target_link_libraries(test_matrix PRIVATE kf GTest::gtest_main)
gtest_discover_tests(test_matrix)

if(KF_WITH_KALMAN)
  add_executable(test_kalman tests/test_kalman.cpp)
  target_link_libraries(test_kalman PRIVATE kf GTest::gtest_main)
  gtest_discover_tests(test_kalman)

  add_executable(kf_demo src/main.cpp)
  target_link_libraries(kf_demo PRIVATE kf)
endif()
