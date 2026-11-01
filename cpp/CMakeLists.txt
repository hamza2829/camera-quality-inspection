cmake_minimum_required(VERSION 3.20)
project(camera_inspection LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(inspection src/inspection.cpp)
target_include_directories(inspection PUBLIC include)
add_executable(inspection_cli src/main.cpp)
target_link_libraries(inspection_cli PRIVATE inspection)

include(CTest)
if(BUILD_TESTING)
  include(FetchContent)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/52eb8108c5bdec04579160ae17225d66034bd723.tar.gz
    URL_HASH SHA256=745c55415660044610f7fcd3af7a6420d5de16a7dbb9ebfe2e131275676232be
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(googletest)
  add_executable(inspection_tests tests/test_inspection.cpp)
  target_link_libraries(inspection_tests PRIVATE inspection GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(inspection_tests)
endif()
