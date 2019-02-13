cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# Define project and related variables
project (tensorflow)
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)

include_directories(./inlcude)
include_directories(./inlcude/thrid_party)
include_directories(./ops)

# Include directories


add_subdirectory ("shim")
