cmake_minimum_required (VERSION 3.8)

add_library (shim SHARED "record_reader.cc")

include_directories(./inlcude)
include_directories(./inlcude/thrid_party)
include_directories(./ops)
