include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(advection advection.cpp)
target_link_libraries( advection PRIVATE cajitafluids)
target_include_directories( advection PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

install(TARGETS advection DESTINATION ${CMAKE_INSTALL_BINDIR})
