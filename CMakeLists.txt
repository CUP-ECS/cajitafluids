# project settings
cmake_minimum_required(VERSION 3.12)

project(CajitaFluids LANGUAGES C CXX VERSION 0.1.0)

include(GNUInstallDirs)

# find dependencies
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# If I want to force it to use a local cabana repo for debugging
#set(Cabana_REQUIRE_MPI ON CACHE BOOL "" FORCE)
#set(Cabana_REQUIRE_HYPRE ON CACHE BOOL "" FORCE)
#add_subdirectory(Cabana)
# find_package(HYPRE)

find_package(Cabana REQUIRED COMPONENTS Cabana::Cajita Cabana::cabanacore)
if( NOT Cabana_ENABLE_MPI )
  message( FATAL_ERROR "Cabana must be compiled with MPI" )
endif()
if( NOT Cabana_ENABLE_CAJITA )
  message( FATAL_ERROR "Cabana must be compiled with Cajita" )
endif()
find_package(Silo REQUIRED)

# find Clang Format
find_package( CLANG_FORMAT 10 )

# library
add_subdirectory(src)


# examples
add_subdirectory(examples)
