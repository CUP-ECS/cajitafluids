set(HEADERS
  BoundaryConditions.hpp
  Mesh.hpp
  ProblemManager.hpp
  Solver.hpp
  )

set(SOURCES
  Mesh.cpp
  ProblemManager.cpp
  )

add_library(cajitafluids ${SOURCES})

if (SILO)
  target_link_libraries(cajitafluids
    Kokkos::kokkos
    MPI::MPI_CXX
    Cabana::cabanacore
    Cabana::Cajita
    ${SILO} )
else ()
  target_link_libraries(cajitafluids
    Kokkos::kokkos
    MPI::MPI_CXX
    Cabana::cabanacore
    Cabana::Cajita
    )
endif()

target_include_directories(cajitafluids
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )
