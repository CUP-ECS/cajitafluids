set(HEADERS
  BodySource.hpp
  BoundaryConditions.hpp
  InflowSource.hpp
  Interpolation.hpp
  Mesh.hpp
  ProblemManager.hpp
  SiloWriter.hpp
  Solver.hpp
  TimeIntegrator.hpp
  VelocityCorrector.hpp
  )

set(SOURCES
  Mesh.cpp
  )

add_library(cajitafluids ${SOURCES})

target_link_libraries(cajitafluids
  Cabana::cabanacore
  Cabana::Cajita
  Silo::silo )

# Add HYPRE:HYPRE to the above if needed for a cabana-local build

target_include_directories(cajitafluids
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )

install(TARGETS cajitafluids LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
