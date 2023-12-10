set(HEADERS
  BodyForce.hpp
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

set(DEPENDS_ON
  Cabana::Core
  Cabana::Grid
  SILO::silo 
  mpi
  )
 
set(INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )
       
blt_add_library( NAME cabanafluids
                 HEADERS ${HEADERS}
                 SOURCES ${SOURCES}
                 INCLUDES ${INCLUDES}
                 DEPENDS_ON ${DEPENDS_ON} 
  )
