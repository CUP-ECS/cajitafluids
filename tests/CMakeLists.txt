blt_add_executable(NAME tstBoundaryConditions
                   SOURCES tstBoundaryConditions.cpp 
                   INCLUDES tstMeshSetup.hpp
                   DEPENDS_ON cajitafluids gtest)
blt_add_test(NAME BoundaryConditions
             COMMAND tstBoundaryConditions)

#blt_add_executable(NAME tstInflowSource
#                   SOURCES tstInflowSource.cpp
#                   INCLUDES tstMeshSetup.hpp
#                   DEPENDS_ON cajitafluids gtest)
#blt_add_test(tstInflowSource)
