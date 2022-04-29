blt_add_executable(NAME tstMesh
                   SOURCES tstMesh.cpp 
                   INCLUDES tstMesh.hpp
                   DEPENDS_ON cajitafluids gtest)
blt_add_test(NAME Mesh
             COMMAND tstMesh)

blt_add_executable(NAME tstProblemManager
                   SOURCES tstProblemManager.cpp 
                   INCLUDES tstProblemManager.hpp tstMesh.hpp
                   DEPENDS_ON cajitafluids gtest)
blt_add_test(NAME ProblemManager
             COMMAND tstProblemManager)

blt_add_executable(NAME tstBoundaryConditions
                   SOURCES tstBoundaryConditions.cpp 
                   INCLUDES tstMesh.hpp tstProblemManager.hpp
                   DEPENDS_ON cajitafluids gtest)
blt_add_test(NAME BoundaryConditions
             COMMAND tstBoundaryConditions)

#blt_add_executable(NAME tstInflowSource
#                   SOURCES tstInflowSource.cpp
#                   INCLUDES tstMeshSetup.hpp
#                   DEPENDS_ON cajitafluids gtest)
#blt_add_test(tstInflowSource)
