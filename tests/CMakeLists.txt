blt_add_executable(NAME tstMesh
                   SOURCES tstMesh.cpp 
                   INCLUDES tstMesh.hpp 
                   DEPENDS_ON cabanafluids gtest)
blt_add_test(NAME MeshTests
             COMMAND tstMesh)

blt_add_executable(NAME tstProblemManager
                   SOURCES tstProblemManager.cpp 
                   INCLUDES tstProblemManager.hpp tstMesh.hpp 
                   DEPENDS_ON cabanafluids gtest)
blt_add_test(NAME ProblemManagerTests
             COMMAND tstProblemManager)

blt_add_executable(NAME tstBoundaryConditions
                   SOURCES tstBoundaryConditions.cpp 
                   INCLUDES tstoundaryConditions.hpp tstProblemManager.hpp tstMesh.hpp 
                   DEPENDS_ON cabanafluids gtest)
blt_add_test(NAME BoundaryConditionTests
             COMMAND tstBoundaryConditions)

#blt_add_executable(NAME tstInflowSource
#                   SOURCES tstInflowSource.cpp
#                   INCLUDES tstMeshSetup.hpp
#                   DEPENDS_ON cajitafluids gtest)
#blt_add_test(tstInflowSource)
