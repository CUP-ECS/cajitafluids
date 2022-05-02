blt_add_executable(NAME tstDriver
                   SOURCES tstDriver.cpp 
                   INCLUDES 
	           tstMesh.hpp 
                   tstProblemManager.hpp
                   tstBoundaryConditions.hpp
                   DEPENDS_ON cajitafluids gtest)
blt_add_test(NAME UnitTests
             COMMAND tstDriver)

#blt_add_executable(NAME tstInflowSource
#                   SOURCES tstInflowSource.cpp
#                   INCLUDES tstMeshSetup.hpp
#                   DEPENDS_ON cajitafluids gtest)
#blt_add_test(tstInflowSource)
