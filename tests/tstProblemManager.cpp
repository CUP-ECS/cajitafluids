#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>
#include <ProblemManager.hpp>

#include <mpi.h>

#include "tstDriver.hpp"
#include "tstMesh.hpp"
#include "tstProblemManager.hpp"

TYPED_TEST_SUITE( ProblemManagerTest, MeshDeviceTypes);

TYPED_TEST(ProblemManagerTest, StateArrayTests)
{
   // The test fixture will have the mesh and matrix objects we're working 
   // with

   // Set up the boundary condition object we're working with - solid edges

   // Apply solid boundary conditions in the I, J, and, if appropriate, K
   // directions

   // Check that the velocity on each edge in the relevant directions are 0

   // ASSERT_EQ(...);
}
