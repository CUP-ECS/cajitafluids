#ifndef _TSTPROBLEMMANGER_HPP_
#define _TSTPROBLEMMANGER_HPP_

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>
#include <ProblemManager.hpp>

#include <mpi.h>

#include "tstMesh.hpp"

template <class T> 
class ProblemManagerTest : public MeshTest<T> {
  protected:
    void SetUp() {
        MeshTest<T>::SetUp();
    }

    void TearDown() {
        MeshTest<T>::TearDown();
    }
};

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

#endif // _TSTPROBLEMMANAGER_HPP_
