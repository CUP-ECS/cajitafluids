#include "gtest/gtest.h"

// Include Statements

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <BoundaryConditions.hpp>

#include "tstProblemManager.hpp"

template <class T> 
class BoundaryConditionsTest : public ProblemManagerTest<T> {
 protected:
  void SetUp() {
    ProblemManagerTest<T>::SetUp();
  }

  void TearDown() {
    ProblemManagerTest<T>::TearDown();
  }
};

using MeshDeviceTypes = ::testing::Types<
#ifdef KOKKOS_ENABLE_OPENMP
    DeviceType<Kokkos::OpenMP, Kokkos::HostSpace>,
#endif
#ifdef KOKKOS_ENABLE_CUDA
    DeviceType<Kokkos::Cuda, Kokkos::CudaSpace>,
#endif
    DeviceType<Kokkos::Serial, Kokkos::HostSpace> >;

TYPED_TEST_SUITE(BoundaryConditionsTest, MeshDeviceTypes);
TYPED_TEST(BoundaryConditionsTest, MeshSolidEdge2D)
{
   // The test fixture will have the mesh and matrix objects we're working 
   // with

   // Set up the boundary condition object we're working with - solid edges

   // Apply solid boundary conditions in the I, J, and, if appropriate, K
   // directions

   // Check that the velocity on each edge in the relevant directions are 0

   // ASSERT_EQ(...);
}

TYPED_TEST(BoundaryConditionsTest, MeshFreeEdge2D)
{
   // The test fixture will have the mesh and matrix objects we're working 
   // with

   // Set up the boundary condition object we're working with - free edges

   // Apply boundary conditions in the I, J, and, if appropriate, K
   // directions

   // Check that the velocity on each edge in every direction are unchanged

   // Test 2 code here...
   // EXPECT_TRUE(...);
}

TYPED_TEST(BoundaryConditionsTest, MatrixSolidEdge2D)
{
   // The test fixture will have the mesh and matrix objects we're working 
   // with

   // Set up the boundary condition object we're working with - solid edges

   // Apply solid boundary conditions in the I, J, and, if appropriate, K
   // directions

   // Check that the matrix values changes appropriately.

   // Test 3 code here...
   // EXPECT_TRUE(...);
}

TYPED_TEST(BoundaryConditionsTest, MatrixFreeEdge2D)
{
   // The test fixture will have the mesh and matrix objects we're working 
   // with

   // Set up the boundary condition object we're working with - free edges

   // Apply solid boundary conditions in the I, J, and, if appropriate, K
   // directions

   // Check that the matrix values changes appropriately.

   // Test 4 code here...
   // EXPECT_TRUE(...);
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );
    ::testing::InitGoogleTest( &argc, argv );
    int return_val = RUN_ALL_TESTS();
    Kokkos::finalize();
    MPI_Finalize();
    return return_val;
}
