#include "gtest/gtest.h"

// Include Statements
#include <BoundaryConditions.hpp>
#include <Solver.hpp>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#if DEBUG
#include <iostream>
#endif


class BoundaryConditionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
  }

  void TearDown() override {
  }

  // The mesh and matrix the boundary conditions will be applied to.
};

TEST_F(BoundaryConditionsTest, MeshSolidEdge2D)
{
   // The test fixture will have the mesh and matrix objects we're working 
   // with

   // Set up the boundary condition object we're working with - solid edges

   // Apply solid boundary conditions in the I, J, and, if appropriate, K
   // directions

   // Check that the velocity on each edge in the relevant directions are 0

   // ASSERT_EQ(...);
}

TEST_F(BoundaryConditionsTest, MeshFreeEdge2D)
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

TEST_F(BoundaryConditionsTest, MatrixSolidEdge2D)
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

TEST_F(BoundaryConditionsTest, MatrixFreeEdge2D)
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
