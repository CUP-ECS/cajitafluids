#include "gtest/gtest.h"

// Include Statements

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <BoundaryConditions.hpp>

#include "tstBoundaryConditions.hpp"
#include "tstDriver.hpp"
#include "tstMesh.hpp"
#include "tstProblemManager.hpp"

TYPED_TEST_SUITE( BoundaryConditionsTest, MeshDeviceTypes );

TYPED_TEST( BoundaryConditionsTest, MeshSolidEdge2D )
{
    // The test fixture will have the mesh and matrix objects we're working
    // with

    // Set up the boundary condition object we're working with - solid edges

    // Apply solid boundary conditions in the I, J, and, if appropriate, K
    // directions

    // Check that the velocity on each edge in the relevant directions are 0

    // ASSERT_EQ(...);
}

TYPED_TEST( BoundaryConditionsTest, MeshFreeEdge2D )
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

TYPED_TEST( BoundaryConditionsTest, MatrixSolidEdge2D )
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

TYPED_TEST( BoundaryConditionsTest, MatrixFreeEdge2D )
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
