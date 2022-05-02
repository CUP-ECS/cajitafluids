#include "gtest/gtest.h"

// Include Statements

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <BoundaryConditions.hpp>

#include "tstDriver.hpp"
#include "tstMesh.hpp"
#include "tstProblemManager.hpp"
#include "tstBoundaryConditions.hpp"

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
