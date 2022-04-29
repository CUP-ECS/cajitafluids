#include "gtest/gtest.h"

#include <Mesh.hpp>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include "tstMesh.hpp"

using MeshDeviceTypes = ::testing::Types<
#ifdef KOKKOS_ENABLE_OPENMP
    DeviceType<Kokkos::OpenMP, Kokkos::HostSpace>,
#endif
#ifdef KOKKOS_ENABLE_CUDA
    DeviceType<Kokkos::Cuda, Kokkos::CudaSpace>,
#endif
    DeviceType<Kokkos::Serial, Kokkos::HostSpace> >;

TYPED_TEST_SUITE( MeshTest, MeshDeviceTypes);

TYPED_TEST(MeshTest, MeshParameters)
{
   // The test fixture will have the mesh and matrix objects we're working 
   // with

   // Set up the boundary condition object we're working with - solid edges

   // Apply solid boundary conditions in the I, J, and, if appropriate, K
   // directions

   // Check that the velocity on each edge in the relevant directions are 0

   // ASSERT_EQ(...);
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
