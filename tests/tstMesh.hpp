#ifndef _TSTMESH_HPP_
#define _TSTMESH_HPP_

#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <Mesh.hpp>

#include <mpi.h>

#include "tstDriver.hpp"

/* 
 * Parameterizing on number of dimensions in here is messy and we 
 * don't do it yet. We'll sort that out when we move to 3D as well.
 * These webpage has some ideas on how to I haven't yet deciphered: 
 * 1. http://www.ashermancinelli.com/gtest-type-val-param
 * 2. https://stackoverflow.com/questions/8507385/google-test-is-there-a-way-to-combine-a-test-which-is-both-type-parameterized-a
 */

template <class T> 
class MeshTest : public ::testing::Test {
  // We need Cajita Arrays 
  // Convenience type declarations
  using Cell = Cajita::Cell;
  using FaceI = Cajita::Face<Cajita::Dim::I>;
  using FaceJ = Cajita::Face<Cajita::Dim::J>;
  using FaceK = Cajita::Face<Cajita::Dim::K>;

  using cell_array =
        Cajita::Array<double, Cajita::Cell, Cajita::UniformMesh<double, 2>,
                      typename T::MemorySpace>;
  using iface_array =
        Cajita::Array<double, Cajita::Face<Cajita::Dim::I>,
                      Cajita::UniformMesh<double, 2>, typename T::MemorySpace>;
  using jface_array =
        Cajita::Array<double, Cajita::Face<Cajita::Dim::J>,
                      Cajita::UniformMesh<double, 2>, typename T::MemorySpace>;
  using mesh_type = CajitaFluids::Mesh<2, typename T::ExecutionSpace, typename T::MemorySpace>;

  protected:
    void SetUp() override {
        // Allocate and initialize the Cajita mesh 
        globalBoundingBox_ = { 0, 0, 1.0, 1.0 };
        globalNumCells_ = {512, 512};
	testMesh_ = std::make_unique<mesh_type>( globalBoundingBox_, 
            globalNumCells_, partitioner_, 1, MPI_COMM_WORLD);
    }

    void TearDown() override {
    }

    Cajita::DimBlockPartitioner<2> partitioner_;
    std::array<int, 2> globalNumCells_;
    Kokkos::Array<double, 4> globalBoundingBox_;

    std::unique_ptr<mesh_type> testMesh_;
};

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

#endif // _TSTMESH_HPP_
