#ifndef _TSTMESH_HPP_
#define _TSTMESH_HPP_

#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <Mesh.hpp>

#include <mpi.h>

#include "tstDriver.hpp"

/*
 * Parameterizing on number of dimensions in here is messy and we
 * don't do it yet. We'll sort that out when we move to 3D as well.
 * These webpage has some ideas on how to I haven't yet deciphered:
 * 1. http://www.ashermancinelli.com/gtest-type-val-param
 * 2.
 * https://stackoverflow.com/questions/8507385/google-test-is-there-a-way-to-combine-a-test-which-is-both-type-parameterized-a
 */

template <class T>
class MeshTest : public ::testing::Test
{
    // We need Cabana Grid Arrays
    // Convenience type declarations
    using Cell = Cabana::Grid::Cell;

    using cell_array =
        Cabana::Grid::Array<double, Cabana::Grid::Cell, Cabana::Grid::UniformMesh<double, 2>,
                      typename T::MemorySpace>;
    using iface_array =
        Cabana::Grid::Array<double, Cabana::Grid::Face<Cabana::Grid::Dim::I>,
                      Cabana::Grid::UniformMesh<double, 2>, typename T::MemorySpace>;
    using jface_array =
        Cabana::Grid::Array<double, Cabana::Grid::Face<Cabana::Grid::Dim::J>,
                      Cabana::Grid::UniformMesh<double, 2>, typename T::MemorySpace>;
    using mesh_type = Cabana::Grid::Mesh<2, typename T::ExecutionSpace,
                                         typename T::MemorySpace>;

  protected:
    const double boxWidth_ = 1.0;
    const int haloWidth_ = 3;
    const int boxCells_ = 512;

    virtual void SetUp() override
    {
        // Allocate and initialize the Cabana::Grid mesh
        globalBoundingBox_ = { 0, 0, boxWidth_, boxWidth_ };
        globalNumCells_ = { boxCells_, boxCells_ };
        testMesh_ = std::make_shared<mesh_type>( globalBoundingBox_,
                                                 globalNumCells_, partitioner_,
                                                 haloWidth_, MPI_COMM_WORLD );
    }

    virtual void TearDown() override { testMesh_ = NULL; }

    Cabana::Grid::DimBlockPartitioner<2> partitioner_;
    std::array<int, 2> globalNumCells_;
    Kokkos::Array<double, 4> globalBoundingBox_;

    std::shared_ptr<mesh_type> testMesh_;
};

#endif // _TSTMESH_HPP_
