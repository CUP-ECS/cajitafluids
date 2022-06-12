#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <Mesh.hpp>

#include <mpi.h>

#include "tstDriver.hpp"
#include "tstMesh.hpp"

TYPED_TEST_SUITE( MeshTest, MeshDeviceTypes );

TYPED_TEST( MeshTest, BasicParameters )
{
    int r;
    EXPECT_EQ( this->testMesh_->cellSize(), this->boxWidth_ / this->boxCells_ );

    auto mins = this->testMesh_->minDomainGlobalCellIndex();
    EXPECT_EQ( mins[0], 0 );
    EXPECT_EQ( mins[1], 0 );
    auto maxs = this->testMesh_->maxDomainGlobalCellIndex();
    EXPECT_EQ( maxs[0], this->boxCells_ - 1 );
    EXPECT_EQ( maxs[1], this->boxCells_ - 1 );

    MPI_Comm_rank( MPI_COMM_WORLD, &r );
    EXPECT_EQ( this->testMesh_->rank(), r );
}

TYPED_TEST( MeshTest, LocalGridSetup )
{
    /* Here we check that the local grid is decomposed like
     * we think it should be. That is, the number of ghosts cells
     * is right, the index spaces for owned, ghost, and boundary
     * cells are right, and so on. */
    auto local_grid = this->testMesh_->localGrid();
    auto global_grid = local_grid->globalGrid();
    for ( int i = 0; i < 2; i++ )
    {
        EXPECT_EQ( this->boxCells_,
                   global_grid.globalNumEntity( Cajita::Cell(), i ) );
    }

    /* Make sure the number of owned cells is our share of what was requested */
    auto own_local_cell_space = local_grid->indexSpace(
        Cajita::Own(), Cajita::Cell(), Cajita::Local() );
    for ( int i = 0; i < 2; i++ )
    {
        EXPECT_EQ( own_local_cell_space.extent( i ),
                   this->boxCells_ / global_grid.dimNumBlock( i ) );
    }

    /*
     * Next we extract the ghosted faces, which encompass the owned faces and
     * the ghosts in each dimension. Note that for faces there is a template
     * dimension parameter - there is a * separate index space for each
     * spatial dimension.
     */
    auto ghost_local_face_space = local_grid->indexSpace(
        Cajita::Ghost(), Cajita::Face<Cajita::Dim::I>(), Cajita::Local() );
    EXPECT_EQ( ghost_local_face_space.extent( 0 ),
               this->boxCells_ / global_grid.dimNumBlock( 1 ) +
                   2 * this->haloWidth_ + 1 );
    EXPECT_EQ( ghost_local_face_space.extent( 1 ),
               this->boxCells_ / global_grid.dimNumBlock( 1 ) +
                   2 * this->haloWidth_ );
}
