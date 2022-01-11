/****************************************************************************
 * Copyright (c) 2021 by the CajitaFluids authors                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the CajitaFluids benchmark. CajitaFluids is         *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITAFLUIDS_MESH_HPP
#define CAJITAFLUIDS_MESH_HPP

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <memory>

#include <mpi.h>

#include <limits>

namespace CajitaFluids
{
//---------------------------------------------------------------------------//
/*!
  \class Mesh
  \brief Logically and spatially uniform Cartesian mesh.
*/
template <int Dim, class MemorySpace>
class Mesh
{
  public:
    using memory_space = MemorySpace;

    // Construct a mesh.
    Mesh( const Kokkos::Array<double, 2*Dim>& global_bounding_box,
          const std::array<int, Dim>& global_num_cell,
          const std::array<bool, Dim>& periodic,
          const Cajita::BlockPartitioner<Dim>& partitioner,
          const int halo_cell_width, const int minimum_halo_cell_width,
          MPI_Comm comm )
    {
        // Make a copy of the global number of cells so we can modify it.
        std::array<int, Dim> num_cell = global_num_cell;

        // Compute the cell size.
        double cell_size =
            ( global_bounding_box[Dim] - global_bounding_box[0] ) / num_cell[0];

        // Because the mesh is uniform check that the domain is evenly
        // divisible by the cell size in each dimension within round-off
        // error. This will let us do cheaper math for particle location.
        for ( int d = 0; d < Dim; ++d )
        {
            double extent = num_cell[d] * cell_size;
            if ( std::abs( extent - ( global_bounding_box[d + Dim] -
                                      global_bounding_box[d] ) ) >
                 double( 10.0 ) * std::numeric_limits<double>::epsilon() )
                throw std::logic_error(
                    "Extent not evenly divisible by uniform cell size" );
        }

        // Create global mesh bounds.
        std::array<double, Dim> global_low_corner, global_high_corner;
        for ( int d = 0; d < Dim; ++d ) {
            global_low_corner[d] = global_bounding_box[d];
            global_high_corner[d] = global_bounding_box[d + Dim];
        }

        for ( int d = 0; d < Dim; ++d )
        {
            _min_domain_global_node_index[d] = 0;
            _max_domain_global_node_index[d] = num_cell[d] + 1;
        }

        // For dimensions that are not periodic we pad by the minimum halo
        // cell width to allow for projections outside of the domain.
        for ( int d = 0; d < Dim; ++d )
        {
            if ( !periodic[d] )
            {
                global_low_corner[d] -= cell_size * minimum_halo_cell_width;
                global_high_corner[d] += cell_size * minimum_halo_cell_width;
                num_cell[d] += 2 * minimum_halo_cell_width;
                _min_domain_global_node_index[d] += minimum_halo_cell_width;
                _max_domain_global_node_index[d] -= minimum_halo_cell_width;
            }
        }

        // Create the global mesh.
        auto global_mesh = Cajita::createUniformGlobalMesh(
            global_low_corner, global_high_corner, num_cell );

        // Build the global grid.
        auto global_grid = Cajita::createGlobalGrid( comm, global_mesh,
                                                     periodic, partitioner );

        // Build the local grid.
        int halo_width = std::max( minimum_halo_cell_width, halo_cell_width );
        _local_grid = Cajita::createLocalGrid( global_grid, halo_width );
    }

    // Get the local grid.
    const std::shared_ptr<Cajita::LocalGrid<Cajita::UniformMesh<double, Dim>>>&
    localGrid() const
    {
        return _local_grid;
    }

    // Get the cell size.
    double cellSize() const
    {
        return _local_grid->globalGrid().globalMesh().cellSize( 0 );
    }

    // Get the minimum node index in the domain.
    Kokkos::Array<int, Dim> minDomainGlobalNodeIndex() const
    {
        return _min_domain_global_node_index;
    }

    // Get the maximum node index in the domain.
    Kokkos::Array<int, Dim> maxDomainGlobalNodeIndex() const
    {
        return _max_domain_global_node_index;
    }

  public:
    std::shared_ptr<Cajita::LocalGrid<Cajita::UniformMesh<double, Dim>>> _local_grid;

    Kokkos::Array<int, Dim> _min_domain_global_node_index;
    Kokkos::Array<int, Dim> _max_domain_global_node_index;
};

//---------------------------------------------------------------------------//

} // end namespace CajitaFluids

#endif // end CAJITAFLUIDS_MESH_HPP
