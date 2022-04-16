/****************************************************************************
 * Copyright (c) 2022 by the CajitaFluids authors                           *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the CajitaFluids library. CajitaFluids is           *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITAFLUIDS_TIMEINTEGRATOR_HPP
#define CAJITAFLUIDS_TIMEINTEGRATOR_HPP

#include <BoundaryConditions.hpp>
#include <Interpolation.hpp>
#include <ProblemManager.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

namespace CajitaFluids
{

// These routines use state in other classes and don't need state themselves,
// so they're just functions in a namespace, not classes or functors.
namespace TimeIntegrator
{

using Cell = Cajita::Cell;
using FaceI = Cajita::Face<Cajita::Dim::I>;
using FaceJ = Cajita::Face<Cajita::Dim::J>;
using FaceK = Cajita::Face<Cajita::Dim::K>;

template <std::size_t NumSpaceDims, class Mesh_t, class View_t>
KOKKOS_FUNCTION void
rk3( const double x0[NumSpaceDims], const Mesh_t& local_mesh, const View_t& u,
     const View_t& v, double delta_t, double trace[NumSpaceDims] )
{
    double v0[NumSpaceDims], x1[NumSpaceDims], v1[NumSpaceDims],
        x2[NumSpaceDims], v2[NumSpaceDims];

    // Velocity at current location.
    Interpolation::interpolateVelocity<NumSpaceDims, 1>( x0, local_mesh, u, v,
                                                         v0 );

    // Velocity half a timestep back
    x1[0] = x0[0] - 0.5 * delta_t * v0[0];
    x1[1] = x0[1] - 0.5 * delta_t * v0[1];
    if constexpr ( NumSpaceDims == 3 )
        x1[2] = x0[2] - 0.5 * delta_t * v0[2];
    Interpolation::interpolateVelocity<NumSpaceDims, 1>( x1, local_mesh, u, v,
                                                         v1 );

    // Velocity three-quarters of a timestep step back
    x2[0] = x0[0] - 0.75 * delta_t * v0[0];
    x2[1] = x0[1] - 0.75 * delta_t * v0[1];
    if constexpr ( NumSpaceDims == 3 )
        x1[2] = x0[2] - 0.5 * delta_t * v0[2];
    Interpolation::interpolateVelocity<NumSpaceDims, 1>( x2, local_mesh, u, v,
                                                         v2 );

    // Final location after a timestep
    trace[0] =
        x0[0] - delta_t * ( ( 2.0 / 9.0 ) * v0[0] + ( 3.0 / 9.0 ) * v1[0] +
                            ( 4.0 / 9.0 ) * v2[0] );
    trace[1] =
        x0[1] - delta_t * ( ( 2.0 / 9.0 ) * v0[1] + ( 3.0 / 9.0 ) * v1[1] +
                            ( 4.0 / 9.0 ) * v2[1] );
    if constexpr ( NumSpaceDims == 3 )
    {
        trace[2] =
            x0[2] - delta_t * ( ( 2.0 / 9.0 ) * v0[2] + ( 3.0 / 9.0 ) * v1[2] +
                                ( 4.0 / 9.0 ) * v2[2] );
    }
}

//---------------------------------------------------------------------------//
// Advect a field in a divergence-free velocity field
template <std::size_t NumSpaceDims, class ProblemManagerType,
          class ExecutionSpace, class Entity_t, class Field_t>
void advect( ExecutionSpace& exec_space, ProblemManagerType& pm, double delta_t,
             const BoundaryCondition<NumSpaceDims>& bc, Entity_t entity,
             Field_t field )
{
    auto field_current = pm.get( entity, field, Version::Current() );
    auto field_next = pm.get( entity, field, Version::Next() );

    auto u = pm.get( FaceI(), Field::Velocity(), Version::Current() );
    auto v = pm.get( FaceJ(), Field::Velocity(), Version::Current() );
    //    auto w = pm.get(FaceK(), Field::Velocity(), Version::Current());

    auto local_grid = pm.mesh()->localGrid();
    auto local_mesh = *( pm.mesh()->localMesh() );

    auto owned_items =
        local_grid->indexSpace( Cajita::Own(), entity, Cajita::Local() );
    parallel_for(
        "advection loop", createExecutionPolicy( owned_items, exec_space ),
        KOKKOS_LAMBDA( int i, int j ) {
            int idx[2] = { i, j };
            double start[NumSpaceDims], trace[NumSpaceDims];
            // 1. Get the location of the entity in question
            local_mesh.coordinates( entity, idx, start );

            // 2. Trace the location back through the velocity field
            rk3<NumSpaceDims>( start, local_mesh, u, v, delta_t, trace );

            // 3. Interpolate the value of the advected quantity at that
            // location
            field_next( i, j, 0 ) =
                Interpolation::interpolateField<NumSpaceDims, 3, Entity_t>(
                    trace, local_mesh, field_current );
        } );
}

//---------------------------------------------------------------------------//
// Take a time step.
template <std::size_t NumSpaceDims, class ProblemManagerType,
          class ExecutionSpace>
void step( const ExecutionSpace& exec_space, ProblemManagerType& pm,
           const double delta_t, const BoundaryCondition<NumSpaceDims>& bc )
{
    Kokkos::Profiling::pushRegion( "TimeIntegrator::Step" );

    // Get up-to-date copies of the fields being advected and the velocity field
    // into the ghost cells so we can interpolate velocity correctly and
    // retrieve the value being advected into owned cells
    Kokkos::Profiling::pushRegion( "TimeIntegrator::Step::Gather" );
    pm.gather( Version::Current() );
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion( "TimeIntegrator::Step::Advect" );
    // Advect the fields we care about into the next versions of the fields
    Kokkos::Profiling::pushRegion( "TimeIntegrator::Step::Advect::Quantity" );
    advect<NumSpaceDims>( exec_space, pm, delta_t, bc, Cell(),
                          Field::Quantity() );
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion(
        "TimeIntegrator::Step::Advect::Velocity::FaceI" );
    advect<NumSpaceDims>( exec_space, pm, delta_t, bc, FaceI(),
                          Field::Velocity() );
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion(
        "TimeIntegrator::Step::Advect::Velocity::FaceJ" );
    advect<NumSpaceDims>( exec_space, pm, delta_t, bc, FaceJ(),
                          Field::Velocity() );
    Kokkos::Profiling::popRegion();

    if constexpr ( NumSpaceDims == 3 )
    {
        Kokkos::Profiling::pushRegion(
            "TimeIntegrator::Step::Advect::Velocity::FaceK" );
        advect<NumSpaceDims>( exec_space, pm, delta_t, FaceK(),
                              Field::Velocity() );
        Kokkos::Profiling::popRegion();
    }
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion( "CajitaFluids::TimeIntegrator::Advance" );
    // Once all calculations with the current versions of the fields (including
    // Velocity!) are done, swap the old values with the new one to finish the
    // time step.
    pm.advance( Cell(), Field::Quantity() );
    pm.advance( FaceI(), Field::Velocity() );
    pm.advance( FaceJ(), Field::Velocity() );
    if constexpr ( NumSpaceDims == 3 )
    {
        pm.advance( FaceK(), Field::Velocity() );
        Kokkos::Profiling::popRegion();
    }

    Kokkos::Profiling::popRegion();
}

//---------------------------------------------------------------------------//

} // end namespace TimeIntegrator
} // end namespace CajitaFluids

#endif // CAJITAFLUIDS_TIMEINTEGRATOR_HPP
