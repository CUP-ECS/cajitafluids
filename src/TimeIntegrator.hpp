/****************************************************************************
 * Copyright (c) 2018-2020 by the ExaMPM authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ExaMPM library. ExaMPM is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITAFLUIDS_TIMEINTEGRATOR_HPP
#define CAJITAFLUIDS_TIMEINTEGRATOR_HPP

#include <BoundaryConditions.hpp>
#include <ProblemManager.hpp>
//#include <VelocityInterpolation.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>

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

//---------------------------------------------------------------------------//
// Advect a field in a divergence-free velocity field
template <std::size_t NumSpaceDims, class ProblemManagerType, class ExecutionSpace, 
          class Entity_t, class Field_t >
void advect(ExecutionSpace &exec_space, ProblemManagerType &pm, double delta_t, 
            const BoundaryCondition<NumSpaceDims> &bc, Entity_t entity, Field_t field)
{
    auto u_current = pm.get(FaceI(), Field::Velocity(), Version::Current() );
    auto v_current = pm.get(FaceJ(), Field::Velocity(), Version::Current() );

    auto field_current = pm.get(entity, field, Version::Current() );
    auto field_next = pm.get(entity, field, Version::Next() );

    auto local_grid = pm.mesh()->localGrid();
    auto local_mesh = *(pm.mesh()->localMesh());

    // XX Only handling 2D for now
    auto owned_items = local_grid->indexSpace( Cajita::Own(), entity, 
                                               Cajita::Local() );
    parallel_for("advection loop", 
        createExecutionPolicy(owned_items, exec_space), 
        KOKKOS_LAMBDA(int i, int j) {
            int idx[2] = {i, j};
            double loc[NumSpaceDims];
            // 1. Get the location of the entity in question
	    local_mesh.coordinates( entity, idx, loc);

            // 2. Trace the location back through the velocity field 
            // rk3(loc, u_current, v_current);

            // 3. Interpolate the value of the advected quantity at that location
            auto new_value = field_current(i, j, 0);

            field_next(i, j, 0) = new_value;
        });
    // Now advect from current to next in the given velocity field
    // XXX
}

//---------------------------------------------------------------------------//
// Take a time step.
template <std::size_t NumSpaceDims, class ProblemManagerType, class ExecutionSpace>
void step( const ExecutionSpace& exec_space, ProblemManagerType& pm,
           const double delta_t, const BoundaryCondition<NumSpaceDims>& bc )
{
    // Get up-to-date copies of the fields being advected and the velocity field
    // into the ghost cells so we can interpolate velocity correctly and retrieve
    // the value being advected into owned cells
    pm.gather( Cell(), Field::Quantity() );
    pm.gather( FaceI(), Field::Velocity() );
    pm.gather( FaceJ(), Field::Velocity() );
    if constexpr (NumSpaceDims == 3) {
        pm.gather( FaceK(), Field::Velocity() );
    }

    // Advect the fields we care about into the next versions of the fields
    advect<NumSpaceDims>(exec_space, pm, delta_t, bc, Cell(), Field::Quantity());
    advect<NumSpaceDims>(exec_space, pm, delta_t, bc, FaceI(), Field::Velocity());
    advect<NumSpaceDims>(exec_space, pm, delta_t, bc, FaceJ(), Field::Velocity());
    if constexpr (NumSpaceDims == 3) {
        advect<NumSpaceDims>(exec_space, pm, delta_t, FaceJ(), Field::Velocity());
    }

    // Once all calculatins with the current versions of the fields (including
    // Velocity!) are done, swap the old values with the new one to finish the 
    // time step.
    pm.advance(Cell(), Field::Quantity());
    pm.advance(FaceI(), Field::Velocity());
    pm.advance(FaceJ(), Field::Velocity());
    if constexpr (NumSpaceDims == 3) {
        pm.advance(FaceK(), Field::Velocity());
    }
}

//---------------------------------------------------------------------------//

} // end namespace TimeIntegrator
} // end namespace ExaMPM

#endif // EXAMPM_TIMEINTEGRATOR_HPP
