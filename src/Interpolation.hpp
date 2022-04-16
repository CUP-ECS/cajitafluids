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

#ifndef CAJITAFLUIDS_INTERPOLATION_HPP
#define CAJITAFLUIDS_INTERPOLATION_HPP

#include <Cajita.hpp>
#include <ProblemManager.hpp>

#include <Kokkos_Core.hpp>

namespace CajitaFluids
{
namespace Interpolation
{
using Cell = Cajita::Cell;
using FaceI = Cajita::Face<Cajita::Dim::I>;
using FaceJ = Cajita::Face<Cajita::Dim::J>;
using FaceK = Cajita::Face<Cajita::Dim::K>;

// Spline to interpolate values for advection.
template <std::size_t NumSpaceDims, std::size_t order, class Entity_t,
          class Mesh_t, class View_t>
KOKKOS_INLINE_FUNCTION double interpolateField( const double loc[NumSpaceDims],
                                                const Mesh_t& local_mesh,
                                                const View_t& field )
{
    double value;
    Cajita::SplineData<double, order, NumSpaceDims, Entity_t> spline;
    Cajita::evaluateSpline( local_mesh, loc, spline );
    Cajita::G2P::value( field, spline, value );
    return value;
}

template <std::size_t NumSpaceDims, std::size_t order, class Mesh_t,
          class View_t>
KOKKOS_INLINE_FUNCTION void interpolateVelocity( const double loc[NumSpaceDims],
                                                 const Mesh_t& local_mesh,
                                                 const View_t u, const View_t v,
                                                 double velocity[NumSpaceDims] )
{
    velocity[0] =
        interpolateField<NumSpaceDims, order, FaceI>( loc, local_mesh, u );
    velocity[1] =
        interpolateField<NumSpaceDims, order, FaceJ>( loc, local_mesh, v );
}

} // end namespace Interpolation
} // end namespace CajitaFluids

#endif // end CAJITAFLUIDS_INTERPOLATION_HPP
