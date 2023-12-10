/****************************************************************************
 * Copyright (c) 2022 by the CabanaFluids authors                           *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the CabanaFluids library. CabanaFluids is           *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANAFLUIDS_INTERPOLATION_HPP
#define CABANAFLUIDS_INTERPOLATION_HPP


#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <ProblemManager.hpp>

namespace CabanaFluids
{
namespace Interpolation
{
using Cell = Cabana::Grid::Cell;
using FaceI = Cabana::Grid::Face<Cabana::Grid::Dim::I>;
using FaceJ = Cabana::Grid::Face<Cabana::Grid::Dim::J>;
using FaceK = Cabana::Grid::Face<Cabana::Grid::Dim::K>;

// Spline to interpolate values for advection.
template <std::size_t NumSpaceDims, std::size_t order, class Entity_t,
          class Mesh_t, class View_t>
KOKKOS_INLINE_FUNCTION double interpolateField( const double loc[NumSpaceDims],
                                                const Mesh_t& local_mesh,
                                                const View_t& field )
{
    double value;
    Cabana::Grid::SplineData<double, order, NumSpaceDims, Entity_t> spline;
    Cabana::Grid::evaluateSpline( local_mesh, loc, spline );
    Cabana::Grid::G2P::value( field, spline, value );
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
} // end namespace CabanaFluids

#endif // end CABANAFLUIDS_INTERPOLATION_HPP
