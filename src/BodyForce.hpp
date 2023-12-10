/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 *
 * @section DESCRIPTION
 * Inflow Sources Conditions for CabanaFluids
 */

#ifndef CABANAFLUIDS_BODYFORCE_HPP
#define CABANAFLUIDS_BODYFORCE_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Mesh.hpp>

#include <Kokkos_Core.hpp>

namespace CabanaFluids
{
/**
 * @struct BodyForce
 * @brief Struct that applies the specified force to the underlying mesh
 */
template <std::size_t NumSpaceDim>
struct BodyForce;

template <>
struct BodyForce<2>
{
    template <class ArrayType>
    KOKKOS_INLINE_FUNCTION void
    operator()( Cabana::Grid::Cell, [[maybe_unused]] ArrayType& d,
                [[maybe_unused]] int i, [[maybe_unused]] int j,
                [[maybe_unused]] double x, [[maybe_unused]] double y,
                [[maybe_unused]] double delta_t,
                [[maybe_unused]] double v ) const
    {
    }

    /* Simple forward Euler for body forces */
    template <class ArrayType>
    KOKKOS_INLINE_FUNCTION void
    operator()( Cabana::Grid::Face<Cabana::Grid::Dim::I>, ArrayType& ux, int i, int j,
                [[maybe_unused]] double x, [[maybe_unused]] double y,
                double delta_t, [[maybe_unused]] double v ) const
    {
        ux( i, j, 0 ) += _force[0] * delta_t;
    }

    template <class ArrayType>
    KOKKOS_INLINE_FUNCTION void
    operator()( Cabana::Grid::Face<Cabana::Grid::Dim::J>, ArrayType& uy, int i, int j,
                [[maybe_unused]] double x, [[maybe_unused]] double y,
                double delta_t, [[maybe_unused]] double v ) const
    {
        uy( i, j, 0 ) += _force[1] * delta_t;
    }

    BodyForce( double fx, double fy )
    {
        _force[0] = fx;
        _force[1] = fy;
    }

    Kokkos::Array<double, 2> _force; /**< Force exerted on all cells. */
};
} // namespace CabanaFluids

#endif
