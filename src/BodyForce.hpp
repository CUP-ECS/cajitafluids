/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 * 
 * @section DESCRIPTION
 * Inflow Sources Conditions for CajitaFluids
 */

#ifndef CAJITAFLUIDS_BODYFORCE_HPP
#define CAJITAFLUIDS_BODYFORCE_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Mesh.hpp>

#include <Kokkos_Core.hpp>

namespace CajitaFluids {
/**
 * @struct BodyForce
 * @brief Struct that applies the specified force to the underlying mesh
 */
template <std::size_t NumSpaceDim> struct BodyForce;

template <> struct BodyForce<2> {
        template <class ArrayType>
        KOKKOS_INLINE_FUNCTION void operator()( Cajita::Cell, ArrayType &d, 
				                const int index[2], const double location[2], 
						double delta_t, double v) const
        {
        }

	/* Simple forward Euler for body forces */
        template <class ArrayType>
        KOKKOS_INLINE_FUNCTION void operator()( Cajita::Face<Cajita::Dim::I>, ArrayType & ux, 
				                const int index[2], const double location[2], 
						double delta_t, double v) const
        {
	    int i = index[0], j = index[1];
            ux(i, j) += _force[0] * delta_t;
        }

        template <class ArrayType>
        KOKKOS_INLINE_FUNCTION void operator()( Cajita::Face<Cajita::Dim::J>, ArrayType &uy, 
				                const int index[2], const double location[2], 
						double delta_t, double v) const
        {
	    int i = index[0], j = index[1];
            uy(i, j) += _force[1] * delta_t;
        }

        BodyForce(double fx, double fy) { _force[0] = fx; _force[1] = fy; }
        
	std::array<double, 2> _force;      /**< Velocity of inflow quantity */
    };
} // namespace CajitaFluids

#endif
