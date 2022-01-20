/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 * 
 * @section DESCRIPTION
 * Inflow Sources Conditions for CajitaFluids
 */

#ifndef CAJITAFLUIDS_INFLOWSOURCE_HPP
#define CAJITAFLUIDS_INFLOWSOURCE_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Mesh.hpp>

#include <Kokkos_Core.hpp>

namespace CajitaFluids {
/**
 * @struct InflowSource
 * @brief Struct that applies the specified Inflow source to the underlying mesh
 */
template <std::size_t NumSpaceDim> struct InflowSource;

template <> struct InflowSource<2> {
        template <class ArrayType>
        KOKKOS_INLINE_FUNCTION void operator()( Cajita::Cell, ArrayType &q, 
				                const int index[2], const double location[2], 
						double delta_t, double v) const
        {
	    int i = index[0], j = index[1];
            /* Check if we're inside the bounding box for the inflow XXX */
	    if (location[0] >= _bounding_box[0] 
	        && location[0] < _bounding_box[2] 
		&& location[1] < _bounding_box[1] 
		&& location[1] < _bounding_box[3]) {
		/* THis should really be a *rate* of inflow, but we're forcing a quantity at
		 * the location to start match the formulation of the original incremental
		 * fluids solver */
		if ( q(i, j) < _quantity ) q(i, j) = _quantity;
	    }
        }

        template <class ArrayType>
        KOKKOS_INLINE_FUNCTION void operator()( Cajita::Face<Cajita::Dim::I>, 
						ArrayType &ux, const int index[2], 
						const double location[2], 
						double delta_t, double v) const
        {
	    int i = index[0], j = index[1];
	    if (location[0] >= _bounding_box[0] 
		&& location[0] < _bounding_box[2] 
		&& location[1] < _bounding_box[1] 
		&& location[1] < _bounding_box[3]) {
		if (ux(i, j) < _velocity[0]) ux(i,j) = _velocity[0];
            }
        }

        template <class ArrayType>
        KOKKOS_INLINE_FUNCTION void operator()( Cajita::Face<Cajita::Dim::J>, ArrayType &uy, 
				                const int index[2], const double location[2], 
						double delta_t, double v) const
        {
	    int i = index[0], j = index[1];
	    if (location[0] >= _bounding_box[0] 
		&& location[0] < _bounding_box[2] 
		&& location[1] < _bounding_box[1] 
		&& location[1] < _bounding_box[3]) {
		if (uy(i, j) < _velocity[1]) uy(i,j) = _velocity[1];
            }
        }
 
        InflowSource(std::array<double, 2> location, std::array<double, 2> size, 
                     std::array<double, 2> velocity, double quantity)
	    : _quantity(quantity)
        {
	    _bounding_box[0] = location[0];
	    _bounding_box[1] = location[1];
	    _bounding_box[2] = location[0] + size[0];
	    _bounding_box[3] = location[1] + size[1];
	    _velocity[0] = velocity[0];
	    _velocity[1] = velocity[1];
        }

        Kokkos::Array<double, 4> _bounding_box; /**< Bounding box on which inflow happens */
        double _quantity;		       /**< Minimum quantity of advected material 
						    to force at inflow location XXX change to rate */
	Kokkos::Array<double, 2> _velocity;      /**< Minimum velocity of inflow quantity to force at location*/
    };

} // namespace CajitaFluids

#endif
