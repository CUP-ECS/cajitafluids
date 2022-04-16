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

namespace CajitaFluids
{
/**
 * @struct InflowSource
 * @brief Struct that applies the specified Inflow source to the underlying mesh
 */
template <std::size_t NumSpaceDim>
struct InflowSource;

template <>
struct InflowSource<2>
{
    template <class ArrayType>
    KOKKOS_INLINE_FUNCTION void operator()( Cajita::Cell, ArrayType& q, int i,
                                            int j, double x, double y,
                                            double delta_t, double v ) const
    {
        /* Check if we're inside the bounding box for the inflow XXX */
        if ( x >= _bounding_box[0] && x < _bounding_box[2] &&
             y >= _bounding_box[1] && y < _bounding_box[3] )
        {
            /* This should really be a *rate* of inflow, and it should be
             * interpolated across the volume of intersecting cells, but
             * we're forcing a quantity at the location to start to match
             * the formulation of the original incremental fluids solver */
            if ( q( i, j, 0 ) < _quantity )
                q( i, j, 0 ) = _quantity;
        }
    }

    template <class ArrayType>
    KOKKOS_INLINE_FUNCTION void
    operator()( Cajita::Face<Cajita::Dim::I>, ArrayType& ux, int i, int j,
                double x, double y, double delta_t, double v ) const
    {
        if ( x >= _bounding_box[0] && x < _bounding_box[2] &&
             y >= _bounding_box[1] && y < _bounding_box[3] )
        {
            if ( fabs( ux( i, j, 0 ) ) < fabs( _velocity[0] ) )
                ux( i, j, 0 ) = _velocity[0];
        }
    }

    template <class ArrayType>
    KOKKOS_INLINE_FUNCTION void
    operator()( Cajita::Face<Cajita::Dim::J>, ArrayType& uy, int i, int j,
                double x, double y, double delta_t, double v ) const
    {
        if ( x >= _bounding_box[0] && x < _bounding_box[2] &&
             y >= _bounding_box[1] && y < _bounding_box[3] )
        {
            if ( fabs( uy( i, j, 0 ) ) < fabs( _velocity[1] ) )
                uy( i, j, 0 ) = _velocity[1];
        }
    }

    InflowSource( std::array<double, 2> location, std::array<double, 2> size,
                  std::array<double, 2> velocity, double quantity )
        : _quantity( quantity )
    {
        _bounding_box[0] = location[0];
        _bounding_box[1] = location[1];
        _bounding_box[2] = location[0] + size[0];
        _bounding_box[3] = location[1] + size[1];
        _velocity[0] = velocity[0];
        _velocity[1] = velocity[1];
    }

    Kokkos::Array<double, 4>
        _bounding_box; /**< Bounding box on which inflow happens */
    double _quantity;  /**< Minimum quantity of advected material
                            to force at inflow location XXX change to rate */
    Kokkos::Array<double, 2> _velocity; /**< Minimum velocity of inflow quantity
                                           to force at location*/
};

} // namespace CajitaFluids

#endif
