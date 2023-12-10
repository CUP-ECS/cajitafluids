/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 *
 * @section DESCRIPTION
 * Boundary Conditions for Cabana Fluid Solver
 */

#ifndef CABANAFLUIDS_BOUNDARY_CONDITIONS_HPP
#define CABANAFLUIDS_BOUNDARY_CONDITIONS_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Mesh.hpp>

#include <Kokkos_Core.hpp>

namespace CabanaFluids
{
/**
 * @struct BoundaryType
 * @brief Struct which contains enums of boundary type options for each boundary
 * In the current system, these are only used for setting up the matrix in the
 * pressure solve.
 */
struct BoundaryType
{
    enum Values
    {
        SOLID = 0,
        FREE = 1,
    };
};

/**
 * @struct BoundaryCondition
 * @brief Struct that applies the specified boundary conditions with
 * Kokkos inline Functions
 */
template <std::size_t NumSpaceDim>
struct BoundaryCondition;

// Note that we loop over teh entire domain here so that later code, which might
// have geometry inside the domain, could use the same approach. This code is
// all cell-centric, not face centric for the moment.
template <>
struct BoundaryCondition<2>
{
    using Cell = Cabana::Grid::Cell;
    using FaceI = Cabana::Grid::Face<Cabana::Grid::Dim::I>;
    using FaceJ = Cabana::Grid::Face<Cabana::Grid::Dim::J>;
    template <class ArrayType>
    KOKKOS_INLINE_FUNCTION void
    build_matrix( const int gi, const int gj, const int i, const int j,
                  ArrayType& matrix, const double scale ) const
    {
        // Correct the equations used in the pressure solve for the boundary
        // conditions. Assumes he matrix has been initialized as a full
        // stencil and we need to adjust the weights for the boundaries.

        if ( gi <= min[0] )
        { // Left Boundary
            matrix( i, j, 1 ) = 0;
            if ( boundary_type[0] == BoundaryType::SOLID )
            {
                matrix( i, j, 0 ) -= scale;
            }
        }
        if ( gi > max[0] - 1 )
        { // Right Boundary
            matrix( i, j, 2 ) = 0;
            if ( boundary_type[2] == BoundaryType::SOLID )
            {
                matrix( i, j, 0 ) -= scale;
            }
        }
        if ( gj <= min[1] )
        { // Bottom Boundary
            matrix( i, j, 3 ) = 0;
            if ( boundary_type[1] == BoundaryType::SOLID )
            {
                matrix( i, j, 0 ) -= scale;
            }
        }
        if ( gj > max[1] - 1 )
        { // Top Boundary
            matrix( i, j, 4 ) = 0;
            if ( boundary_type[3] == BoundaryType::SOLID )
            {
                matrix( i, j, 0 ) -= scale;
            }
        }
    }

    // The functor operator applies velocity boundary conditions. Note that the
    // maxes in the bounding box are in terms of global *cell* indexes, not face
    // indexes. As result the comparisons end up being slighty different.
    template <class UType>
    KOKKOS_INLINE_FUNCTION void operator()( FaceI, UType& u, const int gi,
                                            [[maybe_unused]] const int gj,
                                            const int i, const int j ) const
    {
        if ( ( gi <= min[0] ) && ( boundary_type[0] == BoundaryType::SOLID ) )
        {
            u( i, j, 0 ) = 0;
        }
        if ( ( gi > max[0] ) && ( boundary_type[2] == BoundaryType::SOLID ) )
        {
            u( i, j, 0 ) = 0;
        }
    }
    template <class VType>
    KOKKOS_INLINE_FUNCTION void
    operator()( FaceJ, VType& v, [[maybe_unused]] const int gi, const int gj,
                const int i, const int j ) const
    {
        if ( ( gj <= min[1] ) && ( boundary_type[1] == BoundaryType::SOLID ) )
        {
            v( i, j, 0 ) = 0;
        }
        if ( ( gj > max[1] ) && ( boundary_type[3] == BoundaryType::SOLID ) )
        {
            v( i, j, 0 ) = 0;
        }
    }

    Kokkos::Array<int, 4>
        boundary_type; /**< Boundary condition type on all walls  */
    Kokkos::Array<int, 2> min;
    Kokkos::Array<int, 2> max;
};

} // namespace CabanaFluids

#endif
